"""Classify one UEIKAP chip against footprints from one admitted official catalogue."""

from collections import defaultdict
from datetime import datetime
from math import fsum, isfinite, radians, sin
import re

from shapely.geometry import box, shape
from shapely.ops import unary_union


STAMP = r"[0-9]{8}T[0-9]{6}"
MISSION = r"S1[ABCD]"
PRODUCT = re.compile(
    rf"(?P<stem>(?P<mission>{MISSION})_IW_GRDH_1S(?:DH|DV|SH|SV)_"
    rf"(?P<start>{STAMP})_(?P<end>{STAMP})_(?P<orbit>[0-9]{{6}})_"
    r"(?P<take>[0-9A-F]{6})_[0-9A-F]{4})(?:_COG|-GRD_HD)?(?:\.SAFE)?"
)
PROXY = re.compile(rf"(?P<mission>{MISSION})_(?P<start>{STAMP})")
FAMILY = re.compile(rf"{MISSION}_[0-9]{{6}}_[0-9A-F]{{6}}")
COVERAGE = .95
WINDOW_SECONDS = 120


def parse_time(value):
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%S")
    except ValueError as exc:
        raise ValueError(f"Invalid Sentinel-1 date or time: {value}") from exc


def number(value):
    return type(value) is int or (type(value) is float and isfinite(value))


def validate_ring(ring):
    if not isinstance(ring, (list, tuple)) or len(ring) < 4:
        raise ValueError("Invalid footprint ring")
    for point in ring:
        if (not isinstance(point, (list, tuple)) or len(point) != 2
                or not all(number(value) for value in point)
                or not -180 <= point[0] <= 180 or not -90 <= point[1] <= 90):
            raise ValueError("Out-of-range or invalid WGS84 footprint coordinate")
    if tuple(ring[0]) != tuple(ring[-1]):
        raise ValueError("Invalid unclosed GeoJSON footprint ring")
    if any(abs(right[0] - left[0]) >= 180 for left, right in zip(ring, ring[1:])):
        raise ValueError("Antimeridian-spanning footprint ring")


def footprint(geojson):
    if not isinstance(geojson, dict) or geojson.get("type") not in ("Polygon", "MultiPolygon"):
        raise ValueError("Expected a GeoJSON Polygon or MultiPolygon footprint")
    coordinates = geojson.get("coordinates")
    polygons = [coordinates] if geojson["type"] == "Polygon" else coordinates
    if not isinstance(polygons, (list, tuple)) or not polygons:
        raise ValueError("Empty footprint")
    for rings in polygons:
        if not isinstance(rings, (list, tuple)) or not rings:
            raise ValueError("Empty footprint polygon")
        for ring in rings:
            validate_ring(ring)
    geometry = shape(geojson)
    if geometry.is_empty or not geometry.is_valid or geometry.area <= 0:
        raise ValueError("Invalid or empty footprint geometry")
    west, _, east, _ = geometry.bounds
    if east - west >= 180:
        raise ValueError("Antimeridian-spanning or nonlocal footprint")
    return geometry


def spherical_ring_area(coordinates):
    points = list(coordinates)
    reference = sin(radians(points[0][1]))
    terms = []
    for (lon0, lat0), (lon1, lat1) in zip(points, points[1:]):
        half_delta = radians(lat1 - lat0) / 2
        sinc = sin(half_delta) / half_delta if half_delta else 1.
        mean_sin = sin(radians((lat0 + lat1) / 2)) * sinc
        terms.append(radians(lon1 - lon0) * (mean_sin - reference))
    return abs(fsum(terms))


def spherical_area(geometry):
    """Unit-sphere area for edges straight in lon/lat, after planar clipping."""
    if geometry.is_empty:
        return 0.
    if geometry.geom_type == "Polygon":
        return spherical_ring_area(geometry.exterior.coords) - fsum(
            spherical_ring_area(hole.coords) for hole in geometry.interiors
        )
    if geometry.geom_type in ("MultiPolygon", "GeometryCollection"):
        return fsum(spherical_area(part) for part in geometry.geoms)
    return 0.


def classify_chip(bounds, timestamp_proxy, products, reserved):
    """Return overlap evidence for one source without choosing or promoting a family."""
    if (not isinstance(bounds, (list, tuple)) or len(bounds) != 4 or not all(number(v) for v in bounds)):
        raise ValueError("Expected four finite WGS84 chip bounds")
    west, south, east, north = bounds
    if not (-180 <= west < east <= 180 and east - west < 180 and -90 <= south < north <= 90):
        raise ValueError("Out-of-range, inverted or antimeridian-spanning chip bounds")
    proxy = PROXY.fullmatch(timestamp_proxy) if isinstance(timestamp_proxy, str) else None
    if proxy is None:
        raise ValueError("Expected Sentinel-1 timestamp proxy")
    timestamp = parse_time(proxy["start"])
    if not isinstance(reserved, (set, frozenset)) or any(
        not isinstance(family, str) or FAMILY.fullmatch(family) is None for family in reserved
    ):
        raise ValueError("Expected reserved Sentinel-1 acquisition-family set")

    chip = box(west, south, east, north)
    chip_spherical_area = spherical_area(chip)
    if chip_spherical_area <= 0:
        raise ValueError("Chip has no measurable spherical area")
    geometries, families = {}, defaultdict(dict)
    primary_products, primary_families = [], set()
    for name, geojson in products:
        parsed = PRODUCT.fullmatch(name) if isinstance(name, str) else None
        if parsed is None:
            raise ValueError(f"Invalid full Sentinel-1 IW GRDH product name: {name!r}")
        start = parse_time(parsed["start"])
        if parse_time(parsed["end"]) < start:
            raise ValueError(f"Product ends before its acquisition starts: {name}")
        try:
            geometry = footprint(geojson)
        except ValueError as exc:
            raise ValueError(f"Invalid footprint for {name}: {exc}") from exc
        stem = parsed["stem"]
        if stem in geometries:
            if not geometry.equals(geometries[stem]):
                raise ValueError(f"Conflicting footprints for one product: {stem}")
            continue
        geometries[stem] = geometry
        seconds = (start - timestamp).total_seconds()
        if parsed["mission"] != proxy["mission"] or abs(seconds) > WINDOW_SECONDS:
            continue
        family = "_".join((parsed["mission"], parsed["orbit"], parsed["take"]))
        clipped = chip.intersection(geometry)
        families[family][stem] = clipped
        if seconds == 0 and clipped.area / chip.area >= COVERAGE:
            primary_products.append(stem)
            primary_families.add(family)

    family_overlaps = {}
    secondary, reserved_positive, reserved_boundary = [], [], []
    for family, members in sorted(families.items()):
        combined = unary_union([members[stem] for stem in sorted(members)])
        planar_fraction = min(1., combined.area / chip.area)
        equal_area_fraction = min(1., max(0., spherical_area(combined) / chip_spherical_area))
        boundary_only = not combined.is_empty and combined.area == 0
        family_overlaps[family] = dict(products=sorted(members), planar_fraction=planar_fraction,
                                       equal_area_fraction=equal_area_fraction, boundary_only=boundary_only)
        if equal_area_fraction >= COVERAGE:
            secondary.append(family)
        if family in reserved:
            if planar_fraction > 0:
                reserved_positive.append(family)
            elif boundary_only:
                reserved_boundary.append(family)
    return dict(primary_products=sorted(primary_products), primary_families=sorted(primary_families),
                secondary_families=secondary, reserved_positive_area_families=reserved_positive,
                reserved_boundary_only_families=reserved_boundary, family_overlaps=family_overlaps,
                secondary_area_model="unit_sphere_straight_lon_lat_edges")
