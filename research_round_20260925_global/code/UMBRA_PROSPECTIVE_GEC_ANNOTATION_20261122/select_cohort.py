"""Select a prospective Umbra scene queue using anonymous public metadata only."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from hashlib import sha256
import itertools
import json
from math import asin, cos, isfinite, radians, sin, sqrt
from pathlib import Path
from threading import Lock
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import ProxyHandler, Request, build_opener
from uuid import UUID
from xml.etree import ElementTree as ET

from shapely.geometry import Point, shape

BASE = "https://umbra-open-data-catalog.s3.us-west-2.amazonaws.com/"
PREFIX = "sar-data/tasks/ship_detection_testdata/"
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
MANIFEST_SHA = "b669b72e7d50185d998d8a56d5bab52ba172ed7c6118e6a9c71fc110180c3f5a"
GEOMETRY_SHA = "d7345af229cb815923164b41a616abe2cdb66c5795f05ebf319eb132438b7805"
PILOT_SHAS = (
    "f94e2f3df4a83c9ad0394216141f5fc66d8f4c2f8f09c419c40cc1309d6164a8",
    "88bd84cc8c7529803d70b57357df28ee6948a88d32dfafc6d6419f58c8963d04",
    "4c53a7dfa1363ecd00ef22592ddf9c345dead436f41f9e5f3436c0a05bd19d41",
)
TASKS = 64
QUOTA = 6
MINIMUM = 8
DISTANCE_KM = 100
MAX_BYTES = 8 * 1024**2
MIN_TIME = datetime.fromisoformat("2021-01-01T00:00:00+00:00")


def digest(raw):
    return sha256(raw).hexdigest()


def packed(obj):
    return (json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode()


def require(ok, issue):
    if not ok:
        raise ValueError(issue)


class Transport:
    def __init__(self):
        self.charged = 0
        self.count = 0
        self.lock = Lock()

    def get(self, url, limit):
        require(url.startswith(BASE), "Exact official host required")
        for attempt in range(2):
            try:
                opener = build_opener(ProxyHandler({}))
                request = Request(url, headers={"Accept-Encoding": "identity"})
                with opener.open(request, timeout=18) as reply:
                    require(reply.status == 200 and reply.geturl() == url, "Expected official same-URL response")
                    with self.lock:
                        require(self.charged + limit + 1 <= MAX_BYTES, "Source byte budget exhausted")
                        self.charged += limit + 1
                        self.count += 1
                    body = reply.read(limit + 1)
                    with self.lock:
                        self.charged -= limit + 1 - len(body)
                    require(len(body) <= limit, "Per-response limit exceeded")
                    return body
            except HTTPError as exc:
                if attempt or exc.code not in (429, 500, 502, 503, 504):
                    raise
            except (TimeoutError, URLError):
                if attempt:
                    raise
            sleep(0.6)

    def listing(self, prefix, delimiter=None, continuation=None):
        args = {"list-type": 2, "prefix": prefix}
        if delimiter:
            args["delimiter"] = delimiter
        if continuation:
            args["continuation-token"] = continuation
        raw = self.get(BASE + "?" + urlencode(args), 256 * 1024 if delimiter else 48 * 1024)
        return ET.fromstring(raw), digest(raw)


def pinned_json(path, expected):
    raw = path.read_bytes()
    require(digest(raw) == expected, "Input digest mismatch")
    return json.loads(raw)


def uuid_text(value):
    require(isinstance(value, str), "Missing UUID")
    return str(UUID(value))


def point(coords):
    require(isinstance(coords, (list, tuple)) and len(coords) >= 2, "Invalid center/vertex")
    x, y = coords[:2]
    require(type(x) in (int, float) and type(y) in (int, float) and
            isfinite(x) and isfinite(y) and -180 <= x <= 180 and -90 <= y <= 90,
            "Longitude/latitude bounds")
    return [x, y]


def checked_polygon(feature):
    require(isinstance(feature, dict) and feature.get("type") in ("Polygon", "MultiPolygon"),
            "Expected GeoJSON footprint")
    polygons = [feature["coordinates"]] if feature["type"] == "Polygon" else feature["coordinates"]
    for rings in polygons:
        require(bool(rings), "No footprint rings")
        for ring in rings:
            positions = [point(p) for p in ring]
            require(len(positions) >= 4 and positions[0] == positions[-1] and
                    len({tuple(p) for p in positions}) >= 3, "Open/degenerate footprint ring")
            require(max(p[0] for p in positions) - min(p[0] for p in positions) < 180,
                    "Unresolved antimeridian footprint")
    polygon = shape(feature)
    require(polygon.is_valid and not polygon.is_empty and polygon.area > 0, "Invalid footprint topology")
    return polygon


def distance(left, right):
    x1, y1 = map(radians, left)
    x2, y2 = map(radians, right)
    hav = sin((y2-y1)/2)**2 + cos(y1)*cos(y2)*sin((x2-x1)/2)**2
    return 2 * 6371.0088 * asin(sqrt(min(1, max(0, hav))))


def historic(manifest_path, geometry_path):
    manifest = pinned_json(manifest_path, MANIFEST_SHA)
    geometry = pinned_json(geometry_path, GEOMETRY_SHA)
    rows = manifest["selected"]
    locations = {r["collect_id"]: r for r in geometry["items"]}
    require(geometry["source_manifest_sha256"] == MANIFEST_SHA and
            len(rows) == len(locations) == len(geometry["items"]) == 120 and
            len({r["port"] for r in rows}) == 10, "Historical geometry/manifest coverage")
    centers = []
    for row in rows:
        loc = locations[row["collect_id"]]
        require(loc["stac_sha256"] == row["stac_sha256"] and loc["stac_id"] == row["stac_id"],
                "Historical geometry/manifest identity")
        centroid = checked_polygon(loc["geometry"]).centroid
        centers.append(point([float(centroid.x), float(centroid.y)]))
    return {uuid_text(r["task_id"]) for r in rows}, {uuid_text(r["collect_id"]) for r in rows}, centers


def pilot_ids(transport, receipt_paths):
    tasks, collects = set(), set()
    for path, expected in zip(receipt_paths, PILOT_SHAS):
        receipt = pinned_json(path, expected)
        require(receipt["status"] == "NATIVE_TECHNICAL_PILOT_COMPLETE" and len(receipt["products"]) == 1,
                "Prior pilot receipt status")
        url = receipt["products"][0]["author_url"]
        require(url.startswith(BASE + PREFIX) and url.endswith("_GEC.tif"), "Prior pilot URL")
        task = uuid_text(url.split(PREFIX, 1)[1].split("/", 1)[0])
        j = json.loads(transport.get(url[:-len("_GEC.tif")] + "_METADATA.json", 20 * 1024))
        require(len(j["collects"]) == 1 and uuid_text(j["collects"][0]["taskId"]) == task, "Prior pilot identity")
        tasks.add(task)
        collects.add(uuid_text(j["collects"][0]["id"]))
    require(len(tasks) == len(collects) == 3, "Three distinct prior technical pilots")
    return tasks, collects


def midpoint_indices(length, cap):
    count = min(length, cap)
    return [((2 * i + 1) * length) // (2 * count) for i in range(count)]


def all_tasks(transport):
    prefixes, page_hashes = [], []
    continuation = None
    for _ in range(4):
        page, receipt = transport.listing(PREFIX, delimiter="/", continuation=continuation)
        page_hashes.append(receipt)
        for key in page.findall("s:CommonPrefixes/s:Prefix", NS):
            require(key.text.startswith(PREFIX), "Task listing escaped prefix")
            task = uuid_text(key.text[len(PREFIX):].rstrip("/"))
            require(key.text == PREFIX + task + "/", "Noncanonical task prefix")
            prefixes.append(task)
        if page.findtext("s:IsTruncated", namespaces=NS) == "false":
            require(len(set(prefixes)) == len(prefixes) > 0, "Unique task listing")
            return sorted(prefixes), page_hashes
        continuation = page.findtext("s:NextContinuationToken", namespaces=NS)
        require(continuation, "Missing list continuation")
    raise ValueError("S3 list-page bound exhausted")


def candidate(transport, task):
    task_prefix = PREFIX + task + "/"
    xml, listing_digest = transport.listing(task_prefix)
    require(xml.findtext("s:IsTruncated", namespaces=NS) == "false", "Truncated task object listing")
    objs = {o.findtext("s:Key", namespaces=NS): o for o in xml.findall("s:Contents", NS)}
    assets = [(k, k[:-len("_GEC.tif")] + "_METADATA.json") for k in sorted(objs)
              if k.startswith(task_prefix) and k.endswith("_GEC.tif") and
              k[:-len("_GEC.tif")] + "_METADATA.json" in objs]
    if not assets:
        return None, "no_paired_gec_metadata"
    gec, metadata = assets[0]
    require(int(objs[gec].findtext("s:Size", namespaces=NS)) > 0, "Invalid GEC object size")
    raw = transport.get(BASE + quote(metadata, safe="/"), 20 * 1024)
    j = json.loads(raw)
    ipr = j.get("targetIpr")
    if type(ipr) not in (int, float) or not isfinite(ipr):
        return None, "target_ipr_missing_or_invalid"
    if not 0.35 <= ipr <= 2.0:
        return None, "target_ipr_outside_0p35_to_2"
    collects = j.get("collects", [])
    if not isinstance(collects, list) or len(collects) != 1:
        return None, "collect_identity_ambiguous"
    c = collects[0]
    require(uuid_text(c["taskId"]) == task, "Mismatched task ID")
    when = datetime.fromisoformat(c["startAtUTC"][:-1] + "+00:00"
                                  if c["startAtUTC"].endswith("Z") else c["startAtUTC"])
    require(when.tzinfo is not None and when.utcoffset().total_seconds() == 0, "Expected UTC timestamp")
    if when < MIN_TIME:
        return None, "acquired_before_2021"
    footprint = c.get("footprintPolygonLla")
    if footprint is None:
        return None, "missing_footprint"
    center = c.get("sceneCenterPointLla")
    try:
        polygon = checked_polygon(footprint)
        require(isinstance(center, dict) and center.get("type") == "Point", "Missing center point")
        lonlat = point(center["coordinates"])
        require(polygon.covers(Point(lonlat)), "Center outside supplied footprint")
    except (KeyError, ValueError, TypeError):
        return None, "invalid_footprint_or_center"
    footprint_rings = [[point(v) for v in ring] for ring in footprint["coordinates"]] if footprint["type"] == "Polygon" else [
        [[point(v) for v in ring] for ring in rings] for rings in footprint["coordinates"]]
    depth = j.get("derivedProducts", {}).get("GEC", [])
    dims = depth[0] if len(depth) == 1 else {}
    return {
        "task_id": task,
        "collect_id": uuid_text(c["id"]),
        "start_utc": when.isoformat(),
        "center_lon_lat": lonlat,
        "footprint_type": footprint["type"],
        "footprint_lon_lat": footprint_rings,
        "target_ipr_m": ipr,
        "measured_gec_spacing_m": None,
        "declared_gec_rows": dims.get("numRows"),
        "declared_gec_cols": dims.get("numColumns"),
        "class_group": "coarse_1_to_2_m" if ipr >= 1 else "medium_0p35_to_lt1_m",
        "polarizations": c.get("polarizations"),
        "metadata_url": BASE + quote(metadata, safe="/"),
        "metadata_sha256": digest(raw),
        "task_listing_sha256": listing_digest,
        "author_url": BASE + quote(gec, safe="/"),
        "gec_url": BASE + quote(gec, safe="/"),
        "etag": objs[gec].findtext("s:ETag", namespaces=NS),
        "object_bytes": int(objs[gec].findtext("s:Size", namespaces=NS)),
    }, "qualified"


def choose(candidates, historical, quota=QUOTA):
    decisions = Counter()
    remaining = []
    for row in candidates:
        if any(distance(row["center_lon_lat"], p) < DISTANCE_KM for p in historical):
            decisions["within_100km_historical"] += 1
        else:
            remaining.append(row)
    selected = []
    used_ids = set()

    def try_group(rows, n):
        if n <= 0:
            return
        count = 0
        for row in rows:
            key = row["collect_id"]
            if key in used_ids:
                continue
            if any(distance(row["center_lon_lat"], s["center_lon_lat"]) < DISTANCE_KM for s in selected):
                continue
            selected.append(dict(row, window_id=f"UMB-{len(selected)+1:03d}"))
            used_ids.add(key)
            count += 1
            if count == n:
                break

    try_group((r for r in remaining if r["target_ipr_m"] >= 1), quota)
    try_group((r for r in remaining if r["target_ipr_m"] < 1), quota)
    try_group(remaining, 2 * quota - len(selected))
    for row in remaining:
        if row["collect_id"] not in used_ids:
            reason = ("within_100km_selected" if any(distance(row["center_lon_lat"], s["center_lon_lat"]) < DISTANCE_KM
                       for s in selected) else "qualified_beyond_planned_quota")
            decisions[reason] += 1
    return selected, decisions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--historical-manifest", type=Path, required=True)
    p.add_argument("--historical-geometry", type=Path, required=True)
    p.add_argument("--exclude-pilot-result", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    require(len(args.exclude_pilot_result) == 3, "Supply three chronological pilot receipts")
    old_tasks, old_collects, centers = historic(args.historical_manifest, args.historical_geometry)
    transport = Transport()
    pilots, pilot_collects = pilot_ids(transport, args.exclude_pilot_result)
    tasks, page_hashes = all_tasks(transport)
    idx = midpoint_indices(len(tasks), TASKS)
    sample = [tasks[i] for i in idx]

    def get_task(task):
        if task in pilots or task in old_tasks:
            return None, "historical_or_pilot_task"
        try:
            row, reason = candidate(transport, task)
            if row is not None and row["collect_id"] in (pilot_collects | old_collects):
                return None, "historical_or_pilot_collect"
            return row, reason
        except (HTTPError, URLError, TimeoutError):
            return None, "source_read_failed"
        except (KeyError, ValueError, TypeError, ET.ParseError):
            return None, "invalid_metadata_or_listing"

    with ThreadPoolExecutor(max_workers=10) as pool:
        observations = list(pool.map(get_task, sample))
    counts = Counter(reason for _, reason in observations)
    unique, seen = [], set()
    for row, _ in observations:
        if row is None:
            continue
        if row["collect_id"] in seen:
            counts["duplicate_eligible_collect"] += 1
            continue
        seen.add(row["collect_id"])
        unique.append(row)
    selected, decisions = choose(unique, centers)
    required = {"schema": "umbra_gec_cohort_v1", "items": selected, "scenes": selected,
                "source_snapshot_sha256": digest(packed(tasks)), "sample_task_order_sha256": digest(packed(sample)),
                "frozen_rule_version": "midpoint64_coarse6_medium6_100km_historical120"}
    pairs = list(itertools.combinations(selected, 2))
    historical_distances = [distance(row["center_lon_lat"], c) for row in selected for c in centers]
    overlaps = 0
    for a, b in pairs:
        sa = shape({"type": a["footprint_type"], "coordinates": a["footprint_lon_lat"]})
        sb = shape({"type": b["footprint_type"], "coordinates": b["footprint_lon_lat"]})
        overlaps += int(sa.intersects(sb))
    require(overlaps == 0, "Selected footprints overlap, abort before imagery")
    ips = Counter(str(row["target_ipr_m"]) for row in selected)
    safe = {
        "schema": "umbra_gec_metadata_cohort_safe_v1",
        "status": "planned_12" if len(selected) == 12 else "meets_minimum_8" if len(selected) >= MINIMUM else "below_minimum",
        "public_source": BASE + "?" + urlencode({"list-type":2, "prefix":PREFIX, "delimiter":"/"}),
        "number_of_catalog_tasks": len(tasks),
        "ordered_task_list_sha256": required["source_snapshot_sha256"],
        "s3_list_page_sha256": page_hashes,
        "ordered_sample_task_list_sha256": required["sample_task_order_sha256"],
        "sampled_task_count": len(sample),
        "source_response_reads": transport.count,
        "bounded_payload_bytes_charged": transport.charged,
        "source_read_cap_bytes": MAX_BYTES,
        "task_screening": dict(counts),
        "qualified_unique_before_spatial_filter": len(unique),
        "qualified_resolution_histogram_before_spatial_filter": dict(Counter(str(row["target_ipr_m"]) for row in unique)),
        "spatial_filter_decisions": dict(decisions),
        "historical_exact_tasks": len(old_tasks),
        "historical_exact_collects": len(old_collects),
        "historical_validated_footprint_centroids": len(centers),
        "historical_ports": 10,
        "historical_manifest_sha256": MANIFEST_SHA,
        "historical_geometry_sha256": GEOMETRY_SHA,
        "november_pilot_exact_tasks": len(pilots),
        "selected_scene_count": len(selected),
        "selected_target_ipr_histogram_m": dict(ips),
        "selected_class_groups": dict(Counter(row["class_group"] for row in selected)),
        "selected_date_range_utc": [min(r["start_utc"] for r in selected), max(r["start_utc"] for r in selected)] if selected else [],
        "minimum_pairwise_center_distance_km": min((distance(a["center_lon_lat"], b["center_lon_lat"])
                                                    for a, b in pairs), default=None),
        "minimum_distance_to_any_historical_footprint_centroid_km": min(historical_distances, default=None),
        "selected_footprints_intersecting_pairs": overlaps,
        "selected_measured_gec_spacing_count": 0,
        "image_reads": 0,
        "label_reads": 0,
        "model_forwards": 0,
        "cohort_private_sha256": digest(packed(required)),
        "source_sha256": digest(Path(__file__).read_bytes()),
    }
    args.output.mkdir()
    (args.output / "COHORT_PRIVATE.json").write_bytes(packed(required))
    result = packed(safe)
    (args.output / "RESULT_SAFE.json").write_bytes(result)
    print(json.dumps({"result_sha256": digest(result), "status": safe["status"]}), flush=True)


if __name__ == "__main__":
    main()

