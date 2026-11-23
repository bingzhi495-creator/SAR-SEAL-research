"""Geolocate Sentinel-1 indexed measurement samples using the VV annotation XML.

A column/line index identifies a sample centre. Integer coordinates address the
XML knots directly; no half-pixel corner shift or image resizing is applied.
The forward mapping is bilinear on each (possibly unequally spaced) grid cell.
The inverse reproduces the author's degree-two fit using XML control points as
a reconstruction of the raster TIFF GCPs. Its output is fractional; the author's
crop rounds each coordinate toward zero with int().
"""
from dataclasses import dataclass
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

import numpy as np


def _path(names):
    return "/".join("{*}" + name for name in names.split("/"))


def _field(node, name):
    text = node.findtext(_path(name))
    if text is None or not text.strip():
        raise ValueError(f"Missing XML field {name}")
    return text.strip()


def _utc_seconds(text):
    stamp = text[:-1] + "+00:00" if text.endswith("Z") else text
    moment = datetime.fromisoformat(stamp)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


def _pair(x, y):
    left, right = np.asarray(x), np.asarray(y)
    if left.dtype.kind not in "fiu" or right.dtype.kind not in "fiu":
        raise ValueError("Coordinates must be real numeric values")
    x, y = np.broadcast_arrays(left.astype(np.float64, copy=False), right.astype(np.float64, copy=False))
    if not np.all(np.isfinite(x) & np.isfinite(y)):
        raise ValueError("Nonfinite coordinates")
    return x, y


def _bracket(axis, x):
    if np.any((x < axis[0]) | (x > axis[-1])):
        raise ValueError("Query outside the closed XML knot lattice")
    lower = np.clip(np.searchsorted(axis, x, side="right") - 1, 0, len(axis) - 2)
    fraction = (x - axis[lower]) / (axis[lower + 1] - axis[lower])
    return lower, fraction


def _lerp(left, right, weight):
    # Endpoints reproduce stored knots even for large UTC epoch values.
    return np.where(weight == 1., right, left + weight * (right-left))


def _bilinear(matrix, col, row, fc, fr):
    top = _lerp(matrix[row, col], matrix[row, col+1], fc)
    bottom = _lerp(matrix[row+1, col], matrix[row+1, col+1], fc)
    value = _lerp(top, bottom, fr)
    return value.item() if value.ndim == 0 else value


@dataclass(frozen=True)
class Grid:
    """Sorted source indices and corresponding [row knot, column knot] values."""
    width: int
    height: int
    cols: np.ndarray
    rows: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    azimuth_seconds: np.ndarray
    col_coefficients: np.ndarray
    row_coefficients: np.ndarray
    fit_rank: int

    def _cell(self, cols, rows):
        x, y = _pair(cols, rows)
        i, fx = _bracket(self.cols, x)
        j, fy = _bracket(self.rows, y)
        return i, j, fx, fy

    def forward(self, cols, rows):
        """Return (longitude, latitude) in degrees on the closed XML knot lattice."""
        cell = self._cell(cols, rows)
        return _bilinear(self.lon, *cell), _bilinear(self.lat, *cell)

    def time_seconds(self, cols, rows):
        """Return bilinearly interpolated azimuthTime as UTC seconds since 1970."""
        return _bilinear(self.azimuth_seconds, *self._cell(cols, rows))

    def inverse_polynomial(self, lon, lat):
        """Return fractional (col,row) from the author's unscaled geographic fit.

        As with the author's geographic admission check, coordinates must lie
        within both control-point coordinate ranges. The polynomial is not an
        exact inverse of forward interpolation near a curved scene footprint.
        """
        x, y = _pair(lon, lat)
        if np.any((x < self.lon.min()) | (x > self.lon.max()) | (y < self.lat.min()) | (y > self.lat.max())):
            raise ValueError("Query outside geographic control-point ranges")
        basis = np.stack((np.ones_like(x), x, y, x*y, x**2, y**2), axis=-1)
        c, r = basis @ self.col_coefficients, basis @ self.row_coefficients
        return (float(c), float(r)) if c.ndim == 0 else (c, r)


def read_grid(xml_bytes):
    """Parse one complete rectangular geolocation lattice without extrapolation.

    The immutable intake validates the VV scene, sensing interval and sampling
    provenance. Here image dimensions constrain every knot, but knots need not
    extend to every image edge. Geographic interpolation is specific to scenes
    whose longitude axis does not cross the antimeridian, as in the Gulf set.
    """
    root = ET.fromstring(xml_bytes)
    if root.tag.split("}")[-1] != "product":
        raise ValueError("Expected a Sentinel-1 product annotation")
    width = int(_field(root, "imageAnnotation/imageInformation/numberOfSamples"))
    height = int(_field(root, "imageAnnotation/imageInformation/numberOfLines"))
    listing = root.find(_path("geolocationGrid/geolocationGridPointList"))
    if listing is None or width < 2 or height < 2:
        raise ValueError("Missing geolocation list or invalid image dimensions")
    points = listing.findall(_path("geolocationGridPoint"))
    if not points or len(points) != int(listing.get("count", "-1")):
        raise ValueError("Geolocation point count differs from declared size")

    entries = []
    for p in points:
        col, row = int(_field(p, "pixel")), int(_field(p, "line"))
        lon, lat = float(_field(p, "longitude")), float(_field(p, "latitude"))
        utc = _utc_seconds(_field(p, "azimuthTime"))
        if not (0 <= col < width and 0 <= row < height
                and -180 <= lon <= 180 and -90 < lat < 90 and np.isfinite(utc)):
            raise ValueError("Invalid sample index, geographic coordinate or acquisition time")
        entries.append((col, row, lon, lat, utc))

    # Keep the XML order for the two least-squares solves, before building the
    # sorted interpolation arrays. Use the same basis and separate solves as
    # build_gcp_transform in the source extraction script (31387b3f2dbda).
    cols = np.asarray([v[0] for v in entries], dtype=np.int64)
    rows = np.asarray([v[1] for v in entries], dtype=np.int64)
    lon, lat, times = np.asarray([v[2:] for v in entries], dtype=np.float64).T
    c_axis, ci = np.unique(cols, return_inverse=True)
    r_axis, ri = np.unique(rows, return_inverse=True)
    shape = (len(r_axis), len(c_axis))
    if (min(shape) < 2 or shape[0]*shape[1] != len(entries)
            or len(np.unique(ri*shape[1]+ci)) != len(entries)):
        raise ValueError("Geolocation knots must form a complete rectangular lattice with no duplicates")
    if lon.max() - lon.min() >= 180:
        raise ValueError("Gulf longitude interpolation requires a continuous longitude range")
    basis = np.column_stack((np.ones(len(lon)), lon, lat, lon*lat, lon**2, lat**2))
    cc, _, rank_c, _ = np.linalg.lstsq(basis, cols, rcond=None)
    rc, _, rank_r, _ = np.linalg.lstsq(basis, rows, rcond=None)
    if rank_c != rank_r or not np.isfinite(cc).all() or not np.isfinite(rc).all():
        raise ValueError("Invalid degree-two control-point fit")
    matrices = []
    for values in (lon, lat, times):
        matrix = np.empty(shape, dtype=np.float64)
        matrix[ri, ci] = values
        matrices.append(matrix)
    for a in (c_axis, r_axis, *matrices, cc, rc):
        a.setflags(write=False)
    return Grid(width, height, c_axis, r_axis, *matrices, cc, rc, int(rank_c))
