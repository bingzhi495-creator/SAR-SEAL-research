"""Reference geometry for measuring native mask footprints against NASTaR AIS dimensions.

Georeferencing follows GDAL's corner convention. Measurements use the WGS84
azimuthal equidistant plane centered at the reported AIS reference position.
The measured footprint is the convex extent of the projected pixel corners.
No pixel spacing, grid orientation, CRS, or correction to AIS position is assumed.

Sources:
https://arxiv.org/html/2512.18503v1#S3
https://www.navcen.uscg.gov/ais-class-a-reports
https://gdal.org/en/stable/tutorials/geotransforms_tut.html
https://proj.org/en/stable/tutorials/geodesics.html
"""

import math

import numpy as np
from pyproj import CRS, Geod, Transformer


_WGS84 = CRS.from_epsg(4326)
_GEOD = Geod(ellps="WGS84")


def _position(lon, lat):
    lon, lat = float(lon), float(lat)
    if not -180 <= lon <= 180 or not -90 < lat < 90:
        raise ValueError("Invalid WGS84 AIS reference position or undefined polar axis")
    return lon, lat


def _grid(transform):
    # A GDAL six-tuple and a rasterio.Affine use different coefficient orders.
    coefficients = transform.to_gdal() if hasattr(transform, "to_gdal") else transform
    g = np.asarray(coefficients, dtype=np.float64)
    if g.shape != (6,) or not np.isfinite(g).all():
        raise ValueError("Supply a GDAL six-tuple or rasterio.Affine with finite coefficients")
    matrix = np.array([[g[1], g[2]], [g[4], g[5]]])
    if np.linalg.matrix_rank(matrix) != 2:
        raise ValueError("Singular pixel-to-map transformation")
    return g[[0, 3]], matrix


def ais_pixel_xy(lon, lat, crs, transform):
    """Return fractional (column, row) in pixel-edge coordinates; pixel (0,0) spans [0,1)².

    Use floor to find the containing raster cell. Subtract 0.5 on each axis
    if an API needs integer-valued pixel centers. Coordinates stay unclipped.
    """
    lon, lat = _position(lon, lat)
    origin, matrix = _grid(transform)
    to_grid = Transformer.from_crs(_WGS84, crs, always_xy=True)
    x, y = to_grid.transform(lon, lat, errcheck=True)
    col, row = np.linalg.solve(matrix, np.asarray([x, y]) - origin)
    return float(col), float(row)


def heading_extents_m(mask, lon, lat, heading, crs, transform, valid_mask=None):
    """Return (along-heading length, across-heading beam) in metres, without axis sorting.

    `mask` is a single binary instance on the source raster. `valid_mask` is
    the aligned raster validity band (nonzero = valid). An empty mask or any
    selected invalid cell makes the measurement unavailable and raises ValueError.
    AIS true heading is clockwise from true north; 511 and all missing or
    out-of-range values are rejected. COG is a different recorded quantity.
    """
    lon, lat = _position(lon, lat)
    try:
        heading = float(heading)
    except (TypeError, ValueError) as exc:
        raise ValueError("Missing AIS true heading") from exc
    if not 0 <= heading < 360:
        raise ValueError("AIS true heading must be in [0, 360)")
    origin, matrix = _grid(transform)
    data = np.asarray(mask)
    if data.ndim != 2 or data.dtype.kind not in "buif" or not np.isin(data, (0, 1)).all():
        raise ValueError("A binary two-dimensional source-grid mask is required")
    rows, cols = np.nonzero(data)
    if rows.size == 0:
        raise ValueError("Empty footprint")
    if np.ma.isMaskedArray(mask) and np.any(np.ma.getmaskarray(mask)[rows, cols]):
        raise ValueError("Selected footprint contains a masked cell")
    if valid_mask is not None:
        valid = np.asarray(valid_mask)
        if valid.shape != data.shape or valid.dtype.kind not in "buif":
            raise ValueError("Raster validity mask must match the source grid")
        values = valid[rows, cols]
        if np.any((values == 0) | ~np.isfinite(values)):
            raise ValueError("Selected footprint intersects raster nodata")

    # Integer (col,row) denotes a cell corner. Neighboring cells share vertices.
    stride = data.shape[1] + 1
    top_left = rows * stride + cols
    corner_ids = np.unique(np.concatenate([top_left, top_left+1, top_left+stride, top_left+stride+1]))
    corner_rows, corner_cols = np.divmod(corner_ids, stride)
    x, y = origin[:, None] + matrix @ np.stack((corner_cols, corner_rows))
    to_wgs84 = Transformer.from_crs(crs, _WGS84, always_xy=True)
    corner_lon, corner_lat = to_wgs84.transform(x, y, errcheck=True)
    azimuth, _, distance = _GEOD.inv(np.full(x.shape, lon), np.full(x.shape, lat), corner_lon, corner_lat)
    if not np.isfinite(azimuth).all() or not np.isfinite(distance).all():
        raise ValueError("Footprint cannot be projected onto the AIS local plane")
    relative = np.deg2rad(azimuth) - math.radians(heading)
    along = distance * np.cos(relative)
    across = distance * np.sin(relative)
    return float(np.ptp(along)), float(np.ptp(across))
