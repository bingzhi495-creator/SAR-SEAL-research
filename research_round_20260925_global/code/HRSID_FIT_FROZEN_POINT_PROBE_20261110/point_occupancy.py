"""Fractional bitmap and polygon occupancy at fixed final-grid integer cells."""

import numpy as np
import shapely
from shapely.geometry import Polygon, box

from reference_resize import _compose
from ssdd_pixel_footprint import INT64_MAX, _axis_edges, _intersections
from vector_label_geometry import final_polygons


MAX_LOCAL_INTERSECTION_ELEMENTS = 262_144


def _cells(cells_xy, hw):
    cells = np.asarray(cells_xy)
    height, width = hw
    if cells.ndim != 2 or cells.shape[1] != 2 or cells.dtype.kind not in ("i", "u"):
        raise ValueError("Expected integer N-by-2 final-grid (x, y) cells")
    if (np.any(cells < 0) or np.any(cells[:, 0] >= width) or np.any(cells[:, 1] >= height)):
        raise ValueError("A requested integer cell lies outside the final grid")
    return cells.astype(np.int64, copy=False)


def _axis_intervals(edges, cells, unit):
    lower = np.minimum(edges[cells], edges[cells + 1])
    upper = np.maximum(edges[cells], edges[cells + 1])
    return lower, upper, lower // unit, (upper - 1) // unit + 1


def _axis_weights(intervals, unit, count):
    lower, upper, start, stop = intervals
    max_span = int((stop - start).max())
    source = np.minimum(start[:, None] + np.arange(max_span, dtype=np.int64), count)
    lo = np.maximum(source * unit, lower[:, None])
    hi = np.minimum(np.minimum(source + 1, count) * unit, upper[:, None])
    return np.minimum(source, count - 1), np.maximum(hi - lo, 0)


def bitmap_cell_occupancy(source_bitmap, steps, cells_xy):
    """Integrate original foreground unit squares over each inverse cell.

    Numerators and the common cell denominator use exact integer rational
    units. Only the final division is rounded to float64. The bank preserves
    input order and may contain repeated cells.
    """
    bitmap = np.asarray(source_bitmap)
    if bitmap.ndim != 2 or bitmap.dtype != np.bool_ or min(bitmap.shape) < 1:
        raise ValueError("Expected one nonempty-canvas original Boolean bitmap")
    source_hw, target_hw, slopes, offsets = _compose(steps, bitmap.shape)
    cells = _cells(cells_xy, target_hw)
    if not len(cells):
        return np.empty(0, dtype=np.float64)
    height, width = source_hw
    y_edges, y_unit = _axis_edges(target_hw[0], height, slopes[0], offsets[0])
    x_edges, x_unit = _axis_edges(target_hw[1], width, slopes[1], offsets[1])
    if height * width * y_unit * x_unit > INT64_MAX:
        raise ValueError("Bitmap footprint integration exceeds exact integer range")
    denominator = abs(int(y_edges[1] - y_edges[0]) * int(x_edges[1] - x_edges[0]))
    rows, cols = cells[:, 1], cells[:, 0]
    y_intervals = _axis_intervals(y_edges, rows, y_unit)
    x_intervals = _axis_intervals(x_edges, cols, x_unit)
    max_height = int((y_intervals[3] - y_intervals[2]).max())
    max_width = int((x_intervals[3] - x_intervals[2]).max())

    if len(cells) * max_height * max_width <= MAX_LOCAL_INTERSECTION_ELEMENTS:
        source_y, overlap_y = _axis_weights(y_intervals, y_unit, height)
        source_x, overlap_x = _axis_weights(x_intervals, x_unit, width)
        foreground = bitmap[source_y[:, :, None], source_x[:, None, :]]
        overlap = overlap_y[:, :, None] * overlap_x[:, None, :]
        numerator = (overlap * foreground).sum(axis=(1, 2), dtype=np.int64)
    else:
        numerator = _intersections(bitmap, rows, cols, y_edges, x_edges, y_unit, x_unit)

    if np.any(numerator < 0) or np.any(numerator > denominator):
        raise ArithmeticError("A bitmap intersection lies outside its exact cell footprint")
    return numerator.astype(np.float64) / denominator


def polygon_cell_occupancy(source_segmentation, source_hw, steps, cells_xy):
    """Integrate the valid transformed COCO polygon union on final unit cells.

    Vertices use continuous pixel-edge coordinates, with geometric crop
    clipping and x'=W-x after a horizontal flip. GEOS evaluates areas in
    float64; there is no COCO rasterizer or point-value threshold here.
    """
    rings, target_hw = final_polygons(source_segmentation, source_hw, steps)
    cells = _cells(cells_xy, target_hw)
    if not len(cells):
        return np.empty(0, dtype=np.float64)
    height, width = target_hw
    reference = shapely.union_all([Polygon(ring) for ring in rings]).intersection(box(0, 0, width, height))
    unique, order = np.unique(cells, axis=0, return_inverse=True)
    x, y = unique.T
    footprint = shapely.box(x, y, x + 1, y + 1)
    values = np.asarray(shapely.area(shapely.intersection(footprint, reference)), dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < -1e-10) or np.any(values > 1 + 1e-10):
        raise ArithmeticError("A polygon intersection lies outside its unit cell")
    return np.clip(values, 0., 1.)[order]


def cell_occupancy(source_bitmap, source_segmentation, steps, cells_xy):
    """Return matching bitmap and polygon fractional target arrays."""
    bitmap = np.asarray(source_bitmap)
    return dict(
        bitmap=bitmap_cell_occupancy(bitmap, steps, cells_xy),
        polygon=polygon_cell_occupancy(source_segmentation, bitmap.shape, steps, cells_xy),
    )
