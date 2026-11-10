"""Measure binary target masks against the continuous support of original pixel squares."""

from fractions import Fraction
from math import lcm

import numpy as np


INT64_MAX = np.iinfo(np.int64).max
ARM_FIELDS = ("area", "intersection", "iou", "source_y_error", "abs_source_y_error",
              "fp", "fn", "occupancy_l1", "area_signed_error")


def _axis_edges(count, source_count, scale, offset):
    if not isinstance(scale, Fraction) or not isinstance(offset, Fraction) or not scale:
        raise ValueError("Expected a nonzero exact rational scale and offset.")
    unit = lcm(scale.denominator, offset.denominator)
    first = offset.numerator * (unit // offset.denominator)
    step = scale.numerator * (unit // scale.denominator)
    last = first + count * step
    if min(first, last) < 0 or max(first, last) > source_count * unit:
        raise ValueError("A target cell footprint extends outside the original raster.")
    if source_count * unit > INT64_MAX:
        raise ValueError("Source raster coordinates exceed exact integer range.")
    return first + np.arange(count + 1, dtype=np.int64) * step, unit


def _clipped_axis(source_count, edges, unit):
    starts = np.arange(source_count, dtype=np.int64) * unit
    lower = np.maximum(starts, min(edges[0], edges[-1]))
    upper = np.minimum(starts + unit, max(edges[0], edges[-1]))
    return np.maximum(0, upper - lower), (lower.astype(np.float64) + upper) / (2 * unit)


def _integral(prefix, y, x, y_unit, x_unit):
    """Evaluate the cumulative area at a source edge coordinate in integer units."""
    row, dy = np.divmod(y, y_unit)
    col, dx = np.divmod(x, x_unit)
    next_row = np.minimum(row + 1, prefix.shape[0] - 1)
    next_col = np.minimum(col + 1, prefix.shape[1] - 1)
    left = prefix[row, col] * (y_unit - dy) + prefix[next_row, col] * dy
    right = prefix[row, next_col] * (y_unit - dy) + prefix[next_row, next_col] * dy
    return left * (x_unit - dx) + right * dx


def _intersections(mask, rows, cols, y_edges, x_edges, y_unit, x_unit):
    prefix = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.int64)
    prefix[1:, 1:] = mask.cumsum(axis=0, dtype=np.int64).cumsum(axis=1, dtype=np.int64)
    top = np.minimum(y_edges[rows], y_edges[rows + 1])
    bottom = np.maximum(y_edges[rows], y_edges[rows + 1])
    left = np.minimum(x_edges[cols], x_edges[cols + 1])
    right = np.maximum(x_edges[cols], x_edges[cols + 1])
    return (_integral(prefix, bottom, right, y_unit, x_unit)
            - _integral(prefix, top, right, y_unit, x_unit)
            - _integral(prefix, bottom, left, y_unit, x_unit)
            + _integral(prefix, top, left, y_unit, x_unit))


def measure_masks(raw_masks, candidates, small, steps, compose):
    """Return per-row areas in source pixels squared and vertical errors in source pixels.

    Each original foreground pixel denotes its full unit square. Each positive
    candidate cell denotes its full inverse-mapped rectangle. Integration is
    exact in integer rational units, with float64 output and centroid moments.
    Row selection is the caller's responsibility and no row is removed here.
    """
    raw = np.asarray(raw_masks)
    small = np.asarray(small)
    if raw.ndim != 3 or raw.dtype != np.bool_ or min(raw.shape[1:]) <= 0:
        raise ValueError("Expected original Boolean masks in N,H,W order.")
    count, height, width = raw.shape
    if small.shape != (count,) or small.dtype != np.bool_:
        raise ValueError("Expected one Boolean size stratum per original row.")
    if type(candidates) is not dict or not candidates or any(
            type(name) is not str or not name for name in candidates):
        raise ValueError("Expected named candidate mask arms.")
    initial, final, slopes, offsets = compose(steps, (height, width))
    if initial != (height, width) or len(final) != 2 or min(final) <= 0:
        raise ValueError("Expected a valid original-to-final raster map.")
    candidates = {name: np.asarray(masks) for name, masks in candidates.items()}
    if any(masks.dtype != np.bool_ or masks.shape != (count, *final) for masks in candidates.values()):
        raise ValueError("Every candidate arm must preserve the original row count and final grid.")
    y_edges, y_unit = _axis_edges(final[0], height, slopes[0], offsets[0])
    x_edges, x_unit = _axis_edges(final[1], width, slopes[1], offsets[1])
    area_unit = y_unit * x_unit
    if height * width * area_unit > INT64_MAX:
        raise ValueError("Source raster area exceeds exact integer range.")
    cell_area = abs(int(y_edges[1] - y_edges[0]) * int(x_edges[1] - x_edges[0]))
    y_coverage, source_y = _clipped_axis(height, y_edges, y_unit)
    x_coverage, _ = _clipped_axis(width, x_edges, x_unit)
    cell_y = (y_edges[:-1].astype(np.float64) + y_edges[1:]) / (2 * y_unit)

    truth_area = np.zeros(count, dtype=np.float64)
    truth_y = np.full(count, np.nan)
    truncated = np.zeros(count, dtype=np.bool_)
    original_edge = np.zeros(count, dtype=np.bool_)
    by_arm = {name: {field: np.full(count, np.nan) for field in ARM_FIELDS} for name in candidates}
    for index, mask in enumerate(raw):
        mass_by_row = (mask @ x_coverage) * y_coverage
        visible = int(mass_by_row.sum(dtype=np.int64))
        truth_area[index] = visible / area_unit
        truncated[index] = visible < np.count_nonzero(mask) * area_unit
        original_edge[index] = mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()
        if visible:
            truth_y[index] = (mass_by_row @ source_y) / visible

        active = np.logical_or.reduce([masks[index] for masks in candidates.values()])
        rows, cols = np.nonzero(active)
        overlap = (_intersections(mask, rows, cols, y_edges, x_edges, y_unit, x_unit)
                   if visible and rows.size else np.zeros(rows.size, dtype=np.int64))
        if np.any(overlap < 0) or np.any(overlap > cell_area):
            raise ArithmeticError("A continuous footprint intersection is outside its cell.")
        for name, masks in candidates.items():
            selected = masks[index, rows, cols]
            size = np.count_nonzero(selected)
            area = size * cell_area
            intersection = int(overlap[selected].sum(dtype=np.int64))
            fp, fn = area - intersection, visible - intersection
            if min(fp, fn) < 0:
                raise ArithmeticError("A continuous intersection exceeds a support area.")
            metrics = by_arm[name]
            metrics["area"][index] = area / area_unit
            metrics["intersection"][index] = intersection / area_unit
            metrics["fp"][index] = fp / area_unit
            metrics["fn"][index] = fn / area_unit
            metrics["occupancy_l1"][index] = (fp + fn) / area_unit
            metrics["area_signed_error"][index] = (area - visible) / area_unit
            union = area + fn
            if union:
                metrics["iou"][index] = intersection / union
            if size and visible:
                error = float(cell_y[rows[selected]].mean()) - truth_y[index]
                metrics["source_y_error"][index] = error
                metrics["abs_source_y_error"][index] = abs(error)
    return dict(small=small.copy(), truth_area=truth_area, truth_empty=truth_area == 0, truth_source_y=truth_y,
                crop_truncated=truncated, original_edge=original_edge, by_arm=by_arm)
