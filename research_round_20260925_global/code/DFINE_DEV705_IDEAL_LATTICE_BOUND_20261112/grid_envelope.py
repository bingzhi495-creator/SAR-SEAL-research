"""Ideal half-pixel bilinear mask capacity with relaxed row and column coupling."""

import numpy as np


def axis_cells(start, end, native, coarse):
    """Return the exact floor of the half-pixel source coordinate."""
    if not 0 <= start <= end <= native or coarse < 1:
        raise ValueError("Invalid native/coarse axis or crop")
    index = np.arange(start, end, dtype=np.int64)
    return ((2 * index + 1) * coarse - native) // (2 * native)


def monotone_weight(weights):
    """Best weighted selection per line among binary prefixes and suffixes."""
    prefixes = np.cumsum(weights, axis=1, dtype=np.float64)
    prefixes = np.pad(prefixes, ((0, 0), (1, 0)))
    return np.maximum(prefixes.max(axis=1), prefixes[:, -1] - prefixes.min(axis=1))


def _separable_sum(weights, cells, coarse):
    if coarse == 1:
        return np.maximum(weights.sum(axis=1, dtype=np.float64), 0).sum(dtype=np.float64)

    starts = np.r_[0, np.flatnonzero(np.diff(cells)) + 1]
    ends = np.r_[starts[1:], len(cells)]
    total = 0.
    for start, end in zip(starts, ends, strict=True):
        part = weights[:, start:end]
        if cells[start] in (-1, coarse - 1):
            best = np.maximum(part.sum(axis=1, dtype=np.float64), 0)
        else:
            best = monotone_weight(part)
        total += best.sum(dtype=np.float64)
    return total


def margin_envelope(occupancy, truth_area, threshold, origin_yx, native_hw, coarse_hw):
    """Bound the continuous IoU threshold margin inside the author box crop.

    Under ideal interpolation a binary row segment within one source interval is
    monotone. Relax the coupling between all such segments, then independently
    repeat along columns. The intersection of bounds is tighter than either.
    """
    q = np.asarray(occupancy, dtype=np.float64)
    if q.ndim != 2 or not np.isfinite(q).all() or (q < 0).any() or (q > 1).any():
        raise ValueError("Expected a 2D unit-cell occupancy array")
    if not np.isfinite(truth_area) or truth_area <= 0 or not 0 < threshold < 1:
        raise ValueError("Expected positive truth area and an interior IoU threshold")
    h, w = q.shape
    y, x = origin_yx
    height, width = native_hw
    coarse_h, coarse_w = coarse_hw
    y_cells = axis_cells(y, y + h, height, coarse_h)
    x_cells = axis_cells(x, x + w, width, coarse_w)
    target = threshold * truth_area
    if not q.size:
        return dict(free=-target, rows=-target, columns=-target, upper=-target)
    weights = (1 + threshold) * q - threshold
    free = np.maximum(weights, 0).sum(dtype=np.float64) - target
    rows = _separable_sum(weights, x_cells, coarse_w) - target
    columns = _separable_sum(weights.T, y_cells, coarse_h) - target
    return dict(free=float(free), rows=float(rows), columns=float(columns),
                upper=float(min(free, rows, columns)))
