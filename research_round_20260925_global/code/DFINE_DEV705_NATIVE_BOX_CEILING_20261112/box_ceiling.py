"""Exact continuous-IoU feasibility over native unit-pixel masks."""

import numpy as np


def positive_margins(occupancy, truth_area, origin_yx, box_xyxy, threshold, radii=(0, 1, 2, 4)):
    q = np.asarray(occupancy, dtype=np.float64)
    if q.ndim != 2 or not np.isfinite(q).all() or (q < 0).any() or (q > 1).any():
        raise ValueError("Expected a finite unit-cell occupancy array")
    if not np.isfinite(truth_area) or truth_area <= 0 or not 0 < threshold < 1:
        raise ValueError("Positive truth area and interior threshold required")
    y, x = origin_yx
    h, w = q.shape
    left, top, right, bottom = box_xyxy
    if left >= right or top >= bottom or any(radius < 0 for radius in radii):
        raise ValueError("Nonempty box and nonnegative expansion required")
    positive = np.maximum((1 + threshold) * q - threshold, 0.)
    offset = threshold * truth_area
    out = {"free": float(positive.sum(dtype=float) - offset)}
    for radius in radii:
        xa, xb = max(0, left - radius - x), min(w, right + radius - x)
        ya, yb = max(0, top - radius - y), min(h, bottom + radius - y)
        area = positive[ya:yb, xa:xb].sum(dtype=float) if xa < xb and ya < yb else 0.
        out[str(radius)] = float(area - offset)
    return out
