"""Exact integer-translation diagnostics for one fixed Boolean mask pair."""

from itertools import product
from math import sqrt

import numpy as np


RADIUS = 2
PIXEL_VARIANCE = 1 / 12


def tie_key(candidate):
    dy, dx = candidate["dy"], candidate["dx"]
    return candidate["edits"], dy * dy + dx * dx, dy, dx


def best_shift(base, candidates):
    best = base
    for candidate in candidates:
        union = candidate["tp"] + candidate["fp"] + candidate["fn"]
        best_union = best["tp"] + best["fp"] + best["fn"]
        left, right = candidate["tp"] * best_union, best["tp"] * union
        if left > right or (left == right and tie_key(candidate) < tie_key(best)):
            best = candidate
    return best


def measure(pred, gt):
    """Return fixed and GT-selected shifts on one fixed prediction and target.

    Radius is Chebyshev distance. A shift is admissible only if every predicted
    foreground pixel stays on the original canvas. Fixed shifts are None when
    inadmissible. Signed centroid offsets are prediction minus GT (down/right).
    """
    pred, gt = np.asarray(pred), np.asarray(gt)
    if pred.ndim != 2 or pred.shape != gt.shape or pred.dtype != np.bool_ or gt.dtype != np.bool_:
        raise ValueError("Expected same-shape two-dimensional Boolean masks")
    pred_y, pred_x = np.nonzero(pred)
    gt_y, gt_x = np.nonzero(gt)
    pred_area, gt_area = pred_y.size, gt_y.size
    if not pred_area or not gt_area:
        raise ValueError("Expected two nonempty foreground masks")

    height, width = pred.shape
    y0, y1 = int(pred_y[0]), int(pred_y[-1]) + 1
    x0, x1 = int(pred_x.min()), int(pred_x.max()) + 1
    top, left = max(0, y0 - RADIUS), max(0, x0 - RADIUS)
    bottom, right = min(height, y1 + RADIUS), min(width, x1 + RADIUS)
    original = pred[top:bottom, left:right]
    truth = gt[top:bottom, left:right]
    patch = pred[y0:y1, x0:x1]

    shifts = {}
    for dy, dx in product(range(-RADIUS, RADIUS + 1), repeat=2):
        if not (0 <= y0 + dy and y1 + dy <= height and 0 <= x0 + dx and x1 + dx <= width):
            continue
        dest = (slice(y0 + dy - top, y1 + dy - top), slice(x0 + dx - left, x1 + dx - left))
        # The shifted support is entirely inside dest, so pixels outside it
        # contribute to FN or XOR through the original foreground areas.
        tp = int(np.count_nonzero(patch & truth[dest]))
        unchanged = int(np.count_nonzero(patch & original[dest]))
        shifts[dy, dx] = dict(
            tp=tp, fp=pred_area - tp, fn=gt_area - tp,
            edits=2 * (pred_area - unchanged), dy=dy, dx=dx,
        )

    base = shifts[0, 0]
    oracle1 = best_shift(base, (value for (dy, dx), value in shifts.items() if max(abs(dy), abs(dx)) <= 1))
    oracle2 = best_shift(base, shifts.values())
    gt_center_y, gt_center_x = float(gt_y.mean()), float(gt_x.mean())
    centered = np.column_stack((gt_y - gt_center_y, gt_x - gt_center_x))
    minor, major = np.linalg.eigvalsh(centered.T @ centered / gt_area)
    elongation = sqrt((major + PIXEL_VARIANCE) / (max(0.0, minor) + PIXEL_VARIANCE))
    return dict(
        base=base, oracle1=oracle1, oracle2=oracle2,
        fixed_up=shifts.get((-1, 0)), fixed_down=shifts.get((1, 0)),
        area_impossible=10 * min(pred_area, gt_area) < 9 * max(pred_area, gt_area),
        dy_pixels=float(pred_y.mean() - gt_center_y),
        dx_pixels=float(pred_x.mean() - gt_center_x),
        elongation=elongation,
    )
