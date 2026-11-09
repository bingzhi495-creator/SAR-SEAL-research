"""Ground-truth-assisted geometry bounds on one fixed native mask pair."""

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

SHIFT = 2
BAND = 4


def _counts(mask, truth, prediction, truth_area):
    tp = int(np.count_nonzero(mask & truth))
    return dict(tp=tp, fp=int(np.count_nonzero(mask)) - tp, fn=truth_area - tp,
                edits=int(np.count_nonzero(mask ^ prediction)))


def _iou_ratio(counts):
    union = counts["tp"] + counts["fp"] + counts["fn"]
    return (counts["tp"], union) if union else (1, 1)


def _better(candidate, incumbent):
    n, d = _iou_ratio(candidate)
    old_n, old_d = _iou_ratio(incumbent)
    left, right = n * old_d, old_n * d
    return left > right or (left == right and candidate["edits"] < incumbent["edits"])


def _extent(mask):
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size:
        return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    return None


def _shift(mask, extent, dy, dx):
    """Translate without wrapping, rejecting loss of foreground at canvas edges."""
    if not (dy or dx) or extent is None:
        return mask
    y0, y1, x0, x1 = extent
    height, width = mask.shape
    if not (0 <= y0 + dy < y1 + dy <= height and 0 <= x0 + dx < x1 + dx <= width):
        return None
    moved = np.zeros_like(mask)
    moved[y0 + dy:y1 + dy, x0 + dx:x1 + dx] = mask[y0:y1, x0:x1]
    return moved


def _morph(mask, radius):
    if radius == 0:
        return mask
    operation = maximum_filter if radius > 0 else minimum_filter
    return operation(mask, size=2 * abs(radius) + 1, mode="constant", cval=0)


def _local_optimum(base, fn_available, fp_available, budget):
    # Add missing truth first. Each addition improves IoU by 1/union,
    # at least as much as deleting a false positive from the same state.
    added = min(fn_available, budget)
    # Without intersection, partial FP deletion does not change IoU.
    # The cap is always below union for any nonempty mask pair.
    removed = min(fp_available, budget - added) if base["tp"] + added else 0
    return dict(tp=base["tp"] + added, fp=base["fp"] - removed,
                fn=base["fn"] - added, edits=added + removed)


def analyze_pair(pred, truth):
    """Compare fixed structured transforms and exact local edit oracles.

    Masks are Boolean arrays on the same true image canvas. Structured controls
    shift by at most two pixels per axis, adjust thickness by up to two square
    morphology pixels, or compose morphology then translation. Counts use the
    symmetric difference from the original prediction. Equal IoU prefers fewer
    changes, then first occurrence in the fixed radius and offset enumeration.
    """
    pred, truth = np.asarray(pred), np.asarray(truth)
    if (pred.ndim != 2 or pred.shape != truth.shape or not pred.size
            or pred.dtype != np.bool_ or truth.dtype != np.bool_):
        raise ValueError("Expected nonempty, same-shape two-dimensional Boolean masks")
    extent = _extent(pred | truth)
    if extent is not None:
        y0, y1, x0, x1 = extent
        # Radius two followed by a two-pixel shift can reach four pixels
        # beyond the original mask. Padding keeps those pixels in scope.
        crop = (slice(max(0, y0 - BAND), y1 + BAND), slice(max(0, x0 - BAND), x1 + BAND))
        pred, truth = pred[crop], truth[crop]

    base = _counts(pred, truth, pred, int(np.count_nonzero(truth)))
    truth_area = base["tp"] + base["fn"]
    budget = min(16, (base["tp"] + base["fp"] + base["fn"]) // 10)
    methods = {name: base.copy() for name in ("baseline", "translation", "thickness", "joint", "joint_free")}
    admissible = 0

    for radius in (0, -1, 1, -2, 2):
        layer = _morph(pred, radius)
        foreground = _extent(layer)
        for dy in range(-SHIFT, SHIFT + 1):
            for dx in range(-SHIFT, SHIFT + 1):
                if not (radius or dy or dx):
                    continue
                shifted = _shift(layer, foreground, dy, dx)
                if shifted is None:
                    continue
                candidate = _counts(shifted, truth, pred, truth_area)
                if _better(candidate, methods["joint_free"]):
                    methods["joint_free"] = candidate
                if not 0 < candidate["edits"] <= budget:
                    continue
                admissible += 1
                if _better(candidate, methods["joint"]):
                    methods["joint"] = candidate
                if radius == 0 and _better(candidate, methods["translation"]):
                    methods["translation"] = candidate
                if dy == dx == 0 and _better(candidate, methods["thickness"]):
                    methods["thickness"] = candidate

    band = _morph(pred, BAND) & ~_morph(pred, -BAND)
    band_fn = int(np.count_nonzero(band & truth & ~pred))
    band_fp = int(np.count_nonzero(band & pred & ~truth))
    methods["local_band"] = _local_optimum(base, band_fn, band_fp, budget)
    methods["local_any"] = _local_optimum(base, base["fn"], base["fp"], budget)
    if _better(methods["joint"], methods["local_band"]) or _better(methods["local_band"], methods["local_any"]):
        raise ValueError("Equal-budget geometry bounds are inconsistent")
    return dict(budget=budget, bulk_admissible_count=admissible, local_repair_available=band_fn + band_fp,
                methods=methods)
