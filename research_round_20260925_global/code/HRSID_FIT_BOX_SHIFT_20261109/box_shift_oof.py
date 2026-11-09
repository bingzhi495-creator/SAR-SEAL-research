"""Cross-fit constrained global mask shifts on frozen scene-proxy folds."""

from hashlib import sha256

import numpy as np

FOLDS = 5
RIDGE = .25
MAX_SHIFT = 2


def _group_names(values):
    if isinstance(values, str):
        raise ValueError("Expected a sequence of scene group names")
    groups = list(values)
    if not all(isinstance(group, str) and group for group in groups):
        raise ValueError("Scene groups must be nonempty strings")
    return groups


def _fold_labels(groups, all_groups):
    universe = set(groups if all_groups is None else _group_names(all_groups))
    if len(universe) < FOLDS or not set(groups) <= universe:
        raise ValueError("Expected at least five frozen groups containing every row group")

    def fold_order(group):
        digest = sha256(("HRSID-P196-v1|" + group).encode("utf-8")).digest()
        return digest, group

    assignment = {group: i % FOLDS for i, group in enumerate(sorted(universe, key=fold_order))}
    return np.asarray([assignment[group] for group in groups], dtype=np.int64)


def _feature_matrix(values, n, name):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (n, 2) or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite (n, 2) matrix in (y, x) order")
    return values


def _round_shift(values):
    magnitude = np.abs(np.clip(values, -MAX_SHIFT, MAX_SHIFT))
    whole = np.floor(magnitude)
    rounded = whole + (magnitude - whole >= .5)
    return np.copysign(rounded, values).astype(np.int64)


def _ridge(fit, held, weights, centered_target, intercept):
    """Predict each shift axis from the corresponding standardized gap."""
    mean = np.sum(weights * fit, axis=0)
    centered = fit - mean
    scale = np.sqrt(np.sum(weights * centered**2, axis=0))
    active = (scale > 0) & np.any(fit != fit[0], axis=0)
    fit_standard = np.divide(centered, scale, out=np.zeros_like(fit), where=active)
    held_standard = np.divide(held - mean, scale, out=np.zeros_like(held), where=active)
    covariance = np.sum(weights * fit_standard * centered_target, axis=0)
    variance = np.sum(weights * fit_standard**2, axis=0)
    return intercept + held_standard * (covariance / (variance + RIDGE))


def predict_oof(groups, box_gap, mask_gap, targets, *, all_groups=None, fit_mask=None):
    """Return integer (y, x) shift proposals and each row's fixed held-out fold.

    A fitting group contributes the same total weight as any other fitting
    group. The intercept-only control and both independently standardized
    predictors use the same training rows for each held-out fold. Groups in
    all_groups set the fold assignment even when they have no rows. A false
    fit_mask entry receives predictions but cannot contribute to any fit.
    """
    groups = _group_names(groups)
    n = len(groups)
    if n == 0:
        raise ValueError("Expected at least one prediction row")
    folds = _fold_labels(groups, all_groups)
    fit_mask = np.ones(n, dtype=bool) if fit_mask is None else np.asarray(fit_mask)
    if fit_mask.shape != (n,) or fit_mask.dtype != np.bool_:
        raise ValueError("fit_mask must be a Boolean vector with one value per row")
    box_gap = _feature_matrix(box_gap, n, "box_gap")
    mask_gap = _feature_matrix(mask_gap, n, "mask_gap")
    targets = _feature_matrix(targets, n, "targets")
    if np.any(targets != np.trunc(targets)) or np.any(np.abs(targets) > MAX_SHIFT):
        raise ValueError("Oracle shifts must be integers from -2 to 2 pixels")

    groups = np.asarray(groups)
    output = {name: np.zeros((n, 2), dtype=np.int64) for name in ("identity", "constant", "box", "mask")}
    output["folds"] = folds
    for fold in range(FOLDS):
        held = folds == fold
        if not held.any():
            continue
        fit = fit_mask & ~held
        if not fit.any():
            raise ValueError(f"Fold {fold} has no admissible training rows")
        _, group_index, counts = np.unique(groups[fit], return_inverse=True, return_counts=True)
        weights = 1.0 / counts[group_index, None]
        weights /= weights.sum()
        fit_target = targets[fit]
        intercept = np.sum(weights * fit_target, axis=0)
        centered_target = fit_target - intercept
        output["constant"][held] = _round_shift(intercept)
        output["box"][held] = _round_shift(_ridge(box_gap[fit], box_gap[held], weights, centered_target, intercept))
        output["mask"][held] = _round_shift(_ridge(mask_gap[fit], mask_gap[held], weights, centered_target, intercept))
    return output
