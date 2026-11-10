"""Fixed-budget FIT point allocation and paired frozen-logit diagnostics."""

import numpy as np


POINTS_PER_STRATUM = 4
TARGETS = ("hard_c", "raster", "polygon")
TARGET_PAIRS = (("hard_c", "raster"), ("hard_c", "polygon"), ("raster", "polygon"))


def _vector(value, name, length=None):
    array = np.asarray(value)
    if array.ndim != 1 or not array.size or (length is not None and len(array) != length) or array.dtype.kind not in "bifu":
        raise ValueError(f"Expected aligned nonempty numeric vector for {name}")
    return array


def _hard(value, name, length):
    array = _vector(value, name, length)
    if not np.isin(array, (0, 1)).all():
        raise ValueError("Hard C and B targets must be binary")
    return array.astype(np.bool_)


def _occupancy(value, name, length):
    array = _vector(value, name, length).astype(np.float64)
    if not np.isfinite(array).all() or np.any((array < -1e-12) | (array > 1 + 1e-12)):
        raise ValueError("Soft occupancy must be finite and within [0, 1]")
    return np.clip(array, 0., 1.)


def _banks(strata, proxy, logits, seed):
    tie_seed, random_seed = np.random.SeedSequence(seed).spawn(2)
    ties = np.random.Generator(np.random.PCG64(tie_seed)).permutation(len(strata))
    rng = np.random.Generator(np.random.PCG64(random_seed))
    banks = {name: [] for name in ("phase", "uncertainty", "random")}
    counts, quotas = [], []
    weights = np.zeros(len(strata), dtype=np.float64)
    effective = False
    for group in range(4):
        members = np.flatnonzero(strata == group)
        n = len(members)
        k = min(POINTS_PER_STRATUM, n)
        counts.append(n)
        quotas.append(k)
        if not k:
            continue
        phase_order = np.lexsort((ties[members], -proxy[members].astype(np.int8)))
        uncertainty_order = np.lexsort((ties[members], np.abs(logits[members])))
        banks["phase"].extend(members[phase_order[:k]])
        banks["uncertainty"].extend(members[uncertainty_order[:k]])
        banks["random"].extend(rng.choice(members, size=k, replace=False))
        weights[members] = k / n
        effective |= k < n and 0 < np.count_nonzero(proxy[members]) < n
    k = sum(quotas)
    return {name: np.asarray(sorted(values), dtype=np.int64) for name, values in banks.items()}, counts, quotas, weights / k, bool(effective)


def _measure(targets, logits, probabilities, added, removed, *, indices=None, weights=None):
    k = len(indices) if indices is not None else None

    def mean(values):
        return float(np.mean(values[indices])) if indices is not None else float(np.dot(weights, values))

    hard, raster, polygon = (targets[key] for key in TARGETS)
    raw = dict(
        mean_abs_polygon_minus_raster=mean(np.abs(polygon-raster)),
        mean_abs_raster_minus_hard_c=mean(np.abs(raster-hard)),
        mean_abs_polygon_minus_hard_c=mean(np.abs(polygon-hard)),
        mean_abs_logit=mean(np.abs(logits)), mean_probability=mean(probabilities),
    )
    losses = {}
    signs = {}
    log_positive = np.logaddexp(0., -logits)
    log_negative = np.logaddexp(0., logits)
    for name, target in targets.items():
        if indices is None:
            dice = None
        else:
            p, t = probabilities[indices], target[indices]
            dice = float(1. - (2. * np.dot(p, t) + 1.) / (p.sum() + t.sum() + 1.))
        losses[name] = dict(mean_target=mean(target), mean_bce=mean(target * log_positive + (1.-target) * log_negative),
                            dice_loss=dice, mean_abs_bce_gradient=mean(np.abs(probabilities-target)))
        with np.errstate(divide="ignore"):
            signs[name] = np.sign(logits - (np.log(target) - np.log1p(-target)))

    directions = {}
    for previous, current in TARGET_PAIRS:
        before, after = signs[previous], signs[current]
        directions[current + "_vs_" + previous] = dict(
            positive_to_negative=(before > 0) & (after < 0),
            negative_to_positive=(before < 0) & (after > 0),
            zero_to_nonzero=(before == 0) & (after != 0),
            nonzero_to_zero=(before != 0) & (after == 0),
        )
    count_arrays = dict(proxy_points=added | removed, c_adds=added, c_removes=removed)
    if k is not None:
        counts = {name: int(np.count_nonzero(value[indices])) for name, value in count_arrays.items()}
        flips = {name: {direction: int(np.count_nonzero(value[indices])) for direction, value in group.items()}
                 for name, group in directions.items()}
    else:
        counts = {name: mean(value) for name, value in count_arrays.items()}
        flips = {name: {direction: mean(value) for direction, value in group.items()} for name, group in directions.items()}
    return dict(**raw, **counts, targets=losses, gradient_direction_changes=flips)


def summarize_row(hard_c, hard_b, raster, polygon, logits, strata, seed):
    """Compare phase proxy, low-|logit| and random selection on one common row.

    Input order is the fixed candidate-cell order. Strata 0..3 are supplied by
    the caller. The exact random expectation uses identical per-stratum quotas.
    Targets R and P are evaluated only after every bank has been selected.
    """
    logits = _vector(logits, "logits").astype(np.float64)
    if not np.isfinite(logits).all():
        raise ValueError("Frozen logits must be finite")
    strata = _vector(strata, "strata", len(logits))
    if strata.dtype.kind not in "iu" or not np.isin(strata, range(4)).all():
        raise ValueError("Expected integer strata 0 through 3")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("Expected nonnegative row seed")
    c = _hard(hard_c, "hard_c", len(logits))
    b = _hard(hard_b, "hard_b", len(logits))
    proxy = b != c
    indices, counts, quotas, weights, effective = _banks(strata, proxy, logits, int(seed))
    r = _occupancy(raster, "raster", len(logits))
    p = _occupancy(polygon, "polygon", len(logits))
    added, removed = c & ~b, b & ~c
    probabilities = np.exp(-np.logaddexp(0., -logits))
    targets = dict(hard_c=c.astype(np.float64), raster=r, polygon=p)
    banks = {name: _measure(targets, logits, probabilities, added, removed, indices=selected)
             for name, selected in indices.items()}
    expected = _measure(targets, logits, probabilities, added, removed, weights=weights)
    k = sum(quotas)
    for key in ("proxy_points", "c_adds", "c_removes"):
        expected[key] *= k
    for group in expected["gradient_direction_changes"].values():
        for direction in group:
            group[direction] *= k
    banks["random_expected"] = expected
    return dict(n=len(logits), k=k, strata_counts=counts, strata_quotas=quotas, proxy_points=int(np.count_nonzero(proxy)),
                effective_proxy=effective, banks=banks)
