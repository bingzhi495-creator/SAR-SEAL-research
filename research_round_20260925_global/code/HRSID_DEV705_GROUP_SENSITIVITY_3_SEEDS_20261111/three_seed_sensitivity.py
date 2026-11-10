"""Apply the existing DEV705 group procedure to all three closed seed pairs."""

import math

import numpy as np

from dev705_group_sensitivity import BOOTSTRAP_DRAWS, DRAW_SEED, METRICS, prepare_closed_arm, require


SEEDS = (5201, 5202, 5203)


def metric_pairs(caches, counts):
    results = {}
    for seed, (native, center) in caches.items():
        left, right = native.scores(counts), center.scores(counts)
        require(all(math.isfinite(x) and (x == -1. or 0. <= x <= 1.)
                    for m in METRICS for x in (left[m], right[m])),
                "Finite original COCO score or no-GT sentinel required")
        require(all((left[m] < 0) == (right[m] < 0) for m in METRICS),
                "Both arms must have the same valid GT population")
        results[seed] = {m: None if left[m] < 0 else (left[m], right[m]) for m in METRICS}
    require(all((results[5201][m] is None) == (results[seed][m] is None)
                for seed in SEEDS for m in METRICS),
            "All downstream seeds must share the valid GT population")
    return results


def gain(pair):
    return 100. * (pair[1] - pair[0])


def summarize_caches(caches, *, progress=None):
    """Use the same group multiplicities for both arms of all three completed seeds."""
    require(type(caches) is dict and set(caches) == set(SEEDS), "Exactly the closed 5201, 5202 and 5203 pairs required")
    groups = caches[5201][0].groups
    require(len(groups) >= 2 and all(cache.groups == groups for pair in caches.values() for cache in pair),
            "Exact common group ordering required")
    count = len(groups)
    all_ones = np.ones(count, dtype=np.int64)
    fixed = metric_pairs(caches, all_ones)
    require(all(fixed[s][m] is not None for s in SEEDS for m in METRICS), "Valid fixed primary AP required")
    draws = {s: {m: [] for m in METRICS} for s in SEEDS}
    omissions = {s: {m: [] for m in METRICS} for s in SEEDS}
    rng = np.random.Generator(np.random.PCG64(DRAW_SEED))
    sampled = rng.integers(0, count, size=(BOOTSTRAP_DRAWS, count))
    for ordinal, picks in enumerate(sampled, 1):
        scores = metric_pairs(caches, np.bincount(picks, minlength=count))
        for s in SEEDS:
            for m in METRICS:
                if scores[s][m] is not None:
                    draws[s][m].append(gain(scores[s][m]))
        if progress is not None and ordinal % 256 == 0:
            progress(ordinal)
    for omitted in range(count):
        weights = all_ones.copy()
        weights[omitted] = 0
        scores = metric_pairs(caches, weights)
        for s in SEEDS:
            for m in METRICS:
                if scores[s][m] is not None:
                    omissions[s][m].append(gain(scores[s][m]))

    results = {}
    for s in SEEDS:
        per_seed = {}
        for m in METRICS:
            draws_m = np.asarray(draws[s][m])
            loo_m = np.asarray(omissions[s][m])
            require(draws_m.size > 0 and loo_m.size > 0, "At least one valid proxy resample and omission required")
            lower, median, upper = np.quantile(draws_m, (.025, .5, .975), method="linear")
            a, b = fixed[s][m]
            per_seed[m] = dict(
                fixed_native_fraction=a, fixed_center_fraction=b, fixed_gain_points=gain((a, b)),
                bootstrap=dict(valid_draws=draws_m.size, no_gt_draws=BOOTSTRAP_DRAWS - draws_m.size,
                               positive_draws=int((draws_m > 0).sum()),
                               percentile_025_gain_points=float(lower), median_gain_points=float(median),
                               percentile_975_gain_points=float(upper)),
                leave_one_out=dict(valid_omissions=loo_m.size, no_gt_omissions=count - loo_m.size,
                                   positive_omissions=int((loo_m > 0).sum()),
                                   min_gain_points=float(loo_m.min()), max_gain_points=float(loo_m.max())),
            )
        results[str(s)] = per_seed
    return dict(groups=count, bootstrap_draws=BOOTSTRAP_DRAWS, bootstrap_seed=DRAW_SEED,
                generator="numpy_PCG64", seeds=results)

