"""Pure same-budget RF-DETR mask-matcher attribution on fixed sampled tensors."""
from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

GROUP_WIDTH = 100
GROUPS = 13
K = 1296
TILES = 36
DENSE_SIDE = 576
DENSE_PHASES = ((.5, .5), (.25, .75))
MASK_WEIGHTS = (5., 5.)
THRESHOLDS = {"iou75": (3, 4), "iou90": (9, 10)}


def need(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class DenseCosts:
    bce: np.ndarray
    dice: np.ndarray
    intersection: np.ndarray
    union: np.ndarray


def paired_points(uniform):
    """Reuse the exact K uniforms, once globally and once within equal-area cells."""
    uniform = np.asarray(uniform)
    need(uniform.dtype == np.float32 and uniform.shape == (K, 2)
         and np.isfinite(uniform).all() and np.all((uniform >= 0) & (uniform < 1)),
         "Expected native float32 [1296,2] unit-square uniforms")
    # Point order matches stock [x,y]. Cells are traversed x first.
    y, x = np.divmod(np.arange(K), TILES)
    cells = np.column_stack((x, y))
    lo, hi = cells / TILES, (cells + 1) / TILES
    jitter = ((cells + uniform.astype(np.float64)) / TILES).astype(np.float32)
    # A float32 round at a rational cell edge must not change its tile.
    jitter = np.where(jitter < lo, np.nextafter(jitter, np.float32(np.inf)), jitter)
    jitter = np.where(jitter >= hi, np.nextafter(jitter, np.float32(-np.inf)), jitter)
    need(np.array_equal(np.floor(jitter.astype(np.float64) * TILES), cells),
         "Exactly one point remains inside every tile")
    return uniform, jitter


def uniform_points(seed, count):
    """Deterministic extra draws. The caller supplies the observed native first draw."""
    rng = np.random.default_rng(seed)
    for _ in range(count):
        yield rng.random((K, 2), dtype=np.float32)


def checked_samples(logits, targets, points):
    logits, targets = np.asarray(logits), np.asarray(targets)
    need(logits.ndim == targets.ndim == 2 and logits.shape[1] == targets.shape[1] == points
         and logits.dtype.kind == "f" and np.isfinite(logits).all()
         and np.all((targets == 0) | (targets == 1)), "Aligned finite logits and nearest binary target samples")
    return logits, targets


def mask_cost_terms(logits, targets):
    """Native pairwise BCE and soft-Dice sufficient statistics, before reductions."""
    probs = np.exp(-np.logaddexp(0., -logits))
    positives = np.logaddexp(0., -logits) @ targets.T
    negatives = np.logaddexp(0., logits) @ (1 - targets).T
    return positives + negatives, 2 * (probs @ targets.T), (
        probs.sum(axis=1)[:, None] + targets.sum(axis=1)[None, :])


def sampled_cost(logits, targets):
    """K-point cost with the native additive +1 Dice smoothing."""
    logits, targets = checked_samples(logits, targets, K)
    sums, numerator, denominator = mask_cost_terms(logits.astype(np.float32, copy=False),
                                                  targets.astype(np.float32, copy=False))
    return sums / K, 1 - (numerator + 1) / (denominator + 1)


def dense_cost(sample, *, side=DENSE_SIDE, phase=DENSE_PHASES[0], chunk=2048):
    """Stream a deterministic quadrature and keep smoothing in native K units."""
    need(type(side) is int and side > 0 and type(chunk) is int and chunk > 0
         and len(phase) == 2 and all(0 < a < 1 for a in phase), "Valid dense quadrature")
    sums = None
    shape = None
    n = side * side
    for lo in range(0, n, chunk):
        k = np.arange(lo, min(lo + chunk, n))
        y, x = np.divmod(k, side)
        points = np.column_stack(((x + phase[0]) / side, (y + phase[1]) / side)).astype(np.float32)
        logits, target = checked_samples(*sample(points), len(points))
        current = (len(logits), len(target))
        if sums is None:
            sums = [np.zeros(current, dtype=np.float64) for _ in range(5)]
            shape = current
        need(current == shape, "Quadrature callback returns the same query and target rows")
        logits = logits.astype(np.float64, copy=False)
        target = target.astype(np.float64, copy=False)
        pred = (logits > 0).astype(np.float64)
        inter = pred @ target.T
        union = pred.sum(axis=1)[:, None] + target.sum(axis=1)[None, :] - inter
        for total, add in zip(sums, (*mask_cost_terms(logits, target), inter, union), strict=True):
            total += add
    bce, numerator, denominator, inter, union = sums
    # Adding 1 after raw dense sums would use +1/N in mean units, changing the cost.
    smooth = 1. / K
    return DenseCosts(bce / n, 1 - (numerator / n + smooth) / (denominator / n + smooth),
                      inter.astype(np.int64), union.astype(np.int64))


def combine_costs(nonmask, bce, dice, mask_weights=MASK_WEIGHTS):
    """Preserve native left-associative class/box + mask CE + mask Dice order."""
    bce, dice, nonmask = (np.asarray(x, dtype=np.float32) for x in (bce, dice, nonmask))
    need(nonmask.ndim == 2 and nonmask.shape == bce.shape == dice.shape
         and len(mask_weights) == 2 and all(math.isfinite(x) and x >= 0 for x in mask_weights),
         "Identical query/GT matrix and valid mask weights")
    result = (nonmask + np.float32(mask_weights[0]) * bce) + np.float32(mask_weights[1]) * dice
    need(np.isfinite(result).all(), "Finite grouped matcher costs")
    return result


def solve_groups(cost, *, width=GROUP_WIDTH, groups=GROUPS):
    """Apply one solver to each of the same contiguous query groups."""
    cost = np.asarray(cost)
    need(cost.ndim == 2 and type(width) is type(groups) is int and width > 0 and groups > 0
         and cost.shape[0] == width * groups and cost.shape[1] <= width and np.isfinite(cost).all(),
         "Fixed contiguous query groups and real GT columns")
    targets = cost.shape[1]
    out = np.empty((groups, targets), dtype=np.int64)
    for g in range(groups):
        lo = g * width
        rows, cols = linear_sum_assignment(cost[lo:lo + width])
        need(len(cols) == targets, "Every target assigned exactly once per group")
        out[g, cols] = rows + lo
    return out


def assignment_from_native_pairs(source, target, *, targets, width=GROUP_WIDTH, groups=GROUPS):
    """Re-index observed native (global query, target) pairs into group × GT order."""
    source, target = np.asarray(source), np.asarray(target)
    need(source.shape == target.shape == (groups * targets,) and source.dtype.kind in "iu"
         and target.dtype.kind in "iu" and np.all((source >= 0) & (source < groups * width))
         and np.all((target >= 0) & (target < targets)), "Exact native source and GT index scope")
    out = np.full((groups, targets), -1, dtype=np.int64)
    out[source // width, target] = source
    need(np.all(out >= 0) and all(len(np.unique(row)) == targets for row in out),
         "Original one-to-one grouped assignment")
    return out


def take_assignments(matrix, assigned):
    return matrix[assigned, np.arange(matrix.shape[1])]


def exact_native_anchor(nonmask, native_bce, native_dice, native_total, native_assignment, mask_weights):
    """Audit captured stock recomposition without mixing it into the paired arms."""
    recombined = combine_costs(nonmask, native_bce, native_dice, mask_weights)
    native_total = np.asarray(native_total, dtype=np.float32)
    native_assignment = np.asarray(native_assignment)
    need(native_assignment.ndim == 2 and recombined.shape == native_total.shape
         and native_assignment.shape[1] == native_total.shape[1]
         and native_assignment.shape[0] > 0 and native_total.shape[0] % native_assignment.shape[0] == 0,
         "Native anchor matrix and assignment roles")
    groups = native_assignment.shape[0]
    width = native_total.shape[0] // groups
    recomputed = solve_groups(native_total, width=width, groups=groups)
    observed_objective = take_assignments(native_total, native_assignment).sum(axis=1, dtype=np.float64)
    solver_objective = take_assignments(native_total, recomputed).sum(axis=1, dtype=np.float64)
    return dict(native_total_exact=bool(np.array_equal(recombined, native_total)),
                solver_pairs_exact=bool(np.array_equal(recomputed, native_assignment)),
                max_group_objective_gap=float(np.max(np.abs(observed_objective - solver_objective), initial=0.)))


def reference_counts(good, reference, check, scope, *, width=GROUP_WIDTH):
    """Separate absent masks, one-to-one competition, and dense cost choice."""
    groups, targets = reference.shape
    need(good.shape == (groups * width, targets) and check.shape == reference.shape
         and scope.shape == (targets,), "Grouped good-mask matrix and GT scope")
    # Maximizing scoped good assignments is a cardinality oracle only. Other GT still
    # occupy a unique query but their success is not counted in this scoped capacity.
    achievable = good.reshape(groups, width, targets).any(axis=1)
    oracle = solve_groups(-np.asarray(good & scope[None, :], dtype=np.int8), width=width, groups=groups)
    possible = int((achievable & scope).sum())
    capacity = int((take_assignments(good, oracle) & scope).sum())
    dense = take_assignments(good, reference)
    alternate = take_assignments(good, check)
    dense_count = int((dense & scope).sum())
    need(dense_count <= capacity <= possible, "One-to-one capacity must contain dense successes")
    return dict(
        targets=groups * int(scope.sum()),
        individually_good=possible, maximum_one_to_one_good=capacity,
        no_good_query=groups * int(scope.sum()) - possible, query_competition_gap=possible - capacity,
        dense_objective_gap=capacity - dense_count, dense_good=dense_count,
        alternate_dense_good=int((alternate & scope).sum()),
        stable_dense_good=int((dense & alternate & scope).sum()),
        dense_quality_unstable=int(((dense != alternate) & scope).sum()),
        dense_query_stable=int(((reference == check) & scope).sum()),
    )


def sample_counts(good, reference, check, assigned, scope):
    """Use only quality-stable dense assignments as opportunity or gain anchors."""
    dense, alternate, sampled = (take_assignments(good, x) for x in (reference, check, assigned))
    stable_good = dense & alternate & scope
    stable_bad = ~dense & ~alternate & scope
    return dict(stable_reference_good=int(stable_good.sum()),
                stable_reference_good_lost=int((stable_good & ~sampled).sum()),
                stable_reference_bad_gained=int((stable_bad & sampled).sum()),
                dense_quality_unstable=int(((dense != alternate) & scope).sum()),
                sampled_good=int((sampled & scope).sum()),
                sampled_query_changes=int(((reference != assigned) & scope).sum()))


def diagnose_view(nonmask, sample, is_small, mask_weights, uniform_draws, *,
                  width=GROUP_WIDTH, groups=GROUPS, side=DENSE_SIDE, chunk=2048,
                  native_sampled_cost=None, native_sampled_total=None):
    """Compare stock i.i.d. and whole-grid jitter without changing a frozen forward.

    sample(coords) returns logits and nearest binary target values in the exact
    original query/GT order. If supplied, native_sampled_cost(coords) computes
    the official mask-cost helpers for both arms with those same coordinates.
    A full native cost callback also preserves the observed arithmetic dtypes.
    """
    nonmask, is_small = np.asarray(nonmask), np.asarray(is_small)
    need(nonmask.ndim == 2 and nonmask.shape[0] == width * groups
         and is_small.shape == (nonmask.shape[1],) and is_small.dtype == np.bool_,
         "Fixed native single-image row and augmented small-GT classification")
    scopes = dict(small=is_small, other=~is_small, all=np.ones_like(is_small))
    dense, check = [dense_cost(sample, side=side, phase=phase, chunk=chunk) for phase in DENSE_PHASES]
    reference_cost = combine_costs(nonmask, dense.bce, dense.dice, mask_weights)
    check_cost = combine_costs(nonmask, check.bce, check.dice, mask_weights)
    ref, alt = [solve_groups(c, width=width, groups=groups) for c in (reference_cost, check_cost)]
    good_masks = {
        tag: (dense.union > 0) & (den * dense.intersection >= num * dense.union)
        for tag, (num, den) in THRESHOLDS.items()
    }
    references = {
        tag: {name: reference_counts(good, ref, alt, scope, width=width) for name, scope in scopes.items()}
        for tag, good in good_masks.items()
    }
    reference_objective = take_assignments(reference_cost, ref).sum(dtype=np.float64)
    draws = []
    for uniform in uniform_draws:
        pair = {}
        for label, points in zip(("iid", "jitter"), paired_points(uniform), strict=True):
            if native_sampled_total is not None:
                cost = native_sampled_total(points)
                need(cost.shape == nonmask.shape and cost.dtype == np.float32,
                     "Exact native grouped total cost shape and type")
            else:
                cost_sampler = native_sampled_cost
                if cost_sampler is None:
                    ce, dice = sampled_cost(*sample(points))
                else:
                    ce, dice = cost_sampler(points)
                cost = combine_costs(nonmask, ce, dice, mask_weights)
            assigned = solve_groups(cost, width=width, groups=groups)
            outcomes = {
                tag: {name: sample_counts(good, ref, alt, assigned, scope) for name, scope in scopes.items()}
                for tag, good in good_masks.items()
            }
            outcomes["dense_cost_regret"] = float(
                take_assignments(reference_cost, assigned).sum(dtype=np.float64) - reference_objective)
            pair[label] = outcomes
        draws.append(pair)
    need(len(draws) > 0, "At least the first stock draw is needed")
    cost_source = "numpy_equivalent"
    if native_sampled_total is not None:
        cost_source = "native_total"
    elif native_sampled_cost is not None:
        cost_source = "native_helpers"
    return dict(point_budget=K, groups=groups, group_width=width, dense_side=side,
                dense_phases=[list(x) for x in DENSE_PHASES], dense_smoothing=1. / K,
                sampled_cost_source=cost_source,
                reference=references, draws=draws)
