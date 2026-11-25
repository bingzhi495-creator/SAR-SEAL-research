"""Deterministic same-count writes for two predicted occupancy scenarios."""
import operator

import numpy as np


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


def _posteriors(values, k, name):
    table = np.asarray(values, dtype=np.float64)
    if (table.ndim != 2 or table.shape[1] != 2 or table.shape[0] < k
            or not np.isfinite(table).all()
            or np.any((table < 0.0) | (table > 1.0))):
        raise ValueError(f"{name} requires at least k rows of two finite probabilities in [0, 1]")
    return table


def _incumbent(indices, k, n):
    ordered = tuple(sorted(_integer(index, "incumbent index") for index in indices))
    if len(ordered) != k or len(set(ordered)) != k or any(i < 0 or i >= n for i in ordered):
        raise ValueError("The incumbent must contain exactly k distinct in-range indices per side")
    return ordered


def _top(scores, k, largest):
    order = np.argsort(-scores if largest else scores, kind="stable")
    return tuple(sorted(order[:k].tolist()))


def _candidate_sets(table, incumbent, k, largest):
    # The coefficients are ten times the fixed convex weights. Ranking is unchanged.
    proposals = [incumbent]
    for left in range(11):
        mixture = left * table[:, 0] + (10 - left) * table[:, 1]
        proposals.append(_top(mixture, k, largest))
    return tuple(dict.fromkeys(proposals))


def select(add, drop, incumbent_add, incumbent_drop, *, k):
    """Return k indices per side for four writers over matching scenario columns.

    Each of add and drop is a matrix [native endpoint, scenario], in its
    original deterministic tie order. Its two columns use the same scenario
    order across sides. Each value is a predicted reference occupancy.

    The robust search separately proposes each side's static set and 11
    top-k sets from fixed mixtures. It crosses their deduplicated sets and
    maximizes minimum scenario-wise signed intersection gain relative to
    static. It is exact within this finite proposal search. Including the
    incumbent guarantees a nonnegative predicted minimum change. No
    statement about true overlap or globally optimal subsets follows.

    Scores tie by endpoint input order within a top-k. Within the robust
    search, exact objective ties keep the first pair, with static first
    and subsequent pairs in first-occurrence mixture order.
    """
    k = _integer(k, "k")
    if k < 0:
        raise ValueError("k must be nonnegative")
    add = _posteriors(add, k, "add")
    drop = _posteriors(drop, k, "drop")
    static_add = _incumbent(incumbent_add, k, len(add))
    static_drop = _incumbent(incumbent_drop, k, len(drop))
    add_sets = _candidate_sets(add, static_add, k, largest=True)
    drop_sets = _candidate_sets(drop, static_drop, k, largest=False)
    add_sum = add[np.asarray(add_sets, dtype=np.intp)].sum(axis=1)
    drop_sum = drop[np.asarray(drop_sets, dtype=np.intp)].sum(axis=1)
    incumbent = add_sum[0] - drop_sum[0]
    relative = add_sum[:, None, :] - drop_sum[None, :, :] - incumbent
    objective = relative.min(axis=-1)
    a, d = np.unravel_index(objective.argmax(), objective.shape)
    added_mean = _top(add.mean(axis=1), k, largest=True)
    dropped_mean = _top(drop.mean(axis=1), k, largest=False)
    added_pessimistic = _top(add.min(axis=1), k, largest=True)
    dropped_pessimistic = _top(drop.max(axis=1), k, largest=False)
    return {
        "static": (static_add, static_drop),
        "robust": (add_sets[a], drop_sets[d]),
        "mean": (added_mean, dropped_mean),
        "cell_pessimistic": (added_pessimistic, dropped_pessimistic),
    }

