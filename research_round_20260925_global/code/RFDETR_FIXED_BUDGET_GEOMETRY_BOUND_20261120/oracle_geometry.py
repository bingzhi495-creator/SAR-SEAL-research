"""Fixed-count geometric upper bounds on an explicit native-cell support.

Pure standard-library arithmetic. No files, annotations, models, AP or fitting.
O_U keeps equal add/drop counts; O_F is area-free at the same total count.
O_F here does not mean the experiment's fixed-pair policy SF.
"""
from math import fsum, isfinite
from numbers import Real

ROUND_OFF = 1e-8
OCCUPANCY_ROUND_OFF = 1e-12


def need(ok, message):
    if not ok:
        raise ValueError(message)


def _number(value):
    need(isinstance(value, Real) and not isinstance(value, bool), 'Finite real scalar, not bool')
    value = float(value)
    need(isfinite(value), 'Finite real scalar')
    return value


def _support(values, indices):
    values = tuple(_number(v) for v in values)
    indices = tuple(indices)
    need(len(values) == len(indices) and all(0 <= v <= 1 + OCCUPANCY_ROUND_OFF for v in values),
         'One fractional unit occupancy in [0,1+1e-12] per native pixel')
    need(all(type(i) is int and i >= 0 for i in indices) and len(set(indices)) == len(indices),
         'Unique immutable nonnegative native pixel indices')
    return values, indices


def _prefix(values, indices, descending):
    ordered = sorted(zip(values, indices), key=lambda row: (-row[0] if descending else row[0], row[1]))
    # fsum, not a lower-precision cumulative tensor or rounded comparison.
    sums = [fsum(row[0] for row in ordered[:n]) for n in range(len(ordered) + 1)]
    return ordered, sums


def candidate_oracles(t_add, t_drop, I, P, T, edits, *, add_indices, drop_indices):
    """Return exact feasible-count maxima for one fixed original valid match.

    Inputs are already established disjoint background/foreground cell supports,
    their true occupancies, and original continuous intersection/mask/GT areas.
    ``edits=2*k`` is the observed budget, never selected using these truths.
    The caller owns original match, seed and source identities. Equal float IoUs
    break ties by the sorted immutable edited pixel indices, not input ordering.
    """
    t_add, add_indices = _support(t_add, add_indices)
    t_drop, drop_indices = _support(t_drop, drop_indices)
    need(not set(add_indices).intersection(drop_indices), 'Add and drop supports must be disjoint')
    I, P, T = (_number(v) for v in (I, P, T))
    need(P >= 0 and P.is_integer() and T > 0 and -ROUND_OFF <= I <= min(P, T) + ROUND_OFF,
         'Original integer mask area and physical nonignored GT intersection')
    original_union = P + T - I
    need(original_union > 0, 'Original positive union required')
    need(type(edits) is int and edits >= 0 and edits % 2 == 0, 'Observed nonnegative even edit count')
    k = edits // 2
    na, nd = len(t_add), len(t_drop)
    need(k <= min(na, nd), 'Observed balanced count must fit both original supports')
    need(nd <= P and fsum(t_drop) <= I + ROUND_OFF
         and fsum(1 - q for q in t_drop) <= P - I + ROUND_OFF
         and fsum(t_add) <= T - I + ROUND_OFF,
         'Support occupancies must fit the original intersection, FP and FN mass')
    add, add_sums = _prefix(t_add, add_indices, True)
    drop, drop_sums = _prefix(t_drop, drop_indices, False)
    original_iou = I / original_union

    def at(a, d):
        added, removed = add_sums[a], drop_sums[d]
        after_I = fsum((I, added, -removed))
        after_P = P + a - d
        union = fsum((after_P, T, -after_I))
        need(after_P >= 0 and -ROUND_OFF <= after_I <= min(after_P, T) + ROUND_OFF and union > 0,
             'Physical oracle intersection and positive union')
        selected_add = sorted(row[1] for row in add[:a])
        selected_drop = sorted(row[1] for row in drop[:d])
        return dict(I=after_I, P=after_P, T=T, iou=after_I / union,
                    adds=a, drops=d, edits=a + d, area_delta=a - d,
                    added_intersection=added, removed_intersection=removed,
                    intersection_gain=fsum((added, -removed)),
                    add_indices=selected_add, drop_indices=selected_drop,
                    flipped=sorted(selected_add + selected_drop),
                    gain_over_original=after_I / union - original_iou)

    balanced = at(k, k)
    feasible_adds = range(max(0, edits - nd), min(edits, na) + 1)
    options = [at(a, edits - a) for a in feasible_adds]
    free = min(options, key=lambda row: (-row['iou'], tuple(row['flipped'])))
    need(free['edits'] == balanced['edits'] == edits and free['iou'] >= balanced['iou'],
         'Equal realized capacity and nested feasible sets')
    return dict(schema='RFDETR-fixed-count-support-oracles-v1',
                original=dict(I=I, P=P, T=T, iou=original_iou),
                budget=dict(edits=edits, k=k, k_zero=k == 0, matched_capacity=True,
                            add_support_count=na, drop_support_count=nd,
                            balanced_max_edits=2 * min(na, nd), area_free_max_edits=na + nd,
                            balanced_support_unavailable=min(na, nd) == 0,
                            feasible_area_free_add_counts=list(feasible_adds)),
                oracles=dict(O_U=balanced, O_F=free),
                losses=dict(O_F_minus_O_U=free['iou'] - balanced['iou']),
                comparison='unrounded Python binary64 IoU; exact ties use sorted native pixel indices',
                roundoff_tolerance=ROUND_OFF,
                occupancy_roundoff_tolerance=OCCUPANCY_ROUND_OFF,
                scope='Truth-only fixed-match support bound, not a predictor, AP result or method claim')
