"""UNBOUND pure count oracles for static, distinct, noninteracting pixel flips.

No I/O, models, real labels or scientific execution. These are conditional
fixed-pair IoU calculations, never AP bounds or causal attribution. Callers
must establish candidate visibility and that the fixed PR actions really are
the counted flips; these facts cannot be inferred from intersection/union.
"""
from fractions import Fraction


def _count(name, value):
    if type(value) is not int or value < 0:
        raise ValueError(name + ' must be a nonnegative nonboolean integer')
    return value


def _native(intersection, union, native_fn, native_fp):
    for name, value in (('intersection', intersection), ('union', union),
                        ('native_fn', native_fn), ('native_fp', native_fp)):
        _count(name, value)
    if intersection > union or union != intersection + native_fn + native_fp:
        raise ValueError('Require U = I + native_FN + native_FP and I <= U')
    if union > 0 and intersection + native_fn == 0:
        raise ValueError('This fixed-pair oracle requires nonempty GT')


def _result(intersection, union, budget, native_i, native_u,
            add_fn, remove_fp, remove_tp=0, add_tn=0, selected=None):
    if not 0 <= intersection <= union:
        raise ValueError('Adjusted intersection/union are inconsistent')
    flips = add_fn + remove_fp + remove_tp + add_tn
    selected = flips if selected is None else selected
    empty = native_u == 0
    return dict(status='NA_EMPTY_NATIVE_UNION' if empty else 'OK',
                intersection=intersection, union=union,
                iou=None if empty else Fraction(intersection, union),
                native_iou=None if empty else Fraction(native_i, native_u),
                add_FN=add_fn, remove_FP=remove_fp, remove_TP=remove_tp, add_TN=add_tn,
                flips=flips, selected_count=selected,
                unchanged_selected=selected-flips, unused_budget=budget-selected)


def beneficial_oracle(intersection, union, repairable_fn, repairable_fp, budget,
                      *, native_fn, native_fp):
    """Exact best IoU among at-most-K available beneficial fixed PR flips.

    Put t=min(K,A+R). Beneficial flips cannot lower IoU, so some optimum uses
    t flips. For fixed t and feasible a2>a1, their IoU difference is
    (a2-a1)*(U-I-t)/((U-t+a1)*(U-t+a2)), which is nonnegative because
    t<=A+R<=native_FN+native_FP=U-I. Thus the greatest feasible FN count,
    a=min(A,t), is globally optimal; the rest remove FP. Nonempty GT keeps
    every feasible denominator positive. This does not use local gradients.
    """
    _native(intersection, union, native_fn, native_fp)
    for name, value in (('repairable_fn', repairable_fn),
                        ('repairable_fp', repairable_fp), ('budget', budget)):
        _count(name, value)
    if repairable_fn > native_fn or repairable_fp > native_fp:
        raise ValueError('Available beneficial flips exceed native error counts')
    t = min(budget, repairable_fn + repairable_fp)
    a = min(repairable_fn, t)
    r = t - a
    if t > union-intersection or r > repairable_fp:
        raise ValueError('Infeasible static beneficial action counts')
    return _result(intersection+a, union-r, budget, intersection, union, a, r)


def ideal_candidate_oracle(intersection, union, candidate_fn, candidate_fp, budget,
                           *, native_fn, native_fp):
    """Ideal GT-informed error flips inside C, under the same at-most budget.

    This enlarges action availability, not candidate membership. It is an IoU
    counterfactual for the given fixed pair, not an AP or model upper bound.
    """
    return beneficial_oracle(intersection, union, candidate_fn, candidate_fp, budget,
                             native_fn=native_fn, native_fp=native_fp)


def router_outcome(intersection, union, budget, *, native_fn, native_fp,
                   candidate_fn, candidate_fp, candidate_tp, candidate_tn,
                   b_FN, b_FP, h_TP, h_TN, selected_count):
    """Exact observed action accounting; selected_count includes unchanged slots.

    Candidate categories must come from one distinct fixed C. This function
    checks aggregate feasibility, not the provenance of point identities.
    """
    _native(intersection, union, native_fn, native_fp)
    counts = dict(budget=budget, candidate_fn=candidate_fn, candidate_fp=candidate_fp,
                  candidate_tp=candidate_tp, candidate_tn=candidate_tn,
                  b_FN=b_FN, b_FP=b_FP, h_TP=h_TP, h_TN=h_TN, selected_count=selected_count)
    for name, value in counts.items():
        _count(name, value)
    if (candidate_fn > native_fn or candidate_fp > native_fp or candidate_tp > intersection
            or b_FN > candidate_fn or b_FP > candidate_fp
            or h_TP > candidate_tp or h_TN > candidate_tn):
        raise ValueError('Candidate category or flipped action exceeds its native availability')
    flips = b_FN + b_FP + h_TP + h_TN
    candidate_count = candidate_fn + candidate_fp + candidate_tp + candidate_tn
    if not flips <= selected_count <= min(budget, candidate_count):
        raise ValueError('Need flips <= selected_count <= min(K, distinct C count)')
    return _result(intersection+b_FN-h_TP, union-b_FP+h_TN,
                   budget, intersection, union, b_FN, b_FP, h_TP, h_TN, selected_count)


IOU_THRESHOLDS = tuple(Fraction(n, 20) for n in range(10, 20))


def threshold_crossings(before, after):
    """Ten fixed pair-IoU threshold transitions; never interpret these as AP."""
    for value in (before, after):
        if value is not None and (type(value) not in (int, Fraction) or not 0 <= value <= 1):
            raise ValueError('Use an exact Fraction/int in [0,1] or explicit NA None')
    if before is None or after is None:
        return dict(status='NA', thresholds=IOU_THRESHOLDS, up=None, down=None)
    return dict(status='OK', thresholds=IOU_THRESHOLDS,
                up=tuple(t for t in IOU_THRESHOLDS if before < t <= after),
                down=tuple(t for t in IOU_THRESHOLDS if after < t <= before))
