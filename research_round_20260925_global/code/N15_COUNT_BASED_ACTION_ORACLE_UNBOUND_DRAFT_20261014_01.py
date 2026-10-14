"""UNBOUND_DRAFT: pure integer action oracle, developed with synthetic-only tests.

No SAR data, masks, probabilities, model, Torch, scientific imports or files.
Each candidate point has one FIXED after-action label in exactly one class:
a=FP->TN, b=FN->TP, c=TP->FN, d=TN->FP, z=no change.
The caller supplies real, disjoint candidate counts; no virtual noop is added.
Empty union has IoU=1. Counts and returned rational values are exact integers.
"""
from fractions import Fraction

CLASS_KEYS = ("a", "b", "c", "d", "z")


def _nonnegative_integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(name + " must be a nonnegative integer (bool is not a count)")


def oracle_counts(nativeTP, FP, FN, classes, K, mode):
    """Return optimal fixed-action selections within a real candidate pool.

    atmost allows unused budget. exact uses helpful actions, physical noops,
    then TN->FP before TP->FN if harmful actions are unavoidable.
    The exact mode class priority is b,a,z,d,c; atmost uses b,a and leaves any
    remaining budget unused. Within-class point identities and IoU ties are
    not resolved here; atmost does not promise minimum flips among all optima.
    TN and image size are not inputs: the caller must establish physical class
    disjointness/coverage and available TN. This function checks a<=FP, b<=FN,
    c<=nativeTP and K<=candidate support; it cannot infer a real SAR pool.
    """
    for name, value in (("nativeTP", nativeTP), ("FP", FP), ("FN", FN), ("K", K)):
        _nonnegative_integer(value, name)
    if type(classes) is not dict or set(classes) != set(CLASS_KEYS):
        raise ValueError("classes must contain exactly a,b,c,d,z")
    for key in CLASS_KEYS:
        _nonnegative_integer(classes[key], "classes." + key)
    if classes["a"] > FP or classes["b"] > FN or classes["c"] > nativeTP:
        raise ValueError("a,b,c exceed the matching native FP,FN,TP counts")
    support = sum(classes.values())
    if K > support:
        raise ValueError("K exceeds the real candidate support")
    if type(mode) is not str or mode not in ("atmost", "exact"):
        raise ValueError("mode must be atmost or exact")
    selected = dict.fromkeys(CLASS_KEYS, 0)
    remaining = K
    for key in (("b", "a", "z", "d", "c") if mode == "exact" else ("b", "a")):
        selected[key] = min(classes[key], remaining)
        remaining -= selected[key]
    if mode == "exact" and remaining:
        raise RuntimeError("Internal inconsistency: real pool could not fill exact budget")
    tp = nativeTP + selected["b"] - selected["c"]
    fp = FP - selected["a"] + selected["d"]
    fn = FN - selected["b"] + selected["c"]
    union = tp + fp + fn
    ratio = Fraction(tp, union) if union else Fraction(1, 1)
    flips = selected["a"] + selected["b"] + selected["c"] + selected["d"]
    return dict(mode=mode, K=K, native_counts=dict(TP=nativeTP, FP=FP, FN=FN),
                classes=dict(classes), candidate_support=support,
                legal_counts=True, legal_counts_scope="count constraints only; physical domain is caller evidence",
                selected_counts=selected,
                selected_support=sum(selected.values()), actual_flips=flips,
                selected_noops=selected["z"],
                helpful_flips=selected["a"] + selected["b"],
                harmful_flips=selected["c"] + selected["d"],
                after_counts=dict(TP=tp, FP=fp, FN=fn),
                intersection=tp, union=union, empty_union=(union == 0),
                iou=float(ratio),
                iou_fraction=dict(numerator=ratio.numerator, denominator=ratio.denominator),
                empty_union_convention="IoU=1 when TP+FP+FN=0",
                physical_pool_verified_by_helper=False,
                scope="single-instance fixed-action IoU oracle, not AP or SAR evidence")
