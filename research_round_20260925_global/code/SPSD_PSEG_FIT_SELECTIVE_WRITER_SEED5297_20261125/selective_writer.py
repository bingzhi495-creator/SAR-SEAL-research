"""Fixed-width selective writer features and a group-balanced calibration fit."""
from collections import Counter, defaultdict
import operator

import numpy as np


FEATURE_NAMES = ("predicted_mean_gain", "predicted_half_gap", "native_probability_gain",
                 "log2_mask_area", "detector_score", "exchange_fraction", "point_fraction",
                 "changed_add_fraction", "changed_drop_fraction")
ALPHA = .02
MODEL_STATUS = "SPSD_PSEG_SELECTIVE_WRITER_RIDGE"
TARGET = "normalized_true_min_signed_intersection"


def need(flag, reason):
    if not flag:
        raise ValueError(reason)


def signed_vector(action, static, n, k):
    """Write the signed change from incumbent among shared native endpoints."""
    k = operator.index(k)
    need(0 < k <= n and len(action) == len(static) == 2, "Positive native quota and two signed sides")
    delta = np.zeros((n,2),dtype=np.int8)
    for side, sign in enumerate((1,-1)):
        for sets, coefficient in ((action,sign),(static,-sign)):
            own = np.asarray(sets[side])
            need(own.shape == (k,) and own.dtype.kind in "iu" and len(set(own)) == k
                 and np.all((0 <= own) & (own < n)), "Exact distinct quota-matched side indices")
            delta[own,side] += coefficient
    return delta


def features(raw, posterior, robust, static, k):
    """Return the locked per-candidate nine scalars and raw normalized minimum."""
    raw = np.asarray(raw,dtype=np.float64)
    prob = np.asarray(posterior,dtype=np.float64)
    n = len(raw)
    need(raw.shape == (n,2,14) and prob.shape == (n,2,2) and np.isfinite(raw).all()
         and np.isfinite(prob).all() and np.all((0 <= raw[...,0]) & (raw[...,0] <= 1))
         and np.all((0 <= prob) & (prob <= 1)), "Finite original input and dual probabilities")
    delta = signed_vector(robust, static, n, k)
    expected = (delta[...,None] * prob).sum(axis=(0,1)) / (2*k)
    native = float((delta * raw[...,0]).sum() / (2*k))
    moved = np.count_nonzero(delta,axis=0) / (2*k)
    result = np.array((expected.mean(), abs(expected[0]-expected[1]) / 2, native,
                       raw[0,0,6], raw[0,0,8], k/8, n/32, moved[0], moved[1]), dtype=np.float64)
    return result, float(expected.min())


def true_numerators(occupancy, action, static, k):
    source = np.asarray(occupancy)
    n = len(source)
    need(source.shape == (n,2,2) and np.isin(source,(0,1)).all(), "Exact FIT binary native reference occupancy")
    delta = signed_vector(action, static, n, k)
    return (delta[...,None] * source.astype(np.int8)).sum(axis=(0,1),dtype=np.int64)


def group_weights(groups, images):
    """Equal mass per decoded group, active image within group, and candidate within image."""
    by_group = defaultdict(Counter)
    reverse = {}
    for g,i in zip(groups,images,strict=True):
        need(reverse.setdefault(i,g) == g,"An active image belongs to one group")
        by_group[g][i] += 1
    need(bool(by_group),"Nonempty FIT calibration population")
    return np.array([1/(len(by_group)*len(by_group[g])*by_group[g][i])
                     for g,i in zip(groups,images,strict=True)], dtype=np.float64)


def fit(x, target, groups, images):
    x = np.asarray(x,dtype=np.float64)
    y = np.asarray(target,dtype=np.float64)
    w = group_weights(groups,images)
    n = len(w)
    need(x.shape == (n,len(FEATURE_NAMES)) and y.shape == (n,) and np.isfinite(x).all()
         and np.isfinite(y).all() and np.all((-1 <= y) & (y <= 1)), "Exact finite feature and target table")
    w /= w.sum()
    center = w @ x
    centered = x - center
    scale = np.sqrt(w @ np.square(centered))
    scale[scale < 1e-12] = 1.
    z = centered / scale
    intercept = float(w @ y)
    xtw = z.T * w
    normal = xtw @ z + ALPHA * np.eye(z.shape[1],dtype=np.float64)
    slope = np.linalg.solve(normal, xtw @ (y-intercept))
    need(np.isfinite(slope).all(), "Finite fitted calibration coefficients")
    return dict(status=MODEL_STATUS, feature_names=list(FEATURE_NAMES), center=center.tolist(),
                scale=scale.tolist(), weights=slope.tolist(), intercept=intercept, alpha=ALPHA, target=TARGET)


def score(checkpoint, rows):
    inputs = np.asarray(rows,dtype=np.float64)
    center, scale, weights = (np.asarray(checkpoint[k],dtype=np.float64) for k in ("center","scale","weights"))
    need(checkpoint["status"] == MODEL_STATUS and checkpoint["feature_names"] == list(FEATURE_NAMES)
         and checkpoint["alpha"] == ALPHA and checkpoint["target"] == TARGET
         and len(inputs.shape) in (1,2) and inputs.shape[-1] == len(FEATURE_NAMES)
         and center.shape == scale.shape == weights.shape == (len(FEATURE_NAMES),)
         and all(np.isfinite(v).all() for v in (inputs,center,scale,weights))
         and np.all(scale > 0) and np.isfinite(checkpoint["intercept"]), "Exact finite calibration state")
    return (inputs-center) / scale @ weights + checkpoint["intercept"]
