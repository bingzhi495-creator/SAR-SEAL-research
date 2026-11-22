"""Source-fitted ridge policies and deterministic equal-count quota controls."""

from hashlib import sha256
import json
from typing import NamedTuple

import numpy as np


FEATURES = ("signed_area_delta_over_native", "abs_area_delta_over_native", "expected_add_mass_delta_over_native",
            "expected_drop_mass_delta_over_native", "ratio_change_proxy", "native_area", "native_score",
            "support_expected_signed_area_delta_over_native", "balanced_count")
ARM_NAMES = ("B", "R", "J", "C", "A", "M", "G")
FIT_ARMS = ("A", "M", "G")
REGULARIZER = .05
MODEL_STATUS = "RFDETR_QUOTA_RIDGE_SOURCE_MODEL_V1"
CAPACITY = 25


class Model(NamedTuple):
    scale: np.ndarray
    coefficients: np.ndarray
    constant_signed_offset: int


def check(ok, why):
    if not ok:
        raise ValueError(why)


def basis(raw):
    """Cross five action contrasts with five ordered candidate context terms."""
    raw = np.asarray(raw, dtype=np.float64)
    check(raw.ndim == 2 and raw.shape[1] == len(FEATURES) and np.isfinite(raw).all()
          and np.all((32 <= raw[:, 5]) & (raw[:, 5] < 1024))
          and np.all((0 <= raw[:, 6]) & (raw[:, 6] <= 1))
          and np.all((0 <= raw[:, 8]) & (raw[:, 8] <= 8)), "Original finite action and native context features")
    context = np.column_stack((np.ones(len(raw)), np.log(raw[:, 5])/np.log(1024.), raw[:, 6:8], raw[:, 8]/8.))
    return (raw[:, :5, None]*context[:, None, :]).reshape(len(raw), CAPACITY)


def feasible(k, actions, raw):
    """Validate the exact archived fixed-count action sequence and balanced row."""
    check(isinstance(k, (int, np.integer)) and not isinstance(k, (bool, np.bool_))
          and 0 <= k <= 8 and actions.ndim == 2 and actions.shape[1] == 2 and actions.dtype.kind in "iu"
          and 1 <= len(actions) <= 17 and np.all(actions >= 0)
          and np.all(actions.sum(axis=1) == 2*k) and np.all(np.diff(actions[:, 0]) == 1),
          "Original consecutive feasible allocation at fixed total writes")
    balance = int(k-actions[0, 0])
    check(raw.shape == (len(actions), len(FEATURES)) and 0 <= balance < len(actions)
          and int(actions[balance, 0]) == k and np.all(raw[:, 8] == k)
          and np.all(raw[balance,:5] == 0), "Exactly one balanced allocation at the zero feature anchor")
    return balance


def best_index(worth, adds, k):
    check(worth.shape == adds.shape and np.isfinite(worth).all(), "Finite predicted utilities aligned to feasible actions")
    return int(np.lexsort((adds, np.abs(adds-k), -worth))[0])


def constant_index(adds, k, signed_offset):
    return int(np.clip(k+signed_offset, adds[0], adds[-1])-adds[0])


def select_actions(model, k, actions, features):
    """Return indices into the original moves for all learned and simple controls."""
    actions = np.asarray(actions)
    raw = np.asarray(features, dtype=np.float64)
    balance = feasible(k, actions, raw)
    predicts = (basis(raw)/model.scale) @ model.coefficients.T
    check(np.all(predicts[balance] == 0), "Balanced model predictions remain exactly zero")
    adds = actions[:, 0]
    indices = dict(B=balance, R=best_index(raw[:, 4], adds, k),
                   J=best_index(2*(raw[:, 2]-raw[:, 3])-raw[:, 0], adds, k),
                   C=constant_index(adds, k, model.constant_signed_offset))
    indices.update({name:best_index(predicts[:, j], adds, k) for j,name in enumerate(FIT_ARMS)})
    return indices


def load_model(path, expected_sha):
    """Restore read-only source parameters after their exact digest is verified."""
    with path.open("rb") as stream:
        raw = stream.read(64*1024+1)
    check(len(raw) <= 64*1024 and sha256(raw).hexdigest() == expected_sha, "Pinned bounded source model")
    model = json.loads(raw)
    check(model["status"] == MODEL_STATUS and tuple(model["input_feature_names"]) == FEATURES
          and tuple(model["fit_arms"]) == FIT_ARMS and model["regularizer"] == REGULARIZER
          and model["capacity_per_arm"] == CAPACITY, "Exact fixed-rank quota policy schema")
    scale = np.asarray(model["rms"], dtype=np.float64)
    coef = np.asarray(model["coefficients"], dtype=np.float64)
    constant = model["constant_signed_offset"]
    check(scale.shape == (CAPACITY,) and coef.shape == (len(FIT_ARMS), CAPACITY) and np.isfinite(scale).all()
          and np.isfinite(coef).all() and np.all(scale > 0) and type(constant) is int and -8 <= constant <= 8,
          "Finite fitted source parameters in the declared order")
    scale.setflags(write=False)
    coef.setflags(write=False)
    return Model(scale, coef, constant)
