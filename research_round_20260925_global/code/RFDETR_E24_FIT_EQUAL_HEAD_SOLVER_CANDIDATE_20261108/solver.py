"""Fixed-budget NumPy logistic heads for frozen boundary candidate pixels."""

import numpy as np


_BRANCHES = ("uncertainty", "mean", "rank")
_L2 = 0.01
_STEP = 0.5
_ITERATIONS = 128
_FEATURE_COUNT = 5


def _vector(value, name):
    values = np.asarray(value)
    if values.ndim != 1 or values.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be a one-dimensional real numeric array")
    values = values.astype(np.float64, copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite")
    return values


def _logit_magnitude(features):
    magnitude = _vector(features["abs_logit"], "abs_logit")
    if np.any(magnitude < 0):
        raise ValueError("abs_logit must be nonnegative")
    return magnitude


def _design(features, detector_score, branch):
    if branch not in _BRANCHES:
        raise ValueError("branch must be uncertainty, mean or rank")
    magnitude = _logit_magnitude(features)
    direction = _vector(features["edit_sign"], "edit_sign")
    valid = _vector(features["valid"], "valid")
    score = np.asarray(detector_score)
    if score.ndim == 0:
        score = np.broadcast_to(score, magnitude.shape)
    score = _vector(score, "detector_score")
    if branch == "uncertainty":
        extra = np.exp(-magnitude)
    else:
        key = "mean_contrast" if branch == "mean" else "rank"
        extra = _vector(features[key], key)

    columns = (direction, magnitude, score, valid, extra)
    if any(column.shape != magnitude.shape for column in columns):
        raise ValueError("all feature columns must align")
    if (
        np.any((direction != -1) & (direction != 1))
        or np.any((score < 0) | (score > 1))
        or np.any((valid != 0) & (valid != 1))
        or np.any(np.abs(extra) > 1)
    ):
        raise ValueError("feature column outside its admitted range")
    return np.column_stack(columns)


def _sigmoid(values):
    return np.exp(-np.logaddexp(0.0, -values))


def fit_head(features, detector_score, labels, *, branch, sample_weight=None):
    """Fit one head with a weighted logistic loss and fixed CPU updates.

    Normalize positive row weights, standardize each column using only
    supplied train rows, and add 0.01/2 times squared slopes to the weighted
    mean binary cross entropy. The intercept is unpenalized and starts at
    weighted log odds. Slopes start at zero. All branches use 128 full-batch
    float64 gradient steps of size 0.5 with no early stopping.
    """
    design = _design(features, detector_score, branch)
    labels = _vector(labels, "labels")
    if not len(labels) or len(labels) != len(design) or np.any((labels != 0) & (labels != 1)):
        raise ValueError("nonempty binary labels must align with features")
    if sample_weight is None:
        weights = np.ones(len(labels), dtype=np.float64)
    else:
        weights = _vector(sample_weight, "sample_weight")
        if weights.shape != labels.shape or np.any(weights <= 0):
            raise ValueError("positive weights must align with labels")
    weights = weights / weights.max()
    weights /= weights.sum()
    positives = labels == 1
    positive_mass = weights[positives].sum()
    negative_mass = weights[~positives].sum()
    if positive_mass <= 0 or negative_mass <= 0 or np.any(weights == 0):
        raise ValueError("both labels and every row require positive normalized weight")

    center = np.einsum("i,ij->j", weights, design)
    design -= center
    variance = np.einsum("i,ij,ij->j", weights, design, design)
    if not np.all(np.isfinite(variance)):
        raise ValueError("feature variance must be finite")
    scale = np.where(variance > 1e-24, np.sqrt(variance), 1.0)
    design /= scale
    intercept = np.log(positive_mass) - np.log(negative_mass)
    slopes = np.zeros(_FEATURE_COUNT, dtype=np.float64)
    for _ in range(_ITERATIONS):
        linear = intercept + np.einsum("ij,j->i", design, slopes)
        residual = weights * (_sigmoid(linear) - labels)
        intercept -= _STEP * residual.sum()
        gradient = np.einsum("i,ij->j", residual, design) + _L2 * slopes
        slopes -= _STEP * gradient

    linear = intercept + np.einsum("ij,j->i", design, slopes)
    objective = (
        np.dot(weights, np.logaddexp(0.0, np.where(positives, -linear, linear)))
        + _L2 / 2 * np.dot(slopes, slopes)
    )
    if not np.isfinite(objective):
        raise ValueError("head fit produced a nonfinite objective")
    return {
        "branch": branch,
        "center": center.tolist(),
        "scale": scale.tolist(),
        "slopes": slopes.tolist(),
        "intercept": float(intercept),
        "training_objective": float(objective),
    }


def predict_head(model, features, detector_score):
    """Score aligned candidates with training-only moments and coefficients."""
    design = _design(features, detector_score, model["branch"])
    center = _vector(model["center"], "center")
    scale = _vector(model["scale"], "scale")
    slopes = _vector(model["slopes"], "slopes")
    intercept = float(model["intercept"])
    if (
        any(len(column) != _FEATURE_COUNT for column in (center, scale, slopes))
        or np.any(scale <= 0)
        or not np.isfinite(intercept)
    ):
        raise ValueError("invalid frozen head parameters")
    linear = intercept + np.einsum("ij,j->i", (design - center) / scale, slopes)
    if not np.all(np.isfinite(linear)):
        raise ValueError("head prediction produced nonfinite logits")
    return _sigmoid(linear)


def stock_uncertainty(features):
    """Return the stock negative absolute logit without exponential underflow."""
    return -_logit_magnitude(features)


def select_edits(flat_indices, scores, *, threshold, budget=16):
    """Choose unique pixels by decreasing score then increasing flat index.

    Use threshold=None for a forced-budget diagnostic. A numeric threshold
    is strict, so candidates equal to it abstain. Passing 1.0 abstains on
    both logistic probabilities and stock uncertainty scores.
    """
    flat_indices = np.asarray(flat_indices)
    scores = _vector(scores, "scores")
    if (
        flat_indices.shape != scores.shape
        or flat_indices.dtype.kind not in "iu"
        or (len(flat_indices) and flat_indices[0] < 0)
        or np.any(flat_indices[1:] <= flat_indices[:-1])
    ):
        raise ValueError("flat_indices must be nonnegative, unique, increasing and aligned")
    if (
        isinstance(budget, (bool, np.bool_))
        or not isinstance(budget, (int, np.integer))
        or not 0 <= budget <= 16
    ):
        raise ValueError("budget must be an integer from 0 to 16")

    if threshold is None:
        eligible = np.arange(len(scores))
    else:
        threshold = np.asarray(threshold)
        if threshold.ndim != 0 or threshold.dtype.kind not in "iuf" or not np.isfinite(threshold):
            raise ValueError("threshold must be a finite real scalar or None")
        eligible = np.flatnonzero(scores > float(threshold))
    if not budget or not len(eligible):
        return flat_indices[:0].copy()
    order = np.lexsort((flat_indices[eligible], -scores[eligible]))
    return flat_indices[eligible[order[:budget]]]
