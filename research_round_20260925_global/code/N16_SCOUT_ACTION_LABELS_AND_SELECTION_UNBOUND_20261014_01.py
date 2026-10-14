"""UNBOUND / SYNTHETIC_ONLY development: labels, scores and physical selection.

No file access, model, dataset, weights, assignment generation or training.
Actual target provenance, protocol, numerical/sign qualification and resource
admission are ROOT/caller obligations. Import does not confer qualification.
"""
import numpy as np

UTILITY_NAMES = ('help', 'harm', 'noop')
ERROR_NAMES = ('FP', 'FN', 'correct')
SCORE_DTYPE = 'float64'


def _boolean_shape(**values):
    shape = None
    for name, value in values.items():
        if not isinstance(value, np.ndarray) or np.ma.isMaskedArray(value) or value.dtype != np.bool_:
            raise TypeError(name + ' must be a Boolean NumPy array; no implicit casts')
        if value.ndim == 0:
            raise ValueError(name + ' must have at least one axis')
        if shape is None:
            shape = value.shape
        elif value.shape != shape:
            raise ValueError('All input shapes must be identical')
    return shape


def utility_labels(native_P, G, fixed_after_A):
    """Return INT64 classes0help/1harm/2noop for actual fixed binary actions.

    An uncorrected native FP/FN is noop when A=P; native error is not help.
    Caller supplies qualified FIT targets/valid points or an explicitly locked
    common background-supervision extension. Original mask_weight0 alone does
    not prove an empty target. This helper cannot establish target provenance.
    """
    shape = _boolean_shape(native_P=native_P, G=G, fixed_after_A=fixed_after_A)
    help_mask = (native_P != G) & (fixed_after_A == G)
    harm_mask = (native_P == G) & (fixed_after_A != G)
    noop_mask = native_P == fixed_after_A
    if np.any((help_mask.astype(np.int8) + harm_mask + noop_mask) != 1):
        raise RuntimeError('Actual utility masks must be disjoint and exhaustive')
    labels = np.empty(shape, dtype=np.int64)
    for index, mask in enumerate((help_mask, harm_mask, noop_mask)):
        labels[mask] = index
    return labels


def error_labels(native_P, G):
    """Return INT64 classes0FP/1FN/2correct for the same native binary input."""
    shape = _boolean_shape(native_P=native_P, G=G)
    FP, FN, correct = native_P & ~G, ~native_P & G, native_P == G
    if np.any((FP.astype(np.int8) + FN + correct) != 1):
        raise RuntimeError('Actual error masks must be disjoint and exhaustive')
    labels = np.empty(shape, dtype=np.int64)
    for index, mask in enumerate((FP, FN, correct)):
        labels[mask] = index
    return labels


def _finite_float(value, name):
    if not isinstance(value, np.ndarray) or np.ma.isMaskedArray(value) or value.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise TypeError(name + ' must be a FP32/FP64 NumPy array; no implicit casts')
    if not np.all(np.isfinite(value)):
        raise ValueError(name + ' must be finite')
    return value


def softmax3(logits):
    """Stable canonical FP64 probabilities from finite [...,3] scout logits.

    This dtype is an explicit ordinary proposal. Actual ROOT inference must
    bind the same scoring/ordering path; near ties cannot be waived by allclose.
    """
    _finite_float(logits, 'logits')
    if logits.ndim < 1 or logits.shape[-1] != 3:
        raise ValueError('Scout logits must have final axis length3')
    values = logits.astype(np.float64)
    # Finite extreme opposite values can subtract to -inf; exp(-inf)=0 is
    # mathematically the safe limiting result. No positive overflow is possible.
    with np.errstate(over='ignore', under='ignore', invalid='raise'):
        shifted = values - np.max(values, axis=-1, keepdims=True)
        weights = np.exp(shifted)
    return weights / np.sum(weights, axis=-1, keepdims=True)


def utility_score(logits):
    """p_help-p_harm, with class order help/harm/noop; no GT input."""
    probabilities = softmax3(logits)
    return probabilities[..., 0] - probabilities[..., 1]


def error_score(logits):
    """2*(p_FP+p_FN)-1, with class order FP/FN/correct; no GT input."""
    probabilities = softmax3(logits)
    return 2.0 * (probabilities[..., 0] + probabilities[..., 1]) - 1.0


def choose_candidates(native_L, C, score, K, mode='atmost'):
    """Return a private Boolean selection with no GT input or hidden ranking.

    Main atmost uses score>0 and at mostK. Supplemental exact selects exactlyK
    actual C points, including real zero/negative scores; these are not invented
    noops. Order: score descending, abs(nativeL) ascending, flat index ascending.
    Actual numeric scores are compared directly with no rank/zero tolerance.
    """
    _finite_float(native_L, 'native_L')
    _finite_float(score, 'score')
    _boolean_shape(C=C)
    if native_L.shape != C.shape or score.shape != C.shape:
        raise ValueError('NativeL, C and score must have exactly the same shape')
    if isinstance(K, (bool, np.bool_)) or not isinstance(K, (int, np.integer)):
        raise TypeError('K must be a nonboolean scalar integer')
    K = int(K)
    if K < 0 or K > int(np.count_nonzero(C)):
        raise ValueError('K must satisfy 0<=K<=number of real C points')
    if type(mode) is not str or mode not in ('atmost', 'exact'):
        raise ValueError('mode must be atmost or exact')
    selected = np.zeros(C.shape, dtype=np.bool_)
    if K == 0:
        return selected
    eligible = C & (score > 0) if mode == 'atmost' else C
    indices = np.flatnonzero(eligible)
    order = np.lexsort((indices, np.abs(native_L.ravel()[indices].astype(np.float64)), -score.ravel()[indices].astype(np.float64)))
    selected.ravel()[indices[order[:K]]] = True
    return selected


def apply_selected_residual(native_L, actual_residual, C, selected):
    """Apply fixed PR residual only at selected real C; outside stays bitwise.

    Addition retains caller FP32/FP64 dtype. Mixed operand dtypes are rejected
    rather than silently changing the fixed PR arithmetic.
    """
    _finite_float(native_L, 'native_L')
    _finite_float(actual_residual, 'actual_residual')
    _boolean_shape(C=C, selected=selected)
    if native_L.shape != C.shape or actual_residual.shape != C.shape:
        raise ValueError('NativeL, residual, C and selection must have identical shapes')
    if native_L.dtype != actual_residual.dtype:
        raise TypeError('NativeL and fixed PR residual must have identical dtype')
    if np.any(selected & ~C):
        raise ValueError('Selected support must be a subset of real C')
    output = native_L.copy()
    with np.errstate(over='ignore', invalid='raise'):
        output[selected] = native_L[selected] + actual_residual[selected]
    if not np.all(np.isfinite(output)):
        raise ValueError('Selected PR produced nonfinite combined logits')
    return output


def dense_error_flip_score(actual_delta, error_logits):
    """FP64 ranking score actual_PR_flip*(2*p_error-1), retaining fixed actions.

    actual_delta is Boolean(native_P!=fixed_after_A), never raw signed residual.
    Computing it requires dense PR over C. Selection still applies ORIGINAL PR
    rawu at chosen points; the score must not be used as a new correction.
    """
    _boolean_shape(actual_delta=actual_delta)
    factor = error_score(error_logits)
    if actual_delta.shape != factor.shape:
        raise ValueError('Actual flip shape must equal the error-scout point shape')
    return actual_delta.astype(np.float64) * factor


def stable_sigmoid(values):
    """FP64 piecewise sigmoid for finite FP32/FP64 arrays; exp never overflows."""
    _finite_float(values, 'sigmoid values')
    x = values.astype(np.float64)
    output = np.empty(x.shape, dtype=np.float64)
    positive = x >= 0
    with np.errstate(under='ignore', over='raise', invalid='raise'):
        output[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
        exp_negative = np.exp(x[~positive])
        output[~positive] = exp_negative / (1.0 + exp_negative)
    return output


def magnet_score(native_L, correction_u):
    """Unsmoothed binary MagNet-style adaptation; not author bit reproduction.

    (1-|2*sigmoid(L)-1|)*|2*sigmoid(L+u)-1| is evaluated through its stable
    equivalent 2*sigmoid(-abs(L))*abs(tanh((L+u)/2)). This avoids cancellation
    at positive saturated sigmoid and fixes the actual FP64 ranking expression.
    It is a probability score, not an IoU/AP-optimality guarantee.
    """
    _finite_float(native_L, 'native_L')
    _finite_float(correction_u, 'correction_u')
    if native_L.shape != correction_u.shape:
        raise ValueError('NativeL and correction must have identical shapes')
    L = native_L.astype(np.float64)
    with np.errstate(over='ignore', under='ignore', invalid='raise'):
        refined = L + correction_u.astype(np.float64)
    if not np.all(np.isfinite(refined)):
        raise ValueError('MagNet-style refined logit must remain finite')
    with np.errstate(under='ignore'):
        return (2.0 * stable_sigmoid(-np.abs(L))) * np.abs(np.tanh(refined / 2.0))
