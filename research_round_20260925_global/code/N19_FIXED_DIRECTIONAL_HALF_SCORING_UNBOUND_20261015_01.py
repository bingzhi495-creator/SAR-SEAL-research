# UNBOUND_DRAFT
"""Fixed four GT-free readouts; no IO, model, training, AP or qualification."""
import numpy as np

POLICIES = ('SYMMETRIC', 'DIR_HALF', 'GLOBAL_HALF', 'REVERSED_HALF')


def _finite_float(value, name):
    if (not isinstance(value, np.ndarray) or np.ma.isMaskedArray(value) or
            value.dtype not in (np.dtype('float32'), np.dtype('float64'))):
        raise TypeError(name + ' must be a FP32/FP64 NumPy array; no implicit casts')
    if not np.all(np.isfinite(value)):
        raise ValueError(name + ' must be finite')


def softmax3(logits):
    """Same FP64 subtract-max/exp/sum operations as the N16 canonical path."""
    _finite_float(logits, 'logits')
    if logits.ndim != 2 or logits.shape[1] != 3:
        raise ValueError('logits must have shape [N,3] in help/harm/noop order')
    values = logits.astype(np.float64)
    with np.errstate(over='ignore', under='ignore', invalid='raise'):
        shifted = values - np.max(values, axis=-1, keepdims=True)
        weights = np.exp(shifted)
    return weights / np.sum(weights, axis=-1, keepdims=True)


def score_and_select(logits, native_L_C, C_flat, K):
    """Return policy -> score/selected_flat_sorted/selected_C_positions.

    All points share the caller-bound ascending physical C order. Selection
    uses actual FP64 scores >0, then (score down, abs L up, flat up), at most K.
    Returned flat indices and positions are sorted by ascending physical C.
    No coefficient or threshold parameter is exposed. N=0,K=0 is legal.
    """
    probabilities = softmax3(logits)
    _finite_float(native_L_C, 'native_L_C')
    n = logits.shape[0]
    if native_L_C.shape != (n,):
        raise ValueError('native_L_C must have shape [N]')
    if (not isinstance(C_flat, np.ndarray) or np.ma.isMaskedArray(C_flat) or
            C_flat.dtype != np.dtype('int64')):
        raise TypeError('C_flat must be a NumPy INT64 array')
    if (C_flat.shape != (n,) or np.any(C_flat < 0) or
            not np.all(C_flat[1:] > C_flat[:-1])):
        raise ValueError('C_flat must be [N], nonnegative, ascending and unique')
    if isinstance(K, (bool, np.bool_)) or not isinstance(K, (int, np.integer)):
        raise TypeError('K must be a nonboolean scalar integer')
    K = int(K)
    if K < 0 or K > n:
        raise ValueError('K must satisfy 0<=K<=N')
    ph, pm = probabilities[:, 0], probabilities[:, 1]
    positive = native_L_C > 0
    background_half = ph - 0.5 * pm
    foreground_half = 0.5 * ph - pm
    scores = (ph - pm, np.where(positive, foreground_half, background_half),
              background_half, np.where(positive, background_half, foreground_half))
    result = {}
    for name, score in zip(POLICIES, scores):
        eligible = np.flatnonzero(score > 0)
        order = np.lexsort((C_flat[eligible],
                            np.abs(native_L_C[eligible].astype(np.float64)),
                            -score[eligible].astype(np.float64)))
        positions = np.sort(eligible[order[:K]])
        result[name] = dict(score=score, selected_flat_sorted=C_flat[positions].copy(),
                            selected_C_positions=positions.astype(np.int64, copy=False))
    return result
