"""Strict original-grid Omega coordinate contract; no model or dataset imports."""


def query_plan(np, native_L, union_flat_indices, expected_K):
    if (native_L.dtype != np.float32 or native_L.ndim != 2
            or not np.isfinite(native_L).all()):
        raise ValueError('Finite original FP32 two-dimensional native L required')
    union = union_flat_indices
    if (union.dtype != np.int64 or union.ndim != 1
            or (len(union) and (union[0] < 0 or union[-1] >= native_L.size))
            or (len(union) > 1 and not (union[1:] > union[:-1]).all())):
        raise ValueError('Strictly ascending unique original-grid union required')
    indices = np.flatnonzero(np.abs(native_L.reshape(-1)) <= np.float32(2)).astype(np.int64)
    if type(expected_K) is not int or expected_K != len(indices):
        raise ValueError('Original K must equal complete Omega count')
    positions = np.searchsorted(union, indices).astype(np.int64)
    if len(indices) and ((positions >= len(union)).any()
                         or not np.array_equal(union[positions], indices)):
        raise ValueError('An Omega coordinate has no captured F row')
    return dict(Omega_flat_indices=indices, union_positions=positions, K=len(indices))


def apply_query_logits(np, native_L, indices, corrected_logits):
    if (native_L.dtype != np.float32 or native_L.ndim != 2
            or not np.isfinite(native_L).all()):
        raise ValueError('Finite original FP32 native L required')
    expected = np.flatnonzero(np.abs(native_L.reshape(-1)) <= np.float32(2)).astype(np.int64)
    if (indices.dtype != np.int64 or indices.ndim != 1
            or not np.array_equal(indices, expected)):
        raise ValueError('Only the complete original ascending Omega may be replaced')
    if (corrected_logits.dtype != np.float32 or corrected_logits.shape != indices.shape
            or not np.isfinite(corrected_logits).all()):
        raise ValueError('Finite FP32 corrected logits for each Omega point required')
    result = (native_L > 0).copy()
    result.reshape(-1)[indices] = corrected_logits > 0
    return result
