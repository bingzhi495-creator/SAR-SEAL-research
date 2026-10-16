"""Offline maximum-cardinality association only; never a deployed predictor."""


def maximum_matching(np, ious, threshold):
    if (ious.ndim != 2 or ious.dtype not in (np.dtype('float32'), np.dtype('float64'))
            or not np.isfinite(ious).all() or (ious < 0).any() or (ious > 1).any()
            or not 0 < threshold <= 1):
        raise ValueError('Finite D-by-G IoU matrix and threshold in (0,1] required')
    # GT columns and original score-order rows give a fixed deterministic traversal.
    owner = [-1] * ious.shape[1]

    def augment(dt, seen):
        for gt in range(ious.shape[1]):
            if ious[dt, gt] >= threshold and not seen[gt]:
                seen[gt] = True
                if owner[gt] == -1 or augment(owner[gt], seen):
                    owner[gt] = dt
                    return True
        return False

    for dt in range(ious.shape[0]):
        augment(dt, [False] * ious.shape[1])
    return sorted((dt, gt) for gt, dt in enumerate(owner) if dt != -1)
