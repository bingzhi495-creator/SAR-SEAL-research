"""Pure array scoring diagnostic; its quality input is GT-derived, not deployable."""


def quality_scores(np, ious, original_scores):
    overlaps = np.asarray(ious, dtype=np.float64)
    scores = np.asarray(original_scores, dtype=np.float64)
    if overlaps.ndim != 2 or scores.ndim != 1 or overlaps.shape[0] != scores.shape[0]:
        raise ValueError('IoU rows must align with the original score vector')
    for values in (overlaps, scores):
        if not bool(np.isfinite(values).all()) or bool(((values < 0) | (values > 1)).any()):
            raise ValueError('Finite original scores and IoUs in [0,1] required')
    quality = overlaps.max(axis=1) if overlaps.shape[1] else np.zeros(scores.shape, dtype=np.float64)
    return dict(quality=quality.copy(), GT_PRODUCT=scores * quality,
                GT_IOU_ONLY=quality.copy())
