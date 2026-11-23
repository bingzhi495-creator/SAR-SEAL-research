"""Image-weighted spatial attraction and its squared-distance shell control."""

import numpy as np


def rank_image(rgb):
    """Map the native RGB image to the empirical mid-distribution of RF pair luminance."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or not rgb.size:
        raise ValueError("Expected a nonempty native uint8 RGB image")
    # Match pair_math.image_features, including its float32 operations before ranking.
    weights = np.asarray([.299, .587, .114], dtype=np.float32)
    gray = (rgb.astype(np.float32) @ weights) / np.float32(255)
    _, inverse, counts = np.unique(gray, return_inverse=True, return_counts=True)
    mid = (np.cumsum(counts, dtype=np.int64) - .5 * counts) / gray.size
    return mid[inverse].reshape(gray.shape)


def endpoint_scores(mask, coordinates, ranks, static):
    """Return n-by-2 normalized foreground attraction for static, full and shell.

    The caller supplies the pinned spatial_attraction module and its original endpoints.
    The exterior retains background membership and neutral image affinity 1. Both image
    controls use the same denominator, since the radial weights are constant within
    each shell. Only in-bounds affinities are averaged within a shell.
    """
    mask, coordinates, ranks = map(np.asarray, (mask, coordinates, ranks))
    if (mask.ndim != 2 or mask.dtype.kind != "b" or ranks.shape != mask.shape
            or ranks.dtype != np.float64 or coordinates.ndim != 2 or coordinates.shape[1] != 2
            or coordinates.dtype.kind not in "iu"):
        raise ValueError("Expected a binary native mask, float64 rank image and integer signed pairs")
    if (np.any(coordinates < 0) or np.any(coordinates >= mask.size)
            or np.unique(coordinates).size != coordinates.size or mask.flat[coordinates[:, 0]].any()
            or not mask.flat[coordinates[:, 1]].all()):
        raise ValueError("Invalid native endpoint location or sign")
    y, x = np.divmod(coordinates.astype(np.intp, copy=False), mask.shape[1])
    dr, dc, weights = static._OFF_R, static._OFF_C, static._WEIGHTS
    yy, xx = y[..., None] + dr, x[..., None] + dc
    interior = (yy >= 0) & (yy < mask.shape[0]) & (xx >= 0) & (xx < mask.shape[1])
    yy = np.clip(yy, 0, mask.shape[0]-1)
    xx = np.clip(xx, 0, mask.shape[1]-1)
    foreground = interior & mask[yy, xx]
    affinity = np.where(interior, 1.0 - np.abs(ranks[y, x, None] - ranks[yy, xx]), 1.0)
    if not np.all(np.isfinite(affinity) & (affinity >= 0) & (affinity <= 1)):
        raise ValueError("Invalid native image ranks")
    denominator = np.sum(affinity * weights, axis=-1, dtype=np.float64)
    if np.any(denominator <= 0):
        raise ValueError("Empty affinity mass at a native endpoint")
    shell_affinity = np.ones_like(affinity)
    squared_distances = dr*dr + dc*dc
    for distance in np.unique(squared_distances):
        shell = squared_distances == distance
        valid = interior[..., shell]
        count = np.count_nonzero(valid, axis=-1)
        mass = np.sum(np.where(valid, affinity[..., shell], 0.0), axis=-1, dtype=np.float64)
        mean = np.divide(mass, count, out=np.ones_like(mass), where=count > 0)
        shell_affinity[..., shell] = np.where(valid, mean[..., None], 1.0)

    def occupancy(a):
        return np.sum(foreground * (weights * a), axis=-1, dtype=np.float64) / denominator

    return dict(static=np.sum(foreground * weights, axis=-1, dtype=np.float64) / np.sum(weights),
                full=occupancy(affinity), shell=occupancy(shell_affinity))


def select_scores(scores, k, static):
    """Choose k additions and removals; use the historical raw-weight near-tie scale and row order."""
    scores = np.asarray(scores)
    if (scores.ndim != 2 or scores.shape[1] != 2 or not np.isfinite(scores).all()
            or not isinstance(k, (int, np.integer)) or not 0 <= k <= len(scores)):
        raise ValueError("Expected finite signed endpoint scores and their fixed realized budget")
    raw_scale = scores * np.sum(static._WEIGHTS)
    unused = np.ones(scores.shape, dtype=bool)
    result = []
    for _ in range(k):
        add = static._rank(raw_scale[:, 0], unused[:, 0], ascending=False)
        drop = static._rank(raw_scale[:, 1], unused[:, 1], ascending=True)
        unused[add, 0] = unused[drop, 1] = False
        result.append((add, drop))
    return result
