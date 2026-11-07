"""Equal-support display-gray descriptors at frozen mask boundary pixels."""

import numpy as np


_OFFSETS = np.array(
    [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3) if dy or dx],
    dtype=np.intp,
)
_CHUNK_SIZE = 8192


def _signed_rank_votes(values, inside, outside, outside_count):
    """Sum pairwise wins minus losses using tied groups in sorted gray order."""
    order = np.argsort(values, axis=1)
    values = np.take_along_axis(values, order, axis=1)
    inside = np.take_along_axis(inside, order, axis=1)
    outside = np.take_along_axis(outside, order, axis=1)
    transition = values[:, 1:] != values[:, :-1]
    columns = np.arange(values.shape[1])
    edge = np.ones((len(values), 1), dtype=bool)
    starts = np.concatenate((edge, transition), axis=1)
    ends = np.concatenate((transition, edge), axis=1)
    first = np.maximum.accumulate(np.where(starts, columns, 0), axis=1)
    last = np.minimum.accumulate(
        np.where(ends, columns, values.shape[1])[:, ::-1], axis=1,
    )[:, ::-1]

    outside_prefix = np.zeros((len(values), values.shape[1] + 1), dtype=np.int16)
    np.cumsum(outside, axis=1, dtype=np.int16, out=outside_prefix[:, 1:])
    less = np.take_along_axis(outside_prefix, first, axis=1)
    less_or_equal = np.take_along_axis(outside_prefix, last + 1, axis=1)
    # less - greater gives tied observations zero net vote.
    return ((less + less_or_equal - outside_count[:, None]) * inside).sum(axis=1)


def extract_features(gray, predicted_mask, flat_indices, logits):
    """Return aligned features for the caller's sorted original-grid support.

    Gray is uint8 and predicted_mask is Boolean, on the same nonempty HxW
    grid. Support uses strictly increasing, zero-based row-major indices.
    Logits are the corresponding finite original-grid floating-point values.
    The caller determines the boundary support; rows are never filtered here.
    """
    gray = np.asarray(gray)
    predicted_mask = np.asarray(predicted_mask)
    flat_indices = np.asarray(flat_indices)
    logits = np.asarray(logits)
    if gray.ndim != 2 or not gray.size or gray.dtype != np.uint8:
        raise ValueError("gray must be a nonempty HxW uint8 array")
    if predicted_mask.shape != gray.shape or predicted_mask.dtype != np.bool_:
        raise ValueError("predicted_mask must be Boolean on the gray grid")
    if flat_indices.ndim != 1 or not np.issubdtype(flat_indices.dtype, np.integer):
        raise ValueError("flat_indices must be a one-dimensional integer array")
    if flat_indices.size and (
        flat_indices[0] < 0
        or flat_indices[-1] >= gray.size
        or np.any(flat_indices[1:] <= flat_indices[:-1])
    ):
        raise ValueError("flat_indices must be strictly increasing and in bounds")
    if (
        logits.shape != flat_indices.shape
        or not np.issubdtype(logits.dtype, np.floating)
        or not np.all(np.isfinite(logits))
    ):
        raise ValueError("logits must be finite floats aligned with flat_indices")

    height, width = gray.shape
    indices = flat_indices.astype(np.intp, copy=False)
    row, col = np.divmod(indices, width)
    edit_sign = np.where(predicted_mask[row, col], -1, 1).astype(np.int8)
    valid = np.zeros(len(indices), dtype=bool)
    mean_contrast = np.zeros(len(indices), dtype=np.float32)
    rank = np.zeros(len(indices), dtype=np.float32)

    for start in range(0, len(indices), _CHUNK_SIZE):
        rows = slice(start, start + _CHUNK_SIZE)
        yy = row[rows, None] + _OFFSETS[:, 0]
        xx = col[rows, None] + _OFFSETS[:, 1]
        in_bounds = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
        yy = np.clip(yy, 0, height - 1)
        xx = np.clip(xx, 0, width - 1)
        values = gray[yy, xx]
        predicted_inside = predicted_mask[yy, xx]
        inside = in_bounds & predicted_inside
        outside = in_bounds & ~predicted_inside
        inside_count = inside.sum(axis=1, dtype=np.int32)
        outside_count = outside.sum(axis=1, dtype=np.int32)
        usable = (inside_count >= 2) & (outside_count >= 2)
        valid[rows] = usable

        # Cross-multiplying the sample means keeps gray arithmetic integral.
        inside_scaled = (values * inside).sum(axis=1, dtype=np.int32) * outside_count
        outside_scaled = (values * outside).sum(axis=1, dtype=np.int32) * inside_count
        total = inside_scaled + outside_scaled
        np.divide(
            inside_scaled - outside_scaled, total,
            out=mean_contrast[rows], where=usable & (total != 0),
        )
        votes = _signed_rank_votes(values, inside, outside, outside_count)
        np.divide(votes, inside_count * outside_count, out=rank[rows], where=usable)

    return {
        "flat_indices": flat_indices.copy(),
        "abs_logit": np.abs(logits),
        "edit_sign": edit_sign,
        "valid": valid,
        "mean_contrast": mean_contrast,
        "rank": rank,
    }
