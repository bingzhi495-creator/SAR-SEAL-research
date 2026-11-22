"""Score the complete two-layer signed support using the published native point features."""

import numpy as np


def need(condition, message):
    if not condition:
        raise ValueError(message)


def expanded_features(rank_math, stock, box, area, score, probability, image_features, native_coords):
    """Return two coordinate vectors and two (possibly unequal) float32 feature tables.

    The caller supplies the original nearest-assignment table for S. Its row order
    is the prefix of both sides. Remaining pixels of E follow in row-major order.
    """
    height, width = stock.shape
    need(stock.dtype == np.bool_ and area > 0 and int(stock.sum()) == area,
         "Original nonempty native mask and area")
    need(probability.shape == stock.shape and probability.dtype == np.float32
         and image_features.shape == (5, height, width) and image_features.dtype == np.float32,
         "Native float32 posterior and five image planes")
    box = np.asarray(box)
    need(box.shape == (4,) and np.isfinite(box).all() and np.all(box[:2] >= 0)
         and np.all(box[2:] <= (width, height)) and np.all(box[2:] > box[:2]), "Clipped nondegenerate box")
    need(native_coords.ndim == 2 and native_coords.shape[1] == 2 and native_coords.dtype.kind in "iu",
         "Original add/drop assignment")
    x1, y1, x2, y2 = box
    bx1, by1, bx2, by2 = np.ceil(box).astype(np.int64)
    box_pixels = max(1, (bx2 - bx1) * (by2 - by1))
    pools, _ = rank_math.editable_pools(stock, box)
    sides = [tuple(pools[f"{sign}{depth}"] for depth in (1, 2)) for sign in ("add", "drop")]
    n = min(32, *(sum(len(idx) for idx, _ in layers) for layers in sides))
    need(len(native_coords) == n, "Full original uncertainty-selected prefix length")
    coords, features = [], []
    for side, layers in enumerate(sides):
        indices = np.concatenate([idx for idx, _ in layers])
        distance = np.concatenate([dist for _, dist in layers])
        depth = np.repeat((1, 2), [len(idx) for idx, _ in layers])
        p = probability.ravel()[indices]
        prefix = native_coords[:, side]
        uncertain = np.lexsort((indices, np.abs(p - .5)))[:n]
        need(np.array_equal(np.sort(prefix), np.sort(indices[uncertain])),
             "Original posterior-selected side is present in E")
        ascending = np.argsort(indices)
        sorted_indices = indices[ascending]
        need(np.all(np.diff(sorted_indices) > 0) and np.all(stock.flat[indices] == bool(side)),
             "Unique pixels with the original mask sign")
        prefix_positions = np.searchsorted(sorted_indices, prefix)
        remaining = np.ones(len(indices), dtype=bool)
        remaining[prefix_positions] = False
        order = np.concatenate((ascending[prefix_positions], ascending[remaining]))
        indices = indices[order]
        y, x = np.divmod(indices, width)
        size = len(indices)
        # Keep the original expression dtypes through column_stack and cast once.
        f = np.column_stack((p[order], np.full(size, side), depth[order], distance[order],
                             (x-x1)/(x2-x1), (y-y1)/(y2-y1), np.full(size, np.log2(area)),
                             np.full(size, area/box_pixels), np.full(size, score), image_features[:, y, x].T))
        f = f.astype(np.float32)
        need(np.isfinite(f).all() and np.all((f[:, 0] >= 0) & (f[:, 0] <= 1)), "Finite native point features")
        coords.append(indices)
        features.append(f)
    return tuple(coords), tuple(features)


def predict_sides(predict, features):
    """Use the frozen independent feature head, preserving the historical batch shape on S."""
    add, drop = features
    for table in (add, drop):
        need(table.ndim == 2 and table.shape[1] == 14 and table.dtype == np.float32,
             "Native float32 endpoint features")
    n = min(32, len(add), len(drop))

    def checked(batch):
        result = predict(batch)
        need(result.shape == batch.shape[:2] and result.dtype == np.float32
             and np.isfinite(result).all() and np.all((result >= 0) & (result <= 1)),
             "Finite frozen head posterior for every endpoint")
        return result

    common = checked(np.stack((add[:n], drop[:n]), axis=1)) if n else np.empty((0, 2), np.float32)
    tail = np.concatenate((add[n:], drop[n:]))
    rest = checked(tail[:, None, :])[:, 0] if len(tail) else np.empty(0, np.float32)
    split = len(add) - n
    return np.concatenate((common[:, 0], rest[:split])), np.concatenate((common[:, 1], rest[split:]))


def select(coords, scores, k, *, tie="native"):
    """Choose exactly k additions by decreasing score and k removals by increasing score."""
    need(type(k) is int and 0 <= k <= min(8, *(len(indices) for indices in coords)),
         "Original feasible integer swap count")
    need(tie in ("native", "legacy") and len(coords) == len(scores) == 2, "Signed side and tie policy")
    chosen = []
    for side, (indices, values) in enumerate(zip(coords, scores, strict=True)):
        indices, values = np.asarray(indices), np.asarray(values)
        need(indices.ndim == 1 and indices.dtype.kind in "iu" and indices.shape == values.shape
             and np.isfinite(values).all() and np.all((values >= 0) & (values <= 1)),
             "One endpoint and foreground score per row")
        secondary = indices if tie == "native" else np.arange(len(indices))
        order = np.lexsort((secondary, values if side else -values)) if k else np.empty(0, np.intp)
        chosen.append(indices[order[:k]])
    return tuple(chosen)
