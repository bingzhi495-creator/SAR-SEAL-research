"""Match SegRefiner-LR's signed counts while relocating selected deletions."""
import numpy as np
from scipy.ndimage import distance_transform_edt


def squared_distance(mask):
    return np.rint(np.square(distance_transform_edt(mask))).astype(np.int64)


def deletion_location_masks(raw, stock):
    """Return depth and proximity controls with identical additions and delete counts."""
    if raw.ndim != 2 or raw.shape != stock.shape or raw.dtype != bool or stock.dtype != bool:
        raise ValueError("Expected co-shaped native 2-D Boolean masks")
    removed = raw & ~stock
    deletion_count = int(np.count_nonzero(removed))
    native_pixels = int(np.count_nonzero(raw))
    stock_pixels = int(np.count_nonzero(stock))
    stats = {
        "native_pixels": native_pixels,
        "stock_pixels": stock_pixels,
        "added_pixels": stock_pixels - native_pixels + deletion_count,
        "deleted_pixels": deletion_count,
        "relocated_deletions": 0,
    }
    if deletion_count == 0:
        return stock.copy(), stock.copy(), stats

    # A cropped bounding box plus one outside cell preserves inward distances.
    rows = np.flatnonzero(np.any(raw, axis=1))
    cols = np.flatnonzero(np.any(raw, axis=0))
    box = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
    native = raw[box]
    deleted = removed[box]
    foreground = np.flatnonzero(native)
    inward = squared_distance(np.pad(native, 1))[1:-1, 1:-1].ravel()
    proximity = squared_distance(~deleted).ravel()
    order = np.lexsort((foreground, proximity[foreground], inward[foreground]))
    depth_delete = np.zeros(native.shape, dtype=bool)
    depth_delete.ravel()[foreground[order[:deletion_count]]] = True

    retained = depth_delete & deleted
    moved = deleted & ~retained
    relocation_count = int(np.count_nonzero(moved))
    stats["relocated_deletions"] = relocation_count
    if relocation_count == 0:
        return stock.copy(), stock.copy(), stats

    alternative = np.flatnonzero(native & ~deleted)
    moved_distance = squared_distance(~moved).ravel()
    order = np.lexsort((alternative, moved_distance[alternative]))
    near_delete = retained.copy(order="C")
    near_delete.ravel()[alternative[order[:relocation_count]]] = True

    original_additions = stock[box] & ~native
    periphery = stock.copy()
    periphery[box] = original_additions | (native & ~depth_delete)
    local = stock.copy()
    local[box] = original_additions | (native & ~near_delete)
    return periphery, local, stats
