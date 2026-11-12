"""Disjoint positive continuous-IoU margin contributions at fixed threshold."""

import numpy as np


def action_bonuses(occupancy, stock, inside, expanded, full_stock_area, threshold):
    q = np.asarray(occupancy, dtype=np.float64)
    if not (q.shape == stock.shape == inside.shape == expanded.shape and q.ndim == 2):
        raise ValueError("Aligned native pixel supports required")
    if not 0 < threshold < 1 or (q < 0).any() or (q > 1).any() or not np.isfinite(q).all():
        raise ValueError("Invalid occupancy or IoU threshold")
    local_stock = int(stock.sum())
    if full_stock_area < local_stock or (stock & ~inside).any() or (inside & ~expanded).any():
        raise ValueError("Stock, box and expansion must be nested")
    w = (1 + threshold) * q - threshold
    collar = float(np.maximum(w[expanded & ~inside], 0.).sum(dtype=float))
    remove = float(np.maximum(-w[stock], 0.).sum(dtype=float) + threshold * (full_stock_area - local_stock))
    inner = float(np.maximum(w[inside & ~stock], 0.).sum(dtype=float))
    return collar, remove, inner
