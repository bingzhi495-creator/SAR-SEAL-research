"""A deterministic simple-point constraint for a fixed candidate support."""

import numpy as np


OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def component_count(active, connectivity, cardinal_only):
    remaining = set(active)
    count = 0
    while remaining:
        frontier = [remaining.pop()]
        touches_cardinal = False
        while frontier:
            y, x = frontier.pop()
            touches_cardinal |= abs(y) + abs(x) == 1
            for ny, nx in tuple(remaining):
                dy, dx = abs(y - ny), abs(x - nx)
                adjacent = max(dy, dx) == 1 if connectivity == 8 else dy + dx == 1
                if adjacent:
                    remaining.remove((ny, nx))
                    frontier.append((ny, nx))
        count += not cardinal_only or touches_cardinal
    return count


def make_simple_table():
    table = []
    for pattern in range(256):
        foreground = [p for i, p in enumerate(OFFSETS) if pattern & (1 << i)]
        background = [p for i, p in enumerate(OFFSETS) if not pattern & (1 << i)]
        table.append(component_count(foreground, 8, False) == 1
                     and component_count(background, 4, True) == 1)
    return tuple(table)


SIMPLE = make_simple_table()


def is_simple(mask, y, x):
    key = sum(int(mask[y+dy, x+dx]) << bit for bit, (dy, dx) in enumerate(OFFSETS))
    return SIMPLE[key]


def constrained_pairs(stock, coords, probabilities, cap):
    """Choose positive pairs in score order, checking each flip on the current raster."""
    if stock.ndim != 2 or coords.ndim != 2 or coords.shape[1] != 2 or coords.shape != probabilities.shape:
        raise ValueError("Paired endpoint arrays and one raster required")
    if not 0 <= cap <= len(coords):
        raise ValueError("Bounded original pair budget")
    if not cap:
        return []
    h, w = stock.shape
    if not np.isfinite(probabilities).all() or np.any(coords < 0) or np.any(coords >= stock.size):
        raise ValueError("Finite scores and valid original indexes")
    ys, xs = np.divmod(coords, w)
    if np.any(stock[ys[:,0], xs[:,0]]) or not np.all(stock[ys[:,1], xs[:,1]]):
        raise ValueError("Original background additions and foreground removals")
    y0, y1 = int(ys.min()-1), int(ys.max()+2)
    x0, x1 = int(xs.min()-1), int(xs.max()+2)
    mask = np.pad(stock[max(0,y0):min(h,y1), max(0,x0):min(w,x1)],
                  ((max(0,-y0),max(0,y1-h)), (max(0,-x0),max(0,x1-w)))).astype(bool, copy=False)
    ys = ys-y0
    xs = xs-x0
    order_add = np.argsort(-probabilities[:,0], kind="stable").tolist()
    order_drop = np.argsort(probabilities[:,1], kind="stable").tolist()
    pairs = []
    for _ in range(cap):
        selected = None
        for a in order_add:
            ay, ax = int(ys[a,0]), int(xs[a,0])
            if not is_simple(mask, ay, ax):
                continue
            mask[ay,ax] = True
            for b in order_drop:
                if probabilities[a,0] <= probabilities[b,1]:
                    break
                by, bx = int(ys[b,1]), int(xs[b,1])
                if is_simple(mask, by, bx):
                    selected = a, b
                    mask[by,bx] = False
                    break
            if selected is not None:
                break
            mask[ay,ax] = False
        if selected is None:
            break
        a,b = selected
        order_add.remove(a)
        order_drop.remove(b)
        pairs.append(selected)
    return pairs
