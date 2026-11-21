"""Area-preserving spatial-attraction control on frozen native endpoints."""

import numpy as np


RADIUS = 5
RANGE = 5.0
_GRID = np.arange(-RADIUS, RADIUS + 1)
_DR, _DC = np.meshgrid(_GRID, _GRID, indexing="ij")
_DIST = np.hypot(_DR, _DC)
_SUPPORT = (_DIST > 0) & (_DIST <= RADIUS)
_OFF_R, _OFF_C = _DR[_SUPPORT], _DC[_SUPPORT]
_WEIGHTS = np.exp(-_DIST[_SUPPORT] / RANGE)
_TOL = 1e-12


def _initial(mask, y, x):
    padded = np.pad(mask, RADIUS)
    local = padded[y[..., None] + _OFF_R + RADIUS, x[..., None] + _OFF_C + RADIUS]
    return np.sum(local * _WEIGHTS, axis=-1, dtype=np.float64)


def _effect(y, x, row, col):
    dist = np.hypot(y - row, x - col)
    return np.where((dist > 0) & (dist <= RADIUS), np.exp(-dist / RANGE), 0.0)


def _rank(values, eligible, ascending):
    scores = values if ascending else -values
    best = np.min(scores[eligible])
    return int(np.flatnonzero(eligible & (scores <= best + _TOL))[0])


def select(mask, coordinates, cap):
    """Return dynamic positive-gain pairs and static ranks at that exact count.

    Endpoint column 0 has original background cells, and column 1 foreground.
    All indices are original endpoint ordinals. Used endpoints are retired.
    """
    if not 0 <= cap <= len(coordinates):
        raise ValueError("Invalid per-candidate cap")
    if cap == 0:
        return [], []
    if coordinates.shape != (len(coordinates), 2):
        raise ValueError("Expected paired native linear indices")
    if np.any(coordinates < 0) or np.any(coordinates >= mask.size):
        raise ValueError("Endpoint outside the native grid")
    if len(np.unique(coordinates)) != 2 * len(coordinates):
        raise ValueError("Expected distinct endpoints")
    y, x = np.divmod(coordinates, mask.shape[1])
    if np.any(mask[y[:, 0], x[:, 0]]) or not np.all(mask[y[:, 1], x[:, 1]]):
        raise ValueError("Invalid original endpoint signs")
    initial = _initial(mask, y, x)
    attraction = initial.copy()
    real = []
    unused = np.ones(coordinates.shape, dtype=bool)
    for _ in range(cap):
        a = _rank(attraction[:, 0], unused[:, 0], ascending=False)
        d = _rank(attraction[:, 1], unused[:, 1], ascending=True)
        if attraction[a, 0] - attraction[d, 1] <= _TOL:
            break
        real.append((a, d))
        unused[a, 0] = unused[d, 1] = False
        attraction += _effect(y, x, y[a, 0], x[a, 0])
        attraction -= _effect(y, x, y[d, 1], x[d, 1])
    unused[:] = True
    static = []
    for _ in real:
        a = _rank(initial[:, 0], unused[:, 0], ascending=False)
        d = _rank(initial[:, 1], unused[:, 1], ascending=True)
        static.append((a, d))
        unused[a, 0] = unused[d, 1] = False
    return real, static
