"""Deterministic endpoint graphs and fixed-marginal binary-pair calibration."""

from collections import defaultdict
from hashlib import sha256

import numpy as np


SIGN = np.array((1.0, -1.0, -1.0, 1.0))


def graphs(coords, key):
    """Pair neighboring native endpoints; rewire within each native sign class."""
    coords = np.asarray(coords)
    if coords.ndim != 3 or coords.shape[1:] != (2, 2) or coords.dtype.kind not in "iu":
        raise ValueError("Expected [site, native sign, row or column] integer coordinates")
    cells = [tuple(map(int, row)) for row in coords.reshape(-1, 2)]
    vertices = {cell: position for position, cell in enumerate(cells)}
    if len(vertices) != len(cells) or (coords < 0).any():
        raise ValueError("Signed endpoints must be distinct native cells")
    used = set()
    local = []
    for y, x in sorted(vertices):
        a = vertices[y, x]
        if a in used:
            continue
        for neighbor in ((y, x + 1), (y + 1, x)):
            b = vertices.get(neighbor)
            if b is not None and b not in used:
                used.update((a, b))
                local.append((a, b) if a % 2 <= b % 2 else (b, a))
                break
    bins = defaultdict(list)
    for position, (a, b) in enumerate(local):
        bins[(a % 2, b % 2)].append(position)
    rewired = local.copy()
    for side, slots in sorted(bins.items()):
        if len(slots) < 2:
            continue
        suffix = f"{key}|{side[0]}:{side[1]}"
        offset = 1 + int.from_bytes(sha256(suffix.encode()).digest()[:8], "big") % (len(slots) - 1)
        for i, slot in enumerate(slots):
            rewired[slot] = local[slot][0], local[slots[(i + offset) % len(slots)]][1]
    return {name: np.asarray(edges, dtype=np.intp).reshape(-1, 2)
            for name, edges in (("local", local), ("nonspatial", rewired))}


def pair_tables(q_pairs, y_pairs):
    q = np.asarray(q_pairs, dtype=np.float64)
    y = np.asarray(y_pairs)
    if q.ndim != 2 or q.shape[1] != 2 or q.shape != y.shape or not np.isfinite(q).all():
        raise ValueError("Expected paired occupancy and raster references")
    if np.any((q < 0) | (q > 1)) or np.any((y != 0) & (y != 1)):
        raise ValueError("Occupancy must be probabilistic and references binary")
    p, r = q.T
    independent = np.column_stack(((1-p)*(1-r), (1-p)*r, p*(1-r), p*r))
    min_scale = independent[:, (0, 3)].min(axis=1)
    max_scale = independent[:, (1, 2)].min(axis=1)
    observed = np.eye(4)[2 * y[:, 0].astype(np.intp) + y[:, 1].astype(np.intp)]
    return independent, observed, min_scale, max_scale


def brier(q_pairs, y_pairs, kappa):
    """Four-category Brier loss for each pair, keeping both occupancy marginals."""
    if not np.isfinite(kappa) or not -1 <= kappa <= 1:
        raise ValueError("Kappa must belong to [-1, 1]")
    base, observed, lower, upper = pair_tables(q_pairs, y_pairs)
    scale = upper if kappa >= 0 else lower
    errors = base - observed + kappa * scale[:, None] * SIGN
    return (errors ** 2).sum(axis=1)


def fit_kappa(q_pairs, y_pairs, weights):
    """Minimize the weighted Brier objective on both Fréchet sign branches."""
    base, observed, lower, upper = pair_tables(q_pairs, y_pairs)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (len(base),) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Expected finite nonnegative weights over observed pairs")
    error = base - observed
    residual = -(error @ SIGN)
    choices = [0.0]
    for scale, bounds in ((lower, (-1., 0.)), (upper, (0., 1.))):
        denom = 4 * (w @ (scale ** 2))
        if denom > 0:
            choices.append(float(np.clip(w @ (scale * residual) / denom, *bounds)))
    return min(choices, key=lambda value: w @ brier(q_pairs, y_pairs, value))
