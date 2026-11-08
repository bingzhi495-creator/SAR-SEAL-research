"""Exact gradient trace census for equal-budget, uniform-risk BCE estimators."""
from math import ceil, floor

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import expit


def boundary_quota(points, boundary_size, total):
    if points < 2 or not 0 <= boundary_size <= total or total <= 0:
        raise ValueError("Invalid point or stratum count")
    if boundary_size in (0, total):
        return points if boundary_size else 0
    proportional = floor(points * boundary_size / total + 0.5)
    return min(points - 1, max(ceil(points / 20), proportional))


def _axis(source, target):
    locations = (np.arange(target, dtype=np.float64) + 0.5) * source / target - 0.5
    lower = np.floor(locations).astype(np.int64)
    distance = locations - lower
    return ((lower, 1.0 - distance), (lower + 1, distance))


class BilinearRisk:
    """Gradient with respect to coarse logits under zero-padded bilinear point reads."""

    def __init__(self, coarse_shape, target_shape):
        ch, cw = coarse_shape
        th, tw = target_shape
        rows = np.arange(th * tw).reshape(target_shape)
        ri, ci, values = [], [], []
        for y, wy in _axis(ch, th):
            for x, wx in _axis(cw, tw):
                valid = ((y[:, None] >= 0) & (y[:, None] < ch) &
                         (x[None, :] >= 0) & (x[None, :] < cw))
                ri.append(rows[valid])
                ci.append((y[:, None] * cw + x[None, :])[valid])
                values.append((wy[:, None] * wx[None, :])[valid])
        self.coarse_shape = coarse_shape
        self.target_shape = target_shape
        self.interpolation = csr_matrix((np.concatenate(values), (np.concatenate(ri), np.concatenate(ci))),
                                        shape=(th * tw, ch * cw))
        self.transpose = self.interpolation.T
        self.row_norm_squared = np.asarray(self.interpolation.power(2).sum(axis=1)).ravel()

    def dense_logits(self, logits):
        if logits.shape != self.coarse_shape or not np.isfinite(logits).all():
            raise ValueError("Expected finite fixed coarse logits")
        return (self.interpolation @ logits.ravel()).reshape(self.target_shape)

    def measure(self, logits, target, band, points):
        if target.shape != self.target_shape or band.shape != target.shape or band.dtype != np.bool_:
            raise ValueError("Target and boundary must share the fixed target grid")
        dense = self.dense_logits(logits)
        residual = expit(dense.ravel()) - target.ravel()
        energy = residual**2 * self.row_norm_squared
        n = residual.size
        inside = band.ravel()
        boundary_size = int(inside.sum())
        k_b = boundary_quota(points, boundary_size, n)
        # Group moments use the total pixel count, so mean_b is the contribution to global BCE.
        mean_b = self.transpose @ (residual * inside) / n
        mean_g = self.transpose @ residual / n
        mean_o = mean_g - mean_b
        energy_b = float(energy[inside].sum() / n)
        energy_o = float(energy[~inside].sum() / n)
        mean_g_sq = float(np.dot(mean_g, mean_g))
        mean_b_sq = float(np.dot(mean_b, mean_b))
        mean_o_sq = float(np.dot(mean_o, mean_o))
        w_b = boundary_size / n
        global_u = max(0.0, energy_b + energy_o - mean_g_sq) / points
        boundary_u = max(0.0, energy_b - mean_b_sq) / points
        boundary_s = max(0.0, w_b * energy_b - mean_b_sq) / k_b if k_b else 0.0
        outside_s = (max(0.0, (1 - w_b) * energy_o - mean_o_sq) / (points - k_b)
                     if points > k_b else 0.0)
        return dict(global_uniform=global_u, boundary_uniform=boundary_u, global_stratified=boundary_s + outside_s,
                    boundary_stratified=boundary_s, global_mean_squared=mean_g_sq, boundary_size=boundary_size,
                    boundary_quota=k_b)
