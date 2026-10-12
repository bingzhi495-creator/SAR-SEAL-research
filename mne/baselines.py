# -*- coding: utf-8 -*-
"""Classical CFAR control family for the G1 死活门 (verdict-1 required controls).

All controls share the *same window geometry* as MNE (test box s, guard ring
ceil(s/2), background annulus >= 2 m area) so that differences are attributable
to the statistic, not the geometry.

Controls (weak -> strong):
  box_z          multi-scale box-filter z-score, per-image GLOBAL normalization
                 (the "one depthwise conv + affine" linear-reproducible control)
  ca_cfar        cell-averaging CFAR: local annulus mean/std normalization
  ts_cfar        truncated-statistics CFAR (right-censored annulus moments,
                 Tao et al. TGRS 2016 working point rho=0.25)
  wilcoxon_cfar  single-scale Mann-Whitney rank CFAR, no truncation
                 (arXiv 2402.18579 sliding-window form)
"""
from __future__ import annotations

import numpy as np

from .engine import (SCALES, _box_sum, _onehot_hist, _truncate_hist,
                     annulus_geometry, NBINS)


def box_z(img_u8: np.ndarray, scales=SCALES) -> np.ndarray:
    """(S,H,W) global-affine-normalized box means on log intensity."""
    img = np.log1p(img_u8.astype(np.float64))
    mu, sd = img.mean(), img.std() + 1e-9
    out = []
    for s in scales:
        m = _box_sum(img, s) / (s * s)
        out.append((m - mu) / sd)
    return np.stack(out)


def _annulus_moments(img_u8: np.ndarray, s: int, truncate: bool, oh=None):
    gb, ob = annulus_geometry(s)
    if truncate:
        if oh is None:
            oh = _onehot_hist(img_u8)
        hist = _box_sum(oh, ob) - _box_sum(oh, gb)
        np.clip(hist, 0.0, None, out=hist)
        hist = _truncate_hist(hist)
        n = hist.sum(axis=0)
        vals = np.arange(NBINS, dtype=np.float64)[:, None, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = (hist * vals).sum(axis=0) / n
            var = (hist * vals * vals).sum(axis=0) / n - mu * mu
        return np.nan_to_num(mu), np.sqrt(np.clip(np.nan_to_num(var), 0.0, None))
    img = img_u8.astype(np.float64)
    a1 = _box_sum(img, ob) - _box_sum(img, gb)
    a2 = _box_sum(img * img, ob) - _box_sum(img * img, gb)
    cnt = _box_sum(np.ones_like(img), ob) - _box_sum(np.ones_like(img), gb)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = a1 / cnt
        var = a2 / cnt - mu * mu
    return np.nan_to_num(mu), np.sqrt(np.clip(np.nan_to_num(var), 0.0, None))


def ca_cfar(img_u8: np.ndarray, scales=SCALES) -> np.ndarray:
    """(S,H,W) cell-averaging CFAR z: (mean_T - mu_ann) / sigma_ann."""
    img = img_u8.astype(np.float64)
    out = []
    for s in scales:
        mean_t = _box_sum(img, s) / (s * s)
        mu, sd = _annulus_moments(img_u8, s, truncate=False)
        out.append((mean_t - mu) / (sd + 1e-3))
    return np.stack(out)


def ts_cfar(img_u8: np.ndarray, scales=SCALES) -> np.ndarray:
    """(S,H,W) truncated-statistics CFAR z (right-censored annulus moments)."""
    img = img_u8.astype(np.float64)
    oh = _onehot_hist(img_u8)
    out = []
    for s in scales:
        mean_t = _box_sum(img, s) / (s * s)
        mu, sd = _annulus_moments(img_u8, s, truncate=True, oh=oh)
        out.append((mean_t - mu) / (sd + 1e-3))
    return np.stack(out)


def wilcoxon_cfar(img_u8: np.ndarray, s: int = 7) -> np.ndarray:
    """(H,W) single-scale untruncated Mann-Whitney rank CFAR (mid-rank AUC)."""
    from .engine import rank_auc_map
    return rank_auc_map(img_u8, s, rho=0.0)
