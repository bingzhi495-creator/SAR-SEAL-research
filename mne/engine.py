# -*- coding: utf-8 -*-
"""MNE: Multi-scale Nonparametric Evidence field (NumPy reference implementation).

Channels per scale s in {3, 7, 15, 31}:
  1. Truncated Mann-Whitney rank evidence A_s (mid-rank AUC of test window vs
     right-censored background annulus), standardized to z-score E_s by a
     per-image empirical null.
  2. Truncated local contrast T_s (truncated annulus moments).
  3. Heterogeneity H_cv = log(CV_7 / CV_31).

Design constants (NOT tunable hyperparameters; sensitivity goes to appendix):
  RHO          right-censoring mass of the background annulus  (TS-CFAR working point)
  Q_NULL       background sample count for the empirical null
  GUARD        guard ring half-width = ceil(s/2)               (PSF sidelobe geometry)
  BG_AREA_MIN  background annulus area >= BG_AREA_MIN * s^2

Verdict-6 compliance: the empirical-null background sampler uses an
*independent channel* (coarse intensity quantile + heterogeneity gate), never
the statistic being calibrated.

All histogram operations are exact on 8-bit data (256 bins).
"""
from __future__ import annotations

import numpy as np

SCALES = (3, 7, 15, 31)
RHO = 0.25
Q_NULL = 2000
BG_AREA_MIN = 2.0
NBINS = 256


# ----------------------------------------------------------------------------
# box-sum machinery (exact, integral-image based)
# ----------------------------------------------------------------------------

def _box_sum(arr: np.ndarray, k: int) -> np.ndarray:
    """Exact k x k box sum with zero padding outside the image.

    arr: (H, W) float64 or (B, H, W). Returns same shape, centered window.
    """
    if k == 1:
        return arr.astype(np.float64, copy=True)
    a = arr.astype(np.float64, copy=False)
    pad = k // 2
    if a.ndim == 2:
        ii = np.zeros((a.shape[0] + 1, a.shape[1] + 1))
        ii[1:, 1:] = a.cumsum(0).cumsum(1)
        H, W = a.shape
        r0 = np.clip(np.arange(H) - pad, 0, H)
        r1 = np.clip(np.arange(H) + pad + 1, 0, H)
        c0 = np.clip(np.arange(W) - pad, 0, W)
        c1 = np.clip(np.arange(W) + pad + 1, 0, W)
        return (ii[r1][:, c1] - ii[r0][:, c1] - ii[r1][:, c0] + ii[r0][:, c0])
    # bin-stacked input: keep the big (B,H,W) output in float32 (counts and
    # their box sums are integers < 2^24, exactly representable), compute each
    # 2-D slice in float64
    out = np.empty(a.shape, dtype=np.float32)
    for b in range(a.shape[0]):
        out[b] = _box_sum(a[b], k)
    return out


def _onehot_hist(img_u8: np.ndarray) -> np.ndarray:
    """(NBINS, H, W) float32 one-hot stack of an 8-bit image."""
    H, W = img_u8.shape
    oh = np.zeros((NBINS, H, W), dtype=np.float32)
    np.put_along_axis(oh, img_u8[None].astype(np.int64), 1.0, axis=0)
    return oh


def annulus_geometry(s: int) -> tuple[int, int]:
    """Return (guard_box, outer_box) odd sizes for scale s.

    guard ring width g = ceil(s/2); background annulus chosen as the smallest
    odd outer box with area(outer) - area(guard_box) >= BG_AREA_MIN * s*s.
    """
    g = -(-s // 2)
    gb = s + 2 * g
    ob = gb + 2
    while ob * ob - gb * gb < BG_AREA_MIN * s * s:
        ob += 2
    return gb, ob


def _truncate_hist(hist: np.ndarray, rho: float = RHO) -> np.ndarray:
    """Right-censor the top `rho` mass of per-pixel histograms.

    hist: (NBINS, H, W) counts. Returns truncated counts (fractional at the
    boundary bin, exact mass kept = (1-rho) * N per pixel).
    """
    n = hist.sum(axis=0, keepdims=True)
    keep_total = (1.0 - rho) * n
    cdf = np.cumsum(hist, axis=0)
    cdf_prev = cdf - hist
    kept = np.clip(keep_total - cdf_prev, 0.0, None)
    return np.minimum(kept, hist)


# ----------------------------------------------------------------------------
# per-scale statistics
# ----------------------------------------------------------------------------

def rank_auc_map(img_u8: np.ndarray, s: int, oh: np.ndarray | None = None,
                 rho: float = RHO) -> np.ndarray:
    """Truncated Mann-Whitney mid-rank AUC map A_s(x) in [0, 1].

    A_s(x) = (1/m) * sum_{i in T_s(x)} [ F_B'(I_i-) + 0.5 * p_B'(I_i) ]
    with B' the right-censored background annulus histogram at x.
    """
    if oh is None:
        oh = _onehot_hist(img_u8)
    gb, ob = annulus_geometry(s)
    hist_t = _box_sum(oh, s)                       # test window counts
    hist_bg = _box_sum(oh, ob) - _box_sum(oh, gb)  # annulus counts
    np.clip(hist_bg, 0.0, None, out=hist_bg)
    hist_bp = _truncate_hist(hist_bg, rho)

    n_bp = hist_bp.sum(axis=0)
    cdf_bp = np.cumsum(hist_bp, axis=0)
    # F(I-) + 0.5 p(I) per bin  ==  (cdf_prev + 0.5 * h) / N
    midrank = (cdf_bp - 0.5 * hist_bp)
    m = hist_t.sum(axis=0)  # test count (== s*s in the interior)
    num = (hist_t * midrank).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = num / (m * n_bp)
    return np.nan_to_num(auc, nan=0.5)


def truncated_contrast_map(img_u8: np.ndarray, s: int, oh: np.ndarray | None = None,
                           rho: float = RHO) -> np.ndarray:
    """T_s = (mean_T - mu_b') / (sigma_b' + eps) with truncated annulus moments."""
    if oh is None:
        oh = _onehot_hist(img_u8)
    gb, ob = annulus_geometry(s)
    img = img_u8.astype(np.float64)
    mean_t = _box_sum(img, s) / float(s * s)

    hist_bg = _box_sum(oh, ob) - _box_sum(oh, gb)
    np.clip(hist_bg, 0.0, None, out=hist_bg)
    hist_bp = _truncate_hist(hist_bg, rho)
    n = hist_bp.sum(axis=0)
    vals = np.arange(NBINS, dtype=np.float64)[:, None, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = (hist_bp * vals).sum(axis=0) / n
        var = (hist_bp * vals * vals).sum(axis=0) / n - mu * mu
    sigma = np.sqrt(np.clip(var, 0.0, None))
    return np.nan_to_num((mean_t - mu) / (sigma + 1e-3))


def heterogeneity_map(img_u8: np.ndarray) -> np.ndarray:
    """H_cv = log(CV_7 / CV_31): point scatterers vs textured clutter."""
    img = img_u8.astype(np.float64)

    def cv(k):
        m = _box_sum(img, k) / (k * k)
        m2 = _box_sum(img * img, k) / (k * k)
        sd = np.sqrt(np.clip(m2 - m * m, 0.0, None))
        return sd / (m + 1e-3)

    return np.log((cv(7) + 1e-6) / (cv(31) + 1e-6))


# ----------------------------------------------------------------------------
# empirical-null calibration (verdict-6 compliant background sampler)
# ----------------------------------------------------------------------------

def background_sample_mask(img_u8: np.ndarray) -> np.ndarray:
    """Independent-channel background region: coarse intensity quantile gate
    + heterogeneity gate. Never uses the statistic being calibrated."""
    coarse = _box_sum(img_u8.astype(np.float64), 15) / 225.0
    inten_gate = coarse <= np.quantile(coarse, 0.5)
    h = heterogeneity_map(img_u8)
    het_gate = np.abs(h - np.median(h)) < 2.0 * (np.abs(h - np.median(h)).mean() + 1e-6) * 2.5
    mask = inten_gate & het_gate
    if mask.sum() < 100:  # degenerate image: fall back to intensity gate only
        mask = inten_gate
    return mask


def empirical_null(score: np.ndarray, bg_mask: np.ndarray, q: int = Q_NULL,
                   rng: np.random.Generator | None = None) -> tuple[float, float]:
    """Robust (median, MAD->sigma) of `score` sampled on background positions."""
    rng = rng or np.random.default_rng(0)
    ys, xs = np.nonzero(bg_mask)
    if len(ys) == 0:
        return float(np.median(score)), float(1.4826 * np.median(np.abs(score - np.median(score))) + 1e-9)
    idx = rng.choice(len(ys), size=min(q, len(ys)), replace=False)
    vals = score[ys[idx], xs[idx]]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    return med, 1.4826 * mad + 1e-9


def orientation_field(ev_zmax: np.ndarray, smooth: int = 7) -> np.ndarray:
    """Local orientation (axis angle θ, radians) of the evidence field via the
    structure tensor — for the anisotropic hard-neg band and EBRH boundary head.

    For each pixel: gradients (gx,gy) → structure tensor J=[[Jxx,Jxy],[Jxy,Jyy]]
    box-smoothed over `smooth`; the eigenvector of the LARGER eigenvalue gives
    the gradient (cross-axis) direction, so the structure (ship long-axis)
    orientation is perpendicular. Returns θ in [0,π) = direction of the elongated
    structure (azimuth axis for ships). Pure numpy, CPU.
    """
    a = ev_zmax.astype(np.float64)
    gy, gx = np.gradient(a)
    Jxx = _box_sum(gx * gx, smooth) / (smooth * smooth)
    Jyy = _box_sum(gy * gy, smooth) / (smooth * smooth)
    Jxy = _box_sum(gx * gy, smooth) / (smooth * smooth)
    # principal gradient direction angle = 0.5*atan2(2Jxy, Jxx-Jyy); structure
    # (ridge/long-axis) is perpendicular -> +π/2
    grad_angle = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)
    theta = grad_angle + np.pi / 2.0
    return np.mod(theta, np.pi)


def mne_auc_field(img_u8: np.ndarray, scales=SCALES, rho: float = RHO) -> dict:
    """Memory-frugal field with ONLY the raw rank-AUC maps + background mask.

    For consumers that just need seeding (calibrate_thresholds / extract_seeds):
    skips the E/T/Hcv channels and frees the 256-bin histogram between scales,
    so parallel workers don't blow up RAM. Returns {'A', 'bg_mask'}.
    """
    img_u8 = np.ascontiguousarray(img_u8)
    if img_u8.ndim == 3:
        img_u8 = img_u8[..., 0]
    bg = background_sample_mask(img_u8)
    oh = _onehot_hist(img_u8)
    A = []
    for s in scales:
        A.append(rank_auc_map(img_u8, s, oh=oh, rho=rho))
    del oh
    return {"A": np.stack(A), "bg_mask": bg}


def mne_field(img_u8: np.ndarray, scales=SCALES, rho: float = RHO,
              clip_lo: float = -3.0, clip_hi: float = 10.0) -> dict:
    """Compute the full MNE pyramid for an 8-bit image.

    Returns dict with:
      'E'   : (S, H, W) standardized rank evidence z-maps
      'T'   : (S, H, W) truncated contrast maps
      'Hcv' : (H, W)
      'A'   : (S, H, W) raw AUC maps
      'null': list of (median, sigma) per scale
    """
    img_u8 = np.ascontiguousarray(img_u8)
    if img_u8.ndim == 3:
        img_u8 = img_u8[..., 0]
    oh = _onehot_hist(img_u8)
    bg = background_sample_mask(img_u8)
    rng = np.random.default_rng(0)

    A, E, T, nulls = [], [], [], []
    for s in scales:
        a = rank_auc_map(img_u8, s, oh=oh, rho=rho)
        med, sig = empirical_null(a, bg, rng=rng)
        z = np.clip((a - med) / sig, clip_lo, clip_hi)
        A.append(a)
        E.append(z)
        T.append(truncated_contrast_map(img_u8, s, oh=oh, rho=rho))
        nulls.append((med, sig))
    return {
        "A": np.stack(A),
        "E": np.stack(E),
        "T": np.stack(T),
        "Hcv": heterogeneity_map(img_u8),
        "null": nulls,
        "bg_mask": bg,
    }
