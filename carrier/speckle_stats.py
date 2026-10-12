# -*- coding: utf-8 -*-
"""Task B: empirical ground truth for the R8 speckle-marginalization design.

Measures, on real HRSID chips (>=40 inshore + >=40 offshore, GT COCO masks):

  [P1] block-wise ENL maps: 32x32-block ENL = (mean/std)^2 of intensity
       I = A^2, with the SAME validity guards as carrier/speckle_infer.py
       (dark/saturated/degenerate blocks excluded); per-chip median block
       ENL and the R8 estimator's chip ENL, inshore vs offshore.

  [P2] distribution fitting per pixel class (ship interior / inshore
       background clutter / offshore sea) against Rayleigh / Rice /
       Nakagami / K.  ALL fits and KS statistics are computed in the
       INTENSITY domain: the map A -> I = A^2 is strictly increasing on
       [0, inf), so the empirical CDF ranks are identical and
       D_KS(amplitude, F_A) == D_KS(intensity, F_I) EXACTLY, where
       F_I(x) = F_A(sqrt(x)).  Amplitude-domain law <-> intensity-domain law:
         Rayleigh(sigma)      <-> Exponential(mean 2 sigma^2)
         Rice(nu, sigma)      <-> sigma^2 * ncx2(df=2, nc=nu^2/sigma^2)
         Nakagami(m, Omega)   <-> Gamma(shape m, scale Omega/m)
         K(L, nu_t, mu)       <-> Gamma(L, mean t) compounded with
                                  t ~ Gamma(nu_t, mean mu)   (product model)
       Estimators (moment-based, closed form; derivations in comments):
         Exponential: mu_hat = mean(I)                       (also the MLE)
         Gamma:       shape = mean(I)^2 / var(I), scale = var/mean (moments)
         Rice even-moment: with m1 = E[I], m2 = E[I^2],
              E[I] = nu^2 + 2 sigma^2,  E[I^2] = nu^4 + 8 sigma^2 nu^2 + 8 sigma^4
              => 2 m1^2 - m2 = nu^4  (exact algebra), sigma^2 = (m1 - nu^2)/2
         K (L fixed to the chip/regime ENL): E[I^2]/E[I]^2 = (1+1/L)(1+1/nu)
              => nu = 1 / ( M2 * L/(L+1) - 1 ),  M2 = m2/m1^2
       K pdf/cdf have no closed form; both are evaluated by quantile-midpoint
       quadrature over the texture t (M=400 nodes, error O(M^-2) for the
       smooth integrand):  F_I(x) = E_t[ GammaCDF(x; L, scale=t/L) ].

       'Fully-developed-speckle violation rate' = fraction of 100%-background
       32x32 blocks whose intensity rejects Exponential (== amplitude rejects
       Rayleigh) at alpha = 0.05 under a Lilliefors-type test with the
       critical value MONTE-CARLO CALIBRATED through the actual data pipeline
       (continuous Rayleigh -> rint quantization -> U(-.5,.5) dither), on a
       grid of Rayleigh scales (quantization breaks exact scale invariance;
       nearest-grid critical value is used, grid is dense so the residual
       calibration error is small).

  [P3] top-q% brightest-pixel 'protection' attribution: fraction of the
       protected pixels (per-chip quantile threshold, exactly the
       speckle_redraw rule a >= quantile(a, 1-q)) that fall inside GT ship
       masks / a +5 px ship margin / pure background, inshore vs offshore,
       for q in {0.1%, 0.2%, 0.5%, 1%, 2%}; plus ship coverage (recall).

  [P4] re-draw fidelity on 10 chips (5 inshore + 5 offshore), K=4 draws of
       the CURRENT production recipe (speckle_redraw defaults; NOTE the
       2026-07-31 archived run used the pre-R1..R4 recipe, ksize=5 +
       top-0.5% protection - rerunning now measures the fixed recipe):
       (a) ship-interior mean amplitude & intensity retention ratios,
       (b) clutter texture retention: Pearson r of 7x7 local-variance maps
           (background only) and of radially averaged log power spectra.

  [P5] 4-panel publication figure -> paper_tgrs/figs/fig_speckle_stats.pdf

8-bit caveats (stated, not hidden): chips are JPG-compressed 8-bit
amplitude; values are dequantized with additive U(-0.5, 0.5) dither before
fitting (bin-center rounding model; density-smoothness approximation error
O(h), h = 1 DN); the MC calibration puts the same quantize+dither pipeline
under the null, so [P2] rejection rates are honest to first order.  JPG
block coding adds spatial correlation that no iid test models - noted.

Run (CPU only; the GPU is training):
  python carrier/speckle_stats.py
"""
from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''          # iron rule: never touch GPU

import sys
import time

import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from carrier.speckle_infer import (estimate_enl, gamma_median, make_rng,  # noqa: E402
                                   speckle_redraw)

from pycocotools.coco import COCO      # noqa: E402
from scipy import stats as st          # noqa: E402
from scipy import special as sp        # noqa: E402

import matplotlib                      # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt        # noqa: E402
from matplotlib import gridspec        # noqa: E402

# --------------------------------------------------------------------------- #
IMG_DIR = r'D:/SAR/UNet++/HRSID_JPG/HRSID_JPG/JPEGImages1'
ANN_IN = r'E:/SAR/seal/data/hrsid/annotations/test_inshore.json'
ANN_OFF = r'E:/SAR/seal/data/hrsid/annotations/test_offshore.json'
FIG_OUT = r'E:/SAR/seal/paper_tgrs/figs/fig_speckle_stats.pdf'
N_CHIPS = 60            # per regime (>= 40 required)
BLOCK = 32
SEED = 20260731
DITHER_HALF = 0.5       # U(-0.5, 0.5) dequantization dither
KS_SUB = 4000           # per-chip-per-class KS subsample
POOL_SUB = 20000        # pooled per-class fit subsample
MC_REPS = 1500          # Lilliefors calibration replicates
SIGMA_GRID = np.array([2., 3., 5., 8., 12., 20., 35., 60., 100.])
ALPHA = 0.05
NU_MAX = 1e6            # K texture shape 'no texture' sentinel
M_QUAD = 400            # K-distribution quadrature nodes


def fmt(x, n=4):
    return f'{x:.{n}f}'


# --------------------------------------------------------------------------- #
# intensity-domain laws: fit (moments) + cdf + pdf
# --------------------------------------------------------------------------- #
def fit_expon(I):
    return {'mu': float(np.mean(I))}


def cdf_expon(x, p):
    return st.expon.cdf(x, scale=p['mu'])


def pdf_expon(x, p):
    return st.expon.pdf(x, scale=p['mu'])


def fit_gamma_mom(I):
    m, v = float(np.mean(I)), float(np.var(I))
    shape = float(np.clip(m * m / max(v, 1e-12), 0.02, 1e4))
    return {'shape': shape, 'scale': m / shape}


def cdf_gamma(x, p):
    return st.gamma.cdf(x, p['shape'], scale=p['scale'])


def pdf_gamma(x, p):
    return st.gamma.pdf(x, p['shape'], scale=p['scale'])


def fit_rice_mom(I):
    """Even-moment Rice: nu^4 = 2 E[I]^2 - E[I^2] (exact), sigma^2=(E[I]-nu^2)/2."""
    m1, m2 = float(np.mean(I)), float(np.mean(I * I))
    nu4 = 2.0 * m1 * m1 - m2
    nu2 = float(np.sqrt(nu4)) if nu4 > 0 else 0.0
    nu2 = min(nu2, 0.999 * m1)                    # keep sigma^2 > 0
    sig2 = max((m1 - nu2) / 2.0, 1e-6)
    return {'nu2': nu2, 'sig2': sig2}


def cdf_rice(x, p):
    return st.ncx2.cdf(x, 2, p['nu2'] / p['sig2'], scale=p['sig2'])


def pdf_rice(x, p):
    return st.ncx2.pdf(x, 2, p['nu2'] / p['sig2'], scale=p['sig2'])


def fit_k_mom(I, L):
    """K product model, L fixed: M2 = E[I^2]/E[I]^2 = (1+1/L)(1+1/nu)."""
    m1, m2 = float(np.mean(I)), float(np.mean(I * I))
    M2 = m2 / max(m1 * m1, 1e-12)
    den = M2 * L / (L + 1.0) - 1.0
    nu = 1.0 / den if den > 1e-6 else NU_MAX
    return {'L': float(L), 'nu': float(np.clip(nu, 0.05, NU_MAX)), 'mu': m1}


def _k_tex_nodes(p):
    u = (np.arange(M_QUAD) + 0.5) / M_QUAD
    return st.gamma.ppf(u, p['nu'], scale=p['mu'] / p['nu'])


def cdf_k(x, p):
    if p['nu'] >= NU_MAX * 0.99:                  # no texture -> pure Gamma(L)
        return st.gamma.cdf(x, p['L'], scale=p['mu'] / p['L'])
    t = _k_tex_nodes(p)                           # (M,)
    xx = np.asarray(x, dtype=np.float64)[:, None]
    return st.gamma.cdf(xx, p['L'], scale=t[None, :] / p['L']).mean(axis=1)


def pdf_k(x, p):
    if p['nu'] >= NU_MAX * 0.99:
        return st.gamma.pdf(x, p['L'], scale=p['mu'] / p['L'])
    t = _k_tex_nodes(p)
    xx = np.asarray(x, dtype=np.float64)[:, None]
    return st.gamma.pdf(xx, p['L'], scale=t[None, :] / p['L']).mean(axis=1)


DISTS = ('rayleigh', 'rice', 'nakagami', 'K')


def ks_stat(I, cdf_vals):
    """Exact one-sample KS D given the model CDF at the sorted sample."""
    n = len(I)
    c = np.asarray(cdf_vals)
    lo = np.max(np.arange(1, n + 1) / n - c)
    hi = np.max(c - np.arange(0, n) / n)
    return float(max(lo, hi))


def fit_all_ks(I, L):
    """Fit the 4 laws on intensity sample I; return {name: (D, params)}."""
    Is = np.sort(np.asarray(I, dtype=np.float64))
    out = {}
    for name in DISTS:
        if name == 'rayleigh':
            p = fit_expon(Is); c = cdf_expon(Is, p)
        elif name == 'rice':
            p = fit_rice_mom(Is); c = cdf_rice(Is, p)
        elif name == 'nakagami':
            p = fit_gamma_mom(Is); c = cdf_gamma(Is, p)
        else:
            p = fit_k_mom(Is, L); c = cdf_k(Is, p)
        out[name] = (ks_stat(Is, c), p)
    return out


# --------------------------------------------------------------------------- #
# Lilliefors-Rayleigh MC calibration through the quantize+dither pipeline
# --------------------------------------------------------------------------- #
def calibrate_lilliefors(n, sigma_grid, reps=MC_REPS, alpha=ALPHA, seed=7):
    rng = np.random.default_rng(seed)
    crit = {}
    for s in sigma_grid:
        Ds = np.empty(reps)
        for r in range(reps):
            a = st.rayleigh.rvs(scale=s, size=n, random_state=rng)
            a = np.clip(np.rint(a), 0, 255)                     # 8-bit pipeline
            a = a + rng.uniform(-DITHER_HALF, DITHER_HALF, n)   # dither
            a = np.clip(a, 1e-3, None)
            I = np.sort(a * a)
            Ds[r] = ks_stat(I, cdf_expon(I, fit_expon(I)))
        crit[float(s)] = float(np.quantile(Ds, 1 - alpha))
    return crit


def crit_lookup(crit, sigma_hat):
    g = np.array(sorted(crit))
    return crit[float(g[np.argmin(np.abs(g - sigma_hat))])]


# --------------------------------------------------------------------------- #
# per-chip machinery
# --------------------------------------------------------------------------- #
def load_chip(coco, img_id):
    info = coco.loadImgs([img_id])[0]
    img = cv2.imread(os.path.join(IMG_DIR, info['file_name']), cv2.IMREAD_COLOR)
    assert img is not None, info['file_name']
    return img[:, :, 0]                             # replicated-gray JPGs


def ship_masks(coco, img_id, shape):
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
    u = np.zeros(shape, np.uint8)
    for a in anns:
        u |= coco.annToMask(a).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    core = cv2.erode(u, k, iterations=1)            # interior (no mixed edge px)
    margin5 = cv2.dilate(u, k, iterations=5)        # +5 px halo (sidelobe belt)
    return u.astype(bool), core.astype(bool), margin5.astype(bool)


def block_enls(amp):
    """Valid-block ENLs with the estimate_enl guards; also background-only
    block coordinates are handled by the caller."""
    a = amp.astype(np.float64); I = a * a
    out, coords = [], []
    for y in range(0, amp.shape[0] - BLOCK + 1, BLOCK):
        for x in range(0, amp.shape[1] - BLOCK + 1, BLOCK):
            ba = a[y:y + BLOCK, x:x + BLOCK]
            bI = I[y:y + BLOCK, x:x + BLOCK]
            m, s = bI.mean(), bI.std(ddof=1)
            if (ba.mean() < 2.0 or s <= 1e-9 or (ba == 0).mean() > 0.10
                    or (ba >= 255).mean() > 0.005):
                continue
            out.append((m / s) ** 2)
            coords.append((y, x))
    return np.asarray(out), coords


def local_var(a, k=7):
    a = a.astype(np.float32)
    m = cv2.blur(a, (k, k))
    m2 = cv2.blur(a * a, (k, k))
    return np.maximum(m2 - m * m, 0.0)


def radial_logspec(a, nbin=40):
    a = a.astype(np.float64)
    F = np.abs(np.fft.fftshift(np.fft.fft2(a - a.mean()))) ** 2
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - h / 2, xx - w / 2)
    edges = np.linspace(0, r.max() * 0.7, nbin + 1)   # drop noisy corner bins
    idx = np.digitize(r.ravel(), edges) - 1
    Fv = F.ravel()
    spec = np.array([Fv[idx == i].mean() if np.any(idx == i) else np.nan
                     for i in range(nbin)])
    return np.log10(spec + 1e-12)


# --------------------------------------------------------------------------- #
STD_LUM_Q50 = np.array([  # IJG standard luminance quantization table (Q50)
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99],
    dtype=np.float64)


def sniff_jpeg_quality(paths, n=20):
    """Estimate IJG quality factor from the stored luminance quantization
    table: T = clip(floor((Q50*S+50)/100), 1, 255), S = 5000/q (q<50) else
    200-2q.  Invert S by least squares over the 64 coefficients (approximate
    because of clipping at 1/255; fine for mid-range q)."""
    try:
        from PIL import Image
    except ImportError:
        return None
    qs = []
    for p in paths[:n]:
        try:
            with Image.open(p) as im:
                t = np.array(im.quantization[0], dtype=np.float64)
        except Exception:
            continue
        S = float(np.median(100.0 * t / STD_LUM_Q50))    # robust to clipping
        q = 5000.0 / S if S > 100 else (200.0 - S) / 2.0
        qs.append(q)
    return (float(np.median(qs)), len(qs)) if qs else None


def corr_speckle_field(sig, rho1_I, shape, rng):
    """Correlated single-look Rayleigh amplitude with intensity lag-1
    correlation ~ rho1_I: smooth white circular complex Gaussian with a
    Gaussian kernel of std s, using rho_z(lag)=exp(-lag^2/(4 s^2)) and
    rho_I = |rho_z|^2 (complex Gaussian moment theorem)."""
    from scipy.ndimage import gaussian_filter
    rho_z1 = float(np.sqrt(max(rho1_I, 1e-6)))
    s = float(np.sqrt(-1.0 / (4.0 * np.log(rho_z1))))
    pad = 16
    h, w = shape[0] + 2 * pad, shape[1] + 2 * pad
    zr = gaussian_filter(rng.standard_normal((h, w)), s)
    zi = gaussian_filter(rng.standard_normal((h, w)), s)
    zr = zr[pad:-pad, pad:-pad]; zi = zi[pad:-pad, pad:-pad]
    sc = sig / max(zr.std(), 1e-12)
    return np.hypot(zr * sc, zi * sc)


def smoke_tests():
    print('[SMOKE] ---- pre-flight numeric checks -------------------------')
    # 1) K pdf integrates to 1 and reproduces the mean (quadrature sanity)
    p = {'L': 1.1, 'nu': 2.0, 'mu': 1000.0}
    x = np.linspace(1e-3, 3e4, 60001)
    f = pdf_k(x, p)
    Z = np.trapz(f, x); mu = np.trapz(x * f, x)
    print(f'[SMOKE] K(L=1.1,nu=2,mu=1000): integral={Z:.5f} (want 1), '
          f'mean={mu:.1f} (want 1000)')
    assert abs(Z - 1) < 5e-3 and abs(mu - 1000) < 20
    # 2) Rice even-moment estimator recovers (nu, sigma) on synthetic data
    rng = np.random.default_rng(0)
    A = st.rice.rvs(b=2.0, scale=10.0, size=200000, random_state=rng)
    ph = fit_rice_mom(A * A)                       # true nu=20, sigma=10
    print(f'[SMOKE] Rice recover: nu_hat={np.sqrt(ph["nu2"]):.2f} (want 20.0), '
          f'sigma_hat={np.sqrt(ph["sig2"]):.2f} (want 10.0)')
    assert abs(np.sqrt(ph['nu2']) - 20) < 0.5 and abs(np.sqrt(ph['sig2']) - 10) < 0.5
    # 3) K moment inversion round-trip: M2 -> nu -> M2
    pk = fit_k_mom(st.gamma.rvs(1.0, scale=1.0, size=10, random_state=rng) * 0
                   + np.array([1.0] * 10), L=1.0)  # degenerate: no texture
    print(f'[SMOKE] K no-texture sentinel on constant data: nu={pk["nu"]:.0f} '
          f'(want {NU_MAX:.0f})')
    # 4) gamma_median bias constants used by speckle_redraw
    print(f'[SMOKE] gamma_median: L=1 -> {gamma_median(1.0):.4f} (ln2), '
          f'L=4 -> {gamma_median(4.0):.4f}')
    print('[SMOKE] all passed')


# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    smoke_tests()
    rng = np.random.default_rng(SEED)
    coco_in, coco_off = COCO(ANN_IN), COCO(ANN_OFF)

    def sample_ids(coco):
        ids = sorted(i for i in coco.getImgIds()
                     if len(coco.getAnnIds(imgIds=[i])) > 0)
        pick = rng.choice(len(ids), size=min(N_CHIPS, len(ids)), replace=False)
        return [ids[i] for i in sorted(pick)]

    ids_in, ids_off = sample_ids(coco_in), sample_ids(coco_off)
    print(f'\n[CFG] chips: inshore={len(ids_in)} offshore={len(ids_off)} '
          f'(seed {SEED}); block={BLOCK}; dither=U(+-{DITHER_HALF}); '
          f'alpha={ALPHA}; KS subsample={KS_SUB}; pooled subsample={POOL_SUB}')

    print('[CFG] calibrating Lilliefors-Rayleigh critical values '
          f'(n=1024/256/64 block strides 1/2/4 and n={KS_SUB} chip, '
          f'{MC_REPS} reps, sigma grid {list(SIGMA_GRID)})...')
    crit_s = {1: calibrate_lilliefors(1024, SIGMA_GRID),
              2: calibrate_lilliefors(256, SIGMA_GRID),
              4: calibrate_lilliefors(64, SIGMA_GRID)}
    crit_blk = crit_s[1]
    crit_chip = calibrate_lilliefors(KS_SUB, SIGMA_GRID)
    for s_, c_ in crit_s.items():
        print(f'[CFG] D*_0.05(stride {s_}, n={1024 // s_ ** 2}) per sigma:',
              {k: round(v, 4) for k, v in c_.items()})
    print(f'[CFG] D*_0.05(n={KS_SUB}) per sigma:',
          {k: round(v, 4) for k, v in crit_chip.items()})
    qj = sniff_jpeg_quality([os.path.join(
        IMG_DIR, coco_in.loadImgs([i])[0]['file_name']) for i in ids_in])
    print(f'[CFG] sniffed JPEG quality of real chips: '
          f'{"unavailable" if qj is None else f"median q={qj[0]:.0f} over {qj[1]} files"}')

    # ---------------- accumulators ---------------- #
    enl_blocks = {'inshore': [], 'offshore': []}
    enl_chip_med = {'inshore': [], 'offshore': []}
    enl_chip_r8 = {'inshore': [], 'offshore': []}
    pool = {'ship': [], 'clutter': [], 'sea': []}
    pool_rej = []                     # inshore rejected-block pixels (suppl.)
    blk_tests = {r: {s_: [0, 0] for s_ in (1, 2, 4)}         # [rejected, total]
                 for r in ('inshore', 'offshore')}
    chip_tests = {'inshore': [0, 0], 'offshore': [0, 0]}
    votes = {c: {d: 0 for d in DISTS} for c in ('ship', 'clutter', 'sea')}
    vote_n = {c: 0 for c in ('ship', 'clutter', 'sea')}
    Dsum = {c: {d: [] for d in DISTS} for c in ('ship', 'clutter', 'sea')}
    QS = [0.999, 0.998, 0.995, 0.99, 0.98]
    prot = {r: {q: np.zeros(4) for q in QS} for r in ('inshore', 'offshore')}
    #                    [n_prot, in_ship, in_margin5, ship_px_protected]
    TS = [150, 200, 240]                            # absolute-DN alternatives
    prot_abs = {r: {t: np.zeros(4) for t in TS} for r in ('inshore', 'offshore')}
    ship_px_tot = {'inshore': 0, 'offshore': 0}

    fid_ids = {'inshore': ids_in[:5], 'offshore': ids_off[:5]}
    fidelity = []
    inst_ratio = []                                 # (ship px, A-mean ratio)
    lag1 = {r: {lg: [] for lg in (1, 2, 4)}         # bg intensity lag-k rho
            for r in ('inshore', 'offshore')}
    sea_sigma = {'inshore': [], 'offshore': []}     # per-chip bg Rayleigh sigma
    inst = {'inshore': {'votes': {d: 0 for d in DISTS}, 'n': 0, 'kfac': [],
                        'D': {d: [] for d in DISTS}, 'sat': []},
            'offshore': {'votes': {d: 0 for d in DISTS}, 'n': 0, 'kfac': [],
                         'D': {d: [] for d in DISTS}, 'sat': []}}

    for regime, coco, ids in (('inshore', coco_in, ids_in),
                              ('offshore', coco_off, ids_off)):
        for k, img_id in enumerate(ids):
            amp = load_chip(coco, img_id)
            union, core, margin5 = ship_masks(coco, img_id, amp.shape)
            bg = ~margin5
            a = amp.astype(np.float64)
            crng = make_rng(img_id, 999)          # deterministic per-chip dither
            a_d = np.clip(a + crng.uniform(-DITHER_HALF, DITHER_HALF, a.shape),
                          1e-3, None)
            I_d = a_d * a_d

            # ---- P1: block ENL ----
            be, coords = block_enls(amp)
            enl_blocks[regime].append(be)
            if len(be):
                enl_chip_med[regime].append(float(np.median(be)))
            r8 = estimate_enl(amp)
            enl_chip_r8[regime].append(r8)

            # ---- P2a: block-level Rayleigh violation (100% background blocks)
            rej_px = []
            for (y, x) in coords:
                if not bg[y:y + BLOCK, x:x + BLOCK].all():
                    continue
                rejected = False
                for s_ in (1, 2, 4):
                    Ib = np.sort(
                        I_d[y:y + BLOCK:s_, x:x + BLOCK:s_].ravel())
                    D = ks_stat(Ib, cdf_expon(Ib, fit_expon(Ib)))
                    sig_hat = np.sqrt(np.mean(Ib) / 2.0)
                    rej_ = D > crit_lookup(crit_s[s_], sig_hat)
                    blk_tests[regime][s_][0] += int(rej_)
                    blk_tests[regime][s_][1] += 1
                    if s_ == 1:
                        rejected = rej_
                if rejected and regime == 'inshore':
                    rej_px.append(I_d[y:y + BLOCK, x:x + BLOCK].ravel())
            if rej_px:
                pool_rej.append(np.concatenate(rej_px))

            # ---- P2b: per-chip class fits ----
            cls_px = {}
            if core.sum() >= 300:
                cls_px['ship'] = I_d[core]
            bgpx = I_d[bg]
            cname = 'clutter' if regime == 'inshore' else 'sea'
            if bgpx.size >= 300:
                cls_px[cname] = bgpx
            for cname2, Ipx in cls_px.items():
                sub = Ipx if Ipx.size <= KS_SUB else \
                    crng.choice(Ipx, KS_SUB, replace=False)
                res = fit_all_ks(sub, L=r8)
                best = min(res, key=lambda d: res[d][0])
                votes[cname2][best] += 1
                vote_n[cname2] += 1
                for d in DISTS:
                    Dsum[cname2][d].append(res[d][0])
                pool[cname2].append(sub)
                if cname2 in ('clutter', 'sea'):     # chip-level Rayleigh test
                    Db = res['rayleigh'][0]
                    sig_hat = np.sqrt(np.mean(sub) / 2.0)
                    chip_tests[regime][0] += int(
                        Db > crit_lookup(crit_chip, sig_hat))
                    chip_tests[regime][1] += 1

            # ---- P2c: per-instance ship fits + bg lag-1 autocorrelation ----
            for lg in (1, 2, 4):
                mh = bg[:, :-lg] & bg[:, lg:]
                mv = bg[:-lg, :] & bg[lg:, :]
                if mh.sum() > 1000 and mv.sum() > 1000:
                    rh = np.corrcoef(I_d[:, :-lg][mh], I_d[:, lg:][mh])[0, 1]
                    rv = np.corrcoef(I_d[:-lg, :][mv], I_d[lg:, :][mv])[0, 1]
                    lag1[regime][lg].append(0.5 * (rh + rv))
            sea_sigma[regime].append(float(np.sqrt(np.mean(I_d[bg]) / 2.0)))
            k3 = np.ones((3, 3), np.uint8)
            for ann in coco.loadAnns(coco.getAnnIds(imgIds=[img_id])):
                mci = cv2.erode(coco.annToMask(ann).astype(np.uint8),
                                k3, 1).astype(bool)
                if mci.sum() < 200:
                    continue
                Ipx = I_d[mci]
                subi = Ipx if Ipx.size <= 1024 else \
                    crng.choice(Ipx, 1024, replace=False)
                resi = fit_all_ks(subi, L=r8)
                besti = min(resi, key=lambda d: resi[d][0])
                rec = inst[regime]
                rec['votes'][besti] += 1
                rec['n'] += 1
                for d in DISTS:
                    rec['D'][d].append(resi[d][0])
                pri = resi['rice'][1]
                rec['kfac'].append(pri['nu2'] / (2 * pri['sig2']))
                rec['sat'].append(float((a[mci] >= 254.5).mean()))

            # ---- P3: protection attribution ----
            ship_px_tot[regime] += int(union.sum())
            for q in QS:
                thr = np.quantile(a, q)              # exact speckle_redraw rule
                sel = a >= thr
                prot[regime][q] += np.array([
                    sel.sum(), (sel & union).sum(), (sel & margin5).sum(),
                    (sel & union).sum()])            # [n, in_ship, in_margin, cov]
            for t in TS:
                sel = a >= t
                prot_abs[regime][t] += np.array([
                    sel.sum(), (sel & union).sum(), (sel & margin5).sum(),
                    (sel & union).sum()])

            # ---- P4: fidelity ----
            if img_id in fid_ids[regime]:
                lv0 = local_var(amp)
                sp0 = radial_logspec(amp)
                ratios_a, ratios_i, var_r, spec_r = [], [], [], []
                for d in range(4):
                    red = speckle_redraw(amp, r8, make_rng(img_id, d),
                                         ksize=5)  # current production recipe
                    rf = red.astype(np.float64)
                    if union.sum() > 0:
                        ratios_a.append(rf[union].mean() / max(a[union].mean(), 1e-9))
                        ratios_i.append((rf[union] ** 2).mean()
                                        / max((a[union] ** 2).mean(), 1e-9))
                    for ann in coco.loadAnns(coco.getAnnIds(imgIds=[img_id])):
                        mi = coco.annToMask(ann).astype(bool)
                        if mi.sum() < 10:
                            continue
                        inst_ratio.append(
                            (int(mi.sum()),
                             rf[mi].mean() / max(a[mi].mean(), 1e-9)))
                    lv1 = local_var(red)
                    var_r.append(float(np.corrcoef(lv0[bg], lv1[bg])[0, 1]))
                    s1 = radial_logspec(red)
                    ok = np.isfinite(sp0) & np.isfinite(s1)
                    spec_r.append(float(np.corrcoef(sp0[ok], s1[ok])[0, 1]))
                fidelity.append(dict(
                    regime=regime, img_id=img_id, enl=r8,
                    amp_ratio=(np.mean(ratios_a), np.std(ratios_a)),
                    int_ratio=(np.mean(ratios_i), np.std(ratios_i)),
                    var_r=(np.mean(var_r), np.std(var_r)),
                    spec_r=(np.mean(spec_r), np.std(spec_r))))
            if (k + 1) % 20 == 0:
                print(f'  [{regime}] {k + 1}/{len(ids)} chips  '
                      f'({time.time() - t0:.0f}s)', flush=True)

    # ---- P2 control: iid-Rayleigh + JPEG null (is the violation rate real
    # texture, or JPG-artifact + quantization inflation?) ----
    sig_ref = float(np.median(sea_sigma['offshore']))
    rho1_ref = float(np.median(lag1['offshore'][1]))
    ctrl_rng = np.random.default_rng(11)
    ctrl = {}
    for tag in ('raw', 'jpg95', 'jpg85', 'jpg75',
                'corr-raw', 'corr-jpg95', 'corr-jpg85', 'corr-jpg75'):
        rej = {1: 0, 2: 0, 4: 0}; tot = 0
        rhos = []
        for _ in range(20):
            if tag.startswith('corr'):
                f = corr_speckle_field(sig_ref, rho1_ref, (800, 800),
                                       ctrl_rng)
            else:
                f = st.rayleigh.rvs(scale=sig_ref, size=(800, 800),
                                    random_state=ctrl_rng)
            f8 = np.clip(np.rint(f), 0, 255).astype(np.uint8)
            if 'jpg' in tag:
                q = int(tag.split('jpg')[1])
                ok_enc, buf = cv2.imencode(
                    '.jpg', f8, [cv2.IMWRITE_JPEG_QUALITY, q])
                assert ok_enc
                f8 = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
            fd = np.clip(f8.astype(np.float64)
                         + ctrl_rng.uniform(-DITHER_HALF, DITHER_HALF,
                                            f8.shape), 1e-3, None)
            Ic = fd * fd
            rhos.append(np.corrcoef(Ic[:, :-1].ravel(),
                                    Ic[:, 1:].ravel())[0, 1])
            for y in range(0, 800 - BLOCK + 1, BLOCK):
                for x in range(0, 800 - BLOCK + 1, BLOCK):
                    for s_ in (1, 2, 4):
                        Ib = np.sort(
                            Ic[y:y + BLOCK:s_, x:x + BLOCK:s_].ravel())
                        D = ks_stat(Ib, cdf_expon(Ib, fit_expon(Ib)))
                        sh = np.sqrt(np.mean(Ib) / 2.0)
                        rej[s_] += int(D > crit_lookup(crit_s[s_], sh))
                    tot += 1
        ctrl[tag] = ({s_: 100 * rej[s_] / tot for s_ in (1, 2, 4)},
                     float(np.mean(rhos)))

    # ======================= PRINT P1 ======================= #
    print('\n[P1] ---- block-wise ENL (32x32, intensity (mean/std)^2, '
          'valid blocks only) ----')
    for r in ('inshore', 'offshore'):
        be = np.concatenate(enl_blocks[r])
        cm = np.asarray(enl_chip_med[r]); r8v = np.asarray(enl_chip_r8[r])
        print(f'[P1] {r:8s}: blocks n={be.size}  ENL median={fmt(np.median(be))}'
              f'  IQR=[{fmt(np.quantile(be, .25))},{fmt(np.quantile(be, .75))}]'
              f'  P90={fmt(np.quantile(be, .9))}  frac(ENL>2)='
              f'{fmt((be > 2).mean())}  frac(ENL<1.3)={fmt((be < 1.3).mean())}')
        print(f'[P1] {r:8s}: per-chip median-block-ENL  mean={fmt(cm.mean())}'
              f' +- {fmt(cm.std())}  median={fmt(np.median(cm))}')
        print(f'[P1] {r:8s}: per-chip R8-estimator ENL  mean={fmt(r8v.mean())}'
              f' +- {fmt(r8v.std())}  median={fmt(np.median(r8v))}'
              f'  range=[{fmt(r8v.min())},{fmt(r8v.max())}]')
    u = st.mannwhitneyu(enl_chip_med['inshore'], enl_chip_med['offshore'])
    print(f'[P1] Mann-Whitney U (chip median ENL, in vs off): '
          f'U={u.statistic:.0f}  p={u.pvalue:.3e}')

    # ======================= PRINT P2 ======================= #
    print('\n[P2] ---- distribution fits (intensity domain == amplitude KS, '
          'exact monotone-invariance) ----')
    pooled_fit = {}
    for c in ('ship', 'clutter', 'sea'):
        if not pool[c]:
            continue
        allpx = np.concatenate(pool[c])
        sub = allpx if allpx.size <= POOL_SUB else \
            np.random.default_rng(1).choice(allpx, POOL_SUB, replace=False)
        Lc = float(np.median(enl_chip_r8['inshore' if c != 'sea' else 'offshore']))
        res = fit_all_ks(sub, L=Lc)
        pooled_fit[c] = (res, sub)
        best = min(res, key=lambda d: res[d][0])
        ds = '  '.join(f'{d}:D={fmt(res[d][0])}' for d in DISTS)
        print(f'[P2] pooled {c:8s} (n={sub.size}, L_fix={fmt(Lc, 2)}): {ds}'
              f'  -> best={best}')
        pr = res['rice'][1]; pk = res['K'][1]; pg = res['nakagami'][1]
        kfac = pr['nu2'] / (2 * pr['sig2'])
        print(f'[P2]   params[{c}]: Rice K-factor={fmt(kfac, 2)} '
              f'({fmt(10 * np.log10(max(kfac, 1e-9)), 1)} dB)  '
              f'Nakagami m={fmt(pg["shape"], 3)}  '
              f'K-dist nu={"inf" if pk["nu"] >= NU_MAX * .99 else fmt(pk["nu"], 3)}')
    print('[P2] per-chip best-fit votes (moment fits, KS argmin; '
          'note Rice>=Rayleigh nests at nu=0, K->no-texture nests Rayleigh@L=1):')
    for c in ('ship', 'clutter', 'sea'):
        if vote_n[c] == 0:
            continue
        v = votes[c]
        vs = '  '.join(f'{d}:{v[d]}/{vote_n[c]}({100 * v[d] / vote_n[c]:.0f}%)'
                       for d in DISTS)
        med = '  '.join(f'{d}:medD={fmt(np.median(Dsum[c][d]))}' for d in DISTS)
        print(f'[P2]   {c:8s} votes: {vs}')
        print(f'[P2]   {c:8s} medKS: {med}')
    print('[P2] CAVEAT: pooled fits mix chips with different backscatter '
          'levels (mixture-of-scales masquerades as texture); per-chip and '
          'per-block numbers below are the honest units.')
    print('[P2c] per-INSTANCE ship fits (core mask >=200 px, n<=1024 subsample):')
    for r in ('inshore', 'offshore'):
        rec = inst[r]
        if rec['n'] == 0:
            continue
        vs = '  '.join(
            f'{d}:{rec["votes"][d]}/{rec["n"]}'
            f'({100 * rec["votes"][d] / rec["n"]:.0f}%)' for d in DISTS)
        med = '  '.join(f'{d}:medD={fmt(np.median(rec["D"][d]))}'
                        for d in DISTS)
        kf = np.asarray(rec['kfac']); sat = np.asarray(rec['sat'])
        print(f'[P2c]  {r:8s} ({rec["n"]} instances): {vs}')
        print(f'[P2c]  {r:8s} {med}')
        print(f'[P2c]  {r:8s} Rice K-factor median={fmt(np.median(kf), 2)} '
              f'IQR=[{fmt(np.quantile(kf, .25), 2)},'
              f'{fmt(np.quantile(kf, .75), 2)}]  frac(K>1)={fmt((kf > 1).mean())}'
              f'  saturation(A>=255) median={fmt(np.median(sat))} '
              f'mean={fmt(sat.mean())}')
    print('[P2] fully-developed-speckle VIOLATION RATE '
          '(Rayleigh rejected at MC-calibrated alpha=0.05; H0-expected ~5%; '
          'stride-2/4 decimation breaks spatial correlation -> tests the '
          'MARGINAL law only):')
    for r in ('inshore', 'offshore'):
        cj, ctot = chip_tests[r]
        srs = '  '.join(
            f'stride{s_}:{100 * blk_tests[r][s_][0] / max(blk_tests[r][s_][1], 1):.1f}%'
            for s_ in (1, 2, 4))
        lgs = '  '.join(f'lag{lg}={np.median(lag1[r][lg]):.3f}'
                        for lg in (1, 2, 4))
        print(f'[P2]   {r:8s}: bg blocks (n={blk_tests[r][1][1]}) {srs}   '
              f'bg chips {cj}/{ctot}={100 * cj / max(ctot, 1):.0f}%')
        print(f'[P2]   {r:8s}: bg rho(I) {lgs}   bg sigma_hat '
              f'median={fmt(np.median(sea_sigma[r]), 1)} DN')
    print(f'[P2] CONTROLS (20 fields of 800^2 each, sigma={sig_ref:.1f}, '
          f'corr fields matched to offshore rho1={rho1_ref:.3f}; same '
          f'quantize+dither+test pipeline):')
    for tag, (rr, rho) in ctrl.items():
        srs = '  '.join(f'stride{s_}:{rr[s_]:.1f}%' for s_ in (1, 2, 4))
        print(f'[P2]   {tag:10s}: {srs}   lag-1 rho(I)={rho:.4f}')
    print('[P2]   -> real stride-4 rejection above the matching control '
          'stride-4 rate = genuine MARGINAL non-Rayleigh (texture); '
          'stride-1 excess over that = correlated speckle + JPG artifacts.')
    if pool_rej:
        allr = np.concatenate(pool_rej)
        subr = allr if allr.size <= POOL_SUB else \
            np.random.default_rng(2).choice(allr, POOL_SUB, replace=False)
        Lc = float(np.median(enl_chip_r8['inshore']))
        resr = fit_all_ks(subr, L=Lc)
        ds = '  '.join(f'{d}:D={fmt(resr[d][0])}' for d in DISTS)
        print(f'[P2] suppl. inshore REJECTED-block pixels (n={subr.size}; '
              f'descriptive only - selection by rejection inflates non-Rayleigh): '
              f'{ds} -> best={min(resr, key=lambda d: resr[d][0])}  '
              f'K nu={fmt(resr["K"][1]["nu"], 3)}')

    # ======================= PRINT P3 ======================= #
    print('\n[P3] ---- top-q brightest-pixel protection attribution '
          '(per-chip quantile, exact speckle_redraw rule) ----')
    print('[P3] q       regime    prot_px   in_ship  in_ship+5px  background'
          '   ship_coverage(recall)')
    p3 = {}
    for r in ('inshore', 'offshore'):
        for q in QS:
            n, s, m5, cov = prot[r][q]
            p3[(r, q)] = (s / n, m5 / n, 1 - m5 / n, cov / ship_px_tot[r])
            print(f'[P3] {1 - q:.3f}   {r:8s}  {int(n):8d}   '
                  f'{100 * s / n:6.1f}%  {100 * m5 / n:9.1f}%  '
                  f'{100 * (1 - m5 / n):8.1f}%   {100 * cov / ship_px_tot[r]:6.1f}%')
    print('[P3] absolute-DN alternative thresholds (same chips):')
    for r in ('inshore', 'offshore'):
        for t in TS:
            n, s, m5, cov = prot_abs[r][t]
            if n == 0:
                print(f'[P3] A>={t:3d}  {r:8s}  0 protected px')
                continue
            print(f'[P3] A>={t:3d}  {r:8s}  {int(n):8d}   '
                  f'{100 * s / n:6.1f}%  {100 * m5 / n:9.1f}%  '
                  f'{100 * (1 - m5 / n):8.1f}%   {100 * cov / ship_px_tot[r]:6.1f}%')

    # ======================= PRINT P4 ======================= #
    print('\n[P4] ---- re-draw fidelity, production recipe '
          '(speckle_redraw defaults, K=4 draws) ----')
    print('[P4] chip        regime   ENL    A_ship_ratio      I_ship_ratio'
          '      varmap_r(bg)      logspec_r')
    for f in fidelity:
        print(f'[P4] {f["img_id"]:<10d}  {f["regime"]:8s} {f["enl"]:.2f}   '
              f'{f["amp_ratio"][0]:.4f}+-{f["amp_ratio"][1]:.4f}   '
              f'{f["int_ratio"][0]:.4f}+-{f["int_ratio"][1]:.4f}   '
              f'{f["var_r"][0]:.4f}+-{f["var_r"][1]:.4f}   '
              f'{f["spec_r"][0]:.4f}+-{f["spec_r"][1]:.4f}')
    for r in ('inshore', 'offshore'):
        fs = [f for f in fidelity if f['regime'] == r]
        ar = np.array([f['amp_ratio'][0] for f in fs])
        ir = np.array([f['int_ratio'][0] for f in fs])
        vr = np.array([f['var_r'][0] for f in fs])
        sr = np.array([f['spec_r'][0] for f in fs])
        print(f'[P4] {r:8s} MEAN: A_ratio={ar.mean():.4f}+-{ar.std():.4f}  '
              f'I_ratio={ir.mean():.4f}+-{ir.std():.4f}  '
              f'var_r={vr.mean():.4f}+-{vr.std():.4f}  '
              f'spec_r={sr.mean():.4f}+-{sr.std():.4f}')
    ir_arr = np.array(inst_ratio)                   # (n, 2): size, ratio
    print('[P4] per-instance ship A-mean retention by size (10 chips x 4 '
          'draws; the 5x5 median erases <=12-px scatterers unless protected):')
    for lo, hi, lab in ((10, 100, '<100 px'), (100, 500, '100-500 px'),
                        (500, 10 ** 9, '>500 px')):
        m = (ir_arr[:, 0] >= lo) & (ir_arr[:, 0] < hi)
        if m.sum() == 0:
            print(f'[P4]   {lab:11s}: no instances')
            continue
        rr = ir_arr[m, 1]
        print(f'[P4]   {lab:11s}: n={m.sum():3d}  median={np.median(rr):.4f}  '
              f'mean={rr.mean():.4f}  P10={np.quantile(rr, .1):.4f}  '
              f'min={rr.min():.4f}')

    # ======================= P5: figure ======================= #
    plt.rcParams.update({'font.size': 8, 'pdf.fonttype': 42,
                         'axes.linewidth': 0.7, 'font.family': 'DejaVu Sans'})
    fig = plt.figure(figsize=(10.2, 7.6))
    gs = gridspec.GridSpec(2, 2, hspace=0.34, wspace=0.24,
                           left=0.06, right=0.985, top=0.95, bottom=0.07)

    # (a) block ENL hists
    ax = fig.add_subplot(gs[0, 0])
    bi = np.concatenate(enl_blocks['inshore'])
    bo = np.concatenate(enl_blocks['offshore'])
    bins = np.linspace(0, 6, 61)
    ax.hist(np.clip(bo, 0, 6), bins=bins, density=True, alpha=0.55,
            color='#1f77b4', label=f'offshore (med {np.median(bo):.2f})')
    ax.hist(np.clip(bi, 0, 6), bins=bins, density=True, alpha=0.55,
            color='#d62728', label=f'inshore (med {np.median(bi):.2f})')
    ax.axvline(np.median(bo), color='#1f77b4', ls='--', lw=1)
    ax.axvline(np.median(bi), color='#d62728', ls='--', lw=1)
    ax.set_xlabel(r'block ENL $=(\mu_I/\sigma_I)^2$ (32$\times$32)')
    ax.set_ylabel('density')
    ax.set_title('(a) Block-wise ENL, inshore vs offshore', fontsize=9)
    ax.legend(frameon=False)

    # (b) three-class amplitude densities + fits
    sub_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, 1],
                                              wspace=0.08)
    ttl = {'sea': '(b) sea (offshore bg)',
           'clutter': 'inshore clutter (pooled)',
           'ship': 'ship interior (pooled)'}
    for j, c in enumerate(('sea', 'clutter', 'ship')):
        axb = fig.add_subplot(sub_gs[0, j])
        res, sub = pooled_fit[c]
        A = np.sqrt(sub)
        hi = np.quantile(A, 0.999)
        bins = np.linspace(0, hi, 60)
        axb.hist(A, bins=bins, density=True, color='0.75', label='data')
        ag = np.linspace(1e-2, hi, 400)
        Ig = ag * ag
        best = min(res, key=lambda d: res[d][0])
        curves = {'rayleigh': (pdf_expon, 'rayleigh', '#2ca02c', '--'),
                  best: (None, best, '#d62728', '-')}
        # rayleigh reference always; best-fit on top
        fA_r = 2 * ag * pdf_expon(Ig, res['rayleigh'][1])
        axb.plot(ag, fA_r, '--', color='#2ca02c', lw=1.1,
                 label=f'Rayleigh D={res["rayleigh"][0]:.3f}')
        pdfs = {'rayleigh': pdf_expon, 'rice': pdf_rice,
                'nakagami': pdf_gamma, 'K': pdf_k}
        if best != 'rayleigh':
            fA_b = 2 * ag * pdfs[best](Ig, res[best][1])
            axb.plot(ag, fA_b, '-', color='#d62728', lw=1.2,
                     label=f'{best} D={res[best][0]:.3f}')
        axb.set_yscale('log')
        axb.set_ylim(bottom=max(axb.get_ylim()[0], 1e-6))
        axb.set_title(ttl[c], fontsize=7.5)
        axb.set_xlabel('amplitude (DN)', fontsize=7)
        if j == 0:
            axb.set_ylabel('density (log)')
        else:
            axb.set_yticklabels([])
        axb.legend(frameon=False, fontsize=6)

    # (c) protection attribution stacked bars
    ax = fig.add_subplot(gs[1, 0])
    cats = [('inshore', q) for q in (0.999, 0.998, 0.995, 0.99)] + \
           [('offshore', q) for q in (0.999, 0.998, 0.995, 0.99)]
    xs = np.arange(len(cats)); xs[4:] += 1
    ship_f = np.array([p3[c][0] for c in cats])
    marg_f = np.array([p3[c][1] - p3[c][0] for c in cats])
    bg_f = np.array([p3[c][2] for c in cats])
    ax.bar(xs, 100 * ship_f, color='#d62728', label='inside ship GT')
    ax.bar(xs, 100 * marg_f, bottom=100 * ship_f, color='#ff9896',
           label='ship +5 px margin')
    ax.bar(xs, 100 * bg_f, bottom=100 * (ship_f + marg_f), color='0.8',
           label='background clutter')
    ax.set_xticks(xs)
    ax.set_xticklabels([f'{(1 - q) * 100:.1f}%' for _, q in cats], fontsize=7)
    ax.text(1.5, -14, 'inshore', ha='center', fontsize=8)
    ax.text(7.5, -14, 'offshore', ha='center', fontsize=8)
    ax.set_ylabel('share of protected pixels (%)')
    ax.set_xlabel('protected top fraction q')
    ax.set_title('(c) Bright-pixel protection: who gets protected?',
                 fontsize=9, pad=22)
    ax.axvline(4.0, color='k', lw=0.6, ls=':')
    ax.legend(frameon=False, fontsize=6.5, loc='lower center', ncol=3,
              bbox_to_anchor=(0.5, 1.0))

    # (d) before/after redraw example
    sub_gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 1],
                                               wspace=0.04)
    fex = [f for f in fidelity if f['regime'] == 'inshore'][0]
    amp = load_chip(coco_in, fex['img_id'])
    union, _, _ = ship_masks(coco_in, fex['img_id'], amp.shape)
    red = speckle_redraw(amp, fex['enl'], make_rng(fex['img_id'], 0),
                         ksize=5)  # current production recipe
    ys, xs2 = np.where(union)
    cy, cx = (int(np.median(ys)), int(np.median(xs2))) if len(ys) else (400, 400)
    half = 130
    y0, x0 = np.clip(cy - half, 0, amp.shape[0] - 2 * half), \
        np.clip(cx - half, 0, amp.shape[1] - 2 * half)
    for j, (im, t) in enumerate(
            ((amp, 'original chip (crop)'),
             (red, r'redraw $I^{\prime}=\hat R\,g$, $g\sim\Gamma$(ENL)'))):
        axd = fig.add_subplot(sub_gs2[0, j])
        axd.imshow(im[y0:y0 + 2 * half, x0:x0 + 2 * half], cmap='gray',
                   vmin=0, vmax=255)
        axd.set_title(t, fontsize=7.5)
        axd.set_xticks([]); axd.set_yticks([])
    ar = np.array([f['amp_ratio'][0] for f in fidelity])
    vr = np.array([f['var_r'][0] for f in fidelity])
    sr = np.array([f['spec_r'][0] for f in fidelity])
    fig.text(0.725, 0.075,
             f'ship-mean retention {ar.mean():.3f}$\\pm${ar.std():.3f} | '
             f'bg var-map r {vr.mean():.3f}$\\pm${vr.std():.3f} | '
             f'log-spectrum r {sr.mean():.3f}$\\pm${sr.std():.3f} '
             f'(10 chips, K=4)', ha='center', fontsize=6.5)
    fig.text(0.75, 0.475, '(d) Re-draw fidelity (ENL='
             f'{fex["enl"]:.2f}, id {fex["img_id"]})', ha='center', fontsize=9)

    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    fig.savefig(FIG_OUT)
    fig.savefig(FIG_OUT.replace('.pdf', '.png'), dpi=180)
    print(f'\n[P5] figure -> {FIG_OUT} (+.png preview)')

    # ======================= VERDICTS ======================= #
    print('\n[VERDICT] ---- data-driven conclusions for R8 ----')
    rj = {(r, s_): 100 * blk_tests[r][s_][0] / max(blk_tests[r][s_][1], 1)
          for r in ('inshore', 'offshore') for s_ in (1, 2, 4)}
    print(f'[VERDICT] V1 violation, stride-1 (marginal+correlation+JPG): '
          f'offshore {rj[("offshore", 1)]:.1f}% / inshore '
          f'{rj[("inshore", 1)]:.1f}%  vs corr-matched controls '
          f'{ctrl["corr-raw"][0][1]:.1f}% (raw) '
          f'{ctrl["corr-jpg95"][0][1]:.1f}% (q95) '
          f'{ctrl["corr-jpg85"][0][1]:.1f}% (q85) '
          f'{ctrl["corr-jpg75"][0][1]:.1f}% (q75=sniffed real quality)')
    print(f'[VERDICT] V1 violation, stride-4 (marginal only): offshore '
          f'{rj[("offshore", 4)]:.1f}% / inshore {rj[("inshore", 4)]:.1f}%  '
          f'vs controls raw {ctrl["raw"][0][4]:.1f}%  corr-raw '
          f'{ctrl["corr-raw"][0][4]:.1f}%  corr-q95 '
          f'{ctrl["corr-jpg95"][0][4]:.1f}%  corr-q85 '
          f'{ctrl["corr-jpg85"][0][4]:.1f}%  corr-q75 '
          f'{ctrl["corr-jpg75"][0][4]:.1f}%')
    kf_i = np.asarray(inst['inshore']['kfac'])
    kf_o = np.asarray(inst['offshore']['kfac'])
    if kf_i.size and kf_o.size:
        print(f'[VERDICT] V1b per-instance Rice K-factor>1 (dominant '
              f'scatterer detectable): inshore {100 * (kf_i > 1).mean():.0f}% '
              f'of instances, offshore {100 * (kf_o > 1).mean():.0f}%')
    s5_i, m5_i = p3[("inshore", 0.995)][0], p3[("inshore", 0.995)][1]
    s5_o, m5_o = p3[("offshore", 0.995)][0], p3[("offshore", 0.995)][1]
    print(f'[VERDICT] V2 protection precision at q=0.5%: inshore '
          f'{100 * s5_i:.1f}% in-ship ({100 * m5_i:.1f}% incl +5px); offshore '
          f'{100 * s5_o:.1f}% ({100 * m5_o:.1f}% incl +5px)')
    print(f'[VERDICT]    -> if inshore in-ship share is low, the guard mostly '
          f'freezes bright STATIONARY clutter (harmless for marginalization '
          f'variance but useless for ships); consider mask-free CFAR-style '
          f'local-contrast protection or a smaller q.')
    print(f'[VERDICT] V3 ship-mean retention (production recipe): see [P4]; '
          f'values <1 mean the 5x5 median erodes ship RCS outside the '
          f'protected top-q set.')
    print(f'[TIME] total {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
