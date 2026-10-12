# -*- coding: utf-8 -*-
"""R8: SPECKLE-MARGINALIZATION TTA (Pass B of the offline speckle fusion).

PHYSICS: a single-look-processed SAR amplitude chip is ONE draw of the
multiplicative speckle process I = R * n, n ~ Gamma(L, 1/L) in intensity
(L = equivalent number of looks, ENL). Standard inference CONDITIONS on that
draw; R8 MARGINALIZES it: per image we (1) estimate ENL from homogeneous
blocks, (2) form a crude radar-cross-section estimate R_hat by 5x5 median
speckle filtering of intensity, (3) draw K new speckle fields
I' = R_hat * g, g a spatially CORRELATED unit-mean Gamma(ENL, 1/ENL) field,
A' = sqrt(I'), and run the EXISTING soup checkpoint SINGLE-VIEW on every
re-realization. Geometry is untouched (no resize/flip) so masks map 1:1 to
the original frame - no back-mapping. The K draws join the base wide-TTA
vote-consensus predictions OFFLINE via carrier/ensemble_fuse.py (Pass C,
2 sources, consensus band), exactly the SAHI two-pass pattern (mmdet
TestTimeAug cannot synthesize stochastic views).

THEORY-MANDATED FIXES (docs/THEORY_MARGINALIZATION.md section 7, measured on
real chips by carrier/speckle_stats.py):
  R1 protection: CFAR local-contrast + absolute-DN mask (cfar_protect_mask)
     replaces the global top-0.5% quantile rule, whose inshore protection
     budget went 75.2% to bright land/clutter (only 22.8% inside ships) and
     which no global threshold could fix (precision capped ~26%). Defaults
     re-measured on 25+25 GT chips (2026-07-31 sweep): guard 7 / ring 15 /
     k=5 / abs_dn=200 -> ship-hit precision 29.0% inshore (old 27.8%) at 2x
     the ship coverage (29.9% vs 15.8%), 70.7% offshore (old 31.6%).
  R2 bias: the Gamma-median correction of R_hat is applied ONLY to
     unprotected pixels. Protected regions are scatterer-dominated, not
     fully developed speckle; correcting them lifted ship means +4-6% (P4).
  R3 correlation: g is a spatially correlated Gamma field (Gaussian-copula,
     corr_gamma_field) approximating the measured lag-1 intensity
     correlation (real chips rho_1 ~ 0.42-0.65; the old iid g had rho_1 ~ 0,
     making redraws separable from real chips on the correlation axis).
  R4 ENL: median of (mean/std)^2 over ALL valid blocks. The old
     lowest-CV-quartile pick selected the most JPG-smoothed blocks and
     overestimated ENL (chip median 1.27 vs block median ~1.0).

ESTIMATOR (pure, unit-tested in carrier/tests/cpu_test_speckle.py):
  ENL = (mean/std)^2 of intensity I = A^2 over homogeneous blocks: 32x32 grid,
  keep VALID blocks (not quantization-degenerate: block amplitude mean >= 2,
  intensity std > 0, <=10% zero pixels, <=0.5% saturated pixels), take the
  median of (mean/std)^2 over ALL valid blocks (R4), clamp to [1, 16].
  Falls back to ENL=1 (single-look) if no block is valid.

R_HAT BIAS: the 5x5 median of Gamma-speckled intensity estimates
  R * median(Gamma(L,1/L)) - e.g. 0.693R at L=1, 0.918R at L=4 - so R_hat is
  divided by the theoretical Gamma median at the estimated ENL (scipy ppf),
  on UNPROTECTED pixels only (R2; protected pixels are passed through
  verbatim anyway, so the uncorrected R_hat there is also what a future
  non-passthrough variant should use). medianBlur runs on uint8 AMPLITUDE
  then squares: the median is order-statistic so median(A)^2 == median(A^2)
  exactly (odd window).

SCATTERER PROTECTION (R1): a pixel is protected - passed through UNCHANGED
  (A' = A) - iff it stands out of its LOCAL clutter, A > mean + k*std of a
  guard-7x7 / background-15x15 annulus (CFAR, O(N) via cv2.boxFilter running
  sums), OR exceeds an absolute DN threshold A >= 200 (measured 94.4%
  offshore in-ship precision in the P3 study). MEASURED DEVIATION from the
  theory doc's "k~3": on real chips the correlated-speckle + JPEG-q75 tail
  makes k=3 fire on 0.8-1.1% of ALL pixels (~1% ship precision), and the
  CFAR arm adds ~0 ship coverage beyond the abs-DN arm at these window
  sizes (ships are EXTENDED targets, so the 15x15 background ring is
  ship-contaminated and self-masks) - so the default is k=5, where the arm
  is nearly silent on clutter and kept as cheap insurance for locally
  contrasting scatterers in dark chips that never reach DN 200. Cost:
  protected pixels are identical across draws (no marginalization there);
  inshore the abs-DN arm freezes bright stationary land - harmless for
  consensus fusion (all draws agree there), documented and accepted.

CORRELATED SPECKLE (R3): corr_gamma_field smooths a white Gaussian field
  with a Gaussian kernel (default sigma=0.76, inverted from the measured
  offshore rho_1(I)=0.421 via rho_I=|rho_z|^2, rho_z(l)=exp(-l^2/4s^2)),
  re-standardizes it empirically, and maps it through Phi and the
  Gamma(ENL, 1/ENL) quantile function (copula construction): the marginal
  is EXACTLY the target Gamma law (mean 1, var 1/ENL); the lag-1 Pearson
  correlation of the Gamma field sits slightly below the Gaussian field's
  exp(-1/(4 sigma^2)) = 0.649 because the monotone quantile map shrinks
  Pearson correlation (achieved ~0.5 at ENL~1, printed by the CPU tests -
  inside the real-chip range 0.42-0.65). corr_sigma=0 gives an iid field.

DETERMINISM: draw k of image img_id uses np.random.default_rng(
  SeedSequence([img_id, k])) - byte-identical re-runs, no global RNG state.

ENL SIDECAR (P2 probe): run_split writes the per-image estimated ENL to
  runs/speckle/enl_<split>.json next to the output pkl, so post-hoc
  ENL-bucket analysis of the fusion gains needs no rerun.

8-BIT CAVEAT: chips are 8-bit JPGs, so amplitude is quantized (and JPG
  block-coded); the redraw is written back to uint8. Quantization noise is
  negligible vs speckle at L<=16 but the ENL estimate on very dark chips is
  a mild overestimate (JPG smoothing) - clamp keeps it sane.

SCORE GATE + LAMBDA: keep instances with score >= 0.05; inference runs the
  plain config at infer_iou_lambda=0.0, matching the soup x wide x
  vote-consensus baseline recipe (same score scale as the base pkl).

Usage (GPU pass; launched ONLY by run_speckle_queue.sh):
  python carrier/speckle_infer.py infer \
      --config carrier/configs/c0_maskiou_v2_m2f_hrsid.py \
      --ckpt runs/c0_maskiou_v2/soup_s023/iter_100872.pth \
      --ann data/hrsid/annotations/test_inshore.json \
      --out runs/speckle/spk_inshore.pkl --k 4
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
from carrier.sahi_infer import make_sample  # noqa: E402  (mmdet-free import)

from pycocotools import mask as mask_util  # noqa: E402
from pycocotools.coco import COCO          # noqa: E402


# --------------------------------------------------------------------------- #
# pure functions (unit-tested on CPU, no model involved)
# --------------------------------------------------------------------------- #
def gamma_median(enl: float) -> float:
    """Median of Gamma(shape=enl, scale=1/enl) (mean 1). 0.693@L=1, 0.918@L=4."""
    from scipy.stats import gamma
    return float(gamma.ppf(0.5, float(enl), scale=1.0 / float(enl)))


def estimate_enl(amp: np.ndarray, block: int = 32,
                 clamp=(1.0, 16.0)) -> float:
    """Estimate the equivalent number of looks from an amplitude chip.

    Args:
        amp: (H, W) uint8 (or float) SAR amplitude.
        block: homogeneous-block size (grid, non-overlapping).
        clamp: (lo, hi) physical range for the returned ENL.

    Returns:
        Scalar ENL = median of intensity (mean/std)^2 over ALL valid blocks
        (R4: the former lowest-CV-quartile selection preferred the most
        JPG-smoothed blocks and overestimated ENL); clamp[0] fallback.
    """
    a = np.asarray(amp, dtype=np.float64)
    assert a.ndim == 2, f'amp must be (H, W), got {a.shape}'
    I = a * a
    h, w = I.shape
    enls = []
    for y in range(0, h - block + 1, block):
        for x in range(0, w - block + 1, block):
            blk_a = a[y:y + block, x:x + block]
            blk_i = I[y:y + block, x:x + block]
            m, s = blk_i.mean(), blk_i.std(ddof=1)
            # guards against quantization-degenerate blocks:
            #   dark clipping (JPG zeros), saturation (255 plateaus -> CV ~ 0
            #   would fake a huge ENL), and all-constant blocks (std 0).
            if (blk_a.mean() < 2.0 or s <= 1e-9
                    or (blk_a == 0).mean() > 0.10
                    or (blk_a >= 255).mean() > 0.005):
                continue
            enls.append((m / s) ** 2)
    if not enls:
        return float(clamp[0])
    enl = float(np.median(np.asarray(enls)))
    return float(np.clip(enl, clamp[0], clamp[1]))


def make_rng(img_id: int, draw_idx: int) -> np.random.Generator:
    """Deterministic per-(image, draw) RNG: same seed -> byte-identical draw."""
    return np.random.default_rng(
        np.random.SeedSequence([int(img_id), int(draw_idx)]))


def cfar_protect_mask(amp: np.ndarray, guard: int = 7, ring: int = 15,
                      k: float = 5.0, abs_dn: float = 200.0) -> np.ndarray:
    """R1: scatterer-protection mask = local-contrast CFAR hit OR bright DN.

    A pixel is protected iff
      A > mean(annulus) + k * std(annulus)  - CFAR on the annulus between a
        guard x guard window (target + immediate sidelobes, excluded) and a
        ring x ring background window; fires on locally contrasting
        scatterers at ANY clutter level (the property no global threshold
        has), OR
      A >= abs_dn  - absolute-DN arm (A >= 200 hit ships with 94.4%
        precision offshore in the speckle_stats.py P3 study; inshore it
        freezes bright stationary land, which is harmless for consensus
        fusion because all draws agree there).

    DEFAULT k=5, NOT the naive Rayleigh k~3 (0.5% FA): measured on 25+25 GT
    chips, the heavier correlated-speckle + JPEG-q75 tail makes k=3 fire on
    0.8-1.1% of all pixels with ~1% ship precision, and at guard 7 / ring 15
    the CFAR arm adds ~0 ship coverage beyond abs-DN (extended-target
    self-masking: ships are wider than the guard, so the background ring is
    ship-contaminated). Combined k=5 + abs_dn=200 vs the old top-0.5% rule:
    precision 29.0% vs 27.8% inshore at 2x coverage (29.9% vs 15.8%), 70.7%
    vs 31.6% offshore (2026-07-31 sweep, seed 20260731).

    O(N) via cv2.boxFilter running sums: annulus mean/std come from the
    difference of the ring- and guard-window sums (exact, no approximation).
    Border pixels use the default reflected padding - a slight stat bias in
    the outer ring//2 rows/cols, accepted.

    Args:
        amp: (H, W) uint8 or float amplitude.
        guard: odd guard-window edge (excluded from the background ring).
        ring: odd background-window edge (> guard).
        k: CFAR multiplier on the annulus std.
        abs_dn: absolute amplitude threshold in DN (<= 0 disables that arm).

    Returns:
        (H, W) bool protection mask.
    """
    a = np.asarray(amp, dtype=np.float64)
    assert a.ndim == 2, f'amp must be (H, W), got {a.shape}'
    assert guard % 2 == 1 and ring % 2 == 1 and ring > guard, (guard, ring)
    n_ring = float(ring * ring - guard * guard)
    s1g = cv2.boxFilter(a, -1, (guard, guard), normalize=False)
    s1r = cv2.boxFilter(a, -1, (ring, ring), normalize=False)
    s2g = cv2.boxFilter(a * a, -1, (guard, guard), normalize=False)
    s2r = cv2.boxFilter(a * a, -1, (ring, ring), normalize=False)
    mu = (s1r - s1g) / n_ring
    var = np.maximum((s2r - s2g) / n_ring - mu * mu, 0.0)
    protect = a > mu + float(k) * np.sqrt(var)
    if abs_dn > 0:
        protect |= a >= abs_dn
    return protect


def corr_gamma_field(enl: float, shape, rng: np.random.Generator,
                     corr_sigma: float = 0.76) -> np.ndarray:
    """R3: spatially correlated unit-mean Gamma(ENL, 1/ENL) speckle field.

    Gaussian-copula construction: smooth a white N(0,1) field with a
    Gaussian kernel (std corr_sigma), re-standardize empirically, then map
    through the standard normal CDF and the Gamma(ENL, 1/ENL) quantile
    function. The marginal is EXACTLY the target Gamma law (mean 1,
    var 1/ENL - no variance compensation needed, unlike smoothing the Gamma
    draw itself, which narrows and gaussianizes the marginal and at ENL~1
    would clip mass at 0 after rescaling). Approximation honestly stated:
    the achieved lag-1 Pearson correlation is that of the copula, slightly
    BELOW the Gaussian field's exp(-1/(4*corr_sigma^2)) (= 0.649 at
    sigma=0.76) because a monotone marginal transform shrinks Pearson
    correlation - measured ~0.5 at ENL~1 (see cpu_test_speckle.py test f),
    inside the real-chip range rho_1 = 0.42-0.65 from speckle_stats.py.

    Args:
        enl: Gamma shape (equivalent number of looks).
        shape: (H, W) of the field.
        rng: generator from make_rng (determinism preserved).
        corr_sigma: Gaussian smoothing std in px; 0 -> iid field.

    Returns:
        (H, W) float32 field, marginal Gamma(enl, 1/enl), mean 1.
    """
    from scipy.ndimage import gaussian_filter
    from scipy.special import gammaincinv, ndtr
    z = rng.standard_normal(size=shape)
    if corr_sigma > 0:
        z = gaussian_filter(z, sigma=float(corr_sigma))
        z = (z - z.mean()) / max(float(z.std()), 1e-12)
    u = np.clip(ndtr(z), 1e-12, 1.0 - 1e-12)
    return (gammaincinv(float(enl), u) / float(enl)).astype(np.float32)


def speckle_redraw(amp: np.ndarray, enl: float, rng: np.random.Generator,
                   ksize: int = 5, protect_mask: np.ndarray | None = None,
                   corr_sigma: float = 0.76, cfar_guard: int = 7,
                   cfar_ring: int = 15, cfar_k: float = 5.0,
                   abs_dn: float = 200.0) -> np.ndarray:
    """One new speckle realization of an amplitude chip (geometry untouched).

    I' = R_hat * g with g a spatially correlated unit-mean Gamma(ENL, 1/ENL)
    field (R3, corr_gamma_field); A' = sqrt(I') clipped to [0, 255].
    R_hat = median_{ksize x ksize}(I), divided by gamma_median(ENL) on
    UNPROTECTED pixels only (R2: protected regions are scatterer-dominated,
    not fully developed speckle - bias-correcting them over-brightens).
    Protected pixels (R1: CFAR local contrast OR A >= abs_dn, see
    cfar_protect_mask) are passed through unchanged (ship point scatterers
    keep their true RCS in every draw).

    Args:
        amp: (H, W) uint8 amplitude.
        enl: equivalent number of looks (from estimate_enl).
        rng: per-(image, draw) generator from make_rng.
        ksize: odd median window (5 = fast, edge-preserving enough).
        protect_mask: (H, W) bool; computed via cfar_protect_mask(amp,
            cfar_guard, cfar_ring, cfar_k, abs_dn) if None. Pass it in when
            drawing K times from the same chip (it only depends on amp).
        corr_sigma: speckle correlation kernel std (R3); 0 -> iid.
        cfar_guard/cfar_ring/cfar_k/abs_dn: cfar_protect_mask parameters,
            used only when protect_mask is None.

    Returns:
        (H, W) uint8 re-realized amplitude.
    """
    a8 = np.asarray(amp, dtype=np.uint8)
    assert a8.ndim == 2, f'amp must be (H, W), got {a8.shape}'
    a = a8.astype(np.float32)
    if ksize <= 0:
        # MILD mode (rescue probe): perturb the OBSERVED image directly,
        # I' = I * g at a caller-chosen high L (pass it via `enl`), skipping
        # R_hat estimation entirely. Trades a small ENL reduction for zero
        # despeckle bias -- isolates "R_hat estimator broken" from
        # "marginalization concept broken" (THEORY_MARGINALIZATION Eq. 2.5:
        # measure-mismatch bias is second order).
        g = corr_gamma_field(enl, a.shape, rng, corr_sigma=corr_sigma)
        a_new = np.sqrt((a * a) * g)
        return np.clip(np.rint(a_new), 0, 255).astype(np.uint8)
    if protect_mask is None:
        protect_mask = cfar_protect_mask(a8, guard=cfar_guard, ring=cfar_ring,
                                         k=cfar_k, abs_dn=abs_dn)
    # median is an order statistic: medianBlur(A)^2 == median(A^2) exactly.
    med_a = cv2.medianBlur(a8, ksize).astype(np.float32)
    r_hat = med_a * med_a
    # R2: Gamma-median bias correction only where speckle is fully developed.
    r_hat = np.where(protect_mask, r_hat,
                     r_hat / np.float32(gamma_median(enl)))
    g = corr_gamma_field(enl, a.shape, rng, corr_sigma=corr_sigma)
    a_new = np.sqrt(r_hat * g)
    a_new = np.where(protect_mask, a, a_new)
    return np.clip(np.rint(a_new), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# GPU pass (mmdet imported lazily so the CPU unit test never touches it)
# --------------------------------------------------------------------------- #
def enl_sidecar_path(out_pkl: str) -> str:
    """runs/speckle/spk_<split>.pkl -> runs/speckle/enl_<split>.json."""
    d, base = os.path.split(out_pkl)
    if base.startswith('spk_') and base.endswith('.pkl'):
        return os.path.join(d, 'enl_' + base[4:-4] + '.json')
    return os.path.splitext(out_pkl)[0] + '_enl.json'


def run_split(config, ckpt, ann, out_pkl, img_dir=None, k_draws=4,
              score_thr=0.05, iou_lambda=0.0, device='cuda:0', ksize=5,
              cfar_guard=7, cfar_ring=15, cfar_k=5.0, abs_dn=200.0,
              corr_sigma=0.76, mild_L=16.0):
    from mmcv.transforms import Compose
    from mmdet.apis import inference_detector, init_detector
    from mmdet.utils import get_test_pipeline_cfg

    model = init_detector(
        config, ckpt, device=device,
        cfg_options={'model.panoptic_fusion_head.infer_iou_lambda':
                     float(iou_lambda)})
    if img_dir is None:
        img_dir = model.cfg.test_dataloader.dataset.data_prefix['img']

    # ndarray input + no GT at inference: swap the loader, drop LoadAnnotations
    pipe = get_test_pipeline_cfg(model.cfg.copy())
    pipe[0].type = 'mmdet.LoadImageFromNDArray'
    pipe = [p for p in pipe if 'LoadAnnotations' not in p['type']]
    test_pipeline = Compose(pipe)

    coco = COCO(ann)
    img_ids = sorted(coco.getImgIds())
    out, t0 = [], time.time()
    n_views_total = n_kept = 0
    enls, enl_by_img = [], {}
    for k, img_id in enumerate(img_ids):
        info = coco.loadImgs([img_id])[0]
        path = os.path.join(img_dir, info['file_name'])
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        assert img is not None, f'cannot read {path}'
        ori_h, ori_w = img.shape[:2]
        gray = img[:, :, 0]                    # HRSID JPGs: replicated gray
        if ksize <= 0:
            # MILD mode: enl carries the caller-chosen draw L; no protection.
            enl = float(mild_L)
            pmask = None
        else:
            enl = estimate_enl(gray)
            # R1 protection depends only on the chip: compute once per chip
            pmask = cfar_protect_mask(gray, guard=cfar_guard, ring=cfar_ring,
                                      k=cfar_k, abs_dn=abs_dn)
        enls.append(enl)
        enl_by_img[str(int(img_id))] = float(enl)     # P2 sidecar

        views = []
        for d in range(k_draws):
            red = speckle_redraw(gray, enl, make_rng(img_id, d),
                                 ksize=ksize, protect_mask=pmask,
                                 corr_sigma=corr_sigma)
            views.append(np.repeat(red[:, :, None], 3, axis=2))
        results = inference_detector(model, views, test_pipeline=test_pipeline)
        n_views_total += len(views)

        rles, scores, labels = [], [], []
        for res in results:
            inst = res.pred_instances
            keep = inst.scores >= score_thr
            masks = inst.masks[keep].cpu().numpy().astype(np.uint8)
            scs = inst.scores[keep].cpu().numpy()
            lbs = inst.labels[keep].cpu().numpy()
            for m, s, lab in zip(masks, scs, lbs):
                if not m.any():
                    continue
                assert m.shape == (ori_h, ori_w), (
                    f'mask {m.shape} != ori {(ori_h, ori_w)}')
                rles.append(mask_util.encode(np.asfortranarray(m)))
                scores.append(float(s))
                labels.append(int(lab))
        n_kept += len(rles)
        out.append(make_sample(img_id, (ori_h, ori_w), rles, scores, labels,
                               img_path=path))
        if (k + 1) % 50 == 0 or k + 1 == len(img_ids):
            dt = time.time() - t0
            e = np.asarray(enls)
            print(f'[speckle] {k + 1}/{len(img_ids)} imgs  '
                  f'{n_views_total} views  kept={n_kept}  '
                  f'ENL med={np.median(e):.2f} '
                  f'[{e.min():.2f},{e.max():.2f}]  '
                  f'{dt:.0f}s ({dt / (k + 1):.2f}s/img)', flush=True)

    with open(out_pkl, 'wb') as f:
        pickle.dump(out, f)
    sidecar = enl_sidecar_path(out_pkl)
    with open(sidecar, 'w') as f:
        json.dump(dict(ann=os.path.basename(ann), k_draws=int(k_draws),
                       estimator='R4 median over all valid 32x32 blocks, '
                                 'clamp [1,16]',
                       enl=enl_by_img), f, indent=0, sort_keys=True)
    e = np.asarray(enls)
    print(f'[speckle] wrote {len(out)} imgs -> {out_pkl}  '
          f'(views={n_views_total}, kept={n_kept}, '
          f'ENL med={np.median(e):.2f} mean={e.mean():.2f}); '
          f'per-img ENL sidecar -> {sidecar}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='mode', required=True)
    ai = sub.add_parser('infer',
                        help='speckle re-draw inference -> DumpDetResults pkl')
    ai.add_argument('--config', required=True)
    ai.add_argument('--ckpt', required=True)
    ai.add_argument('--ann', required=True)
    ai.add_argument('--out', required=True)
    ai.add_argument('--img-dir', default=None,
                    help='default: config test_dataloader data_prefix.img')
    ai.add_argument('--k', type=int, default=4, dest='k_draws',
                    help='speckle re-realizations per image')
    ai.add_argument('--score-thr', type=float, default=0.05)
    ai.add_argument('--iou-lambda', type=float, default=0.0,
                    help='matches the soup x wide x vote-consensus baseline')
    ai.add_argument('--ksize', type=int, default=5,
                    help='odd median window for R_hat')
    ai.add_argument('--cfar-guard', type=int, default=7,
                    help='R1 CFAR guard-window edge (odd)')
    ai.add_argument('--cfar-ring', type=int, default=15,
                    help='R1 CFAR background-window edge (odd, > guard)')
    ai.add_argument('--cfar-k', type=float, default=5.0,
                    help='R1 CFAR std multiplier (5, not 3: measured '
                         'correlated-speckle/JPEG tail, see cfar_protect_mask)')
    ai.add_argument('--abs-dn', type=float, default=200.0,
                    help='R1 absolute-DN protection arm (<=0 disables)')
    ai.add_argument('--corr-sigma', type=float, default=0.76,
                    help='R3 speckle correlation kernel std in px (0 = iid)')
    ai.add_argument('--mild-L', type=float, default=16.0,
                    help='MILD-mode draw L (used only when --ksize <= 0: '
                         "I'=I*g, g~Gamma(mild_L,1/mild_L); no R_hat)")
    ai.add_argument('--device', default='cuda:0')
    a = ap.parse_args(argv)
    run_split(a.config, a.ckpt, a.ann, a.out, img_dir=a.img_dir,
              k_draws=a.k_draws, score_thr=a.score_thr,
              iou_lambda=a.iou_lambda, device=a.device, ksize=a.ksize,
              cfar_guard=a.cfar_guard, cfar_ring=a.cfar_ring,
              cfar_k=a.cfar_k, abs_dn=a.abs_dn, corr_sigma=a.corr_sigma,
              mild_L=a.mild_L)
    return 0


if __name__ == '__main__':
    sys.exit(main())
