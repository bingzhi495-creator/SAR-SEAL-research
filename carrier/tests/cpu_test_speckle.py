# -*- coding: utf-8 -*-
"""CPU-ONLY sanity test for R8 speckle-marginalization TTA (speckle_infer.py).

NO GPU, no checkpoints, no mmdet import (speckle_infer imports mmdet lazily).
Exercises the pure physics functions AFTER the theory-mandated fixes R1-R4
(docs/THEORY_MARGINALIZATION.md section 7):

  (a) ENL RECOVERY   synthetic scene with known R (flat intensity 100 + bright
      32x32 square 1000), speckled at L=4 via I = R*g, g~Gamma(4, 1/4),
      A = sqrt(I) quantized to uint8. estimate_enl (R4: median over ALL valid
      blocks) must land in [3, 5].
  (b) REDRAW STATS   speckle_redraw on the synthetic chip preserves mean
      amplitude within 3% (both the original and the redraw carry the same
      sqrt-Gamma amplitude bias, and R_hat is Gamma-median bias-corrected on
      unprotected pixels, R2) and actually re-randomizes pixels:
      Pearson corr(original, redraw) < 0.99.
  (c) DETERMINISM    make_rng(img_id, k) -> byte-identical redraw for the same
      (img_id, k); a different draw index changes the bytes (synthetic chip).
  (d) REAL CHIP      first HRSID test_inshore chip: print estimated ENL
      (expect ~1-5 for single-look HRSID after JPG), time ENL / CFAR mask /
      one redraw, save a side-by-side original|redraw PNG to
      runs/speckle/proof/.
  (e) R1 PROTECTION  on a real inshore chip WITH GT ships: protected-pixel
      count/fraction and in-ship precision of the new CFAR + abs-DN mask vs
      the OLD global top-0.5% quantile rule; the new rule must improve
      ship-hit precision (the P3 study measured old inshore precision ~23%).
  (f) R3 CORRELATION corr_gamma_field at ENL 1.114 and 4.0: achieved lag-1
      Pearson correlation in [0.4, 0.65] (real chips: 0.42-0.65), marginal
      mean within 1% of 1, marginal var within 15% of 1/ENL.
  (g) DETERMINISM (full new pipeline) on the real chip: CFAR mask + copula
      field + R2 bias path is byte-identical for the same (img_id, k) and
      differs for another k.

Run:
  KMP_DUPLICATE_LIB_OK=TRUE CUDA_VISIBLE_DEVICES="" \
  python carrier/tests/cpu_test_speckle.py
"""
from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))
from carrier.speckle_infer import (  # noqa: E402
    cfar_protect_mask, corr_gamma_field, estimate_enl, gamma_median,
    make_rng, speckle_redraw)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ANN = os.path.join(ROOT, 'data', 'hrsid', 'annotations', 'test_inshore.json')
IMG_DIR = 'D:/SAR/UNet++/HRSID_JPG/HRSID_JPG/JPEGImages1/'
PROOF_DIR = os.path.join(ROOT, 'runs', 'speckle', 'proof')


def synth_chip(l_true=4.0, size=256, seed=123):
    """Known-R scene: flat intensity 100 + bright 32x32 square 1000, L looks."""
    R = np.full((size, size), 100.0)
    R[112:144, 112:144] = 1000.0
    g = np.random.default_rng(seed).gamma(shape=l_true, scale=1.0 / l_true,
                                          size=R.shape)
    A = np.sqrt(R * g)
    return np.clip(np.rint(A), 0, 255).astype(np.uint8)


def lag1_corr(f):
    """Mean of horizontal+vertical lag-1 Pearson correlations."""
    f = np.asarray(f, dtype=np.float64)
    rh = np.corrcoef(f[:, :-1].ravel(), f[:, 1:].ravel())[0, 1]
    rv = np.corrcoef(f[:-1, :].ravel(), f[1:, :].ravel())[0, 1]
    return 0.5 * (rh + rv)


def main():
    ok = True

    # ---------------- (a) ENL recovery on synthetic ----------------
    chip = synth_chip(l_true=4.0)
    enl = estimate_enl(chip)
    print(f'(a) synthetic L=4 chip (flat R=100 + square R=1000, uint8): '
          f'estimated ENL = {enl:.3f} (need in [3, 5]; R4 all-valid-block '
          f'median); gamma_median({enl:.2f}) = {gamma_median(enl):.4f}')
    if not (3.0 <= enl <= 5.0):
        print('  !! FAIL (a)'); ok = False

    # ---------------- (b) redraw preserves brightness, changes pixels ------
    redraw = speckle_redraw(chip, enl, make_rng(1, 0))
    ratio = float(redraw.astype(np.float64).mean()
                  / chip.astype(np.float64).mean())
    corr = float(np.corrcoef(chip.ravel().astype(np.float64),
                             redraw.ravel().astype(np.float64))[0, 1])
    print(f'(b) mean amplitude ratio redraw/original = {ratio:.4f} '
          f'(need within [0.97, 1.03]); pixel corr = {corr:.4f} '
          f'(need < 0.99)')
    if not (0.97 <= ratio <= 1.03):
        print('  !! FAIL (b) brightness'); ok = False
    if not (corr < 0.99):
        print('  !! FAIL (b) corr'); ok = False

    # ---------------- (c) determinism (synthetic) ----------------
    r1 = speckle_redraw(chip, enl, make_rng(7, 0))
    r2 = speckle_redraw(chip, enl, make_rng(7, 0))
    r3 = speckle_redraw(chip, enl, make_rng(7, 1))
    same = r1.tobytes() == r2.tobytes()
    diff = r1.tobytes() != r3.tobytes()
    print(f'(c) determinism (synthetic): same (img_id=7, k=0) byte-equal = '
          f'{same} (must be True); different k differs = {diff} '
          f'(must be True)')
    assert same, 'same seed must give byte-identical draw'
    if not diff:
        print('  !! FAIL (c) draw index has no effect'); ok = False

    # ---------------- (d) real HRSID inshore chip ----------------
    from pycocotools.coco import COCO
    coco = COCO(ANN)
    ids = sorted(coco.getImgIds())
    info = coco.loadImgs([ids[0]])[0]
    path = os.path.join(IMG_DIR, info['file_name'])
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    assert img is not None, f'cannot read {path}'
    gray = img[:, :, 0]

    t0 = time.time()
    enl_real = estimate_enl(gray)
    t_enl = time.time() - t0
    t0 = time.time()
    pmask = cfar_protect_mask(gray)
    t_cfar = time.time() - t0
    t0 = time.time()
    red_real = speckle_redraw(gray, enl_real, make_rng(info['id'], 0),
                              protect_mask=pmask)
    t_draw = time.time() - t0
    ratio_real = float(red_real.astype(np.float64).mean()
                       / max(gray.astype(np.float64).mean(), 1e-9))
    print(f"(d) real chip {info['file_name']} (img_id={info['id']}, "
          f"{gray.shape[1]}x{gray.shape[0]}): estimated ENL = "
          f"{enl_real:.3f} (expect ~1-5 for single-look HRSID; R4); "
          f"mean ratio = {ratio_real:.4f}; "
          f"CPU timing: estimate_enl {t_enl * 1000:.0f} ms, "
          f"cfar_protect_mask {t_cfar * 1000:.0f} ms, "
          f"one redraw {t_draw * 1000:.0f} ms")
    if not (1.0 <= enl_real <= 8.0):
        print('  !! FAIL (d) ENL implausible'); ok = False

    os.makedirs(PROOF_DIR, exist_ok=True)
    sep = np.full((gray.shape[0], 8), 255, dtype=np.uint8)
    side = np.hstack([gray, sep, red_real])
    out_png = os.path.join(
        PROOF_DIR, os.path.splitext(info['file_name'])[0]
        + '_orig_vs_redraw.png')
    cv2.imwrite(out_png, side)
    print(f'    side-by-side proof PNG -> {out_png}')

    # ---------------- (e) R1 protection vs old top-0.5% on GT ships --------
    ship_id = next((i for i in ids if coco.getAnnIds(imgIds=[i])), None)
    assert ship_id is not None, 'no inshore chip with GT ships?!'
    sinfo = coco.loadImgs([ship_id])[0]
    simg = cv2.imread(os.path.join(IMG_DIR, sinfo['file_name']),
                      cv2.IMREAD_COLOR)
    assert simg is not None, sinfo['file_name']
    sgray = simg[:, :, 0]
    union = np.zeros(sgray.shape, np.uint8)
    for a_ in coco.loadAnns(coco.getAnnIds(imgIds=[ship_id])):
        union |= coco.annToMask(a_).astype(np.uint8)
    union = union.astype(bool)
    af = sgray.astype(np.float32)
    sel_old = af >= np.quantile(af, 1.0 - 0.005)       # OLD global top-0.5%
    sel_new = cfar_protect_mask(sgray)                 # NEW R1 CFAR + abs-DN
    npx = sgray.size
    n_old, n_new = int(sel_old.sum()), int(sel_new.sum())
    hit_old = int((sel_old & union).sum())
    hit_new = int((sel_new & union).sum())
    prec_old = hit_old / max(n_old, 1)
    prec_new = hit_new / max(n_new, 1)
    cov_old = hit_old / max(int(union.sum()), 1)
    cov_new = hit_new / max(int(union.sum()), 1)
    print(f"(e) R1 protection on chip {sinfo['file_name']} "
          f"(img_id={ship_id}, {int(union.sum())} GT ship px, "
          f"{len(coco.getAnnIds(imgIds=[ship_id]))} instances):")
    print(f'    OLD top-0.5%: protected {n_old} px '
          f'({100.0 * n_old / npx:.2f}% of chip), in-ship {hit_old} '
          f'-> precision {100.0 * prec_old:.1f}%, '
          f'ship coverage {100.0 * cov_old:.1f}%')
    print(f'    NEW CFAR(guard7/ring15/k5)+A>=200: protected {n_new} px '
          f'({100.0 * n_new / npx:.2f}% of chip), in-ship {hit_new} '
          f'-> precision {100.0 * prec_new:.1f}%, '
          f'ship coverage {100.0 * cov_new:.1f}%')
    if not (prec_new > prec_old):
        print('  !! FAIL (e) new rule does not improve ship-hit precision')
        ok = False

    # ---------------- (f) R3 correlated Gamma field ----------------
    for enl_f in (1.114, 4.0):
        g = corr_gamma_field(enl_f, (768, 768), make_rng(42, 0),
                             corr_sigma=0.76)
        rho1 = lag1_corr(g)
        m, v = float(g.mean()), float(g.var())
        v_tgt = 1.0 / enl_f
        ok_rho = 0.4 <= rho1 <= 0.65
        ok_m = abs(m - 1.0) <= 0.01
        ok_v = abs(v - v_tgt) <= 0.15 * v_tgt
        print(f'(f) corr_gamma_field(ENL={enl_f}, 768x768, sigma=0.76): '
              f'lag-1 rho = {rho1:.4f} (need [0.40, 0.65]); '
              f'mean = {m:.4f} (need within 1% of 1); '
              f'var = {v:.4f} vs 1/ENL = {v_tgt:.4f} '
              f'({100.0 * abs(v - v_tgt) / v_tgt:.1f}% off, need <= 15%)')
        if not ok_rho:
            print('  !! FAIL (f) lag-1 correlation out of band'); ok = False
        if not ok_m:
            print('  !! FAIL (f) marginal mean off'); ok = False
        if not ok_v:
            print('  !! FAIL (f) marginal var off'); ok = False

    # ---------------- (g) determinism of the FULL new pipeline -------------
    d1 = speckle_redraw(sgray, enl_real, make_rng(ship_id, 0))
    d2 = speckle_redraw(sgray, enl_real, make_rng(ship_id, 0))
    d3 = speckle_redraw(sgray, enl_real, make_rng(ship_id, 1))
    same_g = d1.tobytes() == d2.tobytes()
    diff_g = d1.tobytes() != d3.tobytes()
    print(f'(g) determinism (real chip, CFAR + copula + R2 path): '
          f'byte-equal same k = {same_g} (must be True); '
          f'different k differs = {diff_g} (must be True)')
    if not (same_g and diff_g):
        print('  !! FAIL (g)'); ok = False

    print('=' * 70)
    print(f'OVERALL: {"ALL PASS" if ok else "FAIL"}')
    print('=' * 70)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
