# -*- coding: utf-8 -*-
"""CPU-ONLY sanity test for the BPR boundary-patch refinement pipeline.

NO GPU, NO checkpoint, NO dataset. Three checks:

  (a) PATCH EXTRACTION: synthetic 200x200 disk GT + shifted coarse mask.
      boundary_centers count must EQUAL the independently recomputed number of
      unique stride-grid cells among boundary pixels; every patch window
      in-bounds and KxK; pasting the coarse-mask patches back (paste_average +
      0.5 threshold) must reproduce the coarse mask EXACTLY wherever patch
      weight > 0 (roundtrip).

  (b) REFINER: UNetRefiner forward on random 4x2x128x128 -> 4x1x128x128,
      bce_dice_loss backward populates a grad for EVERY parameter; param count
      printed.

  (c) BAND GUARD: refine_instance with a RANDOM-init refiner (arbitrary probs)
      on the synthetic instance: every pixel OUTSIDE
      band = dilate(mask,4) XOR erode(mask,4) must be byte-identical to the
      original mask (asserted), including a border-touching instance.

Run:
  KMP_DUPLICATE_LIB_OK=TRUE CUDA_VISIBLE_DEVICES="" PYTHONPATH=E:/SAR/seal \
  python carrier/tests/cpu_test_bpr.py
"""
from __future__ import annotations

import sys

import cv2
import numpy as np
import torch

from carrier.bpr_apply import make_band, refine_instance
from carrier.bpr_patches import (boundary_centers, extract_instance_patches,
                                 paste_average, patch_window)
from carrier.bpr_refiner import UNetRefiner, bce_dice_loss

K, STRIDE = 64, 32


def synth(shift=3, center=(100, 100), r=40, hw=200, seed=0):
    gt = np.zeros((hw, hw), np.uint8)
    cv2.circle(gt, center[::-1], r, 1, -1)
    coarse = np.zeros_like(gt)
    cv2.circle(coarse, (center[1] + shift, center[0]), r, 1, -1)
    rng = np.random.RandomState(seed)
    img = (30 + 120 * gt.astype(np.int32)
           + rng.randint(-20, 21, gt.shape)).clip(0, 255).astype(np.uint8)
    return img, coarse, gt


def test_a():
    img, coarse, gt = synth()
    H, W = coarse.shape
    centers = boundary_centers(coarse, STRIDE)
    # independent recount of unique grid cells among boundary pixels
    er = cv2.erode(coarse, np.ones((3, 3), np.uint8), 1)
    ys, xs = np.nonzero(coarse.astype(bool) & ~er.astype(bool))
    cells = {(y // STRIDE, x // STRIDE) for y, x in zip(ys, xs)}
    assert len(centers) == len(cells), (len(centers), len(cells))
    assert len(centers) > 0
    patches, coords = extract_instance_patches(img, coarse, gt, K, STRIDE,
                                               max_per_inst=999)
    assert len(patches) == len(centers) == len(coords)
    for (p_i, p_c, p_g), (t, l) in zip(patches, coords):
        assert p_i.shape == p_c.shape == p_g.shape == (K, K)
        assert 0 <= t <= H - K and 0 <= l <= W - K
    for cy, cx in centers:
        t, l = patch_window(cy, cx, K, H, W)
        assert t <= cy < t + K and l <= cx < l + K, 'center not inside window'
    # roundtrip paste: coarse patches -> avg -> thr 0.5 == coarse where covered
    avg, wt = paste_average([p[1] for p in patches], coords, K, H, W)
    covered = wt > 0
    rt = (avg >= 0.5).astype(np.uint8)
    assert np.array_equal(rt[covered], coarse[covered]), 'roundtrip mismatch'
    assert covered.sum() > 0
    print('[a] PASS patches=%d unique_cells=%d covered_px=%d roundtrip=EXACT'
          % (len(patches), len(cells), int(covered.sum())))


def test_b():
    torch.manual_seed(0)
    model = UNetRefiner()
    n_par = sum(p.numel() for p in model.parameters())
    x = torch.randn(4, 2, 128, 128)
    y = (torch.rand(4, 1, 128, 128) > 0.5).float()
    out = model(x)
    assert out.shape == (4, 1, 128, 128), out.shape
    loss = bce_dice_loss(out, y)
    loss.backward()
    n_grad = sum(1 for p in model.parameters()
                 if p.grad is not None and torch.isfinite(p.grad).all())
    n_tot = sum(1 for _ in model.parameters())
    assert n_grad == n_tot, (n_grad, n_tot)
    print('[b] PASS params=%d fwd=%s loss=%.4f grads=%d/%d'
          % (n_par, tuple(out.shape), loss.item(), n_grad, n_tot))
    return n_par


def _one_band_case(img, coarse, model, dev, note):
    band = make_band(coarse, 4)
    refined, n_p = refine_instance(img, coarse, model, dev)
    assert n_p > 0
    outside = ~band
    assert np.array_equal(refined[outside], coarse[outside]), \
        'GUARD VIOLATION (%s): pixel outside band changed' % note
    chg = int((refined != coarse).sum())
    assert chg == int((refined[band] != coarse[band]).sum())
    print('[c] PASS (%s) band_px=%d changed_in_band=%d outside_changed=0'
          % (note, int(band.sum()), chg))


def test_c():
    torch.manual_seed(7)
    model = UNetRefiner().eval()   # RANDOM weights: arbitrary probabilities
    dev = 'cpu'
    img, coarse, _ = synth()
    _one_band_case(img, coarse, model, dev, 'centered disk')
    # border-touching instance (window clipping path)
    img2, coarse2, _ = synth(center=(20, 20), r=18)
    _one_band_case(img2, coarse2, model, dev, 'border-touching disk')
    # adversarial prob override: all-ones probs may flood ONLY the band
    img3, coarse3, _ = synth()
    band3 = make_band(coarse3, 4)
    ref3, _ = refine_instance(img3, coarse3, None, dev,
                              _prob_override=lambda b: np.ones(
                                  (b.shape[0], 128, 128), np.float32))
    assert np.array_equal(ref3[~band3], coarse3[~band3]), \
        'GUARD VIOLATION: all-ones override leaked outside band'
    # editable region = band AND patch-covered; there it must be all-foreground;
    # band pixels NOT covered by any patch must keep the original mask
    patches, coords = extract_instance_patches(img3, coarse3, None, K, STRIDE,
                                               max_per_inst=64)
    _, wt = paste_average([np.ones((K, K), np.float32) for _ in patches],
                          coords, K, *coarse3.shape)
    covered = wt > 0
    assert (ref3[band3 & covered] == 1).all()
    assert np.array_equal(ref3[band3 & ~covered], coarse3[band3 & ~covered])
    print('[c] PASS (all-ones override) leak_outside_band=0 '
          'band_covered_px=%d' % int((band3 & covered).sum()))


if __name__ == '__main__':
    test_a()
    test_b()
    test_c()
    print('CPU_TEST_BPR PASS')
    sys.exit(0)
