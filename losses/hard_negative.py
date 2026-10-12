# -*- coding: utf-8 -*-
"""Anisotropic evidence hard-negative focal loss (verdict: anisotropic
exclusion band, D1 graft, unanimous across reviewers).

Targets statistical false positives near (but not on) ships: pixels where the
MNE evidence is high but which are NOT GT ship pixels and lie OUTSIDE an
anisotropic exclusion band around each GT. The band is wider along the azimuth
(range-ambiguity / sidelobe direction, 16 px) than across it (8 px), because
X-band azimuth sidelobes extend further than range — an isotropic band would
either over-suppress the ship ends (hurting Boundary IoU) or leak real FPs.

This module provides:
  build_hardneg_mask(evidence_zmax, gt_masks, gt_thetas) -> bool mask H
  hard_negative_focal(p_ship, H, gamma=2) -> scalar loss

p_ship is the carrier's per-pixel foreground probability (marginalized over
unmatched queries' mask-classification). Pure torch; integrates into the
carrier loss as an extra term gated by H.
"""
from __future__ import annotations

import numpy as np


def _aniso_distance_ok(dist_along, dist_perp, band_along=16.0, band_perp=8.0):
    """Inside the exclusion band iff (a/A)^2 + (p/P)^2 < 1 (elliptical)."""
    return (dist_along / band_along) ** 2 + (dist_perp / band_perp) ** 2 < 1.0


def build_hardneg_mask(evidence_zmax: np.ndarray, gt_masks, gt_thetas,
                       z_thresh: float, band_along=16.0, band_perp=8.0) -> np.ndarray:
    """Boolean hard-negative mask H (numpy reference / offline precompute).

    evidence_zmax : (H, W) max-over-scale MNE z map
    gt_masks      : list of (H, W) bool GT instance masks
    gt_thetas     : list of orientation angles (rad) per instance (azimuth axis)
    z_thresh      : evidence threshold (per-image FA-budget derived)
    Returns H = (evidence high) AND (not GT) AND (outside anisotropic band).
    """
    Hh, Ww = evidence_zmax.shape
    high = evidence_zmax > z_thresh
    on_ship = np.zeros((Hh, Ww), dtype=bool)
    band = np.zeros((Hh, Ww), dtype=bool)

    yy, xx = np.mgrid[0:Hh, 0:Ww]
    for m, th in zip(gt_masks, gt_thetas):
        on_ship |= m
        ys, xs = np.nonzero(m)
        if len(ys) == 0:
            continue
        cy, cx = ys.mean(), xs.mean()
        ca, sa = np.cos(th), np.sin(th)
        # rotate pixel offsets into (along-azimuth, perp) frame
        dx, dy = xx - cx, yy - cy
        d_along = np.abs(dx * ca + dy * sa)
        d_perp = np.abs(-dx * sa + dy * ca)
        # only consider a local bbox for speed
        band |= _aniso_distance_ok(d_along, d_perp, band_along, band_perp)
    return high & (~on_ship) & (~band)


# ---- torch loss (used in training) ----
def hard_negative_focal_torch(p_ship, H_mask, gamma: float = 2.0, eps: float = 1e-6):
    """torch scalar. p_ship,(H_mask): (H,W) tensors; H_mask bool/float gate.

    L = sum_x H * p^gamma * (-log(1-p)) / (sum H + eps)
    Penalizes confident foreground on statistical hard negatives.
    """
    import torch
    p = p_ship.clamp(eps, 1 - eps)
    w = H_mask.float()
    loss = w * (p ** gamma) * (-(1 - p).log())
    return loss.sum() / (w.sum() + eps)


if __name__ == "__main__":
    # self-test: a ship at center with azimuth 0; nearby bright FP along vs perp
    H = W = 64
    ev = np.zeros((H, W)); ev[10, 32] = 5.0   # perp FP (should be hard-neg: outside 8px perp? 22px away -> yes)
    ev[32, 50] = 5.0                          # along FP at 18px -> inside 16px? no, just outside
    m = np.zeros((H, W), bool); m[28:37, 28:37] = True
    Hmask = build_hardneg_mask(ev, [m], [0.0], z_thresh=3.0)
    print("hard-neg pixels:", int(Hmask.sum()), "| at(10,32):", bool(Hmask[10, 32]),
          "| at(32,50):", bool(Hmask[32, 50]))
