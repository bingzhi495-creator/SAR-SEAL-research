# -*- coding: utf-8 -*-
"""Boundary IoU at multiple dilation widths (verdict-4 required change 3).

Cheng et al. CVPR2021 define Boundary IoU with a single d (default 2% of the
image diagonal). The verdict-4 audit showed that on HRSID/PSeg, where almost
all ships have min-dimension < 2*d, standard Boundary IoU degenerates to plain
Mask IoU (G∩G_d = G). So a single-d Boundary IoU is NOT an independent quality
axis on this data.

This module reports Boundary IoU at d ∈ {2, 5, 23} px (23 ≈ 2% of 800²
diagonal), making the degeneracy explicit, plus a 1px-tolerance Boundary F1.
The caller decides what to headline; we never silently use one d.
"""
from __future__ import annotations

import cv2
import numpy as np


def _to_bitmap(rle_or_mask, h=None, w=None):
    if isinstance(rle_or_mask, np.ndarray):
        return rle_or_mask.astype(np.uint8)
    from pycocotools import mask as mu
    r = rle_or_mask
    if isinstance(r, list):  # polygon
        r = mu.merge(mu.frPyObjects(r, h, w))
    return mu.decode(r).astype(np.uint8)


def _boundary_region(m: np.ndarray, d: int) -> np.ndarray:
    """Pixels of m within distance d of its boundary (Cheng et al.)."""
    if m.sum() == 0:
        return m
    k = np.ones((3, 3), np.uint8)
    er = cv2.erode(m, k, iterations=d)
    return m - er


def boundary_iou(gt: np.ndarray, pred: np.ndarray, d: int) -> float:
    gb = _boundary_region(gt, d)
    pb = _boundary_region(pred, d)
    inter = np.logical_and(gb, pb).sum()
    union = np.logical_or(gb, pb).sum()
    return float(inter / union) if union > 0 else 1.0


def mask_iou(gt: np.ndarray, pred: np.ndarray) -> float:
    inter = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    return float(inter / union) if union > 0 else 1.0


def boundary_f1(gt: np.ndarray, pred: np.ndarray, tol: int = 1) -> float:
    """1px-tolerance boundary F1: contour-point precision/recall within tol."""
    def contour(m):
        cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        pts = np.concatenate([c.reshape(-1, 2) for c in cnts], 0) if cnts else np.zeros((0, 2))
        return pts
    gc, pc = contour(gt), contour(pred)
    if len(gc) == 0 and len(pc) == 0:
        return 1.0
    if len(gc) == 0 or len(pc) == 0:
        return 0.0
    # distance via dilated boundary masks (cheap, tol-px)
    gb = _boundary_region(gt, tol + 1)
    pb = _boundary_region(pred, tol + 1)
    # precision: pred contour pts inside gt boundary band; recall: vice versa
    pc_i = pc.astype(int)
    gc_i = gc.astype(int)
    H, W = gt.shape
    prec = gb[np.clip(pc_i[:, 1], 0, H-1), np.clip(pc_i[:, 0], 0, W-1)].mean()
    rec = pb[np.clip(gc_i[:, 1], 0, H-1), np.clip(gc_i[:, 0], 0, W-1)].mean()
    return float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


def standard_d(width, height):
    """Boundary IoU standard dilation = 2% of the image diagonal (Cheng 2021).
    P1-FIX: seal_eval used a fixed d=23 (correct ONLY for 800x800 HRSID); for
    smaller PSeg/SSDD images (~600px diag) the standard d is ~12px, which is why
    the old PSeg degeneracy 89% was wrong (true 67% at the per-image d). Callers
    should compute d per image via this, not hardcode 23."""
    return max(1, round(0.02 * (width ** 2 + height ** 2) ** 0.5))


def report_degeneracy(min_dims, ds=(2, 5, 23)):
    """For an array of instance min-dimensions, report the fraction for which
    Boundary IoU at each d degenerates to Mask IoU (min_dim < 2d).
    NOTE: pass the per-image standard_d() as one of `ds` for the correct
    'standard Boundary IoU' degeneracy (fixed 23 is HRSID-only)."""
    out = {}
    md = np.asarray(min_dims)
    for d in ds:
        out[d] = float((md < 2 * d).mean())
    return out


if __name__ == "__main__":
    # self-test: a 20px square vs a 1px-shifted copy
    a = np.zeros((64, 64), np.uint8); a[20:40, 20:40] = 1
    b = np.zeros((64, 64), np.uint8); b[21:41, 20:40] = 1
    print("mask IoU:", round(mask_iou(a, b), 3))
    for d in (2, 5, 23):
        print(f"boundary IoU d={d}:", round(boundary_iou(a, b, d), 3),
              "(degenerate)" if 20 < 2 * d else "")
    print("boundary F1 (1px tol):", round(boundary_f1(a, b), 3))
    print("degeneracy frac for 20px ships:", report_degeneracy([20] * 10))
