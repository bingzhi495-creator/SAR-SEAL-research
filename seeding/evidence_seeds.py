# -*- coding: utf-8 -*-
"""Evidence seeding v2: budget-direct per-scale threshold calibration.

Replaces the z>const working point (which the s=3 discrete-support ceiling
makes unreachable — measured ceiling z≈2.15 at s=3, 4.54 at s=7) with an
assumption-free rule:

  tau_s = smallest threshold on the raw rank-AUC map A_s such that the number
          of false-seed connected components inside the (independent-channel)
          background region stays within that scale's share of the per-image
          false-alarm budget.

This is exact seed-level FA calibration: spatial correlation, 8-bit ties and
truncation are all absorbed because we count actual CCs on the actual map.
Self-check (G3 v2) is split-half: calibrate tau_s on one half of the
background region, verify the FA count on the other half.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from mne.engine import SCALES

FA_BUDGET_PER_IMAGE = 20.0   # expected false seeds per 800^2 image, all scales
MIN_CC_AREA = 2              # ignore 1-px specks when counting/seeding


def _cc_count_above(score: np.ndarray, region: np.ndarray, thr: float) -> int:
    lab, n = ndimage.label((score > thr) & region)
    if n == 0:
        return 0
    if MIN_CC_AREA > 1:
        counts = np.bincount(lab.ravel())[1:]
        return int((counts >= MIN_CC_AREA).sum())
    return n


def budget_threshold(score: np.ndarray, bg_region: np.ndarray, budget: float,
                     n_grid: int = 24) -> float:
    """Smallest threshold whose background false-seed CC count <= budget.

    Budget is scaled by the background region's share of the image so the
    full-image FA semantics hold approximately.
    """
    area_scale = bg_region.mean()
    eff_budget = max(budget * area_scale, 0.5)
    vals = score[bg_region]
    if vals.size < 100:
        return float(np.max(score)) + 1.0  # degenerate: disable this scale
    qs = np.quantile(vals, np.linspace(0.9, 1.0, n_grid))
    qs = np.unique(qs)
    for thr in qs:  # ascending: first threshold meeting budget is the smallest
        if _cc_count_above(score, bg_region, thr) <= eff_budget:
            return float(thr)
    return float(qs[-1])


def calibrate_thresholds(field: dict, budget_total: float = FA_BUDGET_PER_IMAGE,
                         split_half_check: bool = True, rng=None) -> dict:
    """Per-scale budget thresholds + split-half FA verification (G3 v2).

    Returns {'tau': {s: thr}, 'g3': {s: {...}, 'all_pass': bool}}.
    """
    rng = rng or np.random.default_rng(2)
    bg = field["bg_mask"]
    n_scales = field["A"].shape[0]
    budget_s = budget_total / n_scales

    # split background into two interleaved halves by CC-safe blocks (16px
    # checkerboard) so seeds straddling halves are still counted coherently
    H, W = bg.shape
    by, bx = np.indices((H, W)) // 16
    half_a = ((by + bx) % 2 == 0) & bg
    half_b = ~((by + bx) % 2 == 0) & bg

    tau, g3 = {}, {}
    for k in range(n_scales):
        s = SCALES[k]
        a_map = field["A"][k]
        thr = budget_threshold(a_map, half_a if split_half_check else bg, budget_s)
        tau[s] = thr
        if split_half_check:
            fa_b = _cc_count_above(a_map, half_b, thr)
            allowed = max(budget_s * half_b.mean(), 0.5)
            g3[s] = {"thr": round(thr, 4), "fa_holdout": int(fa_b),
                     "allowed": round(allowed * 3 + 2, 1),  # 3x tolerance + slack
                     "pass": bool(fa_b <= allowed * 3 + 2)}
    g3["all_pass"] = all(v["pass"] for kk, v in g3.items() if isinstance(v, dict))
    return {"tau": tau, "g3": g3}


def extract_seeds(field: dict, tau: dict, max_seeds: int = 200) -> list:
    """Per-scale CC seeding at calibrated thresholds, cross-scale dedup
    (keep the largest supporting scale within 3 px)."""
    seeds = []
    for k, s in enumerate(SCALES):
        if s not in tau:
            continue
        a_map = field["A"][k]
        lab, n = ndimage.label(a_map > tau[s])
        if n == 0:
            continue
        counts = np.bincount(lab.ravel())
        for sl_idx, sl in enumerate(ndimage.find_objects(lab), start=1):
            if sl is None or counts[sl_idx] < MIN_CC_AREA:
                continue
            comp = a_map[sl] * (lab[sl] == sl_idx)
            py, px = np.unravel_index(np.argmax(comp), comp.shape)
            cy, cx = sl[0].start + py, sl[1].start + px
            h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
            y0 = max(0, sl[0].start - int(0.1 * h)); y1 = sl[0].stop + int(0.1 * h)
            x0 = max(0, sl[1].start - int(0.1 * w)); x1 = sl[1].stop + int(0.1 * w)
            seeds.append((cy, cx, y0, x0, y1, x1, s, float(comp.max())))
    seeds.sort(key=lambda t: -t[6])  # larger scale first for dedup
    kept = []
    for sd in seeds:
        if all((sd[0] - kk[0]) ** 2 + (sd[1] - kk[1]) ** 2 > 9 for kk in kept):
            kept.append(sd)
    if len(kept) > max_seeds:  # keep highest-evidence seeds within budget
        kept.sort(key=lambda t: -t[7])
        kept = kept[:max_seeds]
    return kept
