# -*- coding: utf-8 -*-
"""G2/G3 pilot on PSeg-SSDD test (46 inshore + 186 offshore).

G3 (calibration self-check, verdict-6 tail-aware form):
  per image and scale, sample H0 positions from the independent-channel
  background region, standardize by the empirical null, then count #(Z>3).
  Expected under H0: Q * P(Z>3) = Q * 0.00135. Gate: observed within a
  Poisson tolerance band (<= ceil(expect + 4*sqrt(expect)) and the MAD != 0).
  Pass rate reported per stratum; "≥90%" may only ever be quoted from
  *measured* numbers like these (never assumed).

G2 (seed recall pilot, verdict-3 frozen double criterion):
  seeds = per-scale connected components of {Ẽ_s > z_w}, one seed per
  component (peak), box = component bbox x 1.2, cross-scale dedup by center
  distance (keep largest supporting scale). Working point z_w = 4.0.
  Hit criterion (pre-registered): seed box IoU >= 0.3 with GT box, OR seed
  center inside GT box. Recall stratified by subset; per-GT background-ring
  contamination exposure reported (fraction of GT ships whose annulus
  contains > 25% bright pixels, brightness = image Otsu threshold proxy --
  labelled PROXY, real land masks arrive with HRSID+GSHHG).

Usage: python diagnostics\g2_g3_pilot.py [--workers 10]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mne.engine import (mne_field, background_sample_mask, SCALES)  # noqa: E402
from protocol.datasets import test_stems, load_one  # noqa: E402
from seeding.evidence_seeds import calibrate_thresholds, extract_seeds  # noqa: E402

ZW = 4.0          # seed working point (s=3 discrete-support caveat noted)
Q3 = 2000         # null sample size for G3
Z3 = 3.0          # tail threshold for G3 exceedance counting


def _g3_check(field: dict, img: np.ndarray) -> dict:
    """Tail-aware calibration self-check for one image."""
    bg = field["bg_mask"]
    rng = np.random.default_rng(1)
    ys, xs = np.nonzero(bg)
    out = {}
    for k, s in enumerate(SCALES):
        if len(ys) == 0:
            out[f"s{s}"] = {"pass": False, "reason": "no_bg"}
            continue
        idx = rng.choice(len(ys), size=min(Q3, len(ys)), replace=False)
        z = field["E"][k][ys[idx], xs[idx]]
        n_exc = int((z > Z3).sum())
        expect = len(idx) * 0.00135
        tol = expect + 4.0 * np.sqrt(max(expect, 1e-6)) + 1.0
        mad_zero = bool(np.median(np.abs(z - np.median(z))) == 0)
        out[f"s{s}"] = {"n_exc": n_exc, "expect": round(expect, 2),
                        "tol": round(tol, 2), "mad_zero": mad_zero,
                        "pass": bool(n_exc <= tol and not mad_zero)}
    out["all_pass"] = all(v["pass"] for k, v in out.items() if k.startswith("s"))
    return out


def _seeds(field: dict) -> list:
    """Evidence seeds: per-scale CC -> one seed/CC -> cross-scale dedup."""
    seeds = []  # (cy, cx, y0, x0, y1, x1, scale)
    for k, s in enumerate(SCALES):
        lab, n = ndimage.label(field["E"][k] > ZW)
        if n == 0:
            continue
        for sl_idx, sl in enumerate(ndimage.find_objects(lab), start=1):
            comp = field["E"][k][sl] * (lab[sl] == sl_idx)
            py, px = np.unravel_index(np.argmax(comp), comp.shape)
            cy, cx = sl[0].start + py, sl[1].start + px
            h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
            y0 = max(0, sl[0].start - int(0.1 * h)); y1 = sl[0].stop + int(0.1 * h)
            x0 = max(0, sl[1].start - int(0.1 * w)); x1 = sl[1].stop + int(0.1 * w)
            seeds.append((cy, cx, y0, x0, y1, x1, s))
    # cross-scale dedup: keep largest supporting scale within 3 px
    seeds.sort(key=lambda t: -t[6])
    kept = []
    for sd in seeds:
        if all((sd[0] - k[0]) ** 2 + (sd[1] - k[1]) ** 2 > 9 for k in kept):
            kept.append(sd)
    return kept


def _gt_boxes(sample) -> list:
    boxes = []
    for m in sample.instances:
        ys, xs = np.nonzero(m)
        if len(ys) == 0:
            continue
        boxes.append((ys.min(), xs.min(), ys.max() + 1, xs.max() + 1))
    return boxes


def _iou(a, b) -> float:
    iy = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ix = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iy * ix
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


def _ring_contamination(img: np.ndarray, box, factor: float = 1.5) -> float:
    """PROXY: fraction of bright pixels (>= Otsu-like global threshold) in the
    1.5x dilated ring around a GT box."""
    thr = img.mean() + 2.0 * img.std()
    h, w = box[2] - box[0], box[3] - box[1]
    dy, dx = int(h * (factor - 1) / 2), int(w * (factor - 1) / 2)
    Y0, X0 = max(0, box[0] - dy), max(0, box[1] - dx)
    Y1, X1 = min(img.shape[0], box[2] + dy), min(img.shape[1], box[3] + dx)
    ring = np.ones((Y1 - Y0, X1 - X0), dtype=bool)
    ring[box[0] - Y0:box[2] - Y0, box[1] - X0:box[3] - X0] = False
    vals = img[Y0:Y1, X0:X1][ring]
    return float((vals >= thr).mean()) if vals.size else 0.0


def _process(stem_subset) -> dict | None:
    stem, subset = stem_subset
    sample = load_one(stem, subset)
    if sample is None or not sample.instances:
        return None
    field = mne_field(sample.image)
    cal = calibrate_thresholds(field)
    g3 = cal["g3"]
    seeds = extract_seeds(field, cal["tau"])
    gts = _gt_boxes(sample)
    hits, contam = [], []
    for gt in gts:
        hit = False
        for (cy, cx, y0, x0, y1, x1, s, *_rest) in seeds:
            center_in = gt[0] <= cy < gt[2] and gt[1] <= cx < gt[3]
            if center_in or _iou((y0, x0, y1, x1), gt) >= 0.3:
                hit = True
                break
        hits.append(hit)
        contam.append(_ring_contamination(sample.image, gt))
    return {"stem": stem, "subset": subset, "n_gt": len(gts),
            "n_seeds": len(seeds), "n_hit": int(sum(hits)),
            "ring_contam": [round(c, 3) for c in contam],
            "g3": g3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "g2g3_pseg"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from concurrent.futures import ProcessPoolExecutor
    work = test_stems()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_process, work, chunksize=4):
            if r:
                rows.append(r)
    json.dump(rows, open(os.path.join(args.out, "per_image.json"), "w"), indent=1)

    for sub in ["inshore", "offshore", "all"]:
        sel = [r for r in rows if sub == "all" or r["subset"] == sub]
        if not sel:
            continue
        n_gt = sum(r["n_gt"] for r in sel)
        n_hit = sum(r["n_hit"] for r in sel)
        g3_pass = sum(1 for r in sel if r["g3"]["all_pass"])
        seeds_med = float(np.median([r["n_seeds"] for r in sel]))
        contam = [c for r in sel for c in r["ring_contam"]]
        exposed = sum(1 for c in contam if c > 0.25)
        print(f"[{sub}] imgs {len(sel)} | G2 seed recall {n_hit}/{n_gt} = {100*n_hit/max(n_gt,1):.1f}% "
              f"| median seeds/img {seeds_med:.0f} | G3 pass {g3_pass}/{len(sel)} = {100*g3_pass/len(sel):.1f}% "
              f"| GT ring>25% bright: {exposed}/{len(contam)} ({100*exposed/max(len(contam),1):.1f}%)")
    print("written:", args.out)


if __name__ == "__main__":
    main()
