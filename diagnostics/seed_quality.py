# -*- coding: utf-8 -*-
"""Seed quality: precision / recall of the precomputed NQS seeds vs GT boxes,
stratified inshore/offshore. The oracle gate used perfect GT boxes (0.994);
real seeds carry false positives. Seed PRECISION predicts how far SeedDINO-pred
will fall below the oracle ceiling (FP seeds the model must learn to reject).

  recall    : frac of GT boxes with >=1 seed at IoU>=0.3 (or center-in)  [~G2]
  precision : frac of seeds that match some GT at IoU>=0.3 (or center-in)
  seeds/img, fp-seeds/img
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from protocol.datasets import _subset_stems, PSEG_ROOT  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


def center_in(seed, gt):
    cx, cy = (seed[0]+seed[2])/2, (seed[1]+seed[3])/2
    return gt[0] <= cx <= gt[2] and gt[1] <= cy <= gt[3]


def match(seed, gt):
    return iou(seed, gt) >= 0.3 or center_in(seed, gt)


def main():
    seeds = json.load(open(os.path.join(BASE, "data", "seeds", "seeds_pseg_test.json")))
    gt_coco = json.load(open(os.path.join(BASE, "data", "pseg_ssdd", "annotations", "test.json")))
    inshore, _ = _subset_stems(PSEG_ROOT)
    id2name = {im["id"]: os.path.splitext(im["file_name"])[0] for im in gt_coco["images"]}
    gts = {}
    for a in gt_coco["annotations"]:
        x, y, w, h = a["bbox"]
        gts.setdefault(id2name[a["image_id"]], []).append([x, y, x+w, y+h])

    rows = {"inshore": [], "offshore": [], "all": []}
    for stem, gtb in gts.items():
        sb = seeds.get(stem, [])
        sub = "inshore" if stem in inshore else "offshore"
        gt_hit = sum(1 for g in gtb if any(match(s, g) for s in sb))
        seed_hit = sum(1 for s in sb if any(match(s, g) for g in gtb))
        rec = {"n_gt": len(gtb), "n_seed": len(sb), "gt_hit": gt_hit, "seed_hit": seed_hit}
        rows[sub].append(rec); rows["all"].append(rec)

    print(f"{'stratum':<10}{'recall':>9}{'precision':>11}{'seeds/img':>11}{'fp/img':>9}")
    for sub in ["inshore", "offshore", "all"]:
        r = rows[sub]
        ng = sum(x["n_gt"] for x in r); ns = sum(x["n_seed"] for x in r)
        gh = sum(x["gt_hit"] for x in r); sh = sum(x["seed_hit"] for x in r)
        recall = gh/ng if ng else 0
        prec = sh/ns if ns else 0
        spi = ns/len(r) if r else 0
        fppi = (ns-sh)/len(r) if r else 0
        print(f"{sub:<10}{recall:>9.3f}{prec:>11.3f}{spi:>11.1f}{fppi:>9.1f}")
    print("\nNote: low precision = many FP seeds the SeedDINO classifier must reject;")
    print("predicts the gap between SeedDINO-pred and the oracle ceiling (0.994).")


if __name__ == "__main__":
    main()
