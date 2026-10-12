# -*- coding: utf-8 -*-
"""Mechanistic evidence for the refine head: WHY does it move AP75 but not AP50?

Claim (diagnostic core): the final-layer 1/4->1/2 mask recompute lifts the
matched-instance mask-IoU specifically in the band just below the AP75 threshold
(0.75), pushing mass across 0.75 while leaving the 0.50 mass (AP50) unchanged.

For each GT instance, find the best-matching prediction (by mask-IoU) under A0
and under A0R-refine, then characterise the per-instance IoU shift, bucketed by
GT size and by whether it crosses 0.75. Pure CPU on dumped preds (no GPU, no
retrain). Reports PSeg (headline, big gain) and optionally HRSID.

Usage: python diagnostics\\refine_mechanism.py --dataset pseg
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
from pycocotools import mask as mask_util
from pycocotools.coco import COCO

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_preds_by_stem(path):
    preds = pickle.load(open(path, "rb"))
    out = {}
    for p in preds:
        stem = os.path.splitext(os.path.basename(p["img_path"]))[0]
        out[stem] = p["pred_instances"]
    return out


def best_iou_per_gt(gt_masks, inst, score_t):
    scores = inst["scores"].cpu().numpy() if hasattr(inst["scores"], "cpu") else np.asarray(inst["scores"])
    keep = np.nonzero(scores >= score_t)[0]
    pmasks = [mask_util.decode(inst["masks"][k]).astype(bool) for k in keep]
    res = []
    for g in gt_masks:
        best = 0.0
        ga = g.sum()
        for pm in pmasks:
            inter = np.logical_and(g, pm).sum()
            if inter == 0:
                continue
            union = ga + pm.sum() - inter
            iou = inter / union if union else 0.0
            if iou > best:
                best = iou
        res.append(best)
    return np.array(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["pseg", "hrsid"], default="pseg")
    ap.add_argument("--score_t", type=float, default=0.3)
    args = ap.parse_args()
    import torch  # noqa

    if args.dataset == "pseg":
        gt = os.path.join(BASE, "data", "pseg_ssdd", "annotations", "test.json")
        a0p = os.path.join(BASE, "runs", "refine_gate", "A0_stock_preds.pkl")
        a0rp = os.path.join(BASE, "runs", "refine_gate", "A0R_refine_preds.pkl")
    else:
        gt = os.path.join(BASE, "data", "hrsid", "annotations", "official_test.json")
        a0p = os.path.join(BASE, "runs", "a0_hrsid_seed0", "preds.pkl")
        a0rp = os.path.join(BASE, "runs", "a0r_refine_hrsid", "preds.pkl")

    coco = COCO(gt)
    a0 = load_preds_by_stem(a0p)
    a0r = load_preds_by_stem(a0rp)
    name2id = {os.path.splitext(im["file_name"])[0]: i for i, im in coco.imgs.items()}

    iou0, iour, sizes = [], [], []
    for stem, img_id in name2id.items():
        if stem not in a0 or stem not in a0r:
            continue
        im = coco.imgs[img_id]
        h, w = im["height"], im["width"]
        anns = coco.imgToAnns.get(img_id, [])
        if not anns:
            continue
        gmasks, gsz = [], []
        for a in anns:
            seg = a["segmentation"]
            r = seg if isinstance(seg, dict) else mask_util.merge(mask_util.frPyObjects(seg, h, w))
            gmasks.append(mask_util.decode(r).astype(bool))
            gsz.append(a["area"])
        b0 = best_iou_per_gt(gmasks, a0[stem], args.score_t)
        br = best_iou_per_gt(gmasks, a0r[stem], args.score_t)
        iou0.extend(b0); iour.extend(br); sizes.extend(gsz)

    iou0 = np.array(iou0); iour = np.array(iour); sizes = np.array(sizes)
    det = iou0 >= 0.5  # focus on instances A0 already detects (mask-quality lever, not recall)
    print(f"\n===== refine mechanism [{args.dataset}] (n_GT={len(iou0)}, A0-detected={det.sum()}) =====")
    print(f"matched-instance mask-IoU mean: A0={iou0[det].mean():.4f} -> A0R={iour[det].mean():.4f} (delta {iour[det].mean()-iou0[det].mean():+.4f})")
    print(f"median: A0={np.median(iou0[det]):.4f} -> A0R={np.median(iour[det]):.4f}")

    # cross-threshold flow at 0.75 (AP75) vs 0.50 (AP50)
    for thr in (0.50, 0.75):
        up = np.logical_and(iou0 < thr, iour >= thr).sum()    # gained: crossed up
        down = np.logical_and(iou0 >= thr, iour < thr).sum()  # lost: dropped below
        print(f"  IoU>= {thr:.2f}: A0 {np.mean(iou0>=thr):.3f} -> A0R {np.mean(iour>=thr):.3f} | crossed UP {up}, DOWN {down}, net {up-down:+d}")

    # by size bucket (COCO small<32^2=1024, medium<96^2=9216, large)
    print("  by GT size (matched, IoU mean A0->A0R):")
    for name, lo, hi in (("small", 0, 1024), ("medium", 1024, 9216), ("large", 9216, 1e18)):
        m = det & (sizes >= lo) & (sizes < hi)
        if m.sum():
            print(f"    {name:6s} n={m.sum():4d}: {iou0[m].mean():.4f} -> {iour[m].mean():.4f} ({iour[m].mean()-iou0[m].mean():+.4f}); cross0.75 net {(np.logical_and(iou0[m]<0.75,iour[m]>=0.75).sum() - np.logical_and(iou0[m]>=0.75,iour[m]<0.75).sum()):+d}")


if __name__ == "__main__":
    main()
