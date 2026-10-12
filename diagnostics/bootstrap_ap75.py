# -*- coding: utf-8 -*-
"""M4 (P0, zero-GPU): image-level paired bootstrap CI for the refine-head AP75
gain, on the inshore stratum (the headline claim +4.6 inshore AP75 is currently
single-seed, no CI -- statistically the weakest part of the strongest claim).

Resamples the inshore test images with replacement B times; for each resample
computes segm AP75 for A0 and A0R-refine on that image subset; reports the
paired difference distribution (mean, 95% percentile CI). Uses existing preds
(no GPU, no retraining).

Usage: python diagnostics\\bootstrap_ap75.py [--split test_inshore] [--B 2000]
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def preds_to_results(preds, name2id):
    res = []
    for p in preds:
        stem = os.path.splitext(os.path.basename(p["img_path"]))[0]
        if stem not in name2id:
            continue
        inst = p["pred_instances"]
        scores = inst["scores"].cpu().numpy() if hasattr(inst["scores"], "cpu") else np.asarray(inst["scores"])
        masks = inst["masks"]
        for k in range(len(scores)):
            rle = masks[k]
            rle = {"size": rle["size"], "counts": rle["counts"].decode() if isinstance(rle["counts"], bytes) else rle["counts"]}
            res.append({"image_id": name2id[stem], "category_id": 1,
                        "segmentation": rle, "score": float(scores[k])})
    return res


def ap75_on(coco, dt, img_ids):
    ev = COCOeval(coco, dt, "segm")
    ev.params.imgIds = list(img_ids)
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return ev.stats[2]  # AP75


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test_inshore")
    ap.add_argument("--B", type=int, default=2000)
    args = ap.parse_args()
    import torch  # noqa

    gt = os.path.join(BASE, "data", "pseg_ssdd", "annotations", args.split + ".json")
    coco = COCO(gt)
    name2id = {os.path.splitext(im["file_name"])[0]: i for i, im in coco.imgs.items()}
    img_ids = list(coco.imgs.keys())

    a0 = pickle.load(open(os.path.join(BASE, "runs", "refine_gate", "A0_stock_preds.pkl"), "rb"))
    a0r = pickle.load(open(os.path.join(BASE, "runs", "refine_gate", "A0R_refine_preds.pkl"), "rb"))
    dt_a0 = coco.loadRes(preds_to_results(a0, name2id))
    dt_a0r = coco.loadRes(preds_to_results(a0r, name2id))

    base_a0 = ap75_on(coco, dt_a0, img_ids)
    base_a0r = ap75_on(coco, dt_a0r, img_ids)
    print(f"[{args.split}] point AP75: A0={base_a0:.4f} A0R={base_a0r:.4f} Delta={base_a0r-base_a0:+.4f}", flush=True)

    rng = np.random.default_rng(0)
    n = len(img_ids)
    diffs = []
    for b in range(args.B):
        samp = rng.choice(img_ids, size=n, replace=True)
        d = ap75_on(coco, dt_a0r, samp) - ap75_on(coco, dt_a0, samp)
        diffs.append(d)
        if (b + 1) % 200 == 0:
            print(f"  boot {b+1}/{args.B} running mean Delta={np.mean(diffs):+.4f}", flush=True)
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"\n=== {args.split} AP75 Delta (A0R-A0), {args.B} image-level bootstrap ===")
    print(f"  point {base_a0r-base_a0:+.4f} | boot mean {diffs.mean():+.4f} | 95% CI [{lo:+.4f}, {hi:+.4f}] | P(Delta>0)={np.mean(diffs>0):.3f}")


if __name__ == "__main__":
    main()
