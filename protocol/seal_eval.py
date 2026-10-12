# -*- coding: utf-8 -*-
"""Unified SEAL evaluator: one preds.pkl + COCO GT -> the full SEAL metric
table used to compare A0 / A0R / A2 / ... consistently.

Reports (per stratum: all / inshore / offshore / size buckets):
  * COCO mask AP / AP50 / AP75            (via pycocotools)
  * Boundary IoU at d in {2,5,23} + degeneracy fraction (verdict-4)
  * Boundary F1 (1px tol)
  * split / merge rate + mean purity      (verdict §5)
  * detected-mask-IoU median (mask quality of TPs)

Inshore/offshore tagging: PSeg via subset dirs; HRSID via an optional
inshore-stems file (data/hrsid/inshore_stems.txt); absent -> 'all' only.

Usage:
  python protocol\\seal_eval.py --preds runs\\a0_pseg_seed0\\preds.pkl \\
      --gt data\\pseg_ssdd\\annotations\\test.json --dataset pseg --out runs\\a0_pseg_seed0\\seal_metrics.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from protocol.boundary_metric import boundary_iou, boundary_f1, mask_iou, standard_d  # noqa: E402
from protocol.split_merge import split_merge_rates, aggregate  # noqa: E402


def _rle(seg, h, w):
    if isinstance(seg, dict):
        return seg
    r = mask_util.frPyObjects(seg, h, w)
    return mask_util.merge(r) if isinstance(r, list) else r


def _load_inshore(dataset):
    if dataset == "pseg":
        from protocol.datasets import _subset_stems, PSEG_ROOT
        ins, off = _subset_stems(PSEG_ROOT)
        return ins, off
    f = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "hrsid", "inshore_stems.txt")
    if os.path.exists(f):
        ins = set(l.strip() for l in open(f) if l.strip())
        return ins, None
    return None, None


def coco_ap(gt_json, preds, score_t=0.0):
    """Run COCOeval for segm; return dict of AP metrics + per-image dt for slices."""
    coco = COCO(gt_json)
    results = []
    id_by_name = {os.path.splitext(im["file_name"])[0]: im_id for im_id, im in coco.imgs.items()}
    for p in preds:
        stem = os.path.splitext(os.path.basename(p["img_path"]))[0]
        img_id = id_by_name.get(stem)
        if img_id is None:
            continue
        inst = p["pred_instances"]
        scores = inst["scores"].cpu().numpy() if hasattr(inst["scores"], "cpu") else np.asarray(inst["scores"])
        masks = inst["masks"]
        for k in range(len(scores)):
            if scores[k] < score_t:
                continue
            rle = masks[k]
            rle = {"size": rle["size"], "counts": rle["counts"].decode() if isinstance(rle["counts"], bytes) else rle["counts"]}
            results.append({"image_id": img_id, "category_id": 1,
                            "segmentation": rle, "score": float(scores[k])})
    if not results:
        return {}, coco
    dt = coco.loadRes(results)
    ev = COCOeval(coco, dt, "segm")
    ev.evaluate(); ev.accumulate(); ev.summarize()
    s = ev.stats
    return {"AP": s[0], "AP50": s[1], "AP75": s[2], "AP_s": s[3], "AP_m": s[4], "AP_l": s[5]}, coco


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--dataset", choices=["pseg", "hrsid"], default="pseg")
    ap.add_argument("--score_t", type=float, default=0.3, help="for boundary/split-merge (not AP)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch  # noqa: F401
    preds = pickle.load(open(args.preds, "rb"))
    inshore, _ = _load_inshore(args.dataset)

    apm, coco = coco_ap(args.gt, preds)

    # boundary / split-merge / mask quality, per stratum.
    # P1-6 FIX: d=2 is the fine band; "std" is the per-image standard d = 2% of
    # the image diagonal (Cheng 2021). Fixed d=23 was correct ONLY for 800x800
    # HRSID; PSeg (~600px diag) needs d~=12, which is why old PSeg degeneracy 89%
    # was wrong (true 67%). Now per-image.
    by = {"all": [], "inshore": [], "offshore": []}
    biou = {2: [], "std": []}
    biou_strat = {"inshore": {2: [], "std": []}, "offshore": {2: [], "std": []}}
    bf1, det_ious, min_dims, degen_std = [], [], [], []  # degen_std: per-instance min_dim < 2*d_std
    id_by_name = {os.path.splitext(im["file_name"])[0]: im_id for im_id, im in coco.imgs.items()}

    for p in preds:
        stem = os.path.splitext(os.path.basename(p["img_path"]))[0]
        img_id = id_by_name.get(stem)
        if img_id is None:
            continue
        im = coco.imgs[img_id]
        h, w = im["height"], im["width"]
        sub = ("inshore" if (inshore and stem in inshore) else "offshore") if inshore else "all"
        gt_anns = coco.imgToAnns.get(img_id, [])
        gt_masks = [mask_util.decode(_rle(a["segmentation"], h, w)).astype(bool) for a in gt_anns]
        d_std = standard_d(w, h)  # per-image standard Boundary IoU dilation
        for a in gt_anns:
            x, y, bw, bh = a["bbox"]
            min_dims.append(min(bw, bh))
            degen_std.append(min(bw, bh) < 2 * d_std)  # degenerate at the correct per-image d

        inst = p["pred_instances"]
        scores = inst["scores"].cpu().numpy() if hasattr(inst["scores"], "cpu") else np.asarray(inst["scores"])
        keep = np.nonzero(scores >= args.score_t)[0]
        pred_masks = [mask_util.decode(inst["masks"][k]).astype(bool) for k in keep]

        sm = split_merge_rates(pred_masks, gt_masks)
        by["all"].append(sm)
        if sub in ("inshore", "offshore"):
            by[sub].append(sm)

        # per-GT best pred mask-IoU + boundary IoU at the matched pred
        for g in gt_masks:
            best, bi = 0.0, None
            for pm in pred_masks:
                mi = mask_iou(g, pm)
                if mi > best:
                    best, bi = mi, pm
            if bi is not None and best >= 0.5:
                det_ious.append(best)
                gu = g.astype(np.uint8); pu = bi.astype(np.uint8)
                for dk, dv in (("std", d_std), (2, 2)):
                    v = boundary_iou(gu, pu, dv)
                    biou[dk].append(v)
                    if sub in biou_strat:
                        biou_strat[sub][dk].append(v)
                bf1.append(boundary_f1(gu, pu))

    def mean(x):
        return float(np.mean(x)) if len(x) else float("nan")

    out = {
        "coco_ap_segm": apm,
        "boundary_iou": {k: mean(biou[k]) for k in (2, "std")},  # 2=fine, std=per-image 2%-diag
        "boundary_iou_inshore": {k: mean(biou_strat["inshore"][k]) for k in (2, "std")},
        "boundary_iou_offshore": {k: mean(biou_strat["offshore"][k]) for k in (2, "std")},
        "boundary_degeneracy_frac_std": float(np.mean(degen_std)) if degen_std else float("nan"),  # at correct per-image d
        "boundary_f1_1px": mean(bf1),
        "detected_mask_iou_median": float(np.median(det_ious)) if det_ious else float("nan"),
        "split_merge_all": aggregate(by["all"]),
        "split_merge_inshore": aggregate(by["inshore"]) if by["inshore"] else None,
        "split_merge_offshore": aggregate(by["offshore"]) if by["offshore"] else None,
    }
    print(json.dumps(out, indent=1, ensure_ascii=False))
    if args.out:
        json.dump(out, open(args.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print("written:", args.out)


if __name__ == "__main__":
    main()
