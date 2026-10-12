# -*- coding: utf-8 -*-
"""Zero-training split attack: does inference-time mask-NMS dedup cut the
split rate (one GT covered by >=2 predictions = duplicate queries firing)?

Mask2Former relies on bipartite matching to avoid duplicates and ships NO NMS.
On dense/small SAR ships it still produces duplicates (PSeg split 0.303-0.558,
HRSID 0.303). Mask-NMS (drop a mask if it overlaps a higher-scoring kept mask
at IoU>=thr) is a free, evidence-free, training-free split fix. We recompute
split/merge + COCO AP before vs after, on existing preds.pkl.

D2 was WEAK (affinity AUROC 0.652) so split is attacked by geometry, not
evidence -- aligned with the user's "don't rely on evidence discrimination".

Usage: python diagnostics\\mask_nms_split.py --preds runs\\a0_pseg_seed0\\preds.pkl \\
    --gt data\\pseg_ssdd\\annotations\\test.json [--iou 0.5] [--score 0.3]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import sys

import numpy as np
from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from protocol.split_merge import split_merge_rates, aggregate  # noqa: E402


def mask_iou_rle(a, b):
    return float(mask_util.iou([a], [b], [0])[0, 0])


def mask_nms(rles, scores, iou_thr):
    """Greedy mask-NMS: keep highest-score, drop later masks with IoU>=thr."""
    order = np.argsort(-scores)
    keep, suppressed = [], np.zeros(len(scores), bool)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        for j in order:
            if j == i or suppressed[j]:
                continue
            if mask_iou_rle(rles[i], rles[j]) >= iou_thr:
                suppressed[j] = True
    return sorted(keep)


def to_rle(m, h, w):
    if isinstance(m, dict):
        return {"size": m["size"], "counts": m["counts"].decode() if isinstance(m["counts"], bytes) else m["counts"]}
    r = mask_util.encode(np.asfortranarray(m.astype(np.uint8)))
    return {"size": r["size"], "counts": r["counts"].decode() if isinstance(r["counts"], bytes) else r["counts"]}


def coco_eval(gt_json, results):
    coco = COCO(gt_json)
    if not results:
        return {}
    dt = coco.loadRes(results)
    ev = COCOeval(coco, dt, "segm")
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return {"AP": ev.stats[0], "AP50": ev.stats[1], "AP75": ev.stats[2], "AP_s": ev.stats[3]}


def build_results(preds, name2id, score_t, nms_iou=None):
    results, per_img_masks = [], {}
    for p in preds:
        stem = os.path.splitext(os.path.basename(p["img_path"]))[0]
        if stem not in name2id:
            continue
        inst = p["pred_instances"]
        scores = inst["scores"].cpu().numpy() if hasattr(inst["scores"], "cpu") else np.asarray(inst["scores"])
        masks = inst["masks"]
        keep_idx = np.nonzero(scores >= score_t)[0]
        rles = [masks[k] if isinstance(masks[k], dict) else mask_util.encode(np.asfortranarray(np.asarray(masks[k]).astype(np.uint8))) for k in keep_idx]
        sc = scores[keep_idx]
        if nms_iou is not None and len(rles) > 1:
            kept = mask_nms(rles, sc, nms_iou)
            keep_idx, rles, sc = keep_idx[kept], [rles[i] for i in kept], sc[kept]
        per_img_masks[stem] = [mask_util.decode(r).astype(bool) for r in rles]
        for k, r, s in zip(keep_idx, rles, sc):
            results.append({"image_id": name2id[stem], "category_id": 1,
                            "segmentation": to_rle(r, 0, 0), "score": float(s)})
    return results, per_img_masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--score", type=float, default=0.3)
    args = ap.parse_args()
    import torch  # noqa
    preds = pickle.load(open(args.preds, "rb"))
    coco = COCO(args.gt)
    name2id = {os.path.splitext(im["file_name"])[0]: i for i, im in coco.imgs.items()}
    gts = {}
    for a in coco.anns.values():
        stem = os.path.splitext(coco.imgs[a["image_id"]]["file_name"])[0]
        h, w = coco.imgs[a["image_id"]]["height"], coco.imgs[a["image_id"]]["width"]
        seg = a["segmentation"]
        rle = mask_util.merge(mask_util.frPyObjects(seg, h, w)) if isinstance(seg, list) else seg
        gts.setdefault(stem, []).append(mask_util.decode(rle).astype(bool))

    for tag, nms in [("baseline (no NMS)", None), (f"mask-NMS iou>={args.iou}", args.iou)]:
        results, pm = build_results(preds, name2id, args.score, nms)
        sm = aggregate([split_merge_rates(pm.get(s, []), gts.get(s, [])) for s in gts])
        apm = coco_eval(args.gt, results)
        print(f"\n=== {tag} ===")
        print(f"  AP={apm.get('AP',0):.4f} AP50={apm.get('AP50',0):.4f} AP75={apm.get('AP75',0):.4f} AP_s={apm.get('AP_s',0):.4f}")
        print(f"  split={sm['split_rate']:.4f} merge={sm['merge_rate']:.4f} purity={sm['mean_purity']:.4f} n_pred={sm['n_pred']}")


if __name__ == "__main__":
    main()
