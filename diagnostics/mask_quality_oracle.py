# -*- coding: utf-8 -*-
"""Oracle-box mask-quality diagnostic (verdict-4 required change 2).

Question: of A0's AP75 loss on inshore, how much is RESOLUTION-bound (mask
quality of *detected* instances) vs detection-bound (misses / FP)?

Method: for each GT, find A0's best detection by box IoU >= 0.5; for matched
pairs compute mask IoU (the model's mask quality when it DOES detect). Stratify
by COCO size and inshore/offshore. Report:
  * median/mean mask IoU of detected instances
  * fraction of DETECTED instances with mask IoU in [0.5, 0.75) -- these are
    detected but fail AP75 purely on mask quality = the ceiling a refinement
    head could recover
Gate (verdict-4): if that recoverable fraction is < ~30% of the inshore AP75
gap, demote the refinement head to an ordinary ablation row.

Uses the existing preds.pkl + COCO test GT. No GPU, no training.
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np
from pycocotools import mask as mask_util

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol.datasets import _subset_stems, PSEG_ROOT  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDS = os.path.join(BASE, "runs", "a0_pseg_seed0", "preds.pkl")
TEST_JSON = os.path.join(BASE, "data", "pseg_ssdd", "annotations", "test.json")
SCORE_T = 0.3   # detection confidence for "A0 detected it"


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


def main():
    import torch  # noqa: F401  (unpickle)
    preds = pickle.load(open(PREDS, "rb"))
    coco = json.load(open(TEST_JSON))
    inshore, _ = _subset_stems(PSEG_ROOT)

    imgs = {im["id"]: im for im in coco["images"]}
    gts_by_img = {}
    for a in coco["annotations"]:
        gts_by_img.setdefault(a["image_id"], []).append(a)

    # index preds by file stem
    pred_by_stem = {}
    for p in preds:
        stem = os.path.splitext(os.path.basename(p["img_path"]))[0]
        pred_by_stem[stem] = p

    rows = []  # (subset, size_bucket, matched(bool), mask_iou)
    for img_id, im in imgs.items():
        stem = os.path.splitext(im["file_name"])[0]
        subset = "inshore" if stem in inshore else "offshore"
        p = pred_by_stem.get(stem)
        gts = gts_by_img.get(img_id, [])
        if p is None:
            continue
        inst = p["pred_instances"]
        pscores = inst["scores"].cpu().numpy() if hasattr(inst["scores"], "cpu") else np.asarray(inst["scores"])
        pboxes = inst["bboxes"].cpu().numpy() if hasattr(inst["bboxes"], "cpu") else np.asarray(inst["bboxes"])
        pmasks = inst["masks"]  # list of RLE
        keep = np.nonzero(pscores >= SCORE_T)[0]

        for g in gts:
            x, y, w, h = g["bbox"]
            gbox = [x, y, x+w, y+h]
            area = g["area"]
            bucket = "S" if area < 32*32 else ("M" if area < 96*96 else "L")
            # GT mask RLE
            grle = mask_util.frPyObjects(g["segmentation"], im["height"], im["width"])
            grle = mask_util.merge(grle) if isinstance(grle, list) else grle
            best_iou, best_k = 0.0, -1
            for k in keep:
                if box_iou(gbox, pboxes[k]) >= 0.5:
                    miou = float(mask_util.iou([pmasks[k]], [grle], [0])[0, 0])
                    if miou > best_iou:
                        best_iou, best_k = miou, k
            matched = best_k >= 0
            rows.append((subset, bucket, matched, best_iou if matched else 0.0))

    # report
    print(f"A0 mask-quality oracle (score_t={SCORE_T})\n")
    hdr = f"{'stratum':<18}{'n_gt':>6}{'det_rate':>10}{'maskIoU_med':>13}{'maskIoU_mean':>13}{'det&IoU<0.75':>14}"
    print(hdr)
    summary = {}
    for sub in ["inshore", "offshore", "all"]:
        for bk in ["S", "M", "L", "*"]:
            sel = [r for r in rows if (sub == "all" or r[0] == sub) and (bk == "*" or r[1] == bk)]
            if not sel:
                continue
            n = len(sel)
            det = [r for r in sel if r[2]]
            det_rate = len(det)/n
            ious = [r[3] for r in det]
            med = float(np.median(ious)) if ious else 0.0
            mean = float(np.mean(ious)) if ious else 0.0
            # detected but failing AP75 purely on mask quality, as frac of ALL gt
            recoverable = sum(1 for r in det if r[3] < 0.75) / n
            key = f"{sub}/{bk}"
            summary[key] = {"n": n, "det_rate": det_rate, "iou_med": med,
                            "iou_mean": mean, "recoverable_ap75_frac": recoverable}
            label = f"{sub}/{bk}"
            print(f"{label:<18}{n:>6}{det_rate:>10.3f}{med:>13.3f}{mean:>13.3f}{recoverable:>14.3f}")
    json.dump(summary, open(os.path.join(os.path.dirname(PREDS), "mask_quality_oracle.json"), "w"), indent=1)

    # verdict
    print("\n--- VERDICT (refinement head, verdict-4) ---")
    ins = summary.get("inshore/*", {})
    rec = ins.get("recoverable_ap75_frac", 0.0)
    det = ins.get("det_rate", 0.0)
    miss = 1 - det
    print(f"inshore: detection miss rate = {miss:.3f}  |  detected-but-maskIoU<0.75 = {rec:.3f}")
    print(f"  -> of the inshore AP75 gap, ~{rec/(rec+miss+1e-9):.0%} is mask-resolution-bound, ~{miss/(rec+miss+1e-9):.0%} is detection-bound")
    if rec < 0.15:
        print("  -> recoverable fraction LOW: refinement head -> ordinary ablation row; prioritize detection (seeding/hard-neg)")
    else:
        print("  -> recoverable fraction MEANINGFUL: refinement head justified for AP75")


if __name__ == "__main__":
    main()
