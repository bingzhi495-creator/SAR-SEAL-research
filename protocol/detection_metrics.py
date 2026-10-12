# -*- coding: utf-8 -*-
"""Detection metrics in the user's benchmark format, so the SEAL detection
track (SeedDINO / evidence-seeded) drops directly into their comparison table:
  mAP50, mAP50:95, Precision, Recall(Pd), False-alarm(1-P)

COCO mAP via pycocotools; P/R(Pd)/FA reported at the F1-optimal confidence
threshold (YOLO-val convention) so the operating point matches the benchmark's
YOLO rows. Stratifiable by inshore/offshore.

Usage:
  python protocol\\detection_metrics.py --preds runs\\seeddino_pred_seed0\\preds.pkl \\
      --gt data\\pseg_ssdd\\annotations\\test.json [--inshore-split pseg]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


def pr_at_thresholds(preds_by_img, gts_by_img, iou_t=0.5, n_steps=50):
    """Sweep confidence thresholds; return per-threshold P, R, F1 (micro over
    all images). Greedy 1-1 matching at iou_t per image."""
    confs = np.linspace(0.05, 0.95, n_steps)
    out = []
    for c in confs:
        tp = fp = fn = 0
        for img, gts in gts_by_img.items():
            dets = [d for d in preds_by_img.get(img, []) if d[4] >= c]
            dets = sorted(dets, key=lambda d: -d[4])
            matched = set()
            for d in dets:
                best, bj = iou_t, -1
                for j, g in enumerate(gts):
                    if j in matched:
                        continue
                    iou = _iou(d[:4], g)
                    if iou >= best:
                        best, bj = iou, j
                if bj >= 0:
                    tp += 1; matched.add(bj)
                else:
                    fp += 1
            fn += len(gts) - len(matched)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2*p*r/(p+r) if (p+r) else 0.0
        out.append((c, p, r, f1))
    return out


def coco_map(gt_json, results):
    coco = COCO(gt_json)
    if not results:
        return {}, coco
    dt = coco.loadRes(results)
    ev = COCOeval(coco, dt, "bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()
    s = ev.stats
    return {"mAP50_95": float(s[0]), "mAP50": float(s[1]), "mAP75": float(s[2])}, coco


def load_preds(path):
    preds = pickle.load(open(path, "rb"))
    by_img = {}
    results = []
    for p in preds:
        stem = os.path.splitext(os.path.basename(p["img_path"]))[0]
        inst = p["pred_instances"]
        boxes = inst["bboxes"].cpu().numpy() if hasattr(inst["bboxes"], "cpu") else np.asarray(inst["bboxes"])
        scores = inst["scores"].cpu().numpy() if hasattr(inst["scores"], "cpu") else np.asarray(inst["scores"])
        by_img[stem] = [(b[0], b[1], b[2], b[3], float(s)) for b, s in zip(boxes, scores)]
    return preds, by_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--inshore-split", choices=["pseg", "none"], default="none")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import torch  # noqa

    coco = COCO(args.gt)
    id2name = {i: os.path.splitext(im["file_name"])[0] for i, im in coco.imgs.items()}
    name2id = {v: k for k, v in id2name.items()}
    gts_by_img = {}
    for a in coco.anns.values():
        x, y, w, h = a["bbox"]
        gts_by_img.setdefault(id2name[a["image_id"]], []).append([x, y, x+w, y+h])

    preds, by_img = load_preds(args.preds)
    results = []
    for p in preds:
        stem = os.path.splitext(os.path.basename(p["img_path"]))[0]
        if stem not in name2id:
            continue
        inst = p["pred_instances"]
        boxes = inst["bboxes"].cpu().numpy() if hasattr(inst["bboxes"], "cpu") else np.asarray(inst["bboxes"])
        scores = inst["scores"].cpu().numpy() if hasattr(inst["scores"], "cpu") else np.asarray(inst["scores"])
        for b, s in zip(boxes, scores):
            results.append({"image_id": name2id[stem], "category_id": 1,
                            "bbox": [float(b[0]), float(b[1]), float(b[2]-b[0]), float(b[3]-b[1])],
                            "score": float(s)})

    strata = {"all": gts_by_img}
    if args.inshore_split == "pseg":
        from protocol.datasets import _subset_stems, PSEG_ROOT
        inshore, _ = _subset_stems(PSEG_ROOT)
        strata = {
            "all": gts_by_img,
            "inshore": {k: v for k, v in gts_by_img.items() if k in inshore},
            "offshore": {k: v for k, v in gts_by_img.items() if k not in inshore},
        }

    mp, _ = coco_map(args.gt, results)
    report = {"coco_map_all": mp}
    print(f"COCO mAP: mAP50={mp.get('mAP50',0):.4f} mAP50:95={mp.get('mAP50_95',0):.4f}\n")
    print(f"{'stratum':<10}{'P':>8}{'Pd(R)':>8}{'FA':>8}{'F1':>8}{'@conf':>8}")
    for name, gts in strata.items():
        pr = pr_at_thresholds(by_img, gts)
        best = max(pr, key=lambda x: x[3])  # F1-optimal
        c, p, r, f1 = best
        report[name] = {"P": p, "Pd": r, "FA": 1-p, "F1": f1, "conf": c}
        print(f"{name:<10}{p:>8.4f}{r:>8.4f}{(1-p):>8.4f}{f1:>8.4f}{c:>8.2f}")
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)
        print("written:", args.out)


if __name__ == "__main__":
    main()
