# -*- coding: utf-8 -*-
"""DAOM-Seed pre-build KILL GATE (CPU, existing preds only): is HRSID's coverage
miss QUERY-LEVEL (a suppressed/low-ranked query exists near the missed GT, which
one-to-many supervision can rescue) or NO-COVERAGE / budget-bound (no box proposes
the location, which o2m cannot fix)?

Method on the HRSID A0 preds: for each GT box, find the best-IoU predicted box
(IoU>=0.5); record its score. Recall at descending score thresholds:
  recall@0.30 = operating recall
  recall@0.00 = the 100-query COVERAGE ceiling (any box, IoU>=0.5, regardless of score)
The gap recall@0.00 - recall@0.30 = GTs with a SUPPRESSED query = query-level
recoverable headroom (the DAOM premise). Also reports GT-count per image (a budget
check: images with >Nq GTs are query-budget-bound, which o2m does not fix at test).

Caveat: A0 here is Mask2Former (100 queries); the DAOM carrier is DINO (300). This
is a proxy on available HRSID detection preds; read qualitatively.

Usage: PYTHONPATH=repo python diagnostics\\hrsid_coverage_premise.py [--preds ...] [--nq 100]
"""
from __future__ import annotations
import argparse, os, json, pickle
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as mask_util

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def iou_xyxy(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


def main():
    import torch  # noqa
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=os.path.join(BASE, "runs", "a0_hrsid_seed0", "preds.pkl"))
    ap.add_argument("--gt", default=os.path.join(BASE, "data", "hrsid", "annotations", "official_test.json"))
    ap.add_argument("--nq", type=int, default=100, help="query budget of the carrier")
    args = ap.parse_args()

    coco = COCO(args.gt)
    id_by_name = {os.path.splitext(os.path.basename(im["file_name"]))[0]: i for i, im in coco.imgs.items()}
    preds = pickle.load(open(args.preds, "rb"))

    thr = [0.30, 0.10, 0.05, 0.01, 0.0]
    best_scores = []     # per GT: best score among IoU>=0.5 preds, or -1 if none
    gt_per_img = []
    for p in preds:
        stem = os.path.splitext(os.path.basename(p["img_path"]))[0]
        iid = id_by_name.get(stem)
        if iid is None:
            continue
        gts = coco.imgToAnns.get(iid, [])
        gt_per_img.append(len(gts))
        if not gts:
            continue
        inst = p["pred_instances"]
        bx = inst["bboxes"].cpu().numpy() if hasattr(inst["bboxes"], "cpu") else np.asarray(inst["bboxes"])
        sc = inst["scores"].cpu().numpy() if hasattr(inst["scores"], "cpu") else np.asarray(inst["scores"])
        for a in gts:
            x, y, w, h = a["bbox"]
            g = (x, y, x+w, y+h)
            best, bs = 0.5, -1.0
            for k in range(len(bx)):
                i = iou_xyxy(g, bx[k])
                if i >= best:
                    if sc[k] > bs:
                        bs = float(sc[k])
            best_scores.append(bs)

    bs = np.array(best_scores)
    n = len(bs)
    gpi = np.array(gt_per_img)
    print(f"=== HRSID coverage premise: {n} GT instances over {len(gpi)} images ===")
    print(f"  preds: {os.path.relpath(args.preds, BASE)} | query budget Nq={args.nq}")
    print(f"  recall vs score threshold (any pred IoU>=0.5 with score>=t):")
    for t in thr:
        r = float((bs >= t).mean())
        print(f"    recall@{t:<5} = {r:.3f}")
    r_op = float((bs >= 0.30).mean()); r_ceil = float((bs >= 0.0).mean())
    supp = r_ceil - r_op
    print(f"  >> QUERY-LEVEL recoverable headroom (recall@0.0 - recall@0.30) = {supp:+.3f}")
    print(f"     (suppressed queries that one-to-many can rescue; large = DAOM premise holds)")
    no_box = float((bs < 0).mean())
    print(f"  >> GTs with NO covering box at all (IoU>=0.5)             = {no_box:.3f}")
    print(f"     (no-coverage / feature-or-budget gap; o2m cannot fix at test)")
    print(f"  GT-per-image: mean {gpi.mean():.1f}, median {int(np.median(gpi))}, max {gpi.max()}, "
          f">{args.nq}q imgs: {int((gpi > args.nq).sum())}/{len(gpi)} ({100*(gpi>args.nq).mean():.1f}%)")
    out = os.path.join(os.path.dirname(args.preds), "coverage_premise.json")
    json.dump({"n": n, "recall_at": {str(t): float((bs >= t).mean()) for t in thr},
               "suppressed_headroom": supp, "no_box_frac": no_box,
               "gt_per_img": {"mean": float(gpi.mean()), "max": int(gpi.max()),
                              "over_nq_frac": float((gpi > args.nq).mean())}}, open(out, "w"))
    print("written:", os.path.relpath(out, BASE))


if __name__ == "__main__":
    main()
