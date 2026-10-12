# -*- coding: utf-8 -*-
"""DAOM-Seed deciding sub-gate (CPU): do MNE evidence seeds cover the A0-MISSED
HRSID GTs? The recall-threshold gate showed HRSID misses are mostly NO-COVERAGE
(no box at IoU>=0.5), not suppressed queries. DAOM's evidence-seed injection can
only rescue them if the MNE seeds actually PROPOSE those missed locations. This
measures exactly that marginal coverage (mirrors diagnostics/incremental_recall.py
but for HRSID, loading images by img_path and computing MNE on the fly).

Per GT: a0_hit = any A0 box (score>=0.05) IoU>=0.5; seed_hit = MNE seed center in
GT box OR seed box IoU>=0.3. Marginal = fraction of A0-missed GTs covered by seeds
(the number that justifies or kills the DAOM evidence-seed premise on HRSID).

Usage: PYTHONPATH=repo python diagnostics\\hrsid_seed_marginal.py [--limit 300]
"""
from __future__ import annotations
import argparse, os, json, pickle
import numpy as np
from pycocotools.coco import COCO

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from mne.engine import mne_auc_field  # noqa: E402
from seeding.evidence_seeds import calibrate_thresholds, extract_seeds  # noqa: E402

SCORE_T = 0.05


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


def main():
    import torch  # noqa
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--shuffle", action="store_true", help="representative random sample (fix bias)")
    args = ap.parse_args()
    coco = COCO(os.path.join(BASE, "data", "hrsid", "annotations", "official_test.json"))
    id_by_name = {os.path.splitext(os.path.basename(im["file_name"]))[0]: i for i, im in coco.imgs.items()}
    preds = pickle.load(open(os.path.join(BASE, "runs", "a0_hrsid_seed0", "preds.pkl"), "rb"))
    if args.shuffle:
        import random
        random.Random(0).shuffle(preds)

    # median GT-count for density stratification
    counts = []
    for p in preds:
        iid = id_by_name.get(os.path.splitext(os.path.basename(p["img_path"]))[0])
        if iid is not None:
            counts.append(len(coco.imgToAnns.get(iid, [])))
    med = float(np.median([c for c in counts if c > 0])) if counts else 2.0

    rows = []  # (a0_hit, seed_hit, dense)
    done = 0
    for p in preds:
        if done >= args.limit:
            break
        stem = os.path.splitext(os.path.basename(p["img_path"]))[0]
        iid = id_by_name.get(stem)
        if iid is None:
            continue
        gts = coco.imgToAnns.get(iid, [])
        if not gts:
            continue
        dense = len(gts) >= med
        if not os.path.exists(p["img_path"]):
            continue
        img = cv2.imread(p["img_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        field = mne_auc_field(img.astype(np.float32))
        cal = calibrate_thresholds(field)
        seeds = extract_seeds(field, cal["tau"])
        inst = p["pred_instances"]
        bx = inst["bboxes"].cpu().numpy() if hasattr(inst["bboxes"], "cpu") else np.asarray(inst["bboxes"])
        sc = inst["scores"].cpu().numpy() if hasattr(inst["scores"], "cpu") else np.asarray(inst["scores"])
        keep = sc >= SCORE_T
        bxk = bx[keep]
        for a in gts:
            x, y, w, h = a["bbox"]
            g = (x, y, x+w, y+h)
            a0_hit = any(iou(g, pb) >= 0.5 for pb in bxk)
            seed_hit = False
            for s in seeds:
                cy, cx, y0, x0, y1, x1 = s[0], s[1], s[2], s[3], s[4], s[5]
                if (g[1] <= cy < g[3] and g[0] <= cx < g[2]) or iou((x0, y0, x1, y1), g) >= 0.3:
                    seed_hit = True
                    break
            rows.append((a0_hit, seed_hit, dense))
        done += 1
        if done % 50 == 0:
            print(f"[{done}/{args.limit}] images", flush=True)

    def report(sel, tag):
        n = len(sel)
        if n == 0:
            print(f"  [{tag}] no GT"); return None
        a0 = sum(r[0] for r in sel); sd = sum(r[1] for r in sel)
        missed = [r for r in sel if not r[0]]
        marg = sum(r[1] for r in missed)
        m = marg/len(missed) if missed else float('nan')
        print(f"  [{tag:6}] n={n:4} A0recall={a0/n:.3f} seedrecall={sd/n:.3f} "
              f"A0-missed={len(missed)} marginal={m:.3f}")
        return {"n": n, "a0_recall": a0/n, "seed_recall": sd/n,
                "a0_missed": len(missed), "marginal": (m if missed else None)}
    print(f"\n=== HRSID seed-marginal premise ({done} imgs, {len(rows)} GT, "
          f"shuffle={args.shuffle}, density-median={med:.0f}) ===")
    out = {"shuffle": args.shuffle, "median_gt": med,
           "all": report(rows, "all"),
           "dense": report([r for r in rows if r[2]], "dense"),
           "sparse": report([r for r in rows if not r[2]], "sparse")}
    print("  >> DECISION: dense marginal is the decisive number (coverage fails in dense scenes).")
    print("     >=0.4 -> seeds propose the dense misses -> DAOM evidence premise HOLDS;")
    print("     <0.2  -> seeds miss them too -> DAOM headline premise DIES, pivot to fallback.")
    json.dump(out, open(os.path.join(BASE, "runs", "a0_hrsid_seed0", "seed_marginal.json"), "w"))


if __name__ == "__main__":
    main()
