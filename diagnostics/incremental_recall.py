# -*- coding: utf-8 -*-
"""Incremental recall: do evidence seeds cover GT instances that the trained
carrier (A0) misses? This is the decision-relevant gate for the NQS seeding
pillar (verdict-3 required change 2): absolute seed recall is NOT the
investment criterion — marginal coverage over learned proposals is.

Per GT instance:
  A0-hit   : any A0 predicted box with score>=SCORE_T and IoU>=0.5
  seed-hit : v2 criterion (seed box IoU>=0.3 OR seed center inside GT box)
Reported per stratum:
  A0 recall | seed recall | union recall | marginal = seeds' coverage of
  A0-missed GTs (the number that justifies or kills NQS).

Usage: E:\\sar_envs\\env_seal_gpu\\Scripts\\python.exe diagnostics\\incremental_recall.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mne.engine import mne_auc_field  # noqa: E402
from seeding.evidence_seeds import calibrate_thresholds, extract_seeds  # noqa: E402
from protocol.datasets import load_one, _subset_stems, PSEG_ROOT  # noqa: E402

SCORE_T = 0.05
PREDS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "a0_pseg_seed0", "preds.pkl")
TEST_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "pseg_ssdd", "annotations", "test.json")


def _iou_xyxy(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _proc(args):
    stem, subset, pred_boxes = args
    sample = load_one(stem, subset)
    if sample is None or not sample.instances:
        return None
    field = mne_auc_field(sample.image)
    cal = calibrate_thresholds(field)
    seeds = extract_seeds(field, cal["tau"])
    out = []
    for m in sample.instances:
        ys, xs = np.nonzero(m)
        if len(ys) == 0:
            continue
        gt_xyxy = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        a0_hit = any(_iou_xyxy(gt_xyxy, pb) >= 0.5 for pb in pred_boxes)
        seed_hit = False
        for (cy, cx, y0, x0, y1, x1, s, *_r) in seeds:
            center_in = gt_xyxy[1] <= cy < gt_xyxy[3] and gt_xyxy[0] <= cx < gt_xyxy[2]
            if center_in or _iou_xyxy((x0, y0, x1, y1), gt_xyxy) >= 0.3:
                seed_hit = True
                break
        out.append((subset, a0_hit, seed_hit))
    return out


def main():
    import torch  # needed to unpickle mmdet tensors
    with open(PREDS, "rb") as fh:
        preds = pickle.load(fh)
    coco = json.load(open(TEST_JSON))
    id2name = {im["id"]: im["file_name"] for im in coco["images"]}
    inshore, _ = _subset_stems(PSEG_ROOT)

    work = []
    for p in preds:
        name = os.path.basename(p.get("img_path") or id2name[p["img_id"]])
        stem = os.path.splitext(name)[0]
        inst = p["pred_instances"]
        boxes = inst["bboxes"].cpu().numpy() if hasattr(inst["bboxes"], "cpu") else np.asarray(inst["bboxes"])
        scores = inst["scores"].cpu().numpy() if hasattr(inst["scores"], "cpu") else np.asarray(inst["scores"])
        keep = scores >= SCORE_T
        subset = "inshore" if stem in inshore else "offshore"
        work.append((stem, subset, boxes[keep].tolist()))

    # single-process + per-image JSONL checkpoint: 232 imgs x ~1.3s, avoids
    # Windows multiprocessing OOM, and a restart resumes where it left off.
    import time
    ckpt = os.path.join(os.path.dirname(PREDS), "incremental_recall.jsonl")
    done = set()
    rows = []
    if os.path.exists(ckpt):
        with open(ckpt, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    done.add(rec["stem"])
                    rows.extend((r["subset"], r["a0"], r["seed"]) for r in rec["gts"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"resume: {len(done)} images already done", flush=True)

    t0 = time.time()
    todo = [w for w in work if w[0] not in done]
    with open(ckpt, "a", encoding="utf-8") as ck:
        for i, w in enumerate(todo):
            r = _proc(w)
            gts = [{"subset": x[0], "a0": x[1], "seed": x[2]} for x in (r or [])]
            ck.write(json.dumps({"stem": w[0], "gts": gts}) + "\n")
            ck.flush()
            if r:
                rows.extend(r)
            if (i + 1) % 40 == 0:
                print(f"[{i+1}/{len(todo)}] {time.time()-t0:.0f}s", flush=True)

    print(f"{'stratum':<10}{'n_gt':>6}{'A0':>9}{'seeds':>9}{'union':>9}{'A0-missed':>11}{'marginal':>10}")
    for sub in ["inshore", "offshore", "all"]:
        sel = [r for r in rows if sub == "all" or r[0] == sub]
        n = len(sel)
        a0 = sum(1 for r in sel if r[1])
        sd = sum(1 for r in sel if r[2])
        un = sum(1 for r in sel if r[1] or r[2])
        missed = [r for r in sel if not r[1]]
        marg = sum(1 for r in missed if r[2])
        print(f"{sub:<10}{n:>6}{a0/n:>9.3f}{sd/n:>9.3f}{un/n:>9.3f}{len(missed):>11}"
              f"{(marg/len(missed) if missed else float('nan')):>10.3f}")
    out = os.path.join(os.path.dirname(PREDS), "incremental_recall.json")
    json.dump([{"subset": r[0], "a0": r[1], "seed": r[2]} for r in rows], open(out, "w"))
    print("written:", out)


if __name__ == "__main__":
    main()
