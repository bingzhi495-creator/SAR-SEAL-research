# -*- coding: utf-8 -*-
"""Precompute NQS evidence seeds offline for a dataset split.

For each image: MNE AUC field -> budget-calibrated thresholds -> seed boxes.
Saved as seeds_<split>.json: {stem: [[x0,y0,x1,y1], ...]} (abs pixel xyxy).
Consumed at train time by LoadSeeds (carrier/load_seeds.py) for SeedDINO 'pred'
mode. Resumable JSONL checkpoint; single-process to avoid the histogram-field
OOM seen with multiprocessing.

Usage:
  python seeding\\precompute_seeds.py --split pseg_train
  python seeding\\precompute_seeds.py --split pseg_test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mne.engine import mne_auc_field  # noqa: E402
from seeding.evidence_seeds import calibrate_thresholds, extract_seeds  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = {
    "pseg_train": (r"E:\PSeg_SSDD\voc_style\JPEGImages_train", None),
    "pseg_test": (r"E:\PSeg_SSDD\voc_style\JPEGImages_test", None),
    "hrsid_train": (r"D:\SAR\UNet++\HRSID_JPG\HRSID_JPG\JPEGImages1",
                    os.path.join(BASE, "data", "hrsid", "annotations", "official_train.json")),
    "hrsid_test": (r"D:\SAR\UNet++\HRSID_JPG\HRSID_JPG\JPEGImages1",
                   os.path.join(BASE, "data", "hrsid", "annotations", "official_test.json")),
}


def _stems(split):
    img_dir, ann = SPLITS[split]
    if ann:  # HRSID: only images in this split's json
        j = json.load(open(ann))
        return img_dir, [im["file_name"] for im in j["images"]]
    return img_dir, sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".jpg"))


def seeds_for_image(img_path: str):
    img = np.array(Image.open(img_path).convert("L"))
    field = mne_auc_field(img)
    cal = calibrate_thresholds(field)
    seeds = extract_seeds(field, cal["tau"])
    # (cy,cx,y0,x0,y1,x1,scale,evid) -> xyxy abs
    return [[int(s[3]), int(s[2]), int(s[5]), int(s[4])] for s in seeds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=list(SPLITS))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out_dir = os.path.join(BASE, "data", "seeds")
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, f"seeds_{args.split}.jsonl")
    final = os.path.join(out_dir, f"seeds_{args.split}.json")

    img_dir, files = _stems(args.split)
    if args.limit:
        files = files[:args.limit]

    done = {}
    if os.path.exists(ckpt):
        for line in open(ckpt, encoding="utf-8"):
            try:
                r = json.loads(line); done[r["stem"]] = r["seeds"]
            except json.JSONDecodeError:
                pass
        print(f"resume: {len(done)} done", flush=True)

    t0 = time.time()
    with open(ckpt, "a", encoding="utf-8") as ck:
        for i, fn in enumerate(files):
            stem = os.path.splitext(fn)[0]
            if stem in done:
                continue
            try:
                seeds = seeds_for_image(os.path.join(img_dir, fn))
            except Exception as e:
                print(f"ERR {stem}: {e}", flush=True); seeds = []
            done[stem] = seeds
            ck.write(json.dumps({"stem": stem, "seeds": seeds}) + "\n"); ck.flush()
            if (i + 1) % 50 == 0:
                print(f"[{i+1}/{len(files)}] {time.time()-t0:.0f}s", flush=True)

    json.dump(done, open(final, "w"))
    tot = sum(len(v) for v in done.values())
    print(f"done: {len(done)} imgs, {tot} seeds ({tot/max(len(done),1):.1f}/img) -> {final}", flush=True)


if __name__ == "__main__":
    main()
