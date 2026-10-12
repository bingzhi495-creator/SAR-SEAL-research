# -*- coding: utf-8 -*-
"""Precompute compact MNE evidence fields for the v5 segmentation-injection
experiments. Per image saves a float16 .npz with:
  ev_zmax : (H,W)  max-over-scale rank evidence z   -> hard-neg (E2), boundary (E3)
  ev_theta: (H,W)  structure-tensor orientation (rad) -> anisotropic band (E2), TV (E3)
These two are the fields the *data-supported* injections (E2/E3) need, and are
independent of D2's affinity-separability outcome. ev_vec (S,H,W) for affinity
(E1) is added only if D2 passes (separate flag).

float16 + 1/2 downsample keeps PSeg storage ~0.5MB/img (~0.6GB total). HRSID is
deferred until the NumPy MNE is sped up (15s/img is too slow at 5604 imgs).

Usage: python seeding\\precompute_evidence.py --split pseg_train|pseg_test [--vec]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mne.engine import mne_field, orientation_field  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRS = {
    "pseg_train": r"E:\PSeg_SSDD\voc_style\JPEGImages_train",
    "pseg_test": r"E:\PSeg_SSDD\voc_style\JPEGImages_test",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=list(DIRS))
    ap.add_argument("--vec", action="store_true", help="also store ev_vec (S,H,W) for affinity E1")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out = os.path.join(BASE, "data", "evidence", args.split)
    os.makedirs(out, exist_ok=True)
    img_dir = DIRS[args.split]
    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".jpg"))
    if args.limit:
        files = files[:args.limit]

    t0, n = time.time(), 0
    for i, fn in enumerate(files):
        stem = os.path.splitext(fn)[0]
        dst = os.path.join(out, stem + ".npz")
        if os.path.exists(dst):
            n += 1
            continue
        img = np.array(Image.open(os.path.join(img_dir, fn)).convert("L"))
        f = mne_field(img)
        ev_zmax_full = f["E"].max(0)
        ev_theta = orientation_field(ev_zmax_full).astype(np.float16)  # orientation of evidence ridge
        ev_zmax = ev_zmax_full.astype(np.float16)
        payload = {"ev_zmax": ev_zmax, "ev_theta": ev_theta}
        if args.vec:
            payload["ev_vec"] = f["E"].astype(np.float16)
        np.savez_compressed(dst, **payload)
        n += 1
        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(files)}] {time.time()-t0:.0f}s", flush=True)
    print(f"done: {n} evidence fields -> {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
