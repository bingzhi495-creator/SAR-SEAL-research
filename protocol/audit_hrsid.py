# -*- coding: utf-8 -*-
"""HRSID provenance audit (M0/P0): scene parsing, cross-split leakage
quantification, scene-level split proposal.

HRSID chip names encode provenance: P0001_0_800_10190_10990.jpg
  -> scene P0001, chip rows [0,800), cols [10190,10990) in the source scene.
Official split is chip-level -> scenes straddle train/test (leakage).
"""
from __future__ import annotations

import collections
import json
import os

ROOT = r"D:\SAR\UNet++\HRSID_JPG\HRSID_JPG"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "audit_hrsid")


def parse_name(file_name: str):
    stem = file_name.rsplit(".", 1)[0]
    parts = stem.split("_")
    if len(parts) < 5:
        return parts[0], None
    try:
        y0, y1, x0, x1 = (int(parts[-4]), int(parts[-3]), int(parts[-2]), int(parts[-1]))
    except ValueError:
        return parts[0], None
    return "_".join(parts[:-4]), (y0, y1, x0, x1)


def main():
    os.makedirs(OUT, exist_ok=True)
    tr = json.load(open(os.path.join(ROOT, "annotations", "train2017.json")))
    te = json.load(open(os.path.join(ROOT, "annotations", "test2017.json")))

    bad = []
    chips = {"train": collections.defaultdict(list), "test": collections.defaultdict(list)}
    for split, j in (("train", tr), ("test", te)):
        for im in j["images"]:
            scene, box = parse_name(im["file_name"])
            if box is None:
                bad.append((split, im["file_name"]))
            chips[split][scene].append((box, im["file_name"]))
    print(f"unparseable names: {len(bad)}", bad[:5])

    s_tr, s_te = set(chips["train"]), set(chips["test"])
    both = s_tr & s_te
    print(f"scenes: train {len(s_tr)}, test {len(s_te)}, overlap {len(both)}")

    # pixel-level cross-split chip overlap
    n_te = sum(len(v) for v in chips["test"].values())
    overlapped, max_iou_hist = 0, collections.Counter()
    for s in both:
        tr_boxes = [b for b, _ in chips["train"][s] if b]
        for box, name in chips["test"][s]:
            if box is None:
                continue
            y0, y1, x0, x1 = box
            area = (y1 - y0) * (x1 - x0)
            best = 0.0
            for ty0, ty1, tx0, tx1 in tr_boxes:
                iy = max(0, min(y1, ty1) - max(y0, ty0))
                ix = max(0, min(x1, tx1) - max(x0, tx0))
                ov = iy * ix / area if area else 0.0
                best = max(best, ov)
            if best > 0:
                overlapped += 1
            max_iou_hist[round(best, 1)] += 1
    print(f"test chips with ANY pixel overlap vs train chips: {overlapped}/{n_te} ({100*overlapped/n_te:.1f}%)")
    print("max-overlap-fraction histogram (test chips):", dict(sorted(max_iou_hist.items())))

    # scene-level split proposal: whole scenes in/out, stratify by chip count,
    # target ~65/35 chips matching official ratio; deterministic by scene id
    scenes_all = sorted(s_tr | s_te)
    scene_sizes = {s: len(chips["train"].get(s, [])) + len(chips["test"].get(s, [])) for s in scenes_all}
    total = sum(scene_sizes.values())
    order = sorted(scenes_all, key=lambda s: (-scene_sizes[s], s))
    split_tr, split_te, acc = [], [], 0
    for i, s in enumerate(order):  # greedy round-robin to hit 65%
        if acc < 0.65 * total:
            split_tr.append(s); acc += scene_sizes[s]
        else:
            split_te.append(s)
    json.dump({"train_scenes": sorted(split_tr), "test_scenes": sorted(split_te),
               "train_chips": sum(scene_sizes[s] for s in split_tr),
               "test_chips": sum(scene_sizes[s] for s in split_te)},
              open(os.path.join(OUT, "scene_level_split.json"), "w"), indent=1)
    print(f"scene-level split: {len(split_tr)} train scenes ({sum(scene_sizes[s] for s in split_tr)} chips) "
          f"/ {len(split_te)} test scenes ({sum(scene_sizes[s] for s in split_te)} chips)")
    json.dump({"overlapped_test_chips": overlapped, "n_test": n_te,
               "scene_overlap": len(both), "hist": {str(k): v for k, v in max_iou_hist.items()}},
              open(os.path.join(OUT, "leakage_report.json"), "w"), indent=1)
    print("written:", OUT)


if __name__ == "__main__":
    main()
