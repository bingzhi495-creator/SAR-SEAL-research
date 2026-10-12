# -*- coding: utf-8 -*-
"""Generate scene-level-split COCO files for HRSID (verdict-2/3 protocol).

The official HRSID chip split has 92.8% pixel-level cross-split leakage (see
runs/audit_hrsid/leakage_report.json). This builds an alternative split where
whole source scenes go entirely into train or test, using the scene partition
already computed in runs/audit_hrsid/scene_level_split.json.

Outputs under data/hrsid/annotations/:
  scene_train.json / scene_test.json   (scene-level, leakage-free)
  official_train.json / official_test.json  (copies pointing at local images)
Images stay in place (JPEGImages1); file_name carries the relative name.
"""
from __future__ import annotations

import json
import os

ROOT = r"D:\SAR\UNet++\HRSID_JPG\HRSID_JPG"
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "data", "hrsid", "annotations")
SPLIT = os.path.join(BASE, "runs", "audit_hrsid", "scene_level_split.json")


def scene_of(file_name: str) -> str:
    return file_name.split("_")[0]


def main():
    os.makedirs(OUT, exist_ok=True)
    tr = json.load(open(os.path.join(ROOT, "annotations", "train2017.json")))
    te = json.load(open(os.path.join(ROOT, "annotations", "test2017.json")))
    split = json.load(open(SPLIT))
    test_scenes = set(split["test_scenes"])

    # FIX (2026-06-19): official HRSID train2017 and test2017 share OVERLAPPING
    # image-id and annotation-id spaces. Naively merging them collides ids (a
    # train image and a test image with the same id but different files), which
    # corrupts the image<->annotation linkage AND trips mmdet's unique-id assert.
    # Offset test ids past the max train id before merging, updating image_id refs.
    img_off = max(im["id"] for im in tr["images"]) + 1
    ann_off = max(a["id"] for a in tr["annotations"]) + 1
    for im in te["images"]:
        im["id"] += img_off
    for a in te["annotations"]:
        a["id"] += ann_off
        a["image_id"] += img_off

    # merge all images/annotations (now globally unique ids), repartition by scene
    all_imgs = tr["images"] + te["images"]
    all_anns = tr["annotations"] + te["annotations"]
    cats = tr.get("categories") or [{"id": 1, "name": "ship"}]
    assert len({im["id"] for im in all_imgs}) == len(all_imgs), "image ids still not unique"
    assert len({a["id"] for a in all_anns}) == len(all_anns), "ann ids still not unique"

    def subset(keep_scene_fn):
        imgs = [im for im in all_imgs if keep_scene_fn(scene_of(im["file_name"]))]
        ids = {im["id"] for im in imgs}
        anns = [a for a in all_anns if a["image_id"] in ids]
        return {"images": imgs, "annotations": anns, "categories": cats}

    sc_tr = subset(lambda s: s not in test_scenes)
    sc_te = subset(lambda s: s in test_scenes)
    for nm, d in (("scene_train", sc_tr), ("scene_test", sc_te)):
        assert len({im["id"] for im in d["images"]}) == len(d["images"])
        assert len({a["id"] for a in d["annotations"]}) == len(d["annotations"])
    json.dump(sc_tr, open(os.path.join(OUT, "scene_train.json"), "w"))
    json.dump(sc_te, open(os.path.join(OUT, "scene_test.json"), "w"))
    print(f"scene_train: {len(sc_tr['images'])} imgs, {len(sc_tr['annotations'])} anns")
    print(f"scene_test : {len(sc_te['images'])} imgs, {len(sc_te['annotations'])} anns")

    # NOTE: official_train/test.json are intentionally NOT regenerated here -- they
    # already exist, are consistent with every existing A0-HRSID result, and the
    # id-offset above would needlessly renumber them. Scene files only.
    print("(official_train/test.json left untouched)")


if __name__ == "__main__":
    main()
