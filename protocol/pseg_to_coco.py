# -*- coding: utf-8 -*-
"""Convert PSeg-SSDD VOC-style polygon annotations to COCO instance JSON.

Outputs train.json / test.json (+ test_inshore.json / test_offshore.json)
under E:/SAR/seal/data/pseg_ssdd/annotations, with image paths pointing at the
original JPEGImages_* directories (no image copying).
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol.datasets import PSEG_ROOT, _parse_polygons, _subset_stems  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "pseg_ssdd", "annotations")


def polygon_area(pts):
    s = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def build(split: str, stems_subsets, img_dir: str, ann_dir: str) -> dict:
    images, annotations = [], []
    ann_id = 1
    for img_id, (stem, _subset) in enumerate(stems_subsets, start=1):
        xml = os.path.join(PSEG_ROOT, ann_dir, stem + ".xml")
        if not os.path.exists(xml):
            continue
        (h, w), polys = _parse_polygons(xml)
        images.append({"id": img_id, "file_name": stem + ".jpg", "width": w, "height": h})
        for pts in polys:
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            seg = [float(v) for xy in pts for v in xy]
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": 1,
                "segmentation": [seg],
                "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                "area": polygon_area(pts), "iscrowd": 0,
            })
            ann_id += 1
    return {"images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": "ship"}]}


def main():
    os.makedirs(OUT, exist_ok=True)
    inshore, offshore = _subset_stems(PSEG_ROOT)

    def stems_of(d):
        p = os.path.join(PSEG_ROOT, d)
        return sorted(os.path.splitext(f)[0] for f in os.listdir(p) if f.endswith(".jpg"))

    train = [(s, "train") for s in stems_of("JPEGImages_train")]
    test = [(s, "in" if s in inshore else "off") for s in stems_of("JPEGImages_test")]
    sets = {
        "train": (train, "JPEGImages_train", "Annotations_train"),
        "test": (test, "JPEGImages_test", "Annotations_test"),
        "test_inshore": ([t for t in test if t[0] in inshore], "JPEGImages_test", "Annotations_test"),
        "test_offshore": ([t for t in test if t[0] in offshore], "JPEGImages_test", "Annotations_test"),
    }
    for name, (ss, img_dir, ann_dir) in sets.items():
        coco = build(name, ss, img_dir, ann_dir)
        path = os.path.join(OUT, name + ".json")
        json.dump(coco, open(path, "w"))
        print(f"{name}: {len(coco['images'])} imgs, {len(coco['annotations'])} anns -> {path}")


if __name__ == "__main__":
    main()
