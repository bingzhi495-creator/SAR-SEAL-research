# -*- coding: utf-8 -*-
"""Dataset loaders. PSeg-SSDD (VOC-style with <segm> polygon points)."""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

PSEG_ROOT = r"E:\PSeg_SSDD\voc_style"


@dataclass
class Sample:
    stem: str
    image: np.ndarray          # (H, W) uint8
    instances: list            # list of (H, W) bool masks
    subset: str                # 'inshore' | 'offshore' | 'unknown'

    @property
    def ship_mask(self) -> np.ndarray:
        if not self.instances:
            return np.zeros(self.image.shape, dtype=bool)
        return np.any(np.stack(self.instances), axis=0)


def _parse_polygons(xml_path: str):
    root = ET.parse(xml_path).getroot()
    w = int(root.findtext("size/width"))
    h = int(root.findtext("size/height"))
    polys = []
    for obj in root.findall("object"):
        seg = obj.find("segm")
        if seg is None:
            continue
        pts = []
        for p in seg.findall("point"):
            x, y = p.text.split(",")
            pts.append((float(x), float(y)))
        if len(pts) >= 3:
            polys.append(pts)
    return (h, w), polys


def _rasterize(size_hw, polys):
    masks = []
    for pts in polys:
        im = Image.new("1", (size_hw[1], size_hw[0]), 0)
        ImageDraw.Draw(im).polygon(pts, outline=1, fill=1)
        masks.append(np.array(im, dtype=bool))
    return masks


def _subset_stems(root: str):
    inshore = {os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, "JPEGImages_test_inshore"))}
    offshore = {os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, "JPEGImages_test_offshore"))}
    return inshore, offshore


HRSID_ROOT = r"D:\SAR\UNet++\HRSID_JPG\HRSID_JPG"
_HRSID_CACHE: dict = {}


def _hrsid_index(root: str = HRSID_ROOT):
    """image_id -> (file_name, [polygons]) for the HRSID test split."""
    if "idx" in _HRSID_CACHE:
        return _HRSID_CACHE["idx"]
    import json as _json
    j = _json.load(open(os.path.join(root, "annotations", "test2017.json")))
    by_img = {im["id"]: (im["file_name"], []) for im in j["images"]}
    for a in j["annotations"]:
        if a.get("iscrowd"):
            continue
        for seg in a["segmentation"]:
            pts = [(seg[i], seg[i + 1]) for i in range(0, len(seg) - 1, 2)]
            if len(pts) >= 3:
                by_img[a["image_id"]][1].append(pts)
    _HRSID_CACHE["idx"] = by_img
    return by_img


def hrsid_test_stems(limit: int | None = None):
    idx = _hrsid_index()
    stems = sorted(fn for fn, _ in idx.values())
    if limit:
        stems = stems[:limit]
    return [(s, "all") for s in stems]


def load_one_hrsid(file_name: str, subset: str = "all", root: str = HRSID_ROOT) -> Sample | None:
    idx = _hrsid_index(root)
    polys = None
    for fn, ps in idx.values():
        if fn == file_name:
            polys = ps
            break
    img_path = os.path.join(root, "JPEGImages1", file_name)
    if polys is None or not os.path.exists(img_path):
        return None
    img = np.array(Image.open(img_path).convert("L"))
    return Sample(stem=file_name, image=img, instances=_rasterize(img.shape, polys), subset=subset)


def test_stems(root: str = PSEG_ROOT, limit: int | None = None):
    """(stem, subset) pairs for the test split, cheap to compute."""
    img_dir = os.path.join(root, "JPEGImages_test")
    inshore, offshore = _subset_stems(root)
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith(".jpg"))
    if limit:
        stems = stems[:limit]
    return [(s, "inshore" if s in inshore else ("offshore" if s in offshore else "unknown"))
            for s in stems]


def load_one(stem: str, subset: str = "unknown", root: str = PSEG_ROOT) -> Sample | None:
    xml_path = os.path.join(root, "Annotations_test", stem + ".xml")
    img_path = os.path.join(root, "JPEGImages_test", stem + ".jpg")
    if not (os.path.exists(xml_path) and os.path.exists(img_path)):
        return None
    img = np.array(Image.open(img_path).convert("L"))
    _, polys = _parse_polygons(xml_path)
    return Sample(stem=stem, image=img, instances=_rasterize(img.shape, polys), subset=subset)


def load_pseg_ssdd_test(root: str = PSEG_ROOT, limit: int | None = None):
    """Yield Sample for each test image, with inshore/offshore tag."""
    img_dir = os.path.join(root, "JPEGImages_test")
    ann_dir = os.path.join(root, "Annotations_test")
    inshore, offshore = _subset_stems(root)
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith(".jpg"))
    if limit:
        stems = stems[:limit]
    for stem in stems:
        xml_path = os.path.join(ann_dir, stem + ".xml")
        if not os.path.exists(xml_path):
            continue
        img = np.array(Image.open(os.path.join(img_dir, stem + ".jpg")).convert("L"))
        size_hw, polys = _parse_polygons(xml_path)
        masks = _rasterize(img.shape, polys)
        subset = "inshore" if stem in inshore else ("offshore" if stem in offshore else "unknown")
        yield Sample(stem=stem, image=img, instances=masks, subset=subset)
