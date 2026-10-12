# -*- coding: utf-8 -*-
"""LoadSeeds: attach precomputed NQS seed boxes, geometrically aligned to the
augmented image frame.

MNE seeds depend only on intensity statistics, which are equivariant to the
spatial augmentations (flip/resize) -- so seeds computed on the ORIGINAL image
and then geometrically transformed equal seeds computed on the augmented image.
This transform is placed AFTER RandomFlip + Resize in the pipeline; it reads the
recorded transform metadata (flip / flip_direction / scale_factor / ori_shape)
and replays the same ops on the original-coord seed boxes so they land in the
network input frame (where reference_points live). RandomCrop is dropped from
the seeding pipeline (near-no-op on ~450px PSeg images at 512 crop).
"""
from __future__ import annotations

import json
import os

import numpy as np

from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadSeeds(BaseTransform):
    def __init__(self, seeds_json: str):
        with open(seeds_json, encoding="utf-8") as fh:
            self.seeds = json.load(fh)

    def transform(self, results: dict) -> dict:
        stem = os.path.splitext(os.path.basename(results["img_path"]))[0]
        boxes = np.asarray(self.seeds.get(stem, []), dtype=np.float32).reshape(-1, 4)  # ori-coord xyxy

        if boxes.shape[0]:
            # 1) replay horizontal flip (RandomFlip runs before Resize, on ori size)
            if results.get("flip", False) and results.get("flip_direction") == "horizontal":
                ori_w = results["ori_shape"][1]
                x0 = boxes[:, 0].copy()
                boxes[:, 0] = ori_w - boxes[:, 2]
                boxes[:, 2] = ori_w - x0
            elif results.get("flip", False) and results.get("flip_direction") == "vertical":
                ori_h = results["ori_shape"][0]
                y0 = boxes[:, 1].copy()
                boxes[:, 1] = ori_h - boxes[:, 3]
                boxes[:, 3] = ori_h - y0
            # 2) replay resize scale
            sf = results.get("scale_factor", (1.0, 1.0))
            sx, sy = (sf[0], sf[1]) if hasattr(sf, "__len__") else (sf, sf)
            boxes = boxes * np.array([sx, sy, sx, sy], dtype=np.float32)
            # clip to image
            h, w = results["img_shape"][:2]
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)

        results["seeds"] = boxes
        return results
