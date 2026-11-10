import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZIP_STORED, ZipFile

from PIL import Image
import torch
from rfdetr.datasets.coco import CocoDetection, make_coco_transforms_square_div_64

import ssdd_fit_direction_dgx as replay
import ssdd_fit_stage_readback as readback


class DirectionReplayTest(unittest.TestCase):
    def test_grouping_is_bound_to_independent_ordered_jpeg_roster(self):
        group = "a" * 64
        roster = [[7, "b.jpg", 10, group], [2, "a.jpg", 10, group]]
        rows = [dict(image_id=image_id, original_basename=name, original_image_ref=dict(bytes=size, sha256=digest))
                for image_id, name, size, digest in roster]
        raw = replay.compact(dict(rows=rows))
        ordered = sha256(replay.compact([2, 7])).hexdigest()
        receipt = dict(content_roster_sha256=sha256(replay.compact(roster)).hexdigest(),
                       native_sorted_id_order_sha256=ordered)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic_fit.zip"
            with ZipFile(path, "x", compression=ZIP_STORED) as archive:
                archive.writestr("manifest.json", raw)
            with (patch.object(readback, "MANIFEST_BYTES", len(raw)),
                  patch.object(readback, "MANIFEST_SHA", sha256(raw).hexdigest()),
                  patch.object(replay, "FIT_ORDER_SHA", ordered), patch.object(replay, "FIT_IMAGES", 2),
                  patch.object(replay, "CONTENT_GROUPS", 1)):
                self.assertEqual(replay.verified_groups(path, receipt, readback), {2: group, 7: group})
                with self.assertRaisesRegex(ValueError, "JPEG group lineage"):
                    replay.verified_groups(path, dict(receipt, content_roster_sha256="b" * 64), readback)

    def test_stock_views_with_duplicate_image_bytes_share_one_content_group(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            names = ("scene_a.jpg", "scene_b.jpg")
            image = Image.new("RGB", (53, 41), (19, 60, 87))
            for name in names:
                image.save(root / name)
            images = [dict(id=i, file_name=name, width=53, height=41) for i, name in zip((7, 2), names)]
            labels = dict(images=images, categories=[dict(id=1, name="ship")], annotations=[
                dict(id=i, image_id=i, category_id=1, area=600, iscrowd=0,
                     bbox=[11, 5, 30, 20], segmentation=[[11, 5, 41, 5, 41, 25, 11, 25]]) for i in (7, 2)
            ])
            path = root / "annotations.json"
            path.write_text(json.dumps(labels), encoding="utf-8")
            transform = make_coco_transforms_square_div_64("train", 576, patch_size=12, num_windows=1)
            dataset = CocoDetection(root, path, transform, include_masks=True, remap_category_ids=True)
            group = sha256((root / names[0]).read_bytes()).hexdigest()
            self.assertEqual(sha256((root / names[1]).read_bytes()).hexdigest(), group)
            views = []
            original_rng = torch.get_rng_state().clone()
            with patch.object(replay, "FIT_IMAGES", 2), patch.object(replay, "CONTENT_GROUPS", 1):
                result = replay.replay_population(dataset, {2: group, 7: group}, view_sink=views.append)
            self.assertIs(dataset._transforms, transform)
            self.assertTrue(torch.equal(torch.get_rng_state(), original_rng))
            self.assertEqual([(v["image_id"], v["seed"], v["group"]) for v in views],
                             [(2, 5311, group), (7, 5312, group)])
            self.assertEqual(result["direction"]["groups"], 1)
            self.assertEqual(result["direction"]["small"]["rows"], 2)
            self.assertEqual(result["direction"]["decision"], "INSUFFICIENT_COMMON_SMALL_GROUPS")


if __name__ == "__main__":
    unittest.main()
