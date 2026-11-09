import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from rfdetr.datasets.coco import CocoDetection, make_coco_transforms_square_div_64

import ssdd_phase_fit_replay as replay


def rng_snapshot():
    numpy = np.random.get_state()
    return random.getstate(), numpy[0], numpy[1].tobytes(), numpy[2:], torch.get_rng_state().numpy().tobytes()


class ReplayFitTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        folder = Path(temp.name)
        images = []
        for image_id, (width, height) in ((7, (53, 41)), (2, (37, 59))):
            name = f"image_{image_id}.jpg"
            Image.new("RGB", (width, height), (19, 60, 87)).save(folder / name)
            images.append(dict(id=image_id, file_name=name, width=width, height=height))
        labels = dict(images=images, categories=[dict(id=1, name="ship")], annotations=[
            dict(id=1, image_id=7, category_id=1, area=600, iscrowd=0,
                 bbox=[11, 5, 30, 20], segmentation=[[11, 5, 41, 5, 41, 25, 11, 25]]),
        ])
        path = folder / "annotations.json"
        path.write_text(json.dumps(labels), encoding="utf-8")
        self.transform = make_coco_transforms_square_div_64("train", 576, patch_size=12, num_windows=1)
        self.dataset = CocoDetection(folder, path, self.transform, include_masks=True, remap_category_ids=True)

    def run_replay(self, ids=(2, 7)):
        with patch.object(replay, "FIT_IMAGES", 2):
            return replay.replay_fit_population(self.dataset, ids)

    def test_real_stock_rectangular_views_and_empty_targets_are_reproducible(self):
        before = rng_snapshot()
        report = self.run_replay()
        self.assertEqual(report, self.run_replay())
        self.assertEqual(rng_snapshot(), before)
        self.assertIs(self.dataset._transforms, self.transform)
        self.assertEqual(report["views"], 2)
        self.assertEqual(report["rectangular_source_views"], 2)
        self.assertEqual(sum(row["views"] for row in report["by_path"].values()), 2)
        self.assertEqual(report["membership"]["raw_rows"], 1)
        self.assertEqual(report["capacity"]["native_max_present"], 1)
        self.assertEqual(report["capacity"]["center_max_present"], 1)
        self.assertEqual(report["membership"]["common_nonempty"], 1)
        self.assertEqual(sum(row["small"]["rows"] for row in report["by_path"].values()), 1)
        self.assertFalse(report["scene_independence_proven"])

    def test_capacity_stops_at_the_first_failing_stock_ordinal_and_restores_state(self):
        before = rng_snapshot()
        with patch.object(replay, "QUERY_CAPACITY", 0):
            with self.assertRaises(replay.ReplayViewError) as raised:
                self.run_replay()
        self.assertEqual(raised.exception.diagnostic,
                         dict(ordinal=1, image_id=7, seed=5312, stage="query_capacity", cause_type="ValueError"))
        self.assertIn("native=1, center=1", str(raised.exception.__cause__))
        self.assertIs(self.dataset._transforms, self.transform)
        self.assertEqual(rng_snapshot(), before)

    def test_source_dimensions_must_match_decoded_jpeg(self):
        self.dataset.coco.imgs[2]["height"] += 1
        with self.assertRaises(replay.ReplayViewError) as raised:
            self.run_replay()
        self.assertEqual(raised.exception.diagnostic["ordinal"], 0)
        self.assertEqual(raised.exception.diagnostic["stage"], "paired_stock_transform")
        self.assertIn("original variable dimensions", str(raised.exception.__cause__))
        self.assertIs(self.dataset._transforms, self.transform)

    def test_wrong_expected_order_is_rejected_before_stock_access(self):
        with patch.object(self.dataset, "_decode_image", side_effect=AssertionError("must fail before decode")):
            with self.assertRaisesRegex(ValueError, "stock index order"):
                self.run_replay((7, 2))


if __name__ == "__main__":
    unittest.main()
