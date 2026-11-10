import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from rfdetr.datasets import _torchvision as native
from rfdetr.datasets.coco import CocoDetection

from common_mask_supervision import ROW_ID
from one_pass_fit_transform import OnePassFitTransform, substitute_one_pass_masks
from test_paired_fit_transform import compose, draws, set_seed
from trace_resize import capture_transform


class OnePassFitTransformTest(unittest.TestCase):
    def dataset(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        Image.frombytes("RGB", (8, 8), bytes(range(192))).save(root / "sample.png")
        images = [dict(id=31, file_name="sample.png", width=8, height=8)]
        annotations = [
            dict(id=1, image_id=31, category_id=1, area=6, iscrowd=0, bbox=[2, 2, 2, 3],
                 segmentation=[[2, 2, 4, 2, 4, 5, 2, 5]]),
            dict(id=2, image_id=31, category_id=1, area=1, iscrowd=0, bbox=[1, 1, 6, 6],
                 segmentation=[[0, 0, 1, 0, 1, 1, 0, 1]]),
        ]
        path = root / "annotations.json"
        path.write_text(json.dumps(dict(images=images, annotations=annotations,
                                        categories=[dict(id=1, name="ship")])), encoding="utf-8")
        return CocoDetection(root, path, None, include_masks=True, remap_category_ids=True)

    def test_stock_loader_crop_flip_samples_source_once_and_keeps_box_geometry(self):
        dataset = self.dataset()
        image, raw = dataset[0]

        def crop(image_input, target_input):
            return native.crop(image_input, target_input, top=1, left=1, height=3, width=3)

        stock = compose(native.Resize((5, 5)), crop, native.Resize((4, 4)), native.RandomHorizontalFlip(p=1))
        box_carrier = {key: value.clone() for key, value in raw.items() if key != "masks"}
        expected_image, expected_target = stock(image, box_carrier)
        dataset._transforms = OnePassFitTransform(native, stock)
        with patch.object(native, "_apply_to_masks", side_effect=AssertionError("intermediate mask operation")):
            actual_image, actual = dataset[0]
        expected_masks = raw["masks"][:, [2, 3, 4, 5]][:, :, [5, 4, 3, 2]]
        self.assertTrue(torch.equal(actual_image, expected_image))
        self.assertTrue(torch.equal(actual["masks"], expected_masks))
        self.assertTrue(all(torch.equal(actual[key], value) for key, value in expected_target.items()))
        self.assertEqual(actual["masks"].flatten(1).any(dim=1).tolist(), [True, False])
        self.assertNotIn(ROW_ID, actual)

    def test_fixed_stepwise_center_rows_do_not_regain_a_crop_rejection(self):
        dataset = self.dataset()
        image, raw = dataset[0]

        def crop(image_input, target_input):
            return native.crop(image_input, target_input, top=1, left=1, height=3, width=3)

        stock = compose(native.Resize((5, 5)), crop, native.Resize((4, 4)))
        raw[ROW_ID] = torch.arange(len(raw["labels"]), dtype=torch.int64)
        with capture_transform(native, mode="center") as trace:
            _, stepwise = stock(image, raw)
        self.assertEqual(stepwise[ROW_ID].tolist(), [0])
        conditioned = substitute_one_pass_masks(raw["masks"], stepwise, trace)
        self.assertEqual(conditioned[ROW_ID].tolist(), [0])
        self.assertEqual(int(stepwise["masks"].sum()), 3)
        self.assertEqual(int(conditioned["masks"].sum()), 6)
        self.assertTrue(all(torch.equal(conditioned[key], value) for key, value in stepwise.items() if key != "masks"))

    def test_direct_resize_matches_stepwise_center_and_stock_random_stream(self):
        dataset = self.dataset()
        image, raw = dataset[0]
        stock = compose(native.Resize((5, 5)), native.RandomHorizontalFlip(p=.5))
        set_seed()
        with capture_transform(native, mode="center"):
            expected_image, expected = stock(image, raw)
        expected_draws = draws()
        records = []

        def audit(row):
            records.append(row)
            draws()

        set_seed()
        dataset._transforms = OnePassFitTransform(native, stock, audit=audit)
        actual_image, actual = dataset[0]
        self.assertTrue(torch.equal(actual_image, expected_image))
        self.assertTrue(all(torch.equal(actual[key], value) for key, value in expected.items()))
        self.assertEqual(draws(), expected_draws)
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0]["crop"])
        self.assertEqual(records[0]["positive_box_rows"], 2)
        self.assertRegex(records[0]["masks_sha256"], r"^[a-f0-9]{64}$")

    def test_empty_prepared_rows_reach_native_empty_guard_shape(self):
        dataset = self.dataset()
        image, raw = dataset[0]
        empty = {key: value[:0] if key in {"boxes", "labels", "area", "iscrowd", "masks"} else value
                 for key, value in raw.items()}
        _, target = OnePassFitTransform(native, compose(native.Resize((5, 5))))(image, empty)
        self.assertEqual(target["masks"].shape, (0, 5, 5))
        self.assertEqual(target["masks"].dtype, torch.bool)
        self.assertEqual(target["boxes"].shape, (0, 4))


if __name__ == "__main__":
    unittest.main()
