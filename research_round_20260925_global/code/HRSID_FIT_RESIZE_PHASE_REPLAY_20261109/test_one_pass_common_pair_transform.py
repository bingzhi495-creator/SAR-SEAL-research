from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image
import torch
from rfdetr.datasets import _torchvision as native
from rfdetr.datasets.coco import CocoDetection

from common_fit_binding import AUDIT_FIELDS, ConsumedFitLedger
from common_mask_supervision import ROW_ID
from one_pass_common_pair_transform import OnePassCommonPairTransform
from one_pass_fit_transform import OnePassFitTransform
from test_paired_fit_transform import compose, draws, fixture, set_seed
from trace_resize import capture_transform


class OnePassCommonPairTransformTest(unittest.TestCase):
    def dataset(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        Image.frombytes("RGB", (8, 8), bytes(range(192))).save(root / "sample.png")
        images = [dict(id=31, file_name="sample.png", width=8, height=8)]
        polygons = [
            [[3, 3, 6, 3, 6, 6, 3, 6]],
            [[3, 3, 4, 3, 4, 4, 3, 4]],
            [[2, 2, 3, 2, 3, 3, 2, 3], [7, 7, 8, 7, 8, 8, 7, 8]],
        ]
        annotations = [
            dict(id=i, image_id=31, category_id=1, area=area, iscrowd=0,
                 bbox=bbox, segmentation=segmentation)
            for i, (area, bbox, segmentation) in enumerate(zip(
                [9, 1, 2], [[3, 3, 3, 3], [2, 2, 4, 4], [2, 2, 6, 6]], polygons,
            ), start=1)
        ]
        path = root / "annotations.json"
        path.write_text(json.dumps(dict(images=images, annotations=annotations,
                                        categories=[dict(id=1, name="ship")])), encoding="utf-8")
        return CocoDetection(root, path, None, include_masks=True, remap_category_ids=True)

    def test_real_coco_crop_only_delivers_common_rows_and_arm_masks(self):
        dataset = self.dataset()

        def crop(image, target):
            return native.crop(image, target, top=1, left=1, height=2, width=2)

        stock = compose(native.Resize((4, 4)), crop, native.Resize((3, 3)),
                        native.RandomHorizontalFlip(p=1))
        dataset._transforms = OnePassFitTransform(native, stock)
        _, natural = dataset[0]
        self.assertEqual(natural["masks"].flatten(1).any(dim=1).tolist(), [True, False, True])

        audits = []
        outputs = []
        for mode in ("stepwise_center", "one_pass"):
            set_seed()
            dataset._transforms = OnePassCommonPairTransform(native, stock, mode, audit=audits.append)
            outputs.append(dataset[0])
        (stepwise_image, stepwise), (one_pass_image, one_pass) = outputs
        self.assertTrue(torch.equal(stepwise_image, one_pass_image))
        self.assertEqual(stepwise["masks"].sum().item(), 9)
        self.assertEqual(one_pass["masks"].sum().item(), 4)
        self.assertEqual(stepwise["area"].numel(), 1)
        self.assertTrue(all(torch.equal(stepwise[key], one_pass[key]) for key in stepwise if key != "masks"))
        self.assertNotIn(ROW_ID, stepwise)
        self.assertNotIn(ROW_ID, one_pass)
        self.assertEqual(audits[0], audits[1])
        self.assertEqual(set(audits[0]), AUDIT_FIELDS)
        ConsumedFitLedger(StringIO()).fetched(audits[0])
        self.assertTrue(audits[0]["crop"])
        self.assertTrue(audits[0]["flipped"])
        self.assertEqual(
            tuple(audits[0][key] for key in ("native_present", "native_nonempty", "center_present",
                                             "center_nonempty", "common_nonempty", "common_changed_masks")),
            (2, 2, 2, 1, 1, 1),
        )
        self.assertNotEqual(audits[0]["native_common_masks_sha256"], audits[0]["center_common_masks_sha256"])



    def test_direct_uses_one_stock_visit_and_restores_post_transform_rng(self):
        image, raw = fixture()
        augment = compose(native.Resize((5, 8)), native.RandomHorizontalFlip(p=.5))
        calls = []
        audits = []

        def stock(image_input, target_input):
            calls.append(draws())
            return augment(image_input, target_input)

        def audit(row):
            audits.append(row)
            draws()

        set_seed()
        with capture_transform(native, mode="center"):
            expected_image, expected = stock(image.copy(), {key: value.clone() for key, value in raw.items()})
        expected_call, = calls
        expected_rng = draws()
        calls.clear()
        for mode in ("stepwise_center", "one_pass"):
            with self.subTest(mode=mode):
                set_seed()
                selected = OnePassCommonPairTransform(native, stock, mode, audit=audit)
                actual_image, actual = selected(image, raw)
                self.assertTrue(torch.equal(actual_image, expected_image))
                self.assertEqual(set(actual), set(expected))
                self.assertTrue(all(torch.equal(actual[key], expected[key]) for key in expected))
                self.assertEqual(calls, [expected_call])
                self.assertEqual(draws(), expected_rng)
                calls.clear()
        self.assertEqual(audits[0], audits[1])
        self.assertFalse(audits[0]["crop"])
        self.assertEqual(audits[0]["common_changed_masks"], 0)
        self.assertEqual(audits[0]["native_common_masks_sha256"], audits[0]["center_common_masks_sha256"])

    def test_all_lost_common_rows_have_native_empty_shapes(self):
        image, raw = self.dataset()[0]
        single = {key: value[1:2] if key in {"boxes", "labels", "area", "iscrowd", "masks"} else value
                  for key, value in raw.items()}

        def crop(image_input, target_input):
            return native.crop(image_input, target_input, top=1, left=1, height=2, width=2)

        stock = compose(native.Resize((4, 4)), crop, native.Resize((3, 3)))
        for mode in ("stepwise_center", "one_pass"):
            records = []
            transform = OnePassCommonPairTransform(native, stock, mode, audit=records.append)
            _, actual = transform(image, single)
            self.assertEqual(actual["masks"].shape, (0, 3, 3))
            self.assertEqual(actual["masks"].dtype, torch.bool)
            self.assertEqual(actual["boxes"].shape, (0, 4))
            self.assertEqual(actual["labels"].numel(), 0)
            self.assertEqual(records[0]["native_nonempty"], 1)
            self.assertEqual(records[0]["center_nonempty"], 0)
            self.assertEqual(records[0]["common_nonempty"], 0)

    def test_rejected_direct_masks_and_audit_exception_restore_entry_rng(self):
        image, raw = fixture()
        stock = compose(native.Resize((5, 8)), native.RandomHorizontalFlip(p=.5))
        hooks = native.Resize.__call__, native.crop, native.RandomHorizontalFlip.__call__
        original = {key: value.clone() for key, value in raw.items()}

        def bad_direct(image_input, target_input):
            draws()
            augmented, target = stock(image_input, target_input)
            target["masks"][0, 0, 0] = ~target["masks"][0, 0, 0]
            return augmented, target

        def bad_audit(_row):
            draws()
            raise RuntimeError("audit failed")

        set_seed()
        expected_rng = draws()
        for transform, error, reason in (
            (OnePassCommonPairTransform(native, bad_direct, "stepwise_center"),
             ValueError, "Direct FIT masks must match exactly"),
            (OnePassCommonPairTransform(native, stock, "one_pass", audit=bad_audit),
             RuntimeError, "audit failed"),
        ):
            with self.subTest(reason=reason):
                set_seed()
                with self.assertRaisesRegex(error, reason):
                    transform(image, raw)
                self.assertEqual(draws(), expected_rng)
                self.assertEqual(hooks, (native.Resize.__call__, native.crop, native.RandomHorizontalFlip.__call__))
                self.assertTrue(all(torch.equal(raw[key], value) for key, value in original.items()))


if __name__ == "__main__":
    unittest.main()
