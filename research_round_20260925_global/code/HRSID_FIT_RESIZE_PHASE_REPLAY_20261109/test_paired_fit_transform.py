import random
import unittest

import numpy as np
from PIL import Image
import torch
from torchvision.transforms.v2 import ToDtype, ToImage
from rfdetr.datasets import _torchvision as native
from rfdetr.datasets.transforms import Normalize

from common_mask_supervision import ROW_ID
from paired_fit_transform import PairedFitTransform


def fixture(kind="common"):
    image = Image.frombytes("RGB", (8, 8), bytes(range(192)))
    masks = torch.zeros((2, 8, 8), dtype=torch.bool)
    boxes = torch.tensor([[1., 1., 3., 4.], [4., 2., 6., 6.]])
    if kind == "common":
        masks[0, 1:4, 1:3] = True
        masks[1, 2:6, 4:6] = True
    elif kind == "disjoint":
        masks[0, 1, 1:3] = True
        masks[1, 2, 4:6] = True
    target = dict(
        boxes=boxes, labels=torch.zeros(2, dtype=torch.int64), masks=masks,
        area=torch.tensor([6., 8.]), iscrowd=torch.zeros(2, dtype=torch.int64),
        image_id=torch.tensor([31]), size=torch.tensor([8, 8]), orig_size=torch.tensor([8, 8]),
    )
    if kind == "empty":
        for key in ("boxes", "labels", "masks", "area", "iscrowd"):
            target[key] = target[key][:0]
    return image, target


def compose(*transforms):
    return native.Compose([*transforms, ToImage(), ToDtype(torch.float32, scale=True), Normalize()])


def set_seed():
    random.seed(5219)
    np.random.seed(5219)
    torch.random.default_generator.manual_seed(5219)


def draws():
    return random.random(), float(np.random.random()), float(torch.rand(()))


class PairedFitTransformTest(unittest.TestCase):
    def test_success_preserves_stock_native_rng_and_original_inputs(self):
        image, target = fixture()
        snapshot = {key: value.clone() for key, value in target.items()}
        pixels = image.tobytes()
        native_aug = compose(native.Resize((5, 8)), native.RandomHorizontalFlip(p=.5))

        def transform(image_input, target_input):
            draws()
            return native_aug(image_input, target_input)

        set_seed()
        expected_image, _ = transform(image.copy(), {key: value.clone() for key, value in target.items()})
        expected_rng = draws()
        records = []
        choices = []

        def audit(record):
            records.append(record)
            draws()

        for mode in ("native", "center"):
            with self.subTest(mode=mode):
                set_seed()
                adapted = PairedFitTransform(native, transform, mode, audit=audit)
                actual_image, selected = adapted(image, target)
                self.assertTrue(torch.equal(actual_image, expected_image))
                self.assertEqual(draws(), expected_rng)
                self.assertNotIn(ROW_ID, selected)
                self.assertEqual(len(selected["masks"]), 2)
                choices.append(selected["masks"])
                self.assertEqual(records[-1]["common_nonempty"], 2)
                self.assertGreater(records[-1]["common_changed_masks"], 0)
                self.assertEqual(image.tobytes(), pixels)
                self.assertTrue(all(torch.equal(target[key], value) for key, value in snapshot.items()))
        self.assertFalse(torch.equal(*choices))
        self.assertEqual(records[0], records[1])


    def test_crop_filters_original_row_ids_and_records_same_audit_for_both_arms(self):
        image, target = fixture()
        target[ROW_ID] = torch.arange(2, dtype=torch.int64)

        def crop(image_input, target_input):
            return native.crop(image_input, target_input, top=0, left=0, height=5, width=4)

        transform = compose(native.Resize((8, 8)), crop, native.Resize((5, 4)), native.RandomHorizontalFlip(p=1))
        records = []
        for mode in ("native", "center"):
            set_seed()
            normalized, selected = PairedFitTransform(native, transform, mode, audit=records.append)(image, target)
            self.assertEqual(normalized.shape, (3, 5, 4))
            self.assertEqual(selected["masks"].shape, (1, 5, 4))
            self.assertEqual(selected["image_id"].item(), 31)
            self.assertNotIn(ROW_ID, selected)
        self.assertEqual(target[ROW_ID].tolist(), [0, 1])
        self.assertEqual(records[0], records[1])
        self.assertTrue(records[0]["crop"])
        self.assertTrue(records[0]["flipped"])
        self.assertEqual(records[0]["native_present"], 1)
        self.assertEqual(records[0]["center_present"], 1)
        self.assertEqual(records[0]["common_nonempty"], 1)
        self.assertEqual(records[0]["image_id"], 31)
        expected_hashes = {
            "image_sha256", "trace_sha256", "rng_before_sha256", "rng_after_sha256",
            "common_row_ids_sha256", "common_nonmask_sha256",
            "native_common_masks_sha256", "center_common_masks_sha256",
        }
        self.assertEqual({key for key in records[0] if key.endswith("_sha256")}, expected_hashes)
        for key in expected_hashes:
            self.assertRegex(records[0][key], r"^[a-f0-9]{64}$")

    def test_empty_and_disjoint_supervision_follow_the_same_safe_empty_path(self):
        for kind, nonempty, present in (("empty", 0, 0), ("disjoint", 1, 2)):
            image, target = fixture(kind)
            records = []
            transform = compose(native.Resize((5, 8)))
            for mode in ("native", "center"):
                with self.subTest(kind=kind, mode=mode):
                    normalized, selected = PairedFitTransform(native, transform, mode, audit=records.append)(
                        image, target,
                    )
                    self.assertEqual(normalized.shape, (3, 5, 8))
                    self.assertEqual(selected["masks"].shape, (0, 5, 8))
                    self.assertEqual(selected["boxes"].shape, (0, 4))
                    self.assertEqual(selected["labels"].numel(), 0)
                    self.assertEqual(records[-1]["common_nonempty"], 0)
                    self.assertEqual(records[-1]["native_nonempty"], nonempty)
                    self.assertEqual(records[-1]["center_nonempty"], nonempty)
                    self.assertEqual(records[-1]["native_present"], present)
                    self.assertEqual(records[-1]["center_present"], present)

    def test_rejects_wrong_raw_fields_and_preserves_entry_rng(self):
        image, target = fixture()
        transform = compose(native.Resize((5, 8)))
        adapted = PairedFitTransform(native, transform, "native")
        cases = (
            (dict(target, unexpected=torch.tensor(1)), "prepared CPU RF-DETR target"),
            (dict(target, **{ROW_ID: torch.tensor([1, 0])}), "consecutive"),
            (dict(target, size=torch.tensor([9, 8])), "canvas disagree"),
        )
        set_seed()
        expected = draws()
        for supplied, message in cases:
            with self.subTest(message=message):
                set_seed()
                with self.assertRaisesRegex(ValueError, message):
                    adapted(image, supplied)
                self.assertEqual(draws(), expected)

    def test_wrong_common_nonmask_fields_and_random_draws_roll_back(self):
        image, target = fixture()
        transform = compose(native.Resize((5, 8)), native.RandomHorizontalFlip(p=.5))
        original_hooks = native.Resize.__call__, native.crop, native.RandomHorizontalFlip.__call__
        set_seed()
        expected = draws()
        for fault, message in (("labels", "nonmask fields"), ("rng", "CPU random stream")):
            calls = []

            def change(image_input, target_input):
                result_image, result_target = transform(image_input, target_input)
                calls.append(1)
                if len(calls) == 2:
                    if fault == "labels":
                        result_target["labels"] += 1
                    else:
                        np.random.random()
                return result_image, result_target

            with self.subTest(fault=fault):
                set_seed()
                with self.assertRaisesRegex(ValueError, message):
                    PairedFitTransform(native, change, "native")(image, target)
                self.assertEqual(draws(), expected)
                self.assertEqual((native.Resize.__call__, native.crop, native.RandomHorizontalFlip.__call__),
                                 original_hooks)
                self.assertEqual(target["labels"].tolist(), [0, 0])

    def test_nested_replay_and_audit_exception_restore_hooks_and_rng(self):
        image, target = fixture()
        stock = compose(native.Resize((5, 8)), native.RandomHorizontalFlip(p=.5))
        base = PairedFitTransform(native, stock, "center")
        original_hooks = native.Resize.__call__, native.crop, native.RandomHorizontalFlip.__call__
        set_seed()
        expected = draws()

        def nested(image_input, target_input):
            draws()
            return base(image_input, target_input)

        def failed_audit(_):
            draws()
            raise RuntimeError("deliberate audit fault")

        for adapted, message in (
            (PairedFitTransform(native, nested, "native"), "Nested or concurrent"),
            (PairedFitTransform(native, stock, "native", audit=failed_audit), "deliberate audit fault"),
        ):
            with self.subTest(message=message):
                set_seed()
                with self.assertRaisesRegex(RuntimeError, message):
                    adapted(image, target)
                self.assertEqual(draws(), expected)
                self.assertEqual((native.Resize.__call__, native.crop, native.RandomHorizontalFlip.__call__),
                                 original_hooks)
        self.assertEqual(base(image, target)[1]["masks"].shape, (2, 5, 8))


if __name__ == "__main__":
    unittest.main()
