import random
import unittest

import numpy as np
from PIL import Image
import torch
from torchvision.transforms.v2 import functional as tv
from rfdetr.datasets import _torchvision as native

from common_mask_supervision import ROW_ID
from direction_replay import replay_direction_view, summarize_direction


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fixture():
    image = Image.frombytes("RGB", (9, 9), bytes(range(243)))
    mask = torch.zeros((1, 9, 9), dtype=torch.bool)
    mask[0, 1:4, 2:6] = True
    return image, {
        ROW_ID: torch.tensor([0]),
        "masks": mask,
        "boxes": torch.tensor([[2., 1., 6., 4.]]),
        "labels": torch.tensor([0]),
        "area": torch.tensor([12.]),
        "iscrowd": torch.tensor([0]),
        "image_id": torch.tensor([11]),
        "orig_size": torch.tensor([9, 9]),
        "size": torch.tensor([9, 9]),
    }


def normalized(geometry):
    def transform(image, target):
        image, target = geometry(image, target)
        return tv.to_image(image).as_subclass(torch.Tensor).float() / 255., target
    return transform


class DirectionReplayTest(unittest.TestCase):
    def test_native_real_resize_has_opposite_y_direction_on_shared_row(self):
        image, raw = fixture()
        original_resize = native.Resize.__call__
        original_mask_apply = native._apply_to_masks
        result = replay_direction_view(
            native, normalized(native.Resize((5, 5))), set_seed, image, raw, 421
        )
        self.assertEqual(result["path"], "direct")
        self.assertFalse(result["flip"])
        self.assertEqual(result["membership"]["common_nonempty"], 1)
        self.assertAlmostEqual(result["small"]["native_minus_center_y"], 0.9)
        self.assertAlmostEqual(result["small"]["reflected_minus_center_y"], -0.9)
        self.assertEqual(result["small"]["reference_count"], 1)
        self.assertIs(native.Resize.__call__, original_resize)
        self.assertIs(native._apply_to_masks, original_mask_apply)


    def test_crop_and_flip_still_follow_one_native_trace(self):
        image, raw = fixture()
        geometry = native.Compose([
            native.RandomResize([6, 7]),
            native.RandomSizedCrop((4, 6), (5, 5)),
            native.RandomHorizontalFlip(p=1),
        ])
        result = replay_direction_view(native, normalized(geometry), set_seed, image, raw, 421)
        self.assertEqual(result["path"], "crop")
        self.assertTrue(result["flip"])
        self.assertEqual([step["kind"] for step in result["steps"]], ["resize", "crop", "resize", "hflip"])
        self.assertEqual(result["small"]["count"], result["membership"]["common_nonempty"])

    def test_rng_mismatch_fails_with_hooks_restored(self):
        image, raw = fixture()
        original = native.Resize.__call__
        transform = normalized(native.Resize((5, 5)))
        calls = 0

        def changed(image, target):
            nonlocal calls
            calls += 1
            result = transform(image, target)
            if calls == 2:
                torch.rand(())
            return result

        with self.assertRaisesRegex(ValueError, "random stream"):
            replay_direction_view(native, changed, set_seed, image, raw, 421)
        self.assertIs(native.Resize.__call__, original)

    def test_group_equal_summary_and_small_cohort_refusal(self):
        image, raw = fixture()
        template = replay_direction_view(native, normalized(native.Resize((5, 5))), set_seed, image, raw, 421)
        a = {**template, "group": "a"}
        b = {**template, "group": "b", "small": {**template["small"], "native_minus_center_y": 0.1}}
        report = summarize_direction([a, a, a, b], primary=True)
        self.assertAlmostEqual(report["small"]["native_minus_center_y"], 0.5)
        self.assertEqual(report["small"]["rows"], 4)
        self.assertEqual(report["small"]["groups"], 2)
        self.assertEqual(report["decision"], "INSUFFICIENT_COMMON_SMALL_GROUPS")
        self.assertEqual(report["by_path"]["direct_plain"]["views"], 4)


    def test_paired_direction_gate_checks_both_signs(self):
        image, raw = fixture()
        view = replay_direction_view(native, normalized(native.Resize((5, 5))), set_seed, image, raw, 421)
        views = [{**view, "group": str(i)} for i in range(30)]
        good = summarize_direction(views, primary=True)
        self.assertEqual(good["decision"], "SUPPORTS_DIRECTIONAL_LABEL_EFFECT")
        self.assertGreater(good["native_minus_center_y_interval"][0], 0.)
        self.assertLess(good["reflected_minus_center_y_interval"][1], 0.)
        for row in views:
            row["small"] = {**row["small"], "reflected_minus_center_y": 0.9}
        self.assertEqual(summarize_direction(views, primary=True)["decision"], "DIRECTIONAL_LABEL_GATE_NOT_MET")

    def test_empty_shared_support_is_counted_without_geometry(self):
        image, raw = fixture()
        raw["masks"].zero_()
        view = replay_direction_view(native, normalized(native.Resize((5, 5))), set_seed, image, raw, 421)
        self.assertEqual(view["membership"]["common_nonempty"], 0)
        self.assertEqual(view["membership"]["reflected_y_empty"], 1)
        self.assertEqual(view["small"]["count"], 0)
        self.assertEqual(view["small"]["reference_count"], 0)


if __name__ == "__main__":
    unittest.main()
