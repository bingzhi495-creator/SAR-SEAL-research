import unittest

from PIL import Image
from rfdetr.datasets import _torchvision as native
import torch
from torch.nn import functional as F

from phase_patch import mask_resize_phase
from trace_resize import capture_transform


def fixture():
    image = Image.frombytes("RGB", (9, 9), bytes(range(243)))
    masks = torch.zeros((1, 9, 9), dtype=torch.bool)
    masks[0, 1:4, 2:6] = True
    target = {
        "masks": masks,
        "boxes": torch.tensor([[2., 1., 6., 4.]]),
        "labels": torch.tensor([0]),
        "area": torch.tensor([12.]),
        "size": torch.tensor([9, 9]),
    }
    return image, target


class ResizePhaseTests(unittest.TestCase):
    def test_only_center_mask_changes_and_native_resize_is_restored(self):
        image, target = fixture()
        resize = native.Resize((5, 7))
        original = native.Resize.__call__
        baseline_img, baseline = resize(image, target)
        with mask_resize_phase(native, "native"):
            self.assertIs(native.Resize.__call__, original)
            _, stock = resize(image, target)
        self.assertTrue(torch.equal(stock["masks"], baseline["masks"]))
        with mask_resize_phase(native, "center"):
            center_img, center = resize(image, target)
        expected = F.interpolate(target["masks"][:, None].float(), size=(5, 7), mode="nearest-exact")[:, 0].bool()
        self.assertTrue(torch.equal(center["masks"], expected))
        self.assertFalse(torch.equal(center["masks"], baseline["masks"]))
        self.assertEqual(center_img.tobytes(), baseline_img.tobytes())
        self.assertEqual(list(center), list(baseline))
        for key in ("boxes", "labels", "area", "size"):
            self.assertTrue(torch.equal(center[key], baseline[key]))
        self.assertIs(native.Resize.__call__, original)
        _, restored = resize(image, target)
        self.assertTrue(torch.equal(restored["masks"], baseline["masks"]))


    def test_seeded_native_pipeline_matches_replay_center_transform(self):
        image, target = fixture()
        pipeline = native.Compose([
            native.RandomResize([6, 7]),
            native.RandomSizedCrop((4, 6), (5, 5)),
            native.RandomHorizontalFlip(p=1),
        ])
        torch.manual_seed(421)
        with capture_transform(native, mode="center"):
            expected_img, expected_target = pipeline(image, target)
        expected_rng = torch.random.get_rng_state()

        torch.manual_seed(421)
        with mask_resize_phase(native, "center"):
            actual_img, actual_target = pipeline(image, target)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), expected_rng))
        self.assertEqual(actual_img.tobytes(), expected_img.tobytes())
        self.assertEqual(list(actual_target), list(expected_target))
        for key in actual_target:
            self.assertTrue(torch.equal(actual_target[key], expected_target[key]), key)

    def test_exception_restores_phase_and_empty_masks_keep_shape(self):
        image, target = fixture()
        target = {key: value[:0] if key != "size" else value for key, value in target.items()}
        original = native.Resize.__call__
        with self.assertRaisesRegex(RuntimeError, "test exit"):
            with mask_resize_phase(native, "center"):
                _, actual = native.Resize((3, 7))(image, target)
                self.assertEqual(actual["masks"].shape, (0, 3, 7))
                self.assertEqual(actual["masks"].dtype, torch.bool)
                with self.assertRaisesRegex(RuntimeError, "Nested mask resize phase"):
                    with mask_resize_phase(native, "native"):
                        pass
                raise RuntimeError("test exit")
        self.assertIs(native.Resize.__call__, original)


if __name__ == "__main__":
    unittest.main()
