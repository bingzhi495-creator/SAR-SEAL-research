import unittest

from PIL import Image
import torch
from torch.nn import functional as F
from rfdetr.datasets import _torchvision as native

from trace_resize import capture_transform


def fixture():
    image = Image.frombytes("RGB", (9, 9), bytes(range(243)))
    mask = torch.zeros((1, 9, 9), dtype=torch.bool)
    mask[0, 1:4, 2:6] = True
    target = {
        "masks": mask,
        "boxes": torch.tensor([[2., 1., 6., 4.]]),
        "labels": torch.tensor([0]),
        "area": torch.tensor([12.]),
        "size": torch.tensor([9, 9]),
    }
    return image, target


class ResizeTraceTests(unittest.TestCase):
    def test_resize_preserves_native_nonmask_fields(self):
        image, target = fixture()
        resize = native.Resize((5, 7))
        baseline_img, baseline = resize(image, target)
        for mode, sampler in (("native", "nearest"), ("center", "nearest-exact")):
            with capture_transform(native, mode=mode) as trace:
                actual_img, actual = resize(image, target)
            expected = F.interpolate(target["masks"][:, None].float(), size=(5, 7), mode=sampler)[:, 0].bool()
            self.assertTrue(torch.equal(actual["masks"], expected))
            self.assertEqual(actual_img.tobytes(), baseline_img.tobytes())
            for key in ("boxes", "labels", "area", "size"):
                self.assertTrue(torch.equal(actual[key], baseline[key]))
            self.assertEqual(trace, [{"op": "resize", "in_hw": (9, 9), "out_hw": (5, 7)}])

    def test_seeded_crop_trace_preserves_rng_and_avoids_duplicate_events(self):
        image, target = fixture()
        pipeline = native.Compose([
            native.RandomResize([6, 7]),
            native.RandomSizedCrop((4, 6), (5, 5)),
            native.RandomHorizontalFlip(p=0),
            native.RandomHorizontalFlip(p=1),
        ])
        torch.manual_seed(421)
        baseline_img, _ = pipeline(image, target)
        expected_rng = torch.random.get_rng_state()
        traces = []
        for mode in ("native", "center"):
            torch.manual_seed(421)
            with capture_transform(native, mode=mode) as trace:
                actual_img, _ = pipeline(image, target)
            traces.append(trace)
            self.assertEqual(actual_img.tobytes(), baseline_img.tobytes())
            self.assertTrue(torch.equal(torch.random.get_rng_state(), expected_rng))
        self.assertEqual(traces[0], traces[1])
        self.assertEqual([step["op"] for step in traces[0]], ["resize", "crop", "resize", "hflip", "hflip"])
        self.assertEqual(traces[0][1]["in_hw"], traces[0][0]["out_hw"])
        self.assertEqual(traces[0][2]["in_hw"], traces[0][1]["out_hw"])
        self.assertEqual([step["applied"] for step in traces[0][-2:]], [False, True])

    def test_empty_masks_and_nested_trace_guard(self):
        image, target = fixture()
        target["masks"] = target["masks"][:0]
        target["boxes"] = target["boxes"][:0]
        with capture_transform(native, mode="center") as trace:
            with self.assertRaises(RuntimeError):
                with capture_transform(native, mode="native"):
                    pass
            _, actual = native.Resize((3, 7))(image, target)
        self.assertEqual(actual["masks"].shape, (0, 3, 7))
        self.assertEqual(actual["masks"].dtype, torch.bool)
        self.assertEqual(len(trace), 1)
