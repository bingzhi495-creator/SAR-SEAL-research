import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
from pycocotools import mask as codec
import torch

from infer_point_policies import infer_image


def sibling(name):
    here = Path(__file__).resolve().parent
    choices = (here / (name + ".py"),
               here.parent / "RFDETR_POINTREND_SUBDIVISION_20261121" / (name + ".py"))
    spec = importlib.util.spec_from_file_location(name, next(x for x in choices if x.is_file()))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ConstantHead(torch.nn.Module):
    def forward(self, fine, coarse):
        return torch.full_like(coarse, 100.)


class TestHeldImage(unittest.TestCase):
    def setUp(self):
        self.adapter = sibling("roi_point_head")
        self.render = sibling("pointrend_subdivision")
        self.budget = sibling("matched_budget")
        self.basis = torch.zeros(1, 256, 4, 4)
        self.size = (8, 8)
        self.head = ConstantHead().eval()
        mask = np.zeros(self.size, np.uint8, order="F")
        mask[0, 0] = mask[0, 1] = 1
        rle = codec.encode(mask)
        rle["counts"] = rle["counts"].decode("ascii")
        self.row = dict(image_id=7, rank=1, query_id=3, box_xyxy01=[2.5/8, 2/8, 6.5/8, 6/8],
                        coordinates=np.array([[0 * 8 + 4, 0], [3 * 8 + 3, 1]]), k=1,
                        raw_logits=np.full((4, 4), -10, np.float32), rle=rle)

    def infer(self, rows):
        return infer_image(self.head, self.basis, rows, adapter=self.adapter, render=self.render,
                           budget=self.budget, image_hw=self.size, chunk_size=2)

    def test_chunked_policy_raw_and_matched_edits_agree_with_full_native_paste(self):
        second = dict(self.row, rank=2, query_id=4, k=0)
        outputs, counts = self.infer([self.row, second])
        out = outputs[7, 1, 3]
        grid_y, grid_x = torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
        pix = torch.stack((grid_x, grid_y), dim=-1).reshape(1, 64, 2)
        box = torch.tensor([self.row["box_xyxy01"]]) * 8
        expected = self.adapter.paste_probabilities_at_pixels(torch.full((1, 1, 224, 224), 100.),
                                                              box, pix, self.size).reshape(self.size) >= .5
        self.assertTrue(expected[3, 2])  # pixel center is exactly the left box edge
        np.testing.assert_array_equal(codec.decode(out["P"]), expected.numpy())
        self.assertEqual(int(codec.area(out["C"])), 0)
        self.assertEqual(set(out["B"]), {27, 0})
        self.assertEqual(set(out["F"]), {27, 1})
        for policy in ("F", "B", "J"):
            self.assertEqual(len(set(out[policy])), 2)
            self.assertEqual(len(outputs[7, 2, 4][policy]), 0)
        self.assertEqual(counts["point_evaluations"], 2 * 3136)
        self.assertEqual(out["diagnostics"]["P"]["adds"], int(expected.sum()))
        self.assertEqual(out["diagnostics"]["P"]["drops"], 2)
        self.assertGreater(out["diagnostics"]["P"]["nonpositive_flips"], 0)

    def test_missing_raw_requires_empty_support_and_preserves_original_mask(self):
        zero = dict(self.row, raw_logits=None, coordinates=np.empty((0, 2), np.int32), k=0)
        outputs, counts = self.infer([zero])
        self.assertEqual(outputs[7, 1, 3]["P"], self.row["rle"])
        self.assertEqual(counts["fallback"], 1)
        self.assertEqual(counts["point_evaluations"], 0)
        with self.assertRaises(ValueError):
            self.infer([dict(self.row, raw_logits=None)])

    def test_mix_retains_original_candidate_order_and_detects_wrong_endpoint_sides(self):
        absent = dict(self.row, rank=2, query_id=4, raw_logits=None, coordinates=np.empty((0, 2), np.int32), k=0)
        output, _ = self.infer([self.row, absent])
        self.assertEqual(list(output), [(7, 1, 3), (7, 2, 4)])
        bad = dict(self.row, coordinates=self.row["coordinates"][:, ::-1].copy())
        with self.assertRaisesRegex(ValueError, "stock-background"):
            self.infer([bad])


if __name__ == "__main__":
    unittest.main()
