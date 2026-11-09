import unittest

import torch

from box_export_unclipped import attach_native_boxes


class UnclippedNativeBoxTests(unittest.TestCase):
    def test_preserves_outside_centers_and_extent(self):
        boxes = torch.zeros((1, 164, 4))
        boxes[0, 11] = torch.tensor([-0.125, 1.25, 1.5, 0.25])
        baseline = {"native_ids": [11], "native_ranks": [1]}
        self.assertEqual(attach_native_boxes(baseline, boxes)["native_box_cxcywh"],
                         [[-0.125, 1.25, 1.5, 0.25]])
        self.assertNotIn("native_box_cxcywh", baseline)
        for invalid in (-.25, float("inf")):
            boxes[0, 11, 2] = invalid
            with self.assertRaises(ValueError):
                attach_native_boxes(baseline, boxes)


if __name__ == "__main__":
    unittest.main()
