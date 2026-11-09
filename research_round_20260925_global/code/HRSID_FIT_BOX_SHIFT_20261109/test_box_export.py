import unittest

import torch

from box_export import attach_native_boxes


class QueryBoxTests(unittest.TestCase):
    def test_extra_boxes_follow_native_query_order(self):
        values = torch.zeros((1, 164, 4))
        values[0, 3] = torch.tensor([.125, .25, .5, .75])
        values[0, 7] = torch.tensor([.5, .625, .75, .125])
        original = dict(native_ids=[7, 3], native_ranks=[1, 4], all_rles=["original"])
        result = attach_native_boxes(original, values)
        self.assertEqual(result.pop("native_box_cxcywh"), [[.5, .625, .75, .125], [.125, .25, .5, .75]])
        self.assertEqual(result, original)
        self.assertNotIn("native_box_cxcywh", original)
        values[0, 7, 0] = float("nan")
        with self.assertRaises(ValueError):
            attach_native_boxes(original, values)


if __name__ == "__main__":
    unittest.main()
