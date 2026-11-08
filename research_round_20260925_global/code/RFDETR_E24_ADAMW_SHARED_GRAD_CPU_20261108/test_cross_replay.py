"""Exercise the update comparison before applying it to private closed artifacts."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_replay import compare_updates


class CompareUpdatesTest(unittest.TestCase):
    def test_shared_gradient_state_comparison_includes_all_and_mask_head(self):
        left = {"backbone.test": torch.tensor([3.0, 0.0]),
                "segmentation_head.test": torch.tensor([0.0, 4.0])}
        right = {"backbone.test": torch.tensor([0.0, 3.0]),
                 "segmentation_head.test": torch.tensor([0.0, 1.0])}
        result = compare_updates(left, right)
        self.assertAlmostEqual(result["all"]["left_l2"], 5.0)
        self.assertAlmostEqual(result["all"]["right_l2"], 10.0**0.5)
        self.assertAlmostEqual(result["all"]["difference_l2"], 27.0**0.5)
        self.assertAlmostEqual(result["all"]["cosine"], 4.0 / (5.0 * 10.0**0.5))
        self.assertAlmostEqual(result["mask_head"]["right_to_left_l2"], 0.25)
        self.assertAlmostEqual(result["mask_head"]["relative_difference_to_left"], 0.75)
        self.assertAlmostEqual(result["mask_head"]["cosine"], 1.0)


if __name__ == "__main__":
    unittest.main()
