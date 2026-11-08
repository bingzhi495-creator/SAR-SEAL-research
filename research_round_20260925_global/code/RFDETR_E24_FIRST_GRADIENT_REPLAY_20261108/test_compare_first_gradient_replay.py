"""CPU check for exact grouped comparison in the replay summary."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_first_gradient_replay import gradient_comparison


class ReplayComparisonTests(unittest.TestCase):
    def test_projection_accounts_for_all_mismatch_when_rest_is_identical(self):
        projection = "backbone.encoder.patch_embeddings.projection.weight"
        f = {projection: torch.tensor([1., 2.]), "class_embed.weight": torch.tensor([3., 4.])}
        s = {projection: torch.tensor([1., 3.]), "class_embed.weight": torch.tensor([3., 4.])}
        result = gradient_comparison(f, s)
        self.assertEqual(result["all"]["changed_elements"], 1)
        self.assertEqual(result["all"]["diff_l2"], 1.)
        self.assertEqual(result["non_backbone"]["diff_l2"], 0.)
        self.assertEqual(result["patch_projection_weight"]["squared_diff_share"], 1.)


if __name__ == "__main__":
    unittest.main()
