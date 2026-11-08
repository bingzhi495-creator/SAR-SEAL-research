"""Tests for complete boundary capture and DINO-oriented node selection."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from combined_backward_node_trace import collect_tensors, focused_nodes


class CombinedBackwardNodeTraceTest(unittest.TestCase):
    def test_boundary_recursion_marks_trainable_features_and_positions(self):
        feature = torch.ones(2, requires_grad=True)
        position = torch.zeros(2)
        mask = torch.zeros(2, dtype=torch.bool)
        captured = collect_tensors(
            [SimpleNamespace(tensors=feature, mask=mask), [position], None], "joined")
        self.assertIs(captured["joined_0_tensors"], feature)
        self.assertIs(captured["joined_0_mask"], mask)
        self.assertIs(captured["joined_1_0"], position)
        self.assertFalse(captured["joined_1_0"].requires_grad)

    def test_node_region_is_defined_by_observed_feature_boundaries(self):
        source = torch.ones(1, 1, 5, 5, requires_grad=True)
        dino = torch.nn.Conv2d(1, 2, 3)(source)
        joined = torch.nn.Conv2d(2, 2, 1)(dino)
        loss = torch.nn.Conv2d(2, 2, 1)(joined).mean()
        chosen, available, count, regions = focused_nodes(loss.grad_fn, [dino], [joined])
        self.assertGreaterEqual(count, 3)
        names = {regions[ordinal] for ordinal, _, _, operator, _ in chosen if operator == "ConvolutionBackward0"}
        self.assertEqual(names, {"DINO_ENCODER", "BACKBONE_NON_DINO", "OUTSIDE_BACKBONE"})
        self.assertEqual(available["conv"], 3)


if __name__ == "__main__":
    unittest.main()
