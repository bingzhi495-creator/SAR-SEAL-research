"""Tests for direct utility versus pointwise squared error."""
import unittest

import torch

from direct_utility import training_loss


class LossTests(unittest.TestCase):
    def test_direct_cancels_common_point_errors(self):
        logits=torch.zeros((1,2))
        targets=torch.tensor([[.25,.25]])
        self.assertEqual(training_loss("direct",logits,targets).item(),0.)
        self.assertEqual(training_loss("brier",logits,targets).item(),.0625)

    def test_direct_uses_error_difference(self):
        logits=torch.zeros((1,2))
        targets=torch.tensor([[.25,.75]])
        self.assertEqual(training_loss("direct",logits,targets).item(),.125)
        self.assertAlmostEqual(training_loss("bce",logits,targets).item(),.69314718)


if __name__=="__main__":
    unittest.main()
