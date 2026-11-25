"""Support a strict half-probability threshold through the native projection."""
import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_parity import project_native


class NativeParityTest(unittest.TestCase):
    def test_pixel_centers_rescale_then_threshold(self):
        roi = torch.tensor([[2.0, 2.0, 4.0, 4.0, 0.0, 0.9]])
        corner = torch.tensor([[0.0, 0.0]])
        values = torch.tensor([[[[0.5, 0.5, 0.0, 0.0],
                                 [0.5, 0.5, 0.0, 0.0],
                                 [0.75, 0.75, 0.25, 0.25],
                                 [0.75, 0.75, 0.25, 0.25]]]])
        predicted = project_native(roi, values, corner, 4, (4, 4), (2, 2))
        expected = torch.tensor([[[0.5, 0.0], [0.75, 0.25]]])
        torch.testing.assert_close(predicted, expected, atol=0, rtol=0)
        self.assertEqual((predicted > 0.5).sum().item(), 1)


if __name__ == "__main__":
    unittest.main()
