from pathlib import Path
from runpy import run_path
import unittest

import torch


MODULE = Path(__file__).with_name("fit_calibration.py")


class CalibrationTests(unittest.TestCase):
    def test_midpoint_and_gap_separate_the_same_dual_reference_gain(self):
        f = run_path(str(MODULE))
        pairs = torch.tensor([[[.7, -.1], [-.2, -.4], [0., 0.]]], requires_grad=True)
        midpoint = f["component"](pairs, "midpoint")
        gap = f["component"](pairs, "gap")
        torch.testing.assert_close(midpoint, torch.tensor([[.3, -.3, 0.]]))
        torch.testing.assert_close(gap, torch.tensor([[.4, .1, 0.]]))
        torch.testing.assert_close(midpoint - gap, pairs.min(dim=-1).values)
        (midpoint.sum() + gap.sum()).backward()
        self.assertTrue(torch.isfinite(pairs.grad).all().item())


    def test_candidate_balancing_ignores_masked_actions(self):
        f = run_path(str(MODULE))
        true = torch.zeros((2, 2, 2))
        predicted = torch.tensor([[[.4, .2], [10., 20.]], [[.3, -.1], [-.1, -.3]]])
        valid = torch.tensor([[True, False], [True, True]])
        for mode in ("midpoint", "gap"):
            both = f["candidate_loss"](predicted, true, valid, mode)
            per_row = [f["candidate_loss"](predicted[i:i+1], true[i:i+1], valid[i:i+1], mode)
                       for i in range(2)]
            torch.testing.assert_close(both, torch.stack(per_row).mean())

    def test_zero_signed_static_is_independent_of_reference_occupancy(self):
        f = run_path(str(MODULE))
        gain = torch.rand((2, 4, 2))
        same_action = gain - gain
        zero = torch.zeros((2, 4))
        for mode in ("midpoint", "gap"):
            torch.testing.assert_close(f["component"](same_action, mode), zero)


if __name__ == "__main__":
    unittest.main()
