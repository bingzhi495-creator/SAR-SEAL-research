"""CPU fixtures for the first-gradient replay guard and exact diagnostics."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from first_gradient_replay import stop_before_update


class BeforeUpdateTests(unittest.TestCase):
    def test_first_pre_step_interrupts_update_but_keeps_gradient(self):
        p = torch.nn.Parameter(torch.tensor([2.0]))
        opt = torch.optim.AdamW([p], lr=0.1)
        p.square().sum().backward()
        with stop_before_update(opt):
            opt.step()
            self.fail("Optimizer step must be interrupted")
        self.assertTrue(torch.equal(p, torch.tensor([2.0])))
        self.assertTrue(torch.equal(p.grad, torch.tensor([4.0])))
        self.assertEqual(len(opt.state), 0)
        opt.step()
        self.assertFalse(torch.equal(p, torch.tensor([2.0])))


if __name__ == "__main__":
    unittest.main()
