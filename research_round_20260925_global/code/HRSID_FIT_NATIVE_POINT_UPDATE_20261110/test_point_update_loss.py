"""Synthetic matched-budget and differentiation checks for fixed P253 point updates."""

from pathlib import Path
import sys
import unittest

import numpy as np
import torch

P252 = (Path(__file__).resolve().parents[2] / "github_release_20261011/research_round_20260925_global/code"
        / "HRSID_FIT_FROZEN_POINT_PROBE_20261110")
sys.path.insert(0, str(P252))

import point_bank_analysis as previous
from point_update_loss import freeze_row_banks, paired_point_loss


class PointUpdateLossTests(unittest.TestCase):
    def test_frozen_selection_matches_p252_and_survives_target_replacement(self):
        strata = np.repeat(np.arange(4), [3, 9, 5, 1])
        c = strata < 2
        b = (np.arange(len(strata)) % 3) != 0
        logits = np.linspace(-1., 1., len(strata))
        raster = np.linspace(.1, .9, len(strata))
        initial = freeze_row_banks(c, b, raster, logits, strata, 252)
        changed = freeze_row_banks(c, b, 1-raster, logits, strata, 252)
        expected, _, quotas, _, _ = previous._banks(strata, b != c, logits, 252)
        self.assertEqual(set(initial), {"phase", "uncertainty", "random"})
        self.assertEqual(quotas, [3, 4, 4, 1])
        for arm in initial:
            bank = initial[arm]
            np.testing.assert_array_equal(bank.indices, expected[arm])
            np.testing.assert_array_equal(bank.indices, changed[arm].indices)
            np.testing.assert_array_equal(np.bincount(strata[bank.indices], minlength=4), quotas)
            np.testing.assert_array_equal(bank.targets, raster[bank.indices])
            np.testing.assert_array_equal(changed[arm].targets, (1-raster)[bank.indices])
            with self.assertRaises(ValueError):
                bank.indices.setflags(write=True)
            with self.assertRaises(ValueError):
                bank.targets.setflags(write=True)

    def test_bce_uses_equal_rows_and_matches_analytic_gradient(self):
        u = torch.tensor([-.4, .5, 1.2], dtype=torch.float64, requires_grad=True)
        v = torch.tensor([.4, -.7], dtype=torch.float64, requires_grad=True)
        a = np.array([.1, .75, 1.], dtype=np.float64)
        b = torch.tensor([.25, .0], dtype=torch.float64)
        loss = paired_point_loss([(u, a), (v, b)], bce_weight=1., dice_weight=0.)
        expected = sum((torch.logaddexp(torch.zeros_like(z), z)-z*t).mean()
                       for z,t in ((u, torch.tensor(a)), (v,b))) / 2.
        torch.testing.assert_close(loss, expected)
        loss.backward()
        for z,t in ((u, torch.tensor(a)), (v,b)):
            self.assertTrue(torch.isfinite(z.grad).all())
            torch.testing.assert_close(z.grad, (torch.sigmoid(z.detach())-t) / (2 * z.numel()))


if __name__ == "__main__":
    unittest.main()
