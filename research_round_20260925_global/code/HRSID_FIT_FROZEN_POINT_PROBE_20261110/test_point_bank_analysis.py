"""Synthetic equal-budget checks for the fixed per-row FIT point readout."""

from math import exp, log1p
import unittest

import numpy as np

from point_bank_analysis import summarize_row


class PointBankTests(unittest.TestCase):
    def test_phase_proxy_and_uncertainty_have_distinct_geometry_with_exact_random_baseline(self):
        hard_c = np.zeros(8, dtype=bool)
        hard_b = np.arange(8) < 4
        raster = np.r_[np.full(4, .25), np.zeros(4)]
        polygon = np.r_[np.full(4, .75), np.zeros(4)]
        logits = np.r_[np.full(4, .8), np.full(4, .01)]
        row = summarize_row(hard_c, hard_b, raster, polygon, logits, np.zeros(8, dtype=int), 251)
        self.assertEqual((row["n"], row["k"], row["proxy_points"]), (8, 4, 4))
        self.assertTrue(row["effective_proxy"])
        phase, uncertainty, expected = (row["banks"][k] for k in ("phase", "uncertainty", "random_expected"))
        self.assertEqual(phase["mean_abs_polygon_minus_raster"], .5)
        self.assertEqual(uncertainty["mean_abs_polygon_minus_raster"], 0.)
        self.assertEqual(expected["mean_abs_polygon_minus_raster"], .25)
        self.assertEqual(expected["proxy_points"], 2.)
        self.assertAlmostEqual(phase["targets"]["raster"]["mean_bce"], log1p(exp(.8)) - .8 * .25)
        self.assertIsNone(expected["targets"]["raster"]["dice_loss"])
        self.assertEqual(phase["gradient_direction_changes"]["polygon_vs_hard_c"]["positive_to_negative"], 4)

    def test_exact_expected_random_preserves_unequal_stratum_quotas(self):
        strata = np.repeat([0, 1, 2], [3, 9, 5])
        c = np.zeros(len(strata), dtype=bool)
        b = np.arange(len(strata)) < 5
        polygon = np.asarray([.1, .5, .75])[strata]
        args = (c, b, c, polygon, np.zeros(len(strata)), strata, 251)
        row = summarize_row(*args)
        self.assertEqual(row, summarize_row(*args))
        self.assertEqual(row["strata_counts"], [3, 9, 5, 0])
        self.assertEqual(row["strata_quotas"], [3, 4, 4, 0])
        for bank in row["banks"].values():
            self.assertAlmostEqual(bank["mean_abs_polygon_minus_raster"], (3*.1 + 4*.5 + 4*.75)/11)

    def test_soft_reference_changes_do_not_change_bank_membership_proxy_counts(self):
        i = np.arange(12)
        c, b, strata = i < 6, i % 3 == 0, i // 6
        first = summarize_row(c, b, c, b, i - 6., strata, 253)
        second = summarize_row(c, b, np.full(12, .9), np.full(12, .8), i - 6., strata, 253)
        for name, bank in first["banks"].items():
            for field in ("proxy_points", "c_adds", "c_removes", "mean_abs_logit", "mean_probability"):
                self.assertEqual(bank[field], second["banks"][name][field])

    def test_direct_zero_proxy_and_invalid_occupancy(self):
        c = np.asarray([0, 1])
        row = summarize_row(c, c, [.3, .7], [.4, .6], [0, 0], c, 0)
        self.assertFalse(row["effective_proxy"])
        self.assertEqual(row["proxy_points"], 0)
        self.assertEqual(row["banks"]["phase"], row["banks"]["uncertainty"])
        self.assertGreater(row["banks"]["phase"]["mean_abs_polygon_minus_raster"], 0.)
        with self.assertRaisesRegex(ValueError, "Soft occupancy"):
            summarize_row(c, c, [-.1, .7], [.4, .6], [0, 0], c, 0)


if __name__ == "__main__":
    unittest.main()
