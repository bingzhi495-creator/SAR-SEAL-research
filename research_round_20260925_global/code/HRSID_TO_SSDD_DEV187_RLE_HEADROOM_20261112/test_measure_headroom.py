import unittest

import numpy as np

from measure_headroom import oracle_mask


class OracleTest(unittest.TestCase):
    def test_truth_preference_and_row_major_ties(self):
        anchor = np.array([[False, False, False], [True, True, True]])
        truth = np.array([[False, True, True], [True, False, False]])
        allowed = np.ones_like(anchor)
        after = oracle_mask(anchor, truth, allowed, 1, 1)
        expected = np.array([[False, True, False], [True, False, True]])
        np.testing.assert_array_equal(after, expected)

    def test_no_overshoot_when_truth_help_is_scarce(self):
        anchor = np.array([[False, False, False], [True, True, True]])
        truth = np.array([[False, True, False], [True, False, True]])
        allowed = np.ones_like(anchor)
        after = oracle_mask(anchor, truth, allowed, 2, 2)
        self.assertEqual(int((after & ~anchor).sum()), 2)
        self.assertEqual(int((~after & anchor).sum()), 2)
        self.assertEqual(int((after & truth & ~anchor).sum()), 1)
        self.assertEqual(int((~after & ~truth & anchor).sum()), 1)

    def test_untouched_and_unreachable_bounds(self):
        anchor = np.ones((3, 3), dtype=bool)
        blank = np.zeros_like(anchor)
        np.testing.assert_array_equal(oracle_mask(anchor, blank, blank, 0, 0), anchor)
        with self.assertRaisesRegex(ValueError, "Infeasible"):
            oracle_mask(anchor, blank, blank, 0, 1)


if __name__ == "__main__":
    unittest.main()