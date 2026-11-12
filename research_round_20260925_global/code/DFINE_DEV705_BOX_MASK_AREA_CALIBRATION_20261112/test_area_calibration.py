import unittest
import numpy as np
from area_calibration import feasible_splits, select_delta, fit_line, author_box_contains


class FixedCounts(unittest.TestCase):
    def test_same_count_and_smaller_magnitude_tie(self):
        self.assertEqual(feasible_splits(8, 5, 6), [(2, 6), (3, 5), (4, 4), (5, 3)])
        self.assertEqual(select_delta(feasible_splits(8, 5, 6), -1), (4, 4))
        self.assertEqual(select_delta(feasible_splits(8, 5, 6), 1), (4, 4))

    def test_weighted_fit_and_native_box_at_integer_centers(self):
        co = fit_line(np.array([1., 1., 1.]), np.array([.2, .3, .8]), np.array([.25, .25, .5]))
        self.assertAlmostEqual(co[2], .525)
        result = author_box_contains(np.array([1, 2, 3]), np.array([2, 2, 2]), [.5, 1.2, 3., 3.])
        self.assertEqual(result.tolist(), [True, True, False])
