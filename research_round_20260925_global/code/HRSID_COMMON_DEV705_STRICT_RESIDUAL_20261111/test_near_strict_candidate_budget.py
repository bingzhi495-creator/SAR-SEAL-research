from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from near_strict_candidate_budget import assign_budget, minimum_edits, radius_one_capacity


class BudgetTests(unittest.TestCase):
    def test_cardinality_precedes_edit_cost_and_respects_unique_candidates(self):
        cost = np.array([[1, 4], [2, 17]])
        self.assertEqual(assign_budget(cost, 4), (2, 6))
        self.assertEqual(assign_budget(cost, 2), (1, 1))

    def test_false_negative_addition_has_priority(self):
        self.assertEqual(minimum_edits(19, 1, 1), 2)
        self.assertIsNone(minimum_edits(19, 0, 2))

    def test_one_pixel_band_counts_editable_error_and_border(self):
        prediction = np.zeros((4,4),dtype=bool)
        truth = np.zeros_like(prediction)
        prediction[0,0] = True
        truth[1,1] = truth[3,3] = True
        self.assertEqual(radius_one_capacity(prediction,truth), (1,1))


if __name__ == "__main__":
    unittest.main()
