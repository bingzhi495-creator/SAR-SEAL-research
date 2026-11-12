import unittest
import numpy as np
from score_consensus import action_indices


class EqualCountActions(unittest.TestCase):
    def test_band_controls_keep_exact_counts_and_cannot_flip_outside_band(self):
        hard = np.array([False, True, False, True, False, True])
        majority = np.array([True, False, False, True, True, False])
        anchor = np.array([.10, .91, .49, .52, .20, .51])
        mean3 = np.array([.90, .39, .48, .55, .70, .08])
        result = action_indices(hard, majority, anchor, mean3, cap=2)
        self.assertEqual(result["vote"].tolist(), [0, 5])
        self.assertEqual(result["soft"].tolist(), [0, 5])
        self.assertEqual(result["anchor_count"].tolist(), [2, 5])
        for choice in result.values():
            self.assertEqual(int(hard[choice].sum()), 1)
            self.assertEqual(len(choice), 2)

    def test_ties_resolve_by_row_major_and_empty_majority_is_unchanged(self):
        hard = np.array([False, False, True, True])
        majority = np.array([True, True, False, False])
        prob = np.array([.75, .75, .25, .25])
        result = action_indices(hard, majority, prob, prob, cap=1)
        self.assertEqual(result["vote"].tolist(), [0])
        self.assertEqual(result["soft"].tolist(), [0])
        self.assertTrue(all(len(x) == 0 for x in action_indices(hard, hard, prob, prob, 2).values()))
