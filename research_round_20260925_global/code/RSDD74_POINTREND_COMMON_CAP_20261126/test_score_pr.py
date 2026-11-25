"""Basic synthetic guard tests for the PointRend-by-writer crossover."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np

from score_pr import MASK_ARMS, calculate_contrast, cap_feasible, slot_rows


class CrossOver(unittest.TestCase):
    def test_writers_have_shared_paired_difference_in_differences(self):
        arm_names = ("raw", *MASK_ARMS)
        values = np.array([[14., 12., 17., 18., 12., 15., 12.],
                           [10., 11., 15., 14., 12., 12., 20.]])
        interaction = calculate_contrast(values, arm_names, "writer_head_interaction")
        self.assertEqual(interaction.tolist(), [4., -9.])

    def test_cap_uses_existence_of_matching_not_pair_order(self):
        additions = np.array([0,13])
        removals = np.array([14,1])
        self.assertTrue(cap_feasible(additions, removals, 12, 1))
        self.assertFalse(cap_feasible(additions, removals, 12, 0))

    def test_signed_tensor_selection_preserves_original_row_links(self):
        original = np.array([[0,1],[1,0],[0,1],[0,1]],dtype=bool)
        extra = np.array([[0,1],[1,0],[1,0],[0,1]],dtype=bool)
        self.assertEqual(tuple(x.tolist() for x in slot_rows(original,extra)),([2],[2]))
        bad = extra.copy()
        bad[3,1] = False
        with self.assertRaises(ValueError):
            slot_rows(original,bad)


if __name__ == "__main__":
    unittest.main()

