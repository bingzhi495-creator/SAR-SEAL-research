import unittest
import numpy as np
from score_ownership import end_removals, isolated_pairs


class OwnershipTests(unittest.TestCase):
    def test_three_mask_connected_component_is_not_a_pair(self):
        self.assertEqual(isolated_pairs(6, [(0, 1), (1, 2), (3, 4)]), [(3, 4)])

    def test_controls_allocate_disjoint_edits_with_exact_budgets(self):
        ids = np.array([5, 7, 9, 12, 20])
        support = np.array([1., 0., 1., -1., 1.])
        a, b = end_removals(ids, support, 2, 2)
        self.assertEqual(a.tolist(), [12, 7])
        self.assertEqual(b.tolist(), [9, 20])
        self.assertEqual(np.intersect1d(a, b).size, 0)
        _, empty = end_removals(ids, support, 2, 0)
        self.assertEqual(empty.size, 0)


if __name__ == "__main__":
    unittest.main()
