import unittest

import numpy as np

from coverage_probe import best_assignment


class Cardinality(unittest.TestCase):
    def test_cardinality_is_prioritized_over_single_high_overlap(self):
        overlap = np.array([[.99, .50], [.50, .49]])
        rows, cols = best_assignment(overlap, .50)
        self.assertEqual(list(zip(rows, cols)), [(0, 1), (1, 0)])

    def test_later_candidate_only_in_full_pool(self):
        overlap = np.zeros((101, 1))
        overlap[0, 0] = .75
        overlap[100, 0] = .90
        self.assertEqual(len(best_assignment(overlap[:100], .90)[0]), 0)
        self.assertEqual(best_assignment(overlap, .90)[0].tolist(), [100])

    def test_max_cardinality_prefers_max_iou_among_exact_ties(self):
        overlap = np.array([[.91, .92], [.90, .99]])
        rows, cols = best_assignment(overlap, .90)
        self.assertEqual(list(zip(rows, cols)), [(0, 0), (1, 1)])

    def test_empty_side(self):
        rows, cols = best_assignment(np.empty((2, 0)), .50)
        self.assertEqual((len(rows), len(cols)), (0, 0))


if __name__ == "__main__":
    unittest.main()