import unittest
import numpy as np

from equal_prefix import exact_maximum


class MatchingTest(unittest.TestCase):
    def test_size_counts_share_one_maximum_assignment(self):
        matrix = np.array([[.90, .80], [.81, 0.]])
        got = exact_maximum(matrix, .75)
        self.assertEqual(got.tolist(), [True, True])

    def test_conflicted_target_cannot_be_counted_twice(self):
        got = exact_maximum(np.array([[.9, .9]]), .90)
        self.assertEqual(int(got.sum()), 1)

    def test_no_candidates_keeps_uncovered_gt(self):
        self.assertEqual(exact_maximum(np.zeros((0, 3)), .50).tolist(), [False] * 3)


if __name__ == "__main__":
    unittest.main()

