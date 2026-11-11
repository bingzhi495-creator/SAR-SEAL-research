"""Coverage and witness checks for simultaneous unit-capacity donor matching."""

from itertools import product
import unittest

import numpy as np

from dual_flow import common_flow


def brute_cardinality(left, right):
    per_anchor = [
        [None, *product(np.flatnonzero(a), np.flatnonzero(b))]
        for a, b in zip(left, right, strict=True)
    ]
    largest = 0
    for sequence in product(*per_anchor):
        used = [pair for pair in sequence if pair is not None]
        if len({a for a, _ in used}) == len({b for _, b in used}) == len(used):
            largest = max(largest, len(used))
    return largest


class CommonFlowTests(unittest.TestCase):
    def test_joint_match_can_differ_from_separate_first_choices(self):
        left = np.array([[True], [True]])
        right = np.array([[False], [True]])

        self.assertEqual(common_flow(left, right), {1: (0, 0)})

    def test_no_flow_with_empty_or_missing_edges(self):
        for left_shape, right_shape in [((0, 3), (0, 2)), ((2, 0), (2, 3)), ((2, 3), (2, 0)), ((2, 2), (2, 3))]:
            left = np.zeros(left_shape, dtype=bool)
            right = np.ones(right_shape, dtype=bool)

            self.assertEqual(common_flow(left, right), {})

    def test_witness_is_repeatable_and_supports_different_donor_counts(self):
        left = np.ones((4, 2), dtype=bool)
        right = np.ones((4, 3), dtype=bool)
        expected = common_flow(left, right)

        self.assertEqual(len(expected), 2)
        for _ in range(4):
            self.assertEqual(common_flow(left, right), expected)

    def test_all_4096_graphs_with_three_anchors_and_two_donors_each(self):
        for pattern in product([False, True], repeat=12):
            left, right = np.array(pattern, dtype=bool).reshape((2, 3, 2))
            witness = common_flow(left, right)
            self.assertEqual(len(witness), brute_cardinality(left, right), pattern)
            self.assertEqual(len({a for a, _ in witness.values()}), len(witness), pattern)
            self.assertEqual(len({b for _, b in witness.values()}), len(witness), pattern)
            self.assertTrue(all(left[i, a] and right[i, b] for i, (a, b) in witness.items()), pattern)


if __name__ == "__main__":
    unittest.main()
