import json
import unittest

import numpy as np

from candidate_overlap import distinct_pair_cases


class CandidateOverlapTests(unittest.TestCase):
    def test_true_neighbours_and_high_overlap_duplicates_are_separate(self):
        predicted = np.array([
            [1.0, 0.2, 0.8, 0.0],
            [0.2, 1.0, 0.01, 0.1],
            [0.8, 0.01, 1.0, 0.0],
            [0.0, 0.1, 0.0, 1.0],
        ])
        match = [101, 102, None, None]
        overlap = distinct_pair_cases(predicted, match, {frozenset((101, 102))})
        self.assertEqual(json.loads(json.dumps(overlap)), overlap)
        self.assertEqual(overlap["distinct_matched_overlap_pairs"], 1)
        self.assertEqual(overlap["adjacent_distinct_matched_overlap_pairs"], 1)
        self.assertEqual(overlap["unmatched_overlaps_matched_ge50"], 1)
        self.assertEqual(overlap["adjacent_distinct_matched_overlap_indices"], [0, 1])

    def test_disjoint_predictions_and_distant_truth_do_not_count_as_neighbor_conflicts(self):
        predicted = np.eye(3)
        predicted[1, 2] = predicted[2, 1] = 0.1
        overlap = distinct_pair_cases(predicted, [101, 102, 103], {frozenset((101, 102))})
        self.assertEqual(overlap["distinct_matched_overlap_pairs"], 1)
        self.assertEqual(overlap["adjacent_distinct_matched_overlap_pairs"], 0)
        self.assertEqual(overlap["adjacent_distinct_matched_pairs"], 1)


if __name__ == "__main__":
    unittest.main()
