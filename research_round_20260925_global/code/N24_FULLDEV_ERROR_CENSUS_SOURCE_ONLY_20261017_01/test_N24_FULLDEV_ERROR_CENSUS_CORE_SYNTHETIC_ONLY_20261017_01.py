# encoding: utf-8
"""SYNTHETIC_ONLY tracer for maximum-cardinality IoU-threshold matching.

No real predictions, labels, scientific references, or files are loaded here.
Root owns RED/GREEN execution and the implementation.
"""
import unittest

import numpy as np

import N24_FULLDEV_ERROR_CENSUS_CORE_UNBOUND_20261017_01 as core


class FullDevErrorCensusMatchingContractTests(unittest.TestCase):
    def test_augmenting_rematch_retains_both_feasible_detection_GT_pairs(self):
        ious = np.array([[0.7, 0.7], [0.6, 0.0]], dtype=np.float64)

        pairs = core.maximum_matching(np, ious, threshold=0.5)

        self.assertIsInstance(pairs, list)
        self.assertEqual(len(pairs), 2)
        self.assertTrue(all(isinstance(pair, tuple) and len(pair) == 2 for pair in pairs))
        self.assertEqual(len({dt for dt, gt in pairs}), 2)
        self.assertEqual(len({gt for dt, gt in pairs}), 2)
        for dt, gt in pairs:
            self.assertTrue(0 <= dt < ious.shape[0] and 0 <= gt < ious.shape[1])
            self.assertGreaterEqual(ious[dt, gt], 0.5)
        self.assertEqual(set(pairs), {(0, 1), (1, 0)})

    def test_all_2_by_3_binary_graphs_match_independent_brute_force(self):
        import itertools
        for bits in itertools.product((0., 1.), repeat=6):
            matrix = np.array(bits, dtype=np.float64).reshape(2, 3)
            best = 0
            for assignment in itertools.product((-1, 0, 1, 2), repeat=2):
                used = [g for g in assignment if g != -1]
                if len(used) != len(set(used)):
                    continue
                if all(g == -1 or matrix[d, g] >= .5 for d, g in enumerate(assignment)):
                    best = max(best, len(used))
            self.assertEqual(len(core.maximum_matching(np, matrix, .5)), best)

    def test_empty_population_zero_overlap_and_invalid_IoU(self):
        for shape in [(0, 3), (2, 0), (2, 3)]:
            self.assertEqual(core.maximum_matching(np, np.zeros(shape, dtype=np.float64), .5), [])
        with self.assertRaises(ValueError):
            core.maximum_matching(np, np.array([[float('nan')]]), .5)


if __name__ == '__main__':
    unittest.main()
