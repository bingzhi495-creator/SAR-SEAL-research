import unittest

import numpy as np

from sampling_risk import BilinearRisk, boundary_quota


class RiskTests(unittest.TestCase):
    def test_variance_matches_exact_two_point_example(self):
        probe = BilinearRisk((1, 1), (1, 2))
        answer = probe.measure(np.array([[0.0]]), np.array([[1, 0]]),
                               np.array([[1, 0]], dtype=bool), 2)
        self.assertAlmostEqual(answer["global_uniform"], 0.0703125)
        self.assertAlmostEqual(answer["boundary_uniform"], 0.017578125)
        self.assertAlmostEqual(answer["global_stratified"], 0.0)
        self.assertAlmostEqual(answer["boundary_stratified"], 0.0)
        self.assertAlmostEqual(answer["global_mean_squared"], 0.0)

    def test_empty_stratum_falls_back_to_uniform_and_allocations_stay_positive(self):
        probe = BilinearRisk((1, 1), (1, 2))
        answer = probe.measure(np.array([[0.0]]), np.array([[1, 0]]),
                               np.array([[0, 0]], dtype=bool), 2)
        self.assertEqual(answer["global_uniform"], answer["global_stratified"])
        self.assertEqual(answer["boundary_uniform"], 0.0)
        self.assertEqual(boundary_quota(1296, 1, 331776), 65)
        self.assertEqual(boundary_quota(4, 99, 100), 3)
        self.assertEqual(boundary_quota(4, 100, 100), 4)
