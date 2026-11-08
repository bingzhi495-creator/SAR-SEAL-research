from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matcher_cost_math as m


class MatcherCostTests(unittest.TestCase):
    def test_equal_budget_shared_draws_fill_whole_grid(self):
        u = np.zeros((m.K, 2), np.float32)
        u[::2] = np.nextafter(np.float32(1), np.float32(0))
        before = u.copy()
        iid, jittered = m.paired_points(u)
        self.assertEqual(jittered.shape, (1296, 2))
        np.testing.assert_array_equal(iid, before)
        np.testing.assert_array_equal(u, before)
        self.assertEqual(len(np.unique(np.floor(jittered.astype(np.float64) * 36), axis=0)), 1296)
        self.assertTrue(np.all((jittered >= 0) & (jittered < 1)))

    def test_native_terms_and_dense_dice_smoothing(self):
        def flat_sample(points):
            return np.zeros((2, len(points)), np.float32), np.ones((1, len(points)), np.float32)

        points = np.zeros((m.K, 2), np.float32)
        bce, dice = m.sampled_cost(*flat_sample(points))
        dense = m.dense_cost(flat_sample, side=5, phase=(.5, .5), chunk=3)
        np.testing.assert_allclose(bce, np.log(2), atol=2e-6)
        np.testing.assert_allclose(dice, 1 - (m.K + 1) / (1.5 * m.K + 1), atol=2e-7)
        np.testing.assert_allclose(dense.bce, bce, atol=2e-6)
        np.testing.assert_allclose(dense.dice, dice, atol=2e-7)
        np.testing.assert_array_equal(dense.intersection, 0)
        np.testing.assert_array_equal(dense.union, 25)

    def test_grouped_assignments_use_contiguous_queries(self):
        cost = np.array([[2, 0], [0, 2], [0, 2], [2, 0]], dtype=np.float32)
        np.testing.assert_array_equal(m.solve_groups(cost, width=2, groups=2), [[1, 0], [2, 3]])
        np.testing.assert_array_equal(
            m.take_assignments(cost, m.solve_groups(cost, width=2, groups=2)), 0)

    def test_decomposition_distinguishes_competition_and_objective_gap(self):
        good = np.array([[True, True, False], [False, False, False], [False, False, False]])
        dense = np.array([[1, 2, 0]])
        sampled = np.array([[0, 2, 1]])
        scope = np.ones(3, dtype=bool)
        reference = m.reference_counts(good, dense, dense, scope, width=3)
        self.assertEqual(reference["no_good_query"], 1)
        self.assertEqual(reference["query_competition_gap"], 1)
        self.assertEqual(reference["dense_objective_gap"], 1)
        self.assertEqual(reference["maximum_one_to_one_good"], 1)
        outcomes = m.sample_counts(good, dense, dense, sampled, scope)
        self.assertEqual(outcomes["stable_reference_good_lost"], 0)
        self.assertEqual(outcomes["stable_reference_bad_gained"], 1)

    def test_reversal_and_unstable_dense_reference_are_separate(self):
        good = np.array([[True, False], [False, True], [False, False]])
        ref_a = np.array([[0, 1]])
        ref_b = np.array([[0, 2]])
        sampled = np.array([[2, 1]])
        outcomes = m.sample_counts(good, ref_a, ref_b, sampled, np.ones(2, dtype=bool))
        self.assertEqual(outcomes["stable_reference_good_lost"], 1)
        self.assertEqual(outcomes["dense_quality_unstable"], 1)

    def test_view_runs_paired_arms_without_live_dependencies(self):
        def sample(points):
            target = (points[:, 0] < .5)[None, :].astype(np.float32)
            a = 2 - 4 * points[:, 0]
            return np.vstack((a, -a, -a, a)), target

        nonmask = np.zeros((4, 1), np.float32)
        u = np.random.default_rng(515).random((m.K, 2), dtype=np.float32)
        result = m.diagnose_view(nonmask, sample, np.array([True]), (5., 5.), [u],
                                 width=2, groups=2, side=4, chunk=5)
        self.assertEqual(result["point_budget"], m.K)
        reference = result["reference"]["iou75"]["small"]
        self.assertEqual(reference["dense_good"], 2)
        self.assertEqual(reference["stable_dense_good"], 2)
        self.assertEqual(result["draws"][0]["iid"]["iou75"]["small"]["stable_reference_good_lost"], 0)
        self.assertEqual(result["draws"][0]["jitter"]["iou75"]["small"]["stable_reference_good_lost"], 0)


    def test_native_composition_and_solver_observation_are_independent(self):
        nonmask = np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float32)
        ce = np.array([[.1, .2], [.1, .2]], dtype=np.float32)
        dice = np.array([[.05, .15], [.05, .15]], dtype=np.float32)
        total = (nonmask + 5 * ce) + 5 * dice
        observed = m.assignment_from_native_pairs(np.array([1, 0]), np.array([0, 1]),
                                                  targets=2, width=2, groups=1)
        anchor = m.exact_native_anchor(nonmask, ce, dice, total, observed, (5., 5.))
        self.assertTrue(anchor["native_total_exact"])
        self.assertFalse(anchor["solver_pairs_exact"])
        self.assertEqual(anchor["max_group_objective_gap"], 0.)

    def test_empty_augmented_view_is_well_defined(self):
        def sample(points):
            return np.zeros((4, len(points)), np.float32), np.zeros((0, len(points)), np.float32)

        draw = np.zeros((m.K, 2), np.float32)
        result = m.diagnose_view(np.zeros((4, 0), np.float32), sample, np.array([], bool),
                                 (5., 5.), [draw], width=2, groups=2, side=3)
        self.assertEqual(result["reference"]["iou75"]["small"]["targets"], 0)
        self.assertEqual(result["draws"][0]["iid"]["dense_cost_regret"], 0.)


if __name__ == "__main__":
    unittest.main()
