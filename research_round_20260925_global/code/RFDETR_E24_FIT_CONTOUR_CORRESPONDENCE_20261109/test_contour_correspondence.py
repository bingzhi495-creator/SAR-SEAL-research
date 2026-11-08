import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contour_correspondence as c


class ContourTests(unittest.TestCase):
    def test_inward_geometry_and_fixed_axis(self):
        mask = np.zeros((11, 11), dtype=bool)
        mask[5:, :] = True
        positions = np.array([5 * 11 + 5, 4 * 11 + 5])
        ny, nx, strength = c.inward_normal(mask, positions)
        np.testing.assert_array_equal(ny, [1, 1])
        np.testing.assert_array_equal(nx, [0, 0])
        self.assertTrue(np.all(strength > 0))
        gray = np.arange(121, dtype=np.uint8).reshape(mask.shape)
        inward, outward, _ = c.rails(gray, positions, ny, nx)
        fixed_in, fixed_out, _ = c.rails(gray, positions, np.zeros(2), np.ones(2))
        self.assertEqual(inward.shape, (2, 3, 2))
        self.assertEqual(inward.shape, fixed_in.shape)
        self.assertEqual(inward[0, 1, 0], 71)
        self.assertEqual(outward[0, 1, 0], 49)
        self.assertEqual(fixed_in[0, 1, 0], 61)
        self.assertEqual(fixed_out[0, 1, 0], 59)

    def test_zero_normal_is_shared_axis_fallback(self):
        mask = np.zeros((9, 9), dtype=bool)
        ny, nx, strength = c.inward_normal(mask, np.array([0, 40, 80]))
        np.testing.assert_array_equal(ny, [0, 0, 0])
        np.testing.assert_array_equal(nx, [1, 1, 1])
        np.testing.assert_array_equal(strength, [0, 0, 0])

    def test_permutation_preserves_marginals_and_removes_alignment(self):
        inner = np.array([[[4., 4.], [11., 11.], [21., 21.]]] * 2)
        outer = np.array([[[0., 0.], [10., 10.], [20., 20.]]] * 2)
        permuted = c.permute_outer(outer, np.array([0, 1]))
        np.testing.assert_array_equal(np.sort(permuted, axis=1), np.sort(outer, axis=1))
        self.assertTrue(np.all(permuted != outer))
        a, p = [c.pair_features(inner, x, np.array([12., 12.]), np.array([1, 1])) for x in (outer, permuted)]
        np.testing.assert_allclose(a[:, :2], p[:, :2])
        self.assertFalse(np.allclose(a[:, 2:], p[:, 2:]))

    def test_flipping_action_reverses_analytic_signal(self):
        inner = np.full((2, 3, 2), 80.)
        outer = np.full((2, 3, 2), 20.)
        features = c.pair_features(inner, outer, np.array([60., 60.]), np.array([1, -1]))
        np.testing.assert_allclose(features[:, c.ANALYTIC_INDEX], [10 / 255, -10 / 255])

    def test_equal_fit_design_and_predictable_signal(self):
        x = np.arange(24, dtype=float).reshape(12, 2)
        labels = np.arange(12) > 5
        weight = np.ones(12) / 12
        first = c.fit_head(x, labels, weight)
        second = c.fit_head(x.copy(), labels, weight)
        np.testing.assert_allclose(c.predict(x, first), c.predict(x, second))
        self.assertTrue(np.all(np.diff(c.predict(x, first)) > 0))


    def test_design_has_same_taps_and_capacity_across_arms(self):
        gray = np.arange(121, dtype=np.uint8).reshape(11, 11)
        mask = np.zeros_like(gray, dtype=bool)
        mask[:, 5:] = True
        positions = np.array([60, 59])
        features, analytic, counts = c.design(
            gray, mask, positions, np.array([.3, -.1]), np.array([-1, 1]), .7)
        self.assertEqual(set(features), set(c.MODES))
        for mode in c.MODES:
            self.assertEqual(features[mode].shape, (2, len(c.FEATURE_NAMES)))
            self.assertEqual(analytic[mode].shape, (2,))
        np.testing.assert_array_equal(features["aligned"], features["axis"])
        np.testing.assert_array_equal(features["aligned"][:, :7], features["permuted"][:, :7])
        self.assertEqual(counts["zero_normals"], 0)

    def test_same_prefix_budget_and_paired_image_readout(self):
        from types import SimpleNamespace
        signed = c.read_signed_helper()
        positions = np.arange(36)
        labels = positions < 16
        direction = np.where(positions % 2 == 0, 1, -1)
        correct_first = -positions.astype(float)
        wrong_first = positions.astype(float)
        fit_model = dict(center=np.zeros(1), scale=np.ones(1), beta=np.ones(1), bias=0.)
        entry = dict(labels=labels, signs=direction, positions=positions, scores=.8,
                     stock_features={"uncertainty": correct_first},
                     analytic={"aligned": correct_first, "permuted": wrong_first, "axis": wrong_first},
                     features={"aligned": correct_first[:, None], "permuted": wrong_first[:, None],
                               "axis": wrong_first[:, None]})
        solver = SimpleNamespace(stock_uncertainty=lambda f: f["uncertainty"],
                                 predict_head=lambda model, f, score: f["uncertainty"])
        measured = c.measure([[entry], []], solver, {"mean": {}, "rank": {}},
                             dict.fromkeys(c.MODES, fit_model), signed)
        for arm in c.ARMS:
            for budget in c.BUDGETS:
                self.assertEqual(measured[arm]["mixed_direction_prefix"][str(budget)]["combined"]["selected"], budget)
        readout = measured["learned_aligned"]["mixed_direction_prefix"]["16"]
        self.assertEqual(readout["combined"]["correct"], 16)
        self.assertEqual(readout["paired_margin_per_original_instance_vs"]["learned_permuted"]["delta"], 304)
        self.assertEqual(readout["paired_margin_per_original_instance_vs"]["stock"]["delta"], 0)


if __name__ == "__main__":
    unittest.main()
