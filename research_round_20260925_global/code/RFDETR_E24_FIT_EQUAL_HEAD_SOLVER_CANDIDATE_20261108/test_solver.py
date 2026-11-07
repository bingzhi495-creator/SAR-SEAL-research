import unittest

import numpy as np

from solver import fit_head, predict_head, select_edits, stock_uncertainty


def tied_features(rank):
    size = len(rank)
    return {
        "edit_sign": np.tile([-1, 1], size // 2),
        "abs_logit": np.full(size, 0.75),
        "valid": np.ones(size, dtype=bool),
        "mean_contrast": np.full(size, 0.20),
        "rank": np.asarray(rank),
    }


class EqualHeadTests(unittest.TestCase):
    def test_only_branch_with_informative_feature_learns(self):
        features = tied_features([-0.5] * 4 + [0.5] * 4)
        labels = np.array([False] * 4 + [True] * 4)
        query = tied_features([-0.4, 0.4])

        for branch in ("uncertainty", "mean", "rank"):
            with self.subTest(branch=branch):
                model = fit_head(features, 0.8, labels, branch=branch)
                scores = predict_head(model, query, 0.8)
                if branch == "rank":
                    self.assertLess(scores[0], 0.15)
                    self.assertGreater(scores[1], 0.85)
                else:
                    np.testing.assert_allclose(scores, [0.5, 0.5])


    def test_budget_and_abstention_preserve_distinct_pixel_identities(self):
        indices = np.arange(0, 48, 2, dtype=np.int32)
        scores = np.full(len(indices), 0.7)
        scores[17], scores[20] = 0.95, 0.9

        chosen = select_edits(indices, scores, threshold=None)
        np.testing.assert_array_equal(chosen, np.r_[34, 40, indices[:14]])
        self.assertEqual(len(set(chosen)), 16)
        np.testing.assert_array_equal(select_edits(indices, scores, threshold=0.7), [34, 40])
        self.assertEqual(select_edits(indices, scores, threshold=1.0).size, 0)
        self.assertEqual(select_edits(indices, scores, threshold=None, budget=0).size, 0)
        with self.assertRaises(ValueError):
            select_edits(np.array([2, 2]), np.array([0.9, 0.8]), threshold=None)
        with self.assertRaises(ValueError):
            select_edits(indices, scores, threshold=None, budget=17)

    def test_stock_uncertainty_keeps_large_margin_order(self):
        scores = stock_uncertainty({"abs_logit": np.array([1000.0, 900.0])})
        np.testing.assert_array_equal(scores, [-1000.0, -900.0])
        chosen = select_edits(np.array([3, 7]), scores, threshold=None)
        np.testing.assert_array_equal(chosen, [7, 3])


    def test_weighted_intercept_and_deterministic_fit(self):
        features = tied_features([0.0] * 8)
        labels = np.array([0] * 4 + [1] * 4)
        weights = np.array([1.0] * 4 + [3.0] * 4)
        model = fit_head(features, 0.8, labels, branch="mean", sample_weight=weights)
        rescaled = fit_head(features, 0.8, labels, branch="mean", sample_weight=100 * weights)

        self.assertEqual(model, rescaled)
        np.testing.assert_allclose(predict_head(model, features, 0.8), 0.75)
        np.testing.assert_allclose(predict_head(model, features, np.full(8, 0.8)), 0.75)
        with self.assertRaises(ValueError):
            fit_head(features, 0.8, labels, branch="mean", sample_weight=np.zeros(8))
        with self.assertRaises(ValueError):
            fit_head(features, 0.8, np.zeros(8), branch="mean")


if __name__ == "__main__":
    unittest.main()


