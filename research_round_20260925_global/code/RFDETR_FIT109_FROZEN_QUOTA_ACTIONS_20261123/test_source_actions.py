"""Synthetic checks of quota enumeration, features and physical supervision."""

from pathlib import Path
from runpy import run_path
import unittest

import numpy as np


API = run_path(str(Path(__file__).with_name("source_actions.py")))
prepare = API["actions_and_features"]
targets = API["geometric_targets"]


class SourceActionsTest(unittest.TestCase):
    def setUp(self):
        self.coords = np.column_stack((np.arange(4), np.arange(10,14)))
        self.q = np.array([[.9,.1],[.8,.2],[.2,.8],[.1,.9]], dtype=np.float32)
        self.ranks = np.arange(4), np.arange(4)

    def test_every_feasible_quota_and_balanced_anchor(self):
        k, moves, feats = prepare(self.coords, self.q, self.ranks, 50, .45)
        self.assertEqual(k, 2)
        self.assertEqual(moves.tolist(), [[0,4],[1,3],[2,2],[3,1],[4,0]])
        np.testing.assert_array_equal(feats[2,:5], np.zeros(5))
        np.testing.assert_allclose(feats[:,0], [-.08,-.04,0,.04,.08])
        self.assertEqual(feats[0, 5], 50.)

    def test_truth_preserves_continuous_union_and_fixed_k(self):
        k, moves, _ = prepare(self.coords, self.q, self.ranks, 50, .45)
        truth = np.array([[.9,.15],[.75,.2],[.5,.7],[.1,.9]])
        actual = targets(truth, self.ranks, dict(intersection=38, predicted_area=50, truth_area=43), moves, k)
        direct = []
        for a, d in moves:
            inter = 38 + sum(truth[:a,0]) - sum(truth[:d,1])
            direct.append(inter/(50+a-d+43-inter))
        np.testing.assert_allclose(actual, np.asarray(direct)-direct[2], rtol=0, atol=1.e-16)

    def test_zero_budget_only_keeps_original(self):
        q = np.array([[.1,.7],[.2,.8],[.3,.8],[.5,.9]], dtype=np.float32)
        k, moves, feats = prepare(self.coords, q, (np.arange(3,-1,-1), np.arange(4)), 32, .25)
        self.assertEqual(k, 0)
        np.testing.assert_array_equal(moves, [[0,0]])
        np.testing.assert_array_equal(feats[0,:5], np.zeros(5))
        np.testing.assert_array_equal(
            targets(q, self.ranks, dict(intersection=20, predicted_area=32, truth_area=30), moves, k), [0.])


if __name__ == "__main__":
    unittest.main()
