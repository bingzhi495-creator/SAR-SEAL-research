"""Degenerate cases, frozen rank ties and exhaustive agreement for fixed quota."""

from pathlib import Path
from runpy import run_path
import unittest

import numpy as np


API = run_path(str(Path(__file__).with_name("quota.py")))
quota = API["candidate_quota"]


class QuotaTest(unittest.TestCase):
    def test_prefix_ties_retain_original_order(self):
        q = np.array([[.8, .2], [.8, .2], [.3, .3]], dtype=np.float32)
        coords = np.array([[3, 22], [2, 20], [1, 19]])
        add, drop = API["original_ranks"](q)
        API["frozen_prefix"](coords, add, drop, [3, 22, 2, 20])
        API["frozen_prefix"](coords, add, drop, ())
        with self.assertRaisesRegex(ValueError, "archived alternating"):
            API["frozen_prefix"](coords, add, drop, [2, 20, 3, 22])

    def test_zero_budget(self):
        geometry = dict(truth=np.empty((0, 2)), raw_continuous=dict(intersection=28, predicted_area=32, truth_area=40))
        baseline, oracle = quota(geometry, np.array([], dtype=int), np.array([], dtype=int), 0)
        self.assertEqual(baseline, oracle)
        self.assertAlmostEqual(baseline["iou"], 28/(32+40-28))

    def test_exhaustive_nonuniform_rank_and_union(self):
        rng = np.random.default_rng(22)
        for _ in range(128):
            truth = rng.random((5, 2))
            raw = dict(intersection=25, predicted_area=40, truth_area=40)
            edits = 4
            add_order, drop_order = rng.permutation(len(truth)), rng.permutation(len(truth))
            baseline, oracle = quota(dict(truth=truth, raw_continuous=raw), add_order, drop_order, edits)
            results = []
            for add in range(edits+1):
                drop = edits-add
                intersection = 25 + sum(truth[p,0] for p in add_order[:add]) - sum(truth[p,1] for p in drop_order[:drop])
                area = 40+add-drop
                results.append(intersection/(area+40-intersection))
            self.assertAlmostEqual(oracle["iou"], max(results), places=14)
            self.assertAlmostEqual(baseline["iou"], results[edits//2], places=14)
            self.assertEqual(oracle["adds"]+oracle["drops"], edits)


if __name__ == "__main__":
    unittest.main()
