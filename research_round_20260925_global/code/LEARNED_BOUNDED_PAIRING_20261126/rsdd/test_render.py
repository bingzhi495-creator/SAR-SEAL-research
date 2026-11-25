import os
import unittest
from pathlib import Path
import numpy as np

from render import append_bounded, module, SOLVER_SHA

DEFAULT_SOLVER = Path(__file__).resolve().parents[2] / "PHOENIX_BOUNDED_MATCH_20261123/matching.py"
SOLVER = Path(os.environ.get("SAR_BOUNDED_MATCH_SOLVER", DEFAULT_SOLVER))


class BoundedRendererTests(unittest.TestCase):
    def test_cross_pair_can_improve_score_without_allowing_far_free_edit(self):
        solver = module(SOLVER, SOLVER_SHA)
        coords = np.array([[0, 1], [2, 3], [20, 21]], dtype=np.int32)
        posterior = np.array([[.70, .05], [.95, .40], [.04, .02]], dtype=np.float32)
        fixed = [(0, 0)]
        free = [(1, 2)]
        initial = dict(fixed=fixed, free_matched_count=free)
        actions, info = append_bounded(solver, initial, coords, posterior, 32)
        self.assertEqual(actions["bounded"], [(1, 0)])
        self.assertEqual(actions["fixed"], fixed)
        self.assertEqual(actions["free_matched_count"], free)
        self.assertEqual(info["cap_squared_sum"], 1)
        self.assertEqual(info["independent_feasible"], 0)
        self.assertEqual(info["bounded_gain_gt_fixed"], 1)
        self.assertEqual(info["independent_gain_gt_bounded"], 1)


if __name__ == "__main__":
    unittest.main()
