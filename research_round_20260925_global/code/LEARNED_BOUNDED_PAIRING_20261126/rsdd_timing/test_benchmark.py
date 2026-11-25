from pathlib import Path
from types import SimpleNamespace
import os
import unittest
import numpy as np

from benchmark import choose_f, choose_i, choose_l, module, SOLVER_SHA


class ExactTimingPolicyTests(unittest.TestCase):
    def test_equal_counts_and_original_cap_yield_three_distinct_writes(self):
        sibling = Path(__file__).resolve().parents[2] / "PHOENIX_BOUNDED_MATCH_20261123/matching.py"
        path = Path(os.environ.get("SAR_BOUNDED_MATCH_SOLVER", sibling))
        solver = module(path, SOLVER_SHA)
        r = SimpleNamespace(coords=np.asarray([[0, 1], [2, 3], [20, 21]], dtype=np.int32),
                            q=np.asarray([[.70, .05], [.95, .40], [.04, .02]], dtype=np.float32), k=1, cap=1)
        self.assertEqual(tuple(choice.tolist() for choice in choose_f(r)), ([0], [0]))
        self.assertEqual(tuple(choice.tolist() for choice in choose_l(r, solver)), ([1], [0]))
        self.assertEqual(tuple(choice.tolist() for choice in choose_i(r)), ([1], [2]))


if __name__ == "__main__":
    unittest.main()
