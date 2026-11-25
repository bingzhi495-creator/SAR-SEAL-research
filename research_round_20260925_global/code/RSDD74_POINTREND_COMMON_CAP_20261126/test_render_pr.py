"""Synthetic controls for the matched signed-budget PointRend writer."""
from pathlib import Path
import sys
import unittest

import numpy as np
from pycocotools import mask as codec

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_pr import learned_module, pixel_indices, select_pr, write_mask


class WriterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        here = Path(__file__).resolve().parent
        root = here.parent
        code = root if (root / "LEARNED_BOUNDED_PAIRING_20261126").is_dir() else (
            root / "github_release_20261011/research_round_20260925_global/code"
        )
        learned = learned_module(code)
        cls.solver = learned.module(code / "PHOENIX_BOUNDED_MATCH_20261123/matching.py", learned.SOLVER_SHA)

    def test_learned_fixed_cap_is_shared_instead_of_reselected(self):
        coords = np.array([[0, 1], [10, 11], [20, 22]], dtype=np.int64)
        prob = np.array([[.9, .9], [.7, .05], [.8, .1]], dtype=np.float32)
        chosen, cap, stats = select_pr(self.solver, coords, prob, np.array([2]), 32)
        self.assertEqual(cap, 4)
        self.assertEqual(chosen["pr_fixed"], [(2, 2)])
        self.assertEqual(chosen["pr_bounded"], [(2, 2)])
        self.assertEqual(chosen["pr_independent"], [(0, 1)])
        self.assertFalse(stats["independent_feasible"])

    def test_bounded_recombines_vertices_and_preserves_exact_area(self):
        coords = np.array([[0, 3], [4, 1], [8, 11]], dtype=np.int64)
        prob = np.array([[.95, .9], [.7, .05], [.1, .8]], dtype=np.float32)
        chosen, cap, stats = select_pr(self.solver, coords, prob, np.array([0]), 16)
        self.assertEqual(cap, 9)
        self.assertEqual(chosen["pr_fixed"], [(1, 1)])
        self.assertEqual(chosen["pr_bounded"], [(0, 1)])
        self.assertEqual(chosen["pr_independent"], [(0, 1)])
        self.assertTrue(stats["independent_feasible"])
        stock = np.zeros((2, 16), dtype=bool)
        stock.flat[coords[:, 1]] = True
        saved = write_mask(stock, coords, chosen["pr_bounded"], codec)
        adds, drops = pixel_indices(stock, saved, coords, codec)
        self.assertEqual(saved["swaps"], 1)
        self.assertEqual((adds.tolist(), drops.tolist()), ([0], [1]))

    def test_equal_posterior_uses_common_stable_row_order(self):
        coords = np.array([[10, 11], [0, 1], [20, 21]], dtype=np.int64)
        prob = np.array([[.5, .5], [.5, .5], [.5, .5]], dtype=np.float32)
        chosen, cap, _ = select_pr(self.solver, coords, prob, np.array([1]), 32)
        self.assertEqual(cap, 1)
        self.assertEqual(set(tuple(rows) for rows in chosen.values()), {((0, 0),)})


if __name__ == "__main__":
    unittest.main()
