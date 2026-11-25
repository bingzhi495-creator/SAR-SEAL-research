"""Synthetic checks of endpoint counts, whole-reference objective and static fallback."""
from importlib.util import module_from_spec, spec_from_file_location
from itertools import combinations, product
from pathlib import Path
import unittest

import numpy as np

SPEC = spec_from_file_location("oracle", Path(__file__).with_name("oracle.py"))
ORACLE = module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


class OracleTests(unittest.TestCase):
    def test_exhaustive_binary_cases(self):
        rng = np.random.Generator(np.random.PCG64(21877))
        for _ in range(250):
            n = int(rng.integers(1, 7))
            k = int(rng.integers(1, n+1))
            add, drop = rng.integers(4, size=(2, n))
            pg, dg = ORACLE.MEMBERSHIP[add], ORACLE.MEMBERSHIP[drop]
            raw = dg.sum(axis=0) + 2
            total = pg.sum(axis=0) + dg.sum(axis=0) + 4 + n
            static = raw + pg[:k].sum(axis=0) - dg[:k].sum(axis=0)
            denom = rng.integers(1, 6, size=2)
            aa, dd, effect, lower = ORACLE.maximin(add, drop, k, raw, total, static, denom)
            pair_product = product(combinations(range(n), k), repeat=2)
            direct = max(float(((ORACLE.iou(raw + pg[list(a)].sum(axis=0) - dg[list(d)].sum(axis=0), total)
                                 - ORACLE.iou(static, total)) / denom).min())
                         for a, d in pair_product)
            self.assertAlmostEqual(lower, direct, places=12)
            if aa is not None:
                self.assertEqual(int(sum(aa)), k)
                self.assertEqual(int(sum(dd)), k)
                self.assertAlmostEqual(float((effect / denom).min()), direct, places=12)

    def test_zero_margin_prefers_existing_static(self):
        out = ORACLE.maximin(np.array([1, 2]), np.array([0, 0]), 1,
                             np.array([1, 1]), np.array([7, 7]), np.array([2, 1]), np.ones(2))
        self.assertIsNone(out[0])
        self.assertEqual(out[-1], 0.0)

    def test_balanced_disagreement_can_help_both_references(self):
        # Static adds two PSeg-only cells and removes two both-foreground pixels.
        # A balanced pair of one-sided additions and two common-background deletions improves both.
        out = ORACLE.maximin(np.array([1, 1, 2, 2]), np.array([3, 0, 0, 3]), 2,
                             np.array([2, 2]), np.array([8, 8]), np.array([2, 0]), np.ones(2))
        self.assertGreater(out[-1], 0.0)
        self.assertTrue(np.all(out[2] > 0))


if __name__ == "__main__":
    unittest.main()
