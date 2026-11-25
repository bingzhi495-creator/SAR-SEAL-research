"""Check exact fixed and variable signed budgets on synthetic masks."""
import importlib.util
from itertools import combinations
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('oracle', Path(__file__).with_name('oracle.py'))
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class Oracle(unittest.TestCase):
    def test_exact_budgets_against_enumeration(self):
        stock = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
        adds, drops = np.arange(4, 8), np.arange(4)
        rng = np.random.default_rng(5217)
        for _ in range(15):
            actual = rng.integers(0, 2, len(stock)).astype(bool)
            actual[0] = True
            for k in range(5):
                all_scores, balanced = [], []
                for a in range(5):
                    d = 2 * k - a
                    if not 0 <= d <= 4:
                        continue
                    for keep in combinations(adds, a):
                        for cut in combinations(drops, d):
                            raster = stock.copy()
                            raster[list(keep)], raster[list(cut)] = True, False
                            score = (raster & actual).sum() / (raster | actual).sum()
                            all_scores.append(score)
                            if a == k:
                                balanced.append(score)
                self.assertEqual(worker.oracle_pair(actual, int((stock & actual).sum()), 4,
                                                    int(actual.sum()), adds, drops, k),
                                 (max(balanced), max(all_scores)))


if __name__ == '__main__':
    unittest.main()
