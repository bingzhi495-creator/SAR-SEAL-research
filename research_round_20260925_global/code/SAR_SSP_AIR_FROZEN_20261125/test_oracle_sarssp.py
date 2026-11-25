"""Check oracle arithmetic independently by exhaustive synthetic cases."""
import importlib.util
from itertools import combinations
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('oracle_sarssp', Path(__file__).with_name('oracle_sarssp.py'))
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class Oracle(unittest.TestCase):
    def test_intersections_against_enumeration(self):
        rng = np.random.default_rng(5217)
        coords = np.array([[0, 4], [1, 5], [2, 6], [3, 7]])
        stock = np.arange(8) >= 4
        for _ in range(25):
            gt = rng.integers(0, 2, 8).astype(bool)
            for k in range(5):
                paired, free = [], []
                for a in combinations(range(4), k):
                    for d in combinations(range(4), k):
                        pred = stock.copy()
                        pred[coords[list(a), 0]], pred[coords[list(d), 1]] = True, False
                        hits = int((pred & gt).sum())
                        free.append(hits)
                        if a == d:
                            paired.append(hits)
                self.assertEqual(worker.exact_gains(gt, coords, k),
                                 (max(paired) - int((stock & gt).sum()), max(free) - int((stock & gt).sum())))

    def test_frozen_box_can_retain_exterior_foreground(self):
        stock = np.array([[1, 1, 0], [0, 1, 0]], dtype=bool)
        truth = np.array([[1, 0, 1], [0, 0, 1]], dtype=bool)
        output = worker.box_limits(stock, truth, (1, 0, 3, 2), 3, 3, 1)
        self.assertEqual(output, (1., 1., 2 / 3))

    def test_paired_and_free_caps_are_distinct(self):
        gt = np.array([1, 0, 1, 0], dtype=bool)
        coords = np.array([[0, 2], [1, 3]])
        self.assertEqual(worker.exact_gains(gt, coords, 1), (0, 1))
        self.assertEqual(worker.capped_gains(gt, coords, 2), (0, 1))


if __name__ == '__main__':
    unittest.main()
