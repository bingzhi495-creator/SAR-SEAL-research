"""Synthetic check of signed registered-view ordering and source-group resampling."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np


source = Path(__file__).with_name("score_robust.py")
spec = importlib.util.spec_from_file_location("score_robust", source)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class SyntheticOrdering(unittest.TestCase):
    def test_sustained_addition_and_removal(self):
        # One extreme favorable view cannot compensate for two unfavorable ones.
        logits = np.array([[12, 2, 2, -4, -4], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1],
                           [-12, -2, -2, 4, 4], [-1, -1, -1, -1, -1]], dtype=float)
        inside = np.array([False, False, False, True, True])
        result = worker.robust_order(logits, inside)
        np.testing.assert_array_equal(result, [-4, 1, 1, 4, -1])
        self.assertEqual(np.argmax(result[:3]), 1)
        self.assertEqual(np.argmin(result[3:]) + 3, 4)

    def test_group_sampling_keeps_hull_multiplicity(self):
        groups = ["a", "b", "b", "b"]
        delta = np.array([[0.], [4.], [4.], [4.]])
        out = worker.paired_interval(groups, delta)
        np.testing.assert_array_equal(out[0], [3.])
        np.testing.assert_array_equal(out[1], [[0., 4.]])


if __name__ == "__main__":
    unittest.main()
