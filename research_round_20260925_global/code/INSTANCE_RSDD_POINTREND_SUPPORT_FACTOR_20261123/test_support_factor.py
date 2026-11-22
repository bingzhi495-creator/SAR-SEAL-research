"""Synthetic rasters for the support-factor replay."""
from pathlib import Path
from runpy import run_path
import unittest

import numpy as np


factor = run_path(str(Path(__file__).with_name("replay_support_factor.py")))["factor"]


class SplitTests(unittest.TestCase):
    def test_exact_disjoint_splices_with_common_edits(self):
        n = np.array([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=bool)
        b = np.array([[0, 1, 1, 0], [1, 0, 1, 0]], dtype=bool)
        p = np.array([[0, 1, 0, 0], [0, 0, 1, 1]], dtype=bool)
        on, off = factor(n, b, p, np.array([2, 5]), np.array([0, 1]))
        np.testing.assert_array_equal(on, [[0, 1, 0, 0], [1, 0, 1, 0]])
        np.testing.assert_array_equal(off, [[0, 1, 1, 0], [0, 0, 1, 1]])

    def test_empty_support_keeps_native_on_arm_even_if_paste_drops_everything(self):
        n = np.array([[1, 0], [0, 0]], dtype=bool)
        p = np.zeros_like(n)
        empty = np.empty(0, dtype=np.int64)
        on, off = factor(n, n, p, empty, empty)
        np.testing.assert_array_equal(on, n)
        np.testing.assert_array_equal(off, p)

    def test_rejects_balanced_changes_outside_original_endpoint_support(self):
        n = np.array([[1, 0], [0, 0]], dtype=bool)
        b = np.array([[0, 0], [0, 1]], dtype=bool)
        with self.assertRaisesRegex(ValueError, "Balanced arm edits only"):
            factor(n, b, n, np.array([1]), np.array([0]))


if __name__ == "__main__":
    unittest.main()
