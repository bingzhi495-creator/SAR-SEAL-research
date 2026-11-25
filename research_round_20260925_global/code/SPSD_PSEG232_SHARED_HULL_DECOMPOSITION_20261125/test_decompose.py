"""Check exact area-preserving and reference/assignment decompositions."""
from pathlib import Path
from runpy import run_path
import unittest

import numpy as np

functions = run_path(str(Path(__file__).with_name("decompose.py")))
changed_cells = functions["changed_cells"]
effect = functions["effect"]
symmetric = functions["symmetric"]
same_rank = functions["same_rank"]


class TestDecomposition(unittest.TestCase):
    def test_unchanged_area_signal_and_normalization(self):
        before = np.array([[1, 1, 0, 0]], dtype=bool)
        after = np.array([[1, 0, 1, 0]], dtype=bool)
        pseg = np.array([[1, 1, 0, 1]], dtype=bool)
        spsd = np.array([[1, 1, 1, 0]], dtype=bool)
        add, drop = changed_cells(before, after)
        measured = []
        for ref in (pseg, spsd):
            observed = 100 * ((ref & after).sum() / (ref | after).sum()
                              - (ref & before).sum() / (ref | before).sum())
            record = effect(ref, before, add, drop)
            self.assertAlmostEqual(record[0], observed)
            measured.append(record)
        signal, scale, plus, minus = same_rank(*measured, (pseg, spsd), add, drop)
        self.assertAlmostEqual(signal + scale, measured[1][0] - measured[0][0])
        self.assertNotEqual(scale, 0)
        self.assertEqual(plus.tolist(), [0, 0, 1, 0])
        self.assertEqual(minus.tolist(), [0, 0, 0, 1])

    def test_crossed_assignments_with_unequal_denominators(self):
        p_old, p_new, s_old, s_new = (2.5, -1.0, 0.5, -1.5)
        shape, assignment = symmetric(p_old, p_new, s_old, s_new)
        wt_p, wt_s = 1 / 2, 1 / 3
        computed = (wt_p + wt_s) / 2 * (shape + assignment)
        computed += (wt_s - wt_p) / 2 * (s_new + p_old)
        self.assertAlmostEqual(computed, wt_s * s_new - wt_p * p_old)
        self.assertAlmostEqual(shape, -1.25)
        self.assertAlmostEqual(assignment, -2.75)


if __name__ == "__main__":
    unittest.main()
