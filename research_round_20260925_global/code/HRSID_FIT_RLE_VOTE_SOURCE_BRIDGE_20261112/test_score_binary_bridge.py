import unittest
import numpy as np
from score_binary_bridge import choose_binary_edits


class EditGeometry(unittest.TestCase):
    def test_identical_counts_and_priority(self):
        a = np.zeros((8, 9), dtype=bool)
        a[3:6, 3:6] = True
        a[2, 4] = True
        d = a.copy()
        d[1, 2] = True
        d[3, 3] = False
        d[4, 4] = False
        r = choose_binary_edits(a, d, d, 2)
        self.assertEqual(np.flatnonzero(r["vote"] ^ a).tolist(), [11, 30])
        self.assertEqual(int((r["vote"] & ~a).sum()), 1)
        self.assertEqual(int((r["anchor_count"] & ~a).sum()), 1)
        self.assertEqual(int((~r["anchor_count"] & a).sum()), 1)
        self.assertNotEqual(np.flatnonzero(r["anchor_count"] ^ a).tolist(), [11, 30])
