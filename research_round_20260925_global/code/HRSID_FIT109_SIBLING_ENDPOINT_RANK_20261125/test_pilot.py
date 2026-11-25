"""Synthetic tests for the source-only sibling rank pilot."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('pilot', Path(__file__).with_name('pilot.py'))
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class SiblingRank(unittest.TestCase):
    def test_offset_regularization_and_symmetry(self):
        q = np.zeros(4)
        c = np.array([-.6, -.4, .4, .6])
        y = (c > 0).astype(float)
        w = np.full(4, .25)
        b = pilot.fit_coefficient(q, c, y, w)
        self.assertGreater(b, 0)
        self.assertAlmostEqual(b, -pilot.fit_coefficient(q, c, 1-y, w))
        self.assertEqual(pilot.fit_coefficient(q, np.zeros_like(c), y, w), 0)

    def test_controls_keep_sided_support_original_ties(self):
        q = np.array([[.8, .2], [.8, .2], [.9, .4], [.5, .4]])
        coords = np.array([[0, 10], [1, 11], [2, 12], [3, 13]])
        self.assertEqual(pilot.ranked_writes(q, coords, 2).tolist(), [2, 0, 10, 11])

    def test_decoy_preserves_every_local_block(self):
        q = np.tile(np.arange(8, dtype=float)[:,None] * .1, (1, 2))
        depth = np.ones((8, 2), dtype=np.int8)
        depth[4:] = 2
        c = np.arange(16).reshape(8, 2).astype(float)
        result = pilot.scramble(c, depth, q, (1, 2, 3))
        for side in (0, 1):
            for begin in (0, 4):
                self.assertEqual(sorted(c[begin:begin+4, side]), sorted(result[begin:begin+4, side]))
        np.testing.assert_array_equal(result, pilot.scramble(c, depth, q, (1, 2, 3)))


if __name__ == '__main__':
    unittest.main()
