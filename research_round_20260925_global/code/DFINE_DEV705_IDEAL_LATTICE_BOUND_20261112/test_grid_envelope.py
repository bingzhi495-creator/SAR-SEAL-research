import itertools
import unittest

import numpy as np

from grid_envelope import axis_cells, margin_envelope, monotone_weight


class GridEnvelopeTest(unittest.TestCase):
    def test_monotone_solver_matches_exhaustive_patterns(self):
        for n in range(1, 6):
            for values in itertools.product((-1.0, 2.0), repeat=n):
                patterns = [[int(j < split) for j in range(n)] for split in range(n + 1)]
                patterns += [[1 - bit for bit in row] for row in patterns]
                best = max(np.dot(values, pattern) for pattern in patterns)
                self.assertEqual(monotone_weight(np.asarray(values)[None, :])[0], best)

    def test_half_pixel_intervals_and_crop(self):
        self.assertEqual(axis_cells(0, 10, 10, 2).tolist(), [-1, -1, 0, 0, 0, 0, 0, 1, 1, 1])
        self.assertEqual(axis_cells(3, 8, 10, 2).tolist(), [0, 0, 0, 0, 1])

    def test_narrow_foreground_between_two_background_pixels(self):
        q = np.array([[0., 1., 0.]])
        answer = margin_envelope(q, 1., .90, (0, 3), (10, 10), (2, 2))
        self.assertAlmostEqual(answer["free"], .10)
        self.assertAlmostEqual(answer["rows"], -.80)
        self.assertAlmostEqual(answer["columns"], .10)
        self.assertAlmostEqual(answer["upper"], -.80)

    def test_clamped_border_segment_is_constant(self):
        q = np.array([[1., 0.]])
        answer = margin_envelope(q, 1., .90, (0, 0), (10, 10), (2, 2))
        self.assertAlmostEqual(answer["free"], .10)
        self.assertAlmostEqual(answer["rows"], -.80)


    def test_truth_area_outside_author_box_penalizes_both_bounds(self):
        answer = margin_envelope(np.ones((1, 1)), 2., .90, (0, 0), (10, 10), (2, 2))
        self.assertAlmostEqual(answer["free"], -.80)
        self.assertAlmostEqual(answer["upper"], -.80)


if __name__ == "__main__":
    unittest.main()
