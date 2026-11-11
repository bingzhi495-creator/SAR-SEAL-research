import unittest

import numpy as np

from sparse_fit_readback import verify_points


class BoundaryReadbackTest(unittest.TestCase):
    def test_corner_uses_outside_background_for_support_and_clamp_for_logits(self):
        mask = np.array([[True, False], [False, False]])
        shifts = ((0, 0), (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
        positions = [(0, 0, True), (0, 1, False), (1, 0, False), (1, 1, False)]
        points = [dict(y=y, x=x, inside=inside, logits=[1.0 if mask[min(1, max(0, y+dy)),
                   min(1, max(0, x+dx))] else -1.0 for dy, dx in shifts]) for y, x, inside in positions]
        self.assertEqual(verify_points(mask, points, 'ab' * 32, 7), (1, 3))
        points[0]['logits'][1] = -1.0
        with self.assertRaisesRegex(ValueError, 'Neighbor'):
            verify_points(mask, points, 'ab' * 32, 7)


if __name__ == '__main__':
    unittest.main()
