import unittest

import numpy as np

from axis_probe import inside, preferred


class GeometryOracle(unittest.TestCase):
    def test_prefers_zero_for_exact_tie(self):
        self.assertEqual(preferred({-.5: .91, 0.0: .91, .5: .91}), (0.0, .91))
        self.assertEqual(preferred({-.5: .91, .5: .91, 0.0: .90}), (-.5, .91))

    def test_uses_intersection_of_both_axis_border_supports(self):
        axes = (np.array([1., 0.]), np.array([0., 1.]))
        self.assertTrue(inside((1.5, 1.5, 8.5, 8.5), axes, 10, 10))
        self.assertFalse(inside((1.0, 1.5, 8.5, 8.5), axes, 10, 10))


if __name__ == "__main__":
    unittest.main()