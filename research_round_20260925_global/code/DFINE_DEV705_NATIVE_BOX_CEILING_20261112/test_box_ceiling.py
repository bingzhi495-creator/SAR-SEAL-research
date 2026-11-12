import unittest
import numpy as np
from box_ceiling import positive_margins


class BoxCeilingTest(unittest.TestCase):
    def test_one_pixel_box_margin_and_outside_truth(self):
        q = np.array([[0., 1., 1., 0.]])
        limits = positive_margins(q, 2., (0, 0), (1, 0, 2, 1), .90, (0, 1))
        self.assertAlmostEqual(limits["0"], -.8)
        self.assertAlmostEqual(limits["1"], .2)
        self.assertAlmostEqual(limits["free"], .2)

    def test_native_raster_can_be_limiting_with_no_box(self):
        q = np.array([[.5, .5]])
        limits = positive_margins(q, 1., (0, 0), (0, 0, 2, 1), .90, (0, 1))
        self.assertAlmostEqual(limits["free"], -.8)
        self.assertAlmostEqual(limits["0"], limits["free"])

if __name__ == "__main__":
    unittest.main()
