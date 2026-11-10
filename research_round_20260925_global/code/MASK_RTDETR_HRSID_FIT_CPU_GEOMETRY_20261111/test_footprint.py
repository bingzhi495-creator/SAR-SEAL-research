import unittest
import numpy as np

from footprint import exact_footprint_scores

class SourceFootprintTests(unittest.TestCase):
    def test_resized_pixel_footprint_keeps_fractional_overlap_and_signed_centroid(self):
        source = np.zeros((2, 4), dtype=np.uint8)
        source[0, 1] = 1
        first = np.asarray([[1, 0], [0, 0]], dtype=np.uint8)
        second = np.asarray([[0, 1], [0, 0]], dtype=np.uint8)
        scores = exact_footprint_scores(source, (first, second))
        self.assertEqual(scores["reference_area"], .5)
        self.assertEqual(scores["native"]["iou"], .5)
        self.assertEqual(scores["center"]["iou"], 0.)
        self.assertEqual(scores["native"]["signed_x_px"], -.5)
        self.assertEqual(scores["center"]["signed_x_px"], 1.5)
        self.assertEqual(scores["native"]["signed_y_px"], 0.)

