import unittest

import numpy as np

from boundary_metric import boundary_iou, boundary_ring, ring_iou


class BoundaryMetricTest(unittest.TestCase):
    def test_image_border_is_background_at_each_width(self):
        mask = np.ones((5, 7), dtype=bool)
        expected_one = mask.copy()
        expected_one[1:-1, 1:-1] = False
        expected_two = mask.copy()
        expected_two[2:-2, 2:-2] = False
        np.testing.assert_array_equal(boundary_ring(mask, 1), expected_one)
        np.testing.assert_array_equal(boundary_ring(mask, 2), expected_two)


    def test_matches_direct_square_neighborhoods_at_both_widths(self):
        rng = np.random.default_rng(912)
        for height, width in ((1, 8), (8, 1), (3, 4), (6, 11), (10, 8)):
            for density in (.1, .65, .95, 1):
                mask = rng.random((height, width)) < density
                for thickness in (1, 2):
                    with self.subTest(shape=mask.shape, density=density, thickness=thickness):
                        expected = mask.copy()
                        for y in range(thickness, height - thickness):
                            for x in range(thickness, width - thickness):
                                neighborhood = mask[y-thickness:y+thickness+1, x-thickness:x+thickness+1]
                                expected[y, x] = mask[y, x] and not neighborhood.all()
                        np.testing.assert_array_equal(boundary_ring(mask.astype(np.uint8), thickness), expected)

    def test_disconnected_and_internal_boundaries_are_preserved(self):
        mask = np.ones((9, 11), dtype=bool)
        mask[4, 5] = False
        mask[0:3, 7:10] = False
        ring = boundary_ring(mask, 1)
        self.assertFalse(ring[4, 5])
        self.assertTrue(ring[3:6, 4:7][mask[3:6, 4:7]].all())
        self.assertTrue(ring[3, 8])
        self.assertFalse(ring[6, 5])

    def test_pair_iou_on_saturated_rings_and_empty_masks(self):
        reference = np.zeros((8, 8), dtype=bool)
        prediction = reference.copy()
        empty = reference.copy()
        reference[2:4, 2:5] = True
        prediction[3:5, 2:5] = True
        for width in (1, 2):
            np.testing.assert_array_equal(boundary_ring(reference, width), reference)
            np.testing.assert_array_equal(boundary_ring(prediction, width), prediction)
            self.assertEqual(boundary_iou(prediction, reference, width), 1 / 3)
            self.assertEqual(boundary_iou(reference, prediction, width), 1 / 3)
            self.assertEqual(boundary_iou(reference, reference, width), 1)
            self.assertEqual(boundary_iou(empty, reference, width), 0)
            self.assertTrue(np.isnan(boundary_iou(empty, empty, width)))
        with self.assertRaises(ValueError):
            ring_iou(prediction, reference[:, 1:])


if __name__ == "__main__":
    unittest.main()

