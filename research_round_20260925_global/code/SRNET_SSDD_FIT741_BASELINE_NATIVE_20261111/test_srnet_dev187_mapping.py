import unittest

import numpy as np
from pycocotools import mask as codec

from srnet_dev187_mapping import map_native_image


class NativeMaskMappingTests(unittest.TestCase):
    def test_original_grid_and_tied_order_survive(self):
        image = dict(image_id=813, original_hw=[3, 4])
        boxes = np.array([[3, 2, 2, 1, -.4, .25], [1, 2, 2, 1, -.2, .25]],
                         dtype=np.float32)
        first = np.zeros((3, 4), dtype=bool)
        first[2, 3] = True
        second = np.zeros_like(first)
        second[0, 1] = True
        predictions, rotated = map_native_image(image, [boxes], [[first, second]])

        self.assertEqual([row["image_id"] for row in predictions], [813, 813])
        self.assertEqual([row["category_id"] for row in predictions], [1, 1])
        self.assertEqual([row["score"] for row in predictions], [.25, .25])
        self.assertEqual([row["rbox"][0] for row in rotated], [3., 1.])
        self.assertEqual([row["native_rank"] for row in rotated], [1, 2])
        for expected, row in zip((first, second), predictions, strict=True):
            self.assertEqual(set(row), {"image_id", "category_id", "score", "segmentation"})
            self.assertEqual(row["segmentation"]["size"], [3, 4])
            self.assertIsInstance(row["segmentation"]["counts"], str)
            np.testing.assert_array_equal(codec.decode(row["segmentation"]), expected)

    def test_zero_candidate_image_and_zero_pixel_candidate_are_distinct(self):
        image = dict(image_id=904, original_hw=[4, 2])
        no_boxes = np.empty((0, 6), dtype=np.float32)
        self.assertEqual(map_native_image(image, [no_boxes], [[]]), ([], []))

        box = np.array([[1, 2, 2, 1, -.4, .2]], dtype=np.float32)
        empty_mask = np.zeros((4, 2), dtype=bool)
        predictions, rotated = map_native_image(image, [box], [[empty_mask]])
        self.assertEqual(len(predictions), 1)
        self.assertEqual(len(rotated), 1)
        self.assertEqual(int(codec.area(predictions[0]["segmentation"])), 0)

    def test_refuses_shifted_class_slot_and_unpaired_mask(self):
        image = dict(image_id=904, original_hw=[4, 2])
        box = np.array([[1, 2, 2, 1, -.4, .2]], dtype=np.float32)
        empty = np.empty((0, 6), dtype=np.float32)
        mask = np.ones((4, 2), dtype=bool)
        with self.assertRaisesRegex(ValueError, "class index 0"):
            map_native_image(image, [empty, box], [[], [mask]])
        with self.assertRaisesRegex(ValueError, "class index 0"):
            map_native_image(image, {1: box}, {1: [mask]})
        with self.assertRaisesRegex(ValueError, "aligned"):
            map_native_image(image, [box], [[]])
