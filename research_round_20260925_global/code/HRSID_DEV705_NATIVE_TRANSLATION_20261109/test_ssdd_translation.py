from pathlib import Path
import unittest

import numpy as np
from pycocotools import mask

import ssdd_translation
from translation_probe import measure


def rle(h, w, y, x):
    pixels = np.zeros((h, w), dtype=np.uint8, order="F")
    pixels[y, x] = 1
    encoded = mask.encode(pixels)
    return dict(size=[h, w], counts=encoded["counts"].decode("ascii"))


class SsddTranslationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.common = ssdd_translation.load_common()

    def test_variable_image_canvases_preserve_flat_native_order(self):
        first = rle(12, 8, slice(1, 9), slice(2, 5))
        shifted = rle(12, 8, slice(2, 10), slice(2, 5))
        second = rle(6, 9, slice(1, 3), slice(4, 7))
        gt = dict(
            images=[dict(id=7, height=12, width=8), dict(id=3, height=6, width=9)],
            categories=[dict(id=1)],
            annotations=[
                dict(id=1, image_id=7, category_id=1, area=24, segmentation=first),
                dict(id=2, image_id=3, category_id=1, area=6, segmentation=second),
            ],
        )
        predictions = [
            dict(image_id=7, category_id=1, score=.9, segmentation=shifted),
            dict(image_id=3, category_id=1, score=.99, segmentation=second),
        ]

        result = ssdd_translation.summary(gt, predictions, measure, self.common)

        self.assertEqual(result["group_unit"], "logical_images")
        self.assertEqual(result["independent_native_coverage"], {"0.50": 2, "0.75": 2, "0.90": 1})
        self.assertEqual(result["fixed_pairs"], 2)
        self.assertEqual(result["near"]["pairs"], 1)
        self.assertEqual(result["near"]["radius1_recovered"], 1)
        self.assertEqual(result["fixed_one_pixel"]["fixed_up"]["losses"], 1)


    def test_equal_iou_uses_later_gt_annotation_like_hrsid(self):
        first = rle(12, 8, slice(1, 9), slice(2, 5))
        later = rle(12, 8, slice(2, 10), slice(2, 5))
        pred = rle(12, 8, slice(1, 10), slice(2, 5))
        gt = dict(
            images=[dict(id=7, height=12, width=8)],
            categories=[dict(id=1)],
            annotations=[
                dict(id=i, image_id=7, category_id=1, area=24, segmentation=segment)
                for i, segment in enumerate((first, later), 1)
            ],
        )
        predictions = [dict(image_id=7, category_id=1, score=.9, segmentation=pred)]

        result = ssdd_translation.summary(gt, predictions, measure, self.common)

        self.assertEqual(result["independent_native_coverage"]["0.75"], 2)
        self.assertEqual(result["fixed_pairs"], 1)
        self.assertEqual(result["near"]["mean_dy_pixels"], -.5)

    def test_reordered_images_scores_and_wrong_canvas_are_refused(self):
        images = [dict(id=7, height=12, width=8), dict(id=3, height=6, width=9)]
        gt = dict(images=images, categories=[dict(id=1)], annotations=[])
        p = dict(image_id=7, category_id=1, score=.9, segmentation=rle(12, 8, 0, 0))
        q = dict(image_id=3, category_id=1, score=.95, segmentation=rle(6, 9, 0, 0))
        for rows, message in (
            ([q, p], "original flat image order"),
            ([dict(p, score=.1), p], "native descending score order"),
            ([dict(p, segmentation=q["segmentation"])], "original variable canvas"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    ssdd_translation.summary(gt, rows, measure, self.common)


if __name__ == "__main__":
    unittest.main()
