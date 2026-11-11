import unittest

import numpy as np
from pycocotools import mask

from rendered_features import augment_image


class RenderedTest(unittest.TestCase):
    def test_same_points_and_metadata_with_signed_prediction_side_contrast(self):
        pred = np.zeros((6, 6), dtype=np.uint8, order="F")
        pred[:, :3] = 1
        rgb = np.repeat(np.where(pred != 0, 192, 32).astype(np.uint8)[..., None], 3, axis=2)
        rle = mask.encode(pred)
        rle["counts"] = rle["counts"].decode("ascii")
        points = [
            dict(y=2, x=2, inside=True, logits=[1.] * 9),
            dict(y=2, x=3, inside=False, logits=[-1.] * 9),
        ]
        row = dict(native_rank=1, query_id=4, score=.75, category_id=1, segmentation=rle, points=points)
        record = dict(image_id=9, group="P9", encoded_sha256="11"*32, core=True, stress=False, candidates=[row])
        result, counters = augment_image(record, rgb, mask)
        new_points = result["candidates"][0]["points"]
        self.assertEqual(counters["points"], 2)
        self.assertEqual([{k:v for k,v in point.items() if k != "radiometric"} for point in new_points], points)
        self.assertNotIn("radiometric", points[0])
        for original, enriched in zip(points, new_points, strict=True):
            features = enriched["radiometric"]
            self.assertEqual(len(features), 4)
            self.assertGreater(features[0], 0.)
            self.assertAlmostEqual(features[1 if original["inside"] else 2], 0.)
            self.assertLessEqual(max(abs(f) for f in features), 4.)


    def test_edge_window_fallback_and_mask_sign_guard(self):
        pred = np.ones((6, 6), dtype=np.uint8, order="F")
        rgb = np.full((6, 6, 3), 255, dtype=np.uint8)
        rle = mask.encode(pred)
        rle["counts"] = rle["counts"].decode("ascii")
        point = dict(y=0, x=0, inside=True, logits=[1.] * 9)
        row = dict(native_rank=1, query_id=4, score=.75, category_id=1, segmentation=rle, points=[point])
        record = dict(image_id=9, group="P9", encoded_sha256="11"*32, core=True, stress=False, candidates=[row])
        result, counters = augment_image(record, rgb, mask)
        self.assertEqual(result["candidates"][0]["points"][0]["radiometric"], [0.] * 4)
        self.assertEqual(counters["points_missing_background"], 1)
        self.assertEqual(counters["points_clipped_window"], 1)
        point.update(inside=False, logits=[-1.] * 9)
        with self.assertRaisesRegex(ValueError, "native COCO mask"):
            augment_image(record, rgb, mask)

if __name__ == "__main__":
    unittest.main()
