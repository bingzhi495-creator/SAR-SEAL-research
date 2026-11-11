from pathlib import Path
import sys
import unittest

import numpy as np
from pycocotools import mask
from shapely import box

import fixed_match_geometry as app


def rle(pixels):
    original = mask.encode(np.asfortranarray(pixels, dtype=np.uint8))
    return dict(size=original["size"], counts=original["counts"].decode("ascii"))


def shape_row(image_id, pixels, score):
    return dict(image_id=image_id, category_id=1, score=score, segmentation=rle(pixels))


class FixedMatchGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.admission = app.load_module(cls.admission_source, app.ADMISSION_SHA, "_fixture_closed_p392")
        cls.modules = cls.admission.geometry_modules(cls.code_root)
        cls.helpers = app.load_match_helpers(cls.admission, cls.code_root)

    def test_exact_vector_area_centroid_and_hull_projection(self):
        target = box(1.25, 1, 7.25, 3)
        pixels = np.zeros((12, 12), dtype=np.uint8)
        pixels[2:4, 2:8] = 1
        segmentation = rle(pixels)
        hull = app.hull(target)
        self.assertAlmostEqual(hull[0], 3.)
        self.assertAlmostEqual(hull[1], 2.)
        metrics, predicted_centroid = app.measure(target, segmentation, hull, self.modules)
        self.assertEqual(tuple(predicted_centroid), (5., 3.))
        self.assertAlmostEqual(metrics["centroid_abs_short_px"], 1.)
        self.assertAlmostEqual(metrics["centroid_abs_long_px"], .75)
        self.assertAlmostEqual(metrics["continuous_iou_points"], 100 * 5.25 / 18.75)
        self.assertAlmostEqual(metrics["fp_per_gt_area_points"], 56.25)
        self.assertAlmostEqual(metrics["fn_per_gt_area_points"], 56.25)
        rings = [np.asarray(target.exterior.coords)[:-1]]
        error = app.dense_check(rings, segmentation, metrics, predicted_centroid, self.modules)
        self.assertLess(max(error.values()), 1e-10)

    def test_original_associations_preserve_all_rows_and_ties(self):
        gt = dict(images=[dict(id=7, height=12, width=12)], categories=[dict(id=1, name="ship")],
                  annotations=[dict(id=3, image_id=7, category_id=1, iscrowd=0, area=12,
                                    bbox=[1, 1, 6, 2], segmentation=[[1, 1, 7, 1, 7, 3, 1, 3]])])
        pixels = np.zeros((12, 12), dtype=np.uint8)
        pixels[1:3, 1:7] = 1
        blank = np.zeros_like(pixels)
        streams = dict(native=[shape_row(7, pixels, .8), shape_row(7, blank, .8)],
                       center=[shape_row(7, blank, .8), shape_row(7, pixels, .8)])
        priors = {arm: self.modules.geometry.score(gt, rows, self.helpers.adapter)[0]
                  for arm, rows in streams.items()}
        matches = app.original_matches(gt, streams, priors, self.helpers, self.admission)
        self.assertEqual(matches["native"], {3: streams["native"][0]["segmentation"]})
        self.assertEqual(matches["center"], {3: streams["center"][1]["segmentation"]})
        results = app.analyze(gt, matches, self.modules, expected_support=(1, 0), dense_spots=1)
        self.assertEqual(results["supported_coverage"]["all"]["both"], 1)
        self.assertEqual(results["supported_coverage"]["thin"]["both"], 1)
        self.assertEqual(results["paired_geometry"]["all"]["continuous_iou_points"]["delta_center_minus_native"], 0.)
        self.assertEqual(results["paired_geometry"]["all"]["continuous_iou_points"]["ties"], 1)
        priors["center"]["all12"][8] -= .01
        with self.assertRaisesRegex(ValueError, "all12/AP90"):
            app.original_matches(gt, streams, priors, self.helpers, self.admission)


if __name__ == "__main__":
    FixedMatchGeometry.code_root = Path(sys.argv[1]).resolve(strict=True)
    FixedMatchGeometry.admission_source = Path(sys.argv[2]).resolve(strict=True)
    unittest.main(argv=[sys.argv[0]])
