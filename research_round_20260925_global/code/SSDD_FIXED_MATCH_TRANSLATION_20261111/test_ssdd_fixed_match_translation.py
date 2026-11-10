import importlib.util
import os
from pathlib import Path
import unittest

import numpy as np
from pycocotools import mask

spec = importlib.util.spec_from_file_location("p310", Path(__file__).with_name("ssdd_fixed_match_translation.py"))
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def encoded(pixels):
    raster = np.zeros((8, 8), dtype=np.uint8, order="F")
    for y, x in pixels:
        raster[y, x] = 1
    result = mask.encode(raster)
    return dict(size=result["size"], counts=result["counts"].decode("ascii"))


class ShiftFixture(unittest.TestCase):
    def test_edge_pixel_cannot_disappear_from_fixed_predicted_area(self):
        gt = encoded([(2, 0)])
        native = encoded([(2, 0), (2, 1)])
        center = encoded([(2, 0)])
        row = worker.measure_masks((gt, native, center))
        self.assertEqual(row["native"]["pred_area"], 2)
        self.assertEqual(row["native"]["intersection"]["horizontal"], 1)
        self.assertEqual(worker.summarize([row])["horizontal"]["mean_iou"]["native"], .5)
        self.assertEqual(worker.summarize([row])["horizontal"]["mean_iou"]["center"], 1.)


    def test_symmetric_alignment_and_paired_strict_crossing(self):
        gt = encoded([(y, x) for y in range(2, 6) for x in range(2, 6)])
        native = encoded([(y, x) for y in range(3, 7) for x in range(2, 6)])
        center = gt
        source = worker.summarize([worker.measure_masks((gt, native, center))])
        swapped = worker.summarize([worker.measure_masks((gt, center, native))])
        self.assertEqual(source["raw"]["mean_iou"], {"native": .6, "center": 1.})
        self.assertEqual(source["horizontal"]["mean_iou"], source["raw"]["mean_iou"])
        self.assertEqual(source["vertical"]["mean_iou"], {"native": 1., "center": 1.})
        self.assertEqual(source["joint"]["mean_iou"], source["vertical"]["mean_iou"])
        self.assertEqual(source["vertical"]["thresholds"]["0.90"]["gain_from_raw"], {"native": 1, "center": 0})
        for family in worker.OFFSETS:
            self.assertAlmostEqual(source[family]["center_minus_native_iou_points"],
                                   -swapped[family]["center_minus_native_iou_points"])


class CocoFixture(unittest.TestCase):
    def test_exact_original_coco_matches_and_zero_shift_geometry_reconstruct(self):
        code = Path(os.environ["P310_TEST_CODE"]).resolve(strict=True)
        p299 = worker.load_source(code / worker.P299_SOURCE[0], worker.P299_SOURCE[1], "_p310_test_p299")
        helper = worker.load_source(code / p299.GEOMETRY_WRAPPER, p299.GEOMETRY_SHA256, "_p310_test_geometry")
        helpers = helper.load_helpers(code, code / p299.GEOMETRY_CORE)
        gt = dict(images=[dict(id=1, height=8, width=8)], categories=[dict(id=1)],
                  annotations=[dict(id=1, image_id=1, category_id=1, segmentation=[[2, 2, 6, 2, 6, 6, 2, 6]],
                                    area=16, bbox=[2, 2, 4, 4], iscrowd=0)])
        rows = {mode: [dict(image_id=1, category_id=1, score=.75,
                     segmentation=encoded([(y, x) for y in range(2 + offset, 6 + offset) for x in range(2, 6)]))]
                for mode, offset in (("native", 1), ("center", 0))}
        prior = helper.evaluate_pair(gt, rows["native"], rows["center"], helpers)
        result = worker.paired_measurement(gt, rows, helpers, prior)
        self.assertEqual(result["all"], result["small"])
        self.assertEqual(result["all"]["official_iou50_match_census"]["both_matched"], 1)
        self.assertEqual(result["all"]["raw"]["mean_iou"], {"native": .6, "center": 1.})
        self.assertEqual(result["all"]["vertical"]["mean_iou"], {"native": 1., "center": 1.})


if __name__ == "__main__":
    unittest.main()
