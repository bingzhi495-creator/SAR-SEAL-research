import json
from pathlib import Path
import sys
import unittest

import numpy as np

import large_vector_shape_v2 as app


class VectorShape(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vector, cls.math = app.geometry_sources(cls.code_root)

    def test_exact_fractional_reference_and_source_centroid(self):
        polygon = [[2.5, 2, 6.5, 2, 6.5, 4, 2.5, 4]]
        steps = [dict(kind="resize", before=[12, 12], after=[6, 6])]
        a = np.zeros((6, 6), dtype=bool)
        a[1:2, 1:3] = True
        b = np.zeros_like(a)
        b[1:2, 2:4] = True
        rings, hw = self.vector.final_polygons(polygon, (12,12), steps)
        data, audit = app.measure(rings, hw, {"native":a,"center":b}, (2.,2.), self.vector, self.math, dense=True)
        self.assertAlmostEqual(data["native"]["iou_points"], 100*1.75/2.25)
        self.assertAlmostEqual(data["native"]["centroid_abs_x_source_px"], .5)
        self.assertAlmostEqual(data["center"]["centroid_abs_x_source_px"], 1.5)
        self.assertAlmostEqual(data["native"]["fp_per_visible_gt_points"], 12.5)
        self.assertAlmostEqual(data["native"]["fn_per_visible_gt_points"], 12.5)
        self.assertLess(audit, 1e-9)
        entry = dict(prefix="p", image=1, annotation=1, values=data)
        row = app.aggregate([entry], "native", "center", "centroid_abs_x_source_px")
        self.assertIs(type(row["positive_rows"]), int)
        self.assertIs(type(row["negative_rows"]), int)
        self.assertIs(type(row["tied_rows"]), int)
        json.dumps(row, allow_nan=False)

    def test_group_concentration_disallows_consistent_direction(self):
        rows = [
            dict(prefix="first", annotation=1, image=1, values={"native":{"iou_points":.2},"center":{"iou_points":.6}}),
            dict(prefix="first", annotation=1, image=1, values={"native":{"iou_points":.2},"center":{"iou_points":.6}}),
            dict(prefix="second", annotation=2, image=2, values={"native":{"iou_points":.8},"center":{"iou_points":.7}}),
        ]
        result = app.aggregate(rows, "native", "center", "iou_points")
        self.assertEqual(result["rows"], 3)
        self.assertEqual(result["unique_annotations"], 2)
        self.assertEqual(result["prefix_groups"], 2)
        self.assertLess(result["leave_one_prefix_out_min_delta"], 0.)
        self.assertGreater(result["leave_one_prefix_out_max_delta"], 0.)
        self.assertEqual(result["consistent_direction"], "mixed_or_indeterminate")


if __name__ == "__main__":
    VectorShape.code_root = Path(sys.argv[1]).resolve(strict=True)
    unittest.main(argv=[sys.argv[0]])
