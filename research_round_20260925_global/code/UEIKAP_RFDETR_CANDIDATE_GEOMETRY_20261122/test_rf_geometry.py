"""Exhaustive tiny-raster checks for the analytic geometry ceilings."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest

import numpy as np


spec = spec_from_file_location("geometry", Path(__file__).with_name("rf_geometry.py"))
geometry = module_from_spec(spec)
spec.loader.exec_module(geometry)


class GeometryTest(unittest.TestCase):
    def test_analytic_bounds_equal_enumeration_with_foreground_outside_box(self):
        crop = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
        stock = np.array([0, 1, 0, 1, 0, 1], dtype=bool)
        gt = np.array([1, 0, 1, 1, 0, 0], dtype=bool)
        options = [np.array([(v >> bit) & 1 for bit in range(6)], dtype=bool) for v in range(64)]
        def direct_iou(mask):
            return np.count_nonzero(mask & gt) / np.count_nonzero(mask | gt)
        expected = {
            "mask": direct_iou(stock),
            "area_only": max(direct_iou(x) for x in options if x.sum() == stock.sum()),
            "crop_only": max(direct_iou(x) for x in options if not x[~crop].any()),
            "area_and_crop": max(direct_iou(x) for x in options if not x[~crop].any() and x.sum() == stock.sum()),
            "up_to_eight_anywhere": max(direct_iou(x) for x in options if x.sum() == stock.sum()),
            "area_guard_and_up_to_eight_anywhere": direct_iou(stock),
            "within_box_equal_area_preserve_exterior": max(direct_iou(x) for x in options
                if np.array_equal(x[~crop], stock[~crop]) and x.sum() == stock.sum()),
            "within_box_free_area_preserve_exterior": max(direct_iou(x) for x in options
                if np.array_equal(x[~crop], stock[~crop])),
        }
        actual = geometry.ceilings(stock.sum(), gt.sum(), (gt & stock).sum(), crop.sum(),
                                   (crop & gt).sum(), (crop & stock).sum(), (crop & stock & gt).sum())
        self.assertEqual(expected, {k: float(actual[k]) for k in expected})

    def test_infeasible_forced_crop_is_not_an_edit_ceiling(self):
        actual = geometry.ceilings(4, 3, 2, 2, 2, 2, 2)
        self.assertEqual(actual["area_and_crop"], 0)
        self.assertGreaterEqual(actual["within_box_equal_area_preserve_exterior"], actual["mask"])


if __name__ == "__main__":
    unittest.main()
