"""Synthetic checks of independent continuous SSDD pixel-footprint geometry."""

from fractions import Fraction
import unittest

import numpy as np

from reference_resize import _compose
from ssdd_pixel_footprint import measure_masks


def resize(before, after):
    return dict(kind="resize", before=before, after=after)


def raster(rows):
    return np.asarray(rows, dtype=np.bool_)


class PixelFootprintTest(unittest.TestCase):
    def test_identity_measures_exact_source_area_and_centroid(self):
        raw = raster([[[1, 0], [0, 1]]])
        report = measure_masks(raw, {"center": raw.copy()}, raster([True]), [], _compose)

        np.testing.assert_array_equal(report["truth_area"], [2.0])
        np.testing.assert_array_equal(report["truth_source_y"], [1.0])
        np.testing.assert_array_equal(report["truth_empty"], [False])
        np.testing.assert_array_equal(report["crop_truncated"], [False])
        np.testing.assert_array_equal(report["original_edge"], [True])
        np.testing.assert_array_equal(report["small"], [True])
        arm = report["by_arm"]["center"]
        for key, expected in dict(area=2, intersection=2, iou=1, source_y_error=0, abs_source_y_error=0,
                                  fp=0, fn=0, occupancy_l1=0, area_signed_error=0).items():
            np.testing.assert_array_equal(arm[key], [expected])


    def test_fractional_downsample_splits_original_pixel_area(self):
        raw = raster([[[0, 1, 0]]])
        native = raster([[[0, 1]]])
        center = raster([[[0, 0]]])
        report = measure_masks(raw, {"native": native, "center": center}, raster([True]),
                               [resize((1, 3), (1, 2))], _compose)

        np.testing.assert_array_equal(report["truth_area"], [1])
        a, b = (report["by_arm"][name] for name in ("native", "center"))
        for key, expected in dict(area=1.5, intersection=0.5, iou=0.25, fp=1,
                                  fn=0.5, occupancy_l1=1.5, area_signed_error=0.5).items():
            np.testing.assert_array_equal(a[key], [expected])
        np.testing.assert_array_equal(b["iou"], [0.0])
        np.testing.assert_array_equal(b["fn"], [1.0])
        self.assertTrue(np.isnan(b["source_y_error"][0]))

    def test_crop_and_flip_measure_physical_source_support(self):
        raw = raster([[[1, 0, 0], [0, 0, 1]]])
        correct = raster([[[0, 0], [1, 0]]])
        opposite = raster([[[0, 0], [0, 1]]])
        steps = [dict(kind="crop", before=(2, 3), after=(2, 2), top=0, left=1),
                 dict(kind="hflip", before=(2, 2), after=(2, 2))]
        report = measure_masks(raw, {"correct": correct, "opposite": opposite}, raster([False]), steps, _compose)

        np.testing.assert_array_equal(report["crop_truncated"], [True])
        np.testing.assert_array_equal(report["truth_area"], [1.0])
        np.testing.assert_array_equal(report["truth_source_y"], [1.5])
        np.testing.assert_array_equal(report["by_arm"]["correct"]["iou"], [1.0])
        np.testing.assert_array_equal(report["by_arm"]["opposite"]["iou"], [0.0])

    def test_subpixel_crop_centroid_is_not_occupancy_cell_center(self):
        raw = raster([[[0], [1], [0]]])
        candidate = raster([[[1]]])
        steps = [resize((3, 1), (2, 1)),
                 dict(kind="crop", before=(2, 1), after=(1, 1), top=0, left=0)]
        report = measure_masks(raw, {"candidate": candidate}, raster([True]), steps, _compose)

        np.testing.assert_array_equal(report["truth_area"], [0.5])
        np.testing.assert_array_equal(report["truth_source_y"], [1.25])
        np.testing.assert_array_equal(report["crop_truncated"], [True])
        np.testing.assert_array_equal(report["original_edge"], [True])
        arm = report["by_arm"]["candidate"]
        np.testing.assert_array_equal(arm["area"], [1.5])
        np.testing.assert_array_equal(arm["intersection"], [0.5])
        np.testing.assert_array_equal(arm["source_y_error"], [-0.5])
        np.testing.assert_array_equal(arm["abs_source_y_error"], [0.5])

    def test_empty_support_and_empty_candidates_remain_in_census(self):
        raw = raster([[[1]], [[0]], [[0]]])
        candidate = raster([[[0]], [[1]], [[0]]])
        report = measure_masks(raw, {"arm": candidate}, raster([True, False, False]), [], _compose)
        arm = report["by_arm"]["arm"]

        np.testing.assert_array_equal(report["truth_empty"], [False, True, True])
        np.testing.assert_array_equal(report["truth_source_y"], [0.5, np.nan, np.nan])
        np.testing.assert_array_equal(arm["iou"], [0.0, 0.0, np.nan])
        np.testing.assert_array_equal(arm["fp"], [0.0, 1.0, 0.0])
        np.testing.assert_array_equal(arm["fn"], [1.0, 0.0, 0.0])
        self.assertTrue(np.isnan(arm["source_y_error"]).all())

    def test_no_rows_and_invalid_composed_bounds(self):
        empty = np.zeros((0, 2, 3), dtype=np.bool_)
        report = measure_masks(empty, {"arm": empty}, raster([]), [], _compose)
        np.testing.assert_array_equal(report["truth_area"], [])
        np.testing.assert_array_equal(report["by_arm"]["arm"]["iou"], [])

        def outside(_, shape):
            return shape, shape, [Fraction(1), Fraction(1)], [Fraction(-1, 2), Fraction(0)]

        with self.assertRaisesRegex(ValueError, "outside"):
            measure_masks(empty, {"arm": empty}, raster([]), [], outside)

if __name__ == "__main__":
    unittest.main()
