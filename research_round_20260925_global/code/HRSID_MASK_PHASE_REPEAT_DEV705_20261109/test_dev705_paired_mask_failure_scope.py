"""Synthetic CPU tests for fixed paired-mask pixel-error scope."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
from pycocotools import mask

import dev705_paired_mask_failure_scope as subject
import dev705_paired_mask_geometry as geometry
from test_dev705_paired_mask_geometry import annotation, ground_truth, rle, row


def evaluate(gt, native, center, empty_count=0):
    with patch.multiple(subject, DEV_IMAGES=len(gt["images"]), GT_COUNT=len(gt["annotations"]),
                        CANVAS=(14, 14), EMPTY_COUNT=empty_count):
        return subject.evaluate_pair(gt, native, center)


class PairedMaskFailureScopeTests(unittest.TestCase):
    def test_joint_pixel_errors_reconcile_across_original_area_and_gt_edge(self):
        coordinates = ((0, 4), (1, 4), (4, 11), (4, 4), (11, 4), (4, 4), (5, 5))
        areas = (1023.5, 101, 1024, 9215.5, 9216, 10000, 9)
        target = [rle(*coordinate) for coordinate in coordinates]
        gt = ground_truth([annotation(i, i, segmentation, area)
                           for i, (segmentation, area) in enumerate(zip(target, areas, strict=True), 1)]
                          + [annotation(8, 8, rle(), 0)])
        native = [row(i, value) for i, value in enumerate(
            (target[0], rle(0, 4), target[2], target[3], rle(10, 4), target[5], target[6], rle(4, 4)), 1)]
        center = [row(i, value) for i, value in enumerate(
            (rle(1, 4), target[1], rle(5, 11), target[3], target[4], rle(5, 4), rle(10, 10), rle(4, 4)), 1)]
        gt["annotations"][0]["segmentation"] = [[4, 0, 7, 0, 7, 3, 4, 3]]
        gt["annotations"][1]["bbox"] = [0, 0, 10, 10]
        original = deepcopy((gt, native, center))
        result = evaluate(gt, native, center, empty_count=1)
        self.assertEqual((gt, native, center), original)
        self.assertEqual(result["exclusions"], dict(empty=1, crowd=0, ignore=0))
        self.assertEqual(result["retained_prefix_rows"], 8)
        strata = result["strata"]
        self.assertEqual({key: data["center_minus_native"]["fn_pixels"] for key, data in strata.items()},
                         dict(all=3, small_edge=3, small_interior=-3, medium_edge=3,
                              medium_interior=0, large_edge=-3, large_interior=3))
        whole = strata["all"]
        self.assertEqual((whole["paired_gt"], whole["gt_area_pixels"]), (6, 54))
        self.assertEqual(whole["fn_delta_counts"], dict(increase=3, decrease=2, tie=1))
        self.assertEqual((whole["native"]["fp_pixels"], whole["native"]["fn_pixels"]), (6, 6))
        self.assertEqual((whole["center"]["fp_pixels"], whole["center"]["fn_pixels"]), (9, 9))
        delta = whole["center_minus_native"]
        self.assertEqual((delta["fn_pixels"], delta["fp_pixels"], delta["total_error_pixels"]), (3, 3, 6))
        self.assertAlmostEqual(delta["micro_fn_per_gt_area"], 3 / 54)
        self.assertAlmostEqual(delta["macro_fn_per_gt_area"], 3 / 54)
        self.assertAlmostEqual(delta["mean_iou_change"], -.5 / 6)
        self.assertEqual(strata["small_interior"]["paired_gt"], 1)
        concentration = result["fn_concentration"]["non_small"]
        self.assertEqual(concentration["largest_five_positive_fn_delta_pixels"], 6)
        self.assertEqual(concentration["net_without_largest_five_positive_fn_delta_pixels"], -3)



    def test_largest_five_use_signed_fn_and_leave_other_positive_errors(self):
        def reduced_square(side, pixels_to_remove):
            foreground = np.ones(side * side, dtype=np.uint8)
            foreground[-pixels_to_remove:] = 0
            bitmap = np.zeros((14, 14), dtype=np.uint8, order="F")
            bitmap[4:4 + side, 4:4 + side] = foreground.reshape(side, side)
            result = mask.encode(bitmap)
            return dict(size=result["size"], counts=result["counts"].decode("ascii"))

        target = rle(4, 4, 4)
        small = rle(4, 4)
        gt = ground_truth([annotation(i, i, target, 1024 if i <= 6 else 100) for i in range(1, 8)]
                          + [annotation(8, 8, small, 1024)])
        native = [row(i, target) for i in range(1, 8)] + [row(8, reduced_square(3, 4))]
        center = [row(i, reduced_square(4, i)) for i in range(1, 8)] + [row(8, small)]
        result = evaluate(gt, native, center)
        all_gt = result["strata"]["all"]
        delta = all_gt["center_minus_native"]
        self.assertEqual((all_gt["gt_area_pixels"], delta["fn_pixels"], delta["fp_pixels"]), (121, 24, 0))
        self.assertAlmostEqual(delta["micro_fn_per_gt_area"], 24 / 121)
        self.assertAlmostEqual(delta["macro_fn_per_gt_area"], (28 / 16 - 4 / 9) / 8)
        whole, non_small = (result["fn_concentration"][key] for key in ("all", "non_small"))
        self.assertEqual((whole["total_positive_fn_delta_pixels"], whole["total_negative_fn_delta_magnitude_pixels"]),
                         (28, 4))
        self.assertEqual((whole["largest_one_positive_fn_delta_pixels"], whole["largest_five_positive_fn_delta_pixels"]),
                         (7, 25))
        self.assertEqual((whole["net_without_largest_one_positive_fn_delta_pixels"],
                          whole["net_without_largest_five_positive_fn_delta_pixels"]), (17, -1))
        self.assertEqual((non_small["net_fn_delta_pixels"], non_small["largest_five_positive_fn_delta_pixels"],
                          non_small["net_without_largest_five_positive_fn_delta_pixels"]), (17, 20, -3))
        with patch.multiple(geometry, DEV_IMAGES=8, GT_COUNT=8, CANVAS=(14, 14), EMPTY_COUNT=0):
            prior = geometry.evaluate_pair(gt, native, center)["strata"]["all"]
        self.assertEqual(prior["both_matched"], all_gt["paired_gt"])
        for arm in ("native", "center"):
            for field in ("fp_pixels", "fn_pixels"):
                self.assertEqual(all_gt[arm][field], prior["paired"][arm][f"{field}_total"])


if __name__ == "__main__":
    unittest.main()
