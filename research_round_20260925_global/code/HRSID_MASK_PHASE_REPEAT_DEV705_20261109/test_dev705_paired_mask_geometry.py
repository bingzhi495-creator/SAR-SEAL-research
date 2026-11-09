"""Synthetic CPU tests for fixed .50-pair native-grid mask geometry."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
from pycocotools import mask

import dev705_paired_mask_geometry as subject


def rle(y=None, x=None, side=3):
    bitmap = np.zeros((14, 14), dtype=np.uint8, order="F")
    if y is not None:
        bitmap[y:y+side, x:x+side] = 1
    result = mask.encode(bitmap)
    return dict(size=result["size"], counts=result["counts"].decode("ascii"))


def annotation(aid, iid, segmentation, area=9):
    return dict(id=aid, image_id=iid, category_id=1, iscrowd=0, segmentation=segmentation,
                bbox=mask.toBbox(segmentation).tolist(), area=area)


def row(iid, segmentation, score=.9):
    return dict(image_id=iid, category_id=1, score=score, segmentation=segmentation)


def ground_truth(annotations):
    ids = [entry["image_id"] for entry in annotations]
    return dict(info={}, images=[dict(id=i, height=14, width=14) for i in ids],
                annotations=annotations, categories=[dict(id=1, name="ship")])


def evaluate(gt, native, center, empty_count=0):
    with patch.multiple(subject, DEV_IMAGES=len(gt["images"]), GT_COUNT=len(gt["annotations"]),
                        CANVAS=(14, 14), EMPTY_COUNT=empty_count):
        return subject.evaluate_pair(gt, native, center)


class PairedGeometryTests(unittest.TestCase):
    def test_directional_recovery_uses_shared_interior_and_original_area_strata(self):
        gt = ground_truth([annotation(1, 7, rle(4, 4)), annotation(2, 8, rle(0, 4), 1024),
                           annotation(3, 9, rle(5, 5)), annotation(4, 10, rle(), 0)])
        native = [row(7, rle(5, 4)), row(8, rle(0, 4)), row(9, rle(5, 5)), row(10, rle(5, 5))]
        center = [row(7, rle(3, 4)), row(8, rle(1, 4)), row(9, rle(10, 10)), row(10, rle(5, 5))]
        frozen = deepcopy((gt, native, center))
        result = evaluate(gt, native, center, empty_count=1)
        self.assertEqual((gt, native, center), frozen)
        self.assertEqual(result["retained_prefix_rows"], 4)
        self.assertEqual(result["exclusions"], dict(empty=1, crowd=0, ignore=0))
        all_gt, small = (result["strata"][key] for key in ("all", "small"))
        self.assertEqual((all_gt["eligible_gt"], all_gt["both_matched"], all_gt["native_only"]), (3, 2, 1))
        self.assertEqual((small["eligible_gt"], small["both_matched"], small["native_only"]), (2, 1, 1))
        directional = all_gt["directional"]
        self.assertEqual((directional["shared_interior"], directional["boundary_excluded"]), (1, 1))
        self.assertAlmostEqual(directional["native"]["up"]["mean_delta_iou"], .5)
        self.assertAlmostEqual(directional["center"]["down"]["mean_delta_iou"], .5)
        self.assertAlmostEqual(directional["native"]["down"]["mean_delta_iou"], -.3)
        self.assertAlmostEqual(directional["center"]["up"]["mean_delta_iou"], -.3)
        self.assertEqual(directional["native"]["up"]["gains_strict90"], 1)
        self.assertEqual(directional["center"]["down"]["gains_strict90"], 1)
        self.assertAlmostEqual(small["paired"]["native"]["mean_fp_per_gt_area"], 1 / 3)
        self.assertAlmostEqual(small["paired"]["native"]["mean_signed_dy_pixels"], 1.)
        self.assertAlmostEqual(small["paired"]["center"]["mean_signed_dy_pixels"], -1.)



    def test_equal_prefix_preserves_official_score_ties_and_no_geometry_reselection(self):
        gt = ground_truth([annotation(5, 11, rle(4, 4))])
        gt["annotations"][0]["segmentation"] = [[4, 4, 7, 4, 7, 7, 4, 7]]
        native = [row(11, rle(5, 4)), row(11, rle(4, 4)), row(11, rle(8, 8), .1)]
        center = [row(11, rle(), .99), row(11, rle(4, 4)), row(11, rle(8, 8))]
        result = evaluate(gt, native, center)
        self.assertEqual(result["retained_prefix_rows"], 2)
        both = result["strata"]["all"]
        self.assertEqual(both["both_matched"], 1)
        self.assertEqual(both["paired"]["native"]["mean_iou"], .5)
        self.assertEqual(both["paired"]["center"]["mean_iou"], 1.)
        self.assertEqual(both["directional"]["native"]["up"]["gains_strict90"], 1)
        self.assertEqual(both["directional"]["center"]["up"]["losses_strict90"], 1)

    def test_bottom_edge_excludes_both_arms_and_empty_prefix_retains_unmatched_gt(self):
        gt = ground_truth([annotation(6, 12, rle(10, 4)), annotation(7, 13, rle(4, 4))])
        native = [row(12, rle(10, 4)), row(13, rle(4, 4))]
        center = [row(12, rle(11, 4)), row(13, rle())]
        result = evaluate(gt, native, center)
        all_gt = result["strata"]["all"]
        self.assertEqual((result["images_with_prefix_rows"], result["images_without_prefix_rows"]), (1, 1))
        self.assertEqual((all_gt["eligible_gt"], all_gt["both_matched"], all_gt["neither"]), (2, 1, 1))
        directional = all_gt["directional"]
        self.assertEqual((directional["shared_interior"], directional["boundary_excluded"]), (0, 1))
        self.assertIsNone(directional["native"]["up"]["mean_delta_iou"])
        self.assertEqual(directional["center"]["down"]["gains_strict90"], 0)


if __name__ == "__main__":
    unittest.main()

