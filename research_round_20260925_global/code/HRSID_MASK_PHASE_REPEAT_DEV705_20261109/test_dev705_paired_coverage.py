import unittest
from unittest.mock import patch

import numpy as np
from pycocotools import mask

import dev705_paired_coverage as coverage


def rle(*bits):
    array = np.asfortranarray([[int(pixel) for pixel in row] for row in bits], dtype=np.uint8)
    value = mask.encode(array)
    return dict(size=value["size"], counts=value["counts"].decode("ascii"))


def prediction(image_id, segmentation, score=.9):
    return dict(image_id=image_id, category_id=1, score=score, segmentation=segmentation)


class PairedCoverageTest(unittest.TestCase):
    def setUp(self):
        left = rle("11110000", "00000000")
        right = rle("00001111", "00000000")
        wide = rle("11111111", "00000000")
        empty = rle("00000000", "00000000")
        self.gt = dict(images=[dict(id=i, width=8, height=2) for i in (7, 2)], categories=[dict(id=1)],
                       annotations=[dict(id=i, image_id=image, category_id=1, segmentation=segment, area=area)
                                    for i, (image, segment, area) in enumerate(
                                        [(7, left, 4), (7, right, 1024), (7, empty, 0), (2, left, 4)], 1)])
        self.native = [prediction(7, empty), prediction(7, wide), prediction(7, right)]
        self.center = [prediction(7, right), prediction(7, wide), prediction(2, left)]

    def evaluate(self, *, gt_count=4):
        with patch.multiple(coverage, DEV_IMAGES=2, GT_COUNT=gt_count, EMPTY_COUNT=1, CANVAS=(8, 2)):
            return coverage.evaluate_pair(self.gt, self.native, self.center)

    def test_native_tie_order_and_equal_mapped_prefix_are_both_preserved(self):
        result = self.evaluate()
        self.assertEqual(result["exclusions"], dict(empty=1, crowd=0, ignore=0))
        self.assertEqual(result["eligible_gt_by_coco_area"], dict(small=2, non_small=1))
        full = result["full"]
        a = full["arms"]["native"]["thresholds"]["0.50"]
        b = full["arms"]["center"]["thresholds"]["0.50"]
        self.assertEqual((a["independent"], a["maximum_one_to_one"], a["native_greedy"], a["rank_or_assignment_gap"]),
                         (2, 2, 1, 1))
        self.assertEqual((b["independent"], b["maximum_one_to_one"], b["native_greedy"]), (3, 3, 3))
        self.assertEqual(full["any_candidate_transitions"]["0.90"]["small"],
                         dict(both=0, native_only=0, center_only=1, neither=1))
        prefix = result["equal_mapped_prefix"]
        self.assertEqual(prefix["rows_per_arm"], 2)
        self.assertEqual(prefix["arms"]["center"]["thresholds"]["0.50"]["native_greedy"], 2)
        self.assertEqual(prefix["any_candidate_transitions"]["0.90"]["small"],
                         dict(both=0, native_only=0, center_only=0, neither=2))
        self.assertEqual(prefix["arms"]["native"]["thresholds"]["0.50"]["maximum_one_to_one"], 1)
        nonempty = result["equal_nonempty_prefix"]
        self.assertEqual(nonempty["rows_per_arm"], 2)
        self.assertEqual(nonempty["arms"]["native"]["thresholds"]["0.50"]["maximum_one_to_one"], 2)
        self.assertEqual(nonempty["arms"]["native"]["empty_predictions"], 0)
        self.assertFalse(result["COCO_AP"])

    def test_crowd_and_ignored_targets_do_not_enter_the_diagnostic_denominator(self):
        self.gt["annotations"][2].update(iscrowd=1, ignore=1)
        target = self.gt["annotations"][0]
        self.gt["annotations"].extend((dict(target, id=5, iscrowd=1, ignore=1), dict(target, id=6, ignore=1)))
        result = self.evaluate(gt_count=6)
        self.assertEqual(result["exclusions"], dict(empty=1, crowd=1, ignore=1))
        self.assertEqual(result["eligible_gt"], 3)
        self.assertEqual(sum(result["full"]["any_candidate_transitions"]["0.90"]["all"].values()), 3)

    def test_rejects_out_of_order_rows_and_wrong_empty_gt_census(self):
        self.center.reverse()
        with self.assertRaisesRegex(ValueError, "original image|Original image"):
            self.evaluate()
        self.center.reverse()
        self.native[1]["score"] = .95
        with self.assertRaisesRegex(ValueError, "nonincreasing native score"):
            self.evaluate()
        self.native[1]["score"] = .9
        self.gt["annotations"][2]["segmentation"] = self.gt["annotations"][0]["segmentation"]
        with self.assertRaisesRegex(ValueError, "GT exclusions and denominator"):
            self.evaluate()

    def test_no_emitted_rows_retains_zero_coverage_and_the_full_gt_denominator(self):
        self.native = self.center = []
        result = self.evaluate()
        for scope in ("full", "equal_mapped_prefix", "equal_nonempty_prefix"):
            scores = result[scope]["arms"]["native"]["thresholds"]["0.90"]
            self.assertEqual((scores["independent"], scores["candidate_quality_gap"]), (0, 3))
            self.assertEqual(result[scope]["any_candidate_transitions"]["0.90"]["all"]["neither"], 3)


if __name__ == "__main__":
    unittest.main()
