"""Generated tests for the Gaofen-3 frozen score integration."""

from pathlib import Path
import sys
import unittest

import numpy as np
from pycocotools import mask

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_gf3 import matched_masks


def rle(cells):
    raster = np.zeros((512, 512), dtype=np.uint8, order="F")
    for y, x in cells:
        raster[y, x] = 1
    encoded = mask.encode(raster)
    encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded


class FrozenMaskTest(unittest.TestCase):
    def test_equal_count_and_zero_fallback(self):
        native = rle([(2, 2), (3, 3), (4, 4)])
        equal = {
            "5217/fixed": {"swaps": 1, "rle": rle([(1, 1), (3, 3), (4, 4)])},
            "5217/free_matched_count": {"swaps": 1, "rle": rle([(5, 5), (2, 2), (4, 4)])},
        }
        picked, quotas = matched_masks(mask, native, equal, (5217, 5218))
        self.assertEqual(quotas, {5217: 1, 5218: 0})
        self.assertEqual(picked["5218/fixed"], native)
        self.assertEqual(picked["5218/free_matched_count"], native)
        self.assertNotEqual(picked["5217/fixed"], picked["5217/free_matched_count"])

    def test_same_area_invalid_budget_fails(self):
        native = rle([(0, 0), (0, 1)])
        neither = rle([(3, 3), (4, 4)])
        invalid = {"5217/fixed": {"swaps": 1, "rle": neither},
                   "5217/free_matched_count": {"swaps": 1, "rle": neither}}
        with self.assertRaisesRegex(ValueError, "Written signed changes"):
            matched_masks(mask, native, invalid, (5217,))

    def test_unpaired_policy_and_unequal_quotas_fail(self):
        native = rle([(0, 0), (0, 1)])
        edit = rle([(1, 0), (0, 1)])
        invalid = {"5217/fixed": {"swaps": 1, "rle": edit}}
        with self.assertRaisesRegex(ValueError, "Orphan signed writer"):
            matched_masks(mask, native, invalid, (5217,))
        invalid["5217/free_matched_count"] = {"swaps": 2, "rle": edit}
        with self.assertRaisesRegex(ValueError, "Different signed budgets"):
            matched_masks(mask, native, invalid, (5217,))

    def test_grouped_continuous_parity_checks_historical_intervals(self):
        from score_gf3 import geometric_parity

        seeds = (5217, 5218)
        arms = ("native", "5217/fixed", "5217/free_matched_count",
                "5218/fixed", "5218/free_matched_count")
        rows = {g: [[84, 84.5, 85.0, 85.5, 87]] for g in range(68)}
        expected = {}
        for seed, fixed, free in zip(seeds, (84.5, 85.5), (85., 87.), strict=True):
            hist = {arm: {"continuous_iou_points": value} for arm, value in
                    zip(("raw", "fixed", "free_matched_count"), (84, fixed, free), strict=True)}
            contrast = {"continuous_iou_points": {"delta": free - fixed, "interval_95": [free - fixed, free - fixed]}}
            expected[seed] = {"stats": {"group_equal_arms": hist, "paired_contrasts_points":
                                        {"free_matched_count minus fixed": contrast}}}
        actual = geometric_parity(rows, expected, 5202, arms, seeds)
        self.assertEqual(actual["head_mean"]["delta_points"], 1.)
        self.assertEqual(actual["head_mean"]["interval_95"], [1., 1.])
        expected[5218]["stats"]["paired_contrasts_points"]["free_matched_count minus fixed"][
            "continuous_iou_points"]["interval_95"] = [1.44, 1.46]
        with self.assertRaisesRegex(ValueError, "geometry or interval differs"):
            geometric_parity(rows, expected, 5202, arms, seeds)

    def test_cohort_coco_metric_legacy_missing_signature(self):
        from score_gf3 import agree_native_metrics

        agree_native_metrics({"AP": .613, "APl": -1}, {"AP": .613, "APl": None})
        with self.assertRaisesRegex(ValueError, "Exact original native COCO"):
            agree_native_metrics({"AP": .614}, {"AP": .613})


if __name__ == "__main__":
    unittest.main()
