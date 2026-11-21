"""Synthetic-only tests. CODE_ROOT and SHAPELY_SITE name ordinary published code and runtime."""
import os
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
from pycocotools import mask as codec

from fit109_score import ARMS, evaluate_fit109, gate, load_helpers, same_count_edit


def encoded(mask):
    value = codec.encode(np.asfortranarray(mask, dtype=np.uint8))
    value["counts"] = value["counts"].decode("ascii")
    return value


def fixture():
    images, annotations, candidates, rows, predictions, budgets = [], [], {}, [], {}, {}
    for image_id in (12, 43):
        hw = (18, 22)
        images.append(dict(id=image_id, height=18, width=22))
        annotations.append(dict(id=image_id+100, image_id=image_id, category_id=1,
                                segmentation=[[3, 5, 13, 5, 13, 9, 3, 9]],
                                bbox=[3, 5, 10, 4], area=40, iscrowd=0))
        stock = np.zeros(hw, dtype=bool)
        stock[5:9, 3:13] = True
        stock[5, 4] = False
        stock[9, 7] = True
        slot = dict(rank=2, query_id=11, score=.75, area=40, rle=encoded(stock))
        # Empty original raster and equal-score native tie remain in the COCO stream.
        empty = dict(rank=4, query_id=9, score=.75, area=0, rle=encoded(np.zeros(hw)))
        candidates[image_id] = [slot, empty]
        coordinates = (np.array([[5*22+4, 9*22+7], [4*22+7, 5*22+7]], dtype=np.int32)
                       if image_id == 12 else np.empty((0, 2), dtype=np.int32))
        key = (image_id, 2, 11)
        rows.append(dict(image_id=image_id, rank=2, query_id=11, group=str(image_id),
                         coordinates=coordinates))
        good = np.array([[1, 0], [0, 1]], dtype=np.float32)[:len(coordinates)]
        predictions[key] = {arm: good.copy() if arm == "Q" else 1-good for arm in ARMS}
        budgets[key] = int(bool(len(coordinates)))
    # Keep even original annotations whose native raster has zero area in pooled AP.
    annotations.append(dict(id=900, image_id=12, category_id=1, iscrowd=0, area=.01,
                            bbox=[1, 1, .1, .1], segmentation=[[1, 1, 1.1, 1, 1, 1.1]]))
    gt = dict(images=images, annotations=annotations, categories=[dict(id=1, name="ship")])
    return gt, candidates, rows, predictions, budgets


class ScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helpers = load_helpers(Path(os.environ["CODE_ROOT"]), Path(os.environ["SHAPELY_SITE"]))

    def score(self, args=None, **kwargs):
        return evaluate_fit109(*(fixture() if args is None else args), self.helpers, expected_images=2, **kwargs)

    def test_per_arm_fixed_count_independent_sides_and_ties(self):
        coordinates = np.array([[8, 3], [1, 12], [10, 6]], dtype=np.int32)
        p = np.array([[.1, .8], [.3, .2], [.3, .1]], np.float32)
        edit = same_count_edit(coordinates, p, 1)
        self.assertEqual(edit["flipped"].tolist(), [1, 6])
        self.assertEqual(edit["area_delta"], 0)
        self.assertEqual(same_count_edit(coordinates, 1-p, 0)["edits"], 0)
        with self.assertRaises(ValueError):
            same_count_edit(coordinates, p.astype(float), 1)

    def test_original_pooled_geometry_and_short_axis(self):
        args = fixture()
        original_gt = deepcopy(args[0])
        result = self.score(args)
        self.assertEqual(args[0], original_gt)
        self.assertEqual(result["counts"]["mapped_candidates"], 4)
        self.assertEqual(result["counts"]["annotations"], 3)
        self.assertEqual(result["counts"]["valid"], 2)
        self.assertEqual(result["counts"]["valid_unsupported_noop"], 1)
        self.assertEqual(result["counts"]["source_groups_with_valid"], 2)
        self.assertEqual(result["arms"]["Q"]["edits"], 2)
        self.assertEqual(result["arms"]["Q"]["strict90"]["lost"], 0)
        self.assertEqual(result["arms"]["F"]["strict90"]["lost"], 0)
        area = result["arms"]["Q"]["small_slender"]
        self.assertEqual(area["valid_instances"], 2)
        self.assertAlmostEqual(area["mean_abs_short_px"], .05, places=10)
        self.assertEqual(area["worse_than_stock_fraction"], 0)
        self.assertGreater(result["arms"]["Q"]["AP"], result["arms"]["F"]["AP"])
        self.assertAlmostEqual(result["comparisons"]["Q_minus_F"]["geometry_delta_points"],
                               50*(1-38/42))
        self.assertEqual(result["comparisons"]["Q_minus_F"]["geometry_ci95_points"][0], 0.)
        self.assertFalse(result["gate"]["advance"])

    def test_population_or_support_tampering_fails(self):
        for change in ("drop_zero", "bad_rank", "bad_query", "bad_group", "duplicate_group", "wrong_orientation",
                       "outside_cap", "swapped_witness", "foreign_latent"):
            args = fixture()
            _, candidates, records, predictions, budgets = args
            kwargs = {}
            if change == "drop_zero":
                records.pop()
            elif change == "bad_rank":
                candidates[12].reverse()
            elif change == "bad_query":
                candidates[12][0]["query_id"] = 33
            elif change == "bad_group":
                records[0]["group"] = ""
            elif change == "duplicate_group":
                records[1]["group"] = records[0]["group"]
            elif change == "wrong_orientation":
                records[0]["coordinates"][:, [0,1]] = records[0]["coordinates"][:, [1,0]]
            elif change == "outside_cap":
                budgets[(12,2,11)] = 5
            elif change == "swapped_witness":
                kwargs["expected_flipped"] = {key: {a: [] for a in ARMS} for key in budgets}
            else:
                predictions[(12,2,11)]["Q"][0,0] = np.nan
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.score(args, **kwargs)

    def test_governed_gate_boundaries(self):
        arms = {a: dict(AP=.9, AP75=.93, strict90=dict(lost=1)) for a in ARMS}
        comparisons = {"Q_minus_"+a: dict(AP_delta_points=.30, geometry_delta_points=.20,
                                           geometry_ci95_points=[.01, .40]) for a in ("Z", "F", "QP")}
        self.assertTrue(gate(arms, comparisons)["advance"])
        for key, field, value in (("Q_minus_Z","AP_delta_points",.2999),
                                  ("Q_minus_F","geometry_delta_points",.1999),
                                  ("Q_minus_QP","geometry_ci95_points",[0., .30])):
            partial = deepcopy(comparisons)
            partial[key][field] = value
            self.assertFalse(gate(arms, partial)["advance"])
        for ap75, loss in ((.929, 1), (.93, 2), (-1,1)):
            weakened = deepcopy(arms)
            weakened["Q"].update(AP75=ap75, strict90=dict(lost=loss))
            self.assertFalse(gate(weakened, comparisons)["advance"])


if __name__ == "__main__":
    unittest.main()
