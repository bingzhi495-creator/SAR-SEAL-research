"""Verify rectangular native rows, exact budgets and COCO anchor identity."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest

import numpy as np
from pycocotools import mask

spec = spec_from_file_location("_test_psg_score", Path(__file__).with_name("score.py"))
score = module_from_spec(spec)
spec.loader.exec_module(score)


def rle(binary):
    r = mask.encode(np.asfortranarray(binary))
    return {"size": r["size"], "counts": r["counts"].decode("ascii")}


class FrozenRows(unittest.TestCase):
    def test_exact_ranks_one_area_and_primary_anchors(self):
        pix = np.zeros((34, 61), np.uint8)
        pix[5:13, 3:11] = 1
        moved = pix.copy()
        moved[5, 3], moved[5, 11] = 0, 1
        raw = {"image_id": 9, "category_id": 1, "score": .81, "segmentation": rle(pix)}
        cap = {"image_id": 9, "stock_prediction_start": 0, "stock_prediction_count": 1,
               "native_ranks": [3], "original_hw": [34,61]}
        plans = {"image_id":9, "rgb_sha256":"1"*64,
                 "changes":{"3":{arm: {"swaps":1,"rle":rle(moved)} for arm in score.ARMS[1:]}}}
        blocks, counts = score.cohort([raw], [cap], [plans])
        self.assertEqual(counts[score.ARMS[5]+"_swaps"], 1)
        got = {"info": {}, "images": [{"id":9,"height":34,"width":61}],
               "categories": [{"id":1,"name":"ship"}],
               "annotations": [{"id":43,"image_id":9,"category_id":1,"segmentation":[[3,5,11,5,11,13,3,13]],
                                "iscrowd":0,"area":64,"bbox":[3,5,8,8]}]}
        metrics, pairs = score.score_one(got, blocks, "raw", anchors=True)
        self.assertAlmostEqual(metrics["AP"], 1)
        self.assertEqual(pairs[43][0:2], (9,3))
        plans["changes"]["3"][score.ARMS[5]]["swaps"] = 2
        with self.assertRaisesRegex(ValueError, "Count matched"):
            score.cohort([raw], [cap], [plans])


if __name__ == "__main__":
    unittest.main()

