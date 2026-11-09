import unittest

import numpy as np
from pycocotools import mask

from candidate_coverage import census, graph_counts


def encoded(array):
    value = mask.encode(np.asfortranarray(array, dtype=np.uint8))
    return {"size": value["size"], "counts": value["counts"].decode("ascii")}


class CoverageTests(unittest.TestCase):
    def test_distinguishes_competition_and_order(self):
        overlap = np.array([[.91, .89, 0], [.88, 0, 0], [0, 0, 0]])
        self.assertEqual(graph_counts(overlap, .75), (2, 2, 1))
        self.assertEqual(graph_counts(overlap[:1], .75), (2, 1, 1))
        self.assertEqual(graph_counts(overlap, .90), (1, 1, 1))
        self.assertEqual(graph_counts(np.empty((0, 3)), .75), (0, 0, 0))
        self.assertEqual(graph_counts(np.empty((2, 0)), .75), (0, 0, 0))
        ties = np.array([[.80, .80], [0, .81]])
        self.assertEqual(graph_counts(ties, .75), (2, 2, 1))

    def test_real_coco_rle_denominators_and_precedence(self):
        empty = np.zeros((8, 8), dtype=np.uint8)
        ship = empty.copy()
        ship[1:3, 1:3] = 1
        gt = dict(images=[dict(id=i, width=8, height=8) for i in (1, 2)], categories=[dict(id=1)], annotations=[
            dict(id=i, image_id=image, category_id=1, area=area, segmentation=encoded(rle),
                 iscrowd=crowd, ignore=ignore)
            for i, (image, rle, area, crowd, ignore) in enumerate([
                (1, ship, 4, 0, 0), (1, empty, 0, 1, 1), (1, ship, 4, 1, 1),
                (1, ship, 4, 0, 1), (2, ship, 1024, 0, 0),
            ], 1)
        ])
        blocks = [
            dict(image_id=1, coco=[dict(image_id=1, category_id=1, score=score, segmentation=encoded(rle))
                                   for score, rle in ((.9, empty), (.1, ship))], native_ranks=[1, 4]),
            dict(image_id=2, coco=[], native_ranks=[]),
        ]
        out = census(gt, blocks)
        self.assertEqual((out["images"], out["annotations"], out["eligible_gt"]), (2, 5, 2))
        self.assertEqual(out["exclusions"], dict(empty=1, crowd=1, ignore=1))
        self.assertEqual((out["predictions"], out["empty_predictions"]), (2, 1))
        self.assertEqual(out["best_iou_bins"], [1, 0, 0, 0, 1])
        self.assertEqual(out["best_iou_bins_by_coco_area"]["small"], [0, 0, 0, 0, 1])
        self.assertEqual(out["best_iou_bins_by_coco_area"]["non_small"], [1, 0, 0, 0, 0])
        self.assertEqual(out["thresholds"]["0.90"]["native_greedy"], 1)
        self.assertEqual(out["thresholds"]["0.90"]["candidate_quality_gap"], 1)
        with self.assertRaises(ValueError):
            census(gt, blocks[::-1])


if __name__ == "__main__":
    unittest.main()
