import unittest

import numpy as np
from pycocotools import mask

from paired_geometry import evaluate_pair


def rle(*rows):
    array = np.asfortranarray([[int(pixel) for pixel in row] for row in rows], dtype=np.uint8)
    encoded = mask.encode(array)
    return dict(size=encoded["size"], counts=encoded["counts"].decode("ascii"))


def gt_for(width, height, *masks):
    return dict(
        images=[dict(id=7, width=width, height=height)],
        categories=[dict(id=1)],
        annotations=[
            dict(id=i, image_id=7, category_id=1, segmentation=segmentation, **flags)
            for i, (segmentation, flags) in enumerate(masks, 1)
        ],
    )


def block(*masks):
    return [
        dict(
            image_id=7,
            raw_query_count=164,
            native_count=100,
            unmapped_slot1_count=100 - len(masks),
            native_ranks=list(range(1, len(masks) + 1)),
            coco=[
                dict(image_id=7, category_id=1, score=1 / rank, segmentation=segmentation)
                for rank, segmentation in enumerate(masks, 1)
            ],
        )
    ]


class PairedGeometryTest(unittest.TestCase):
    def test_strict_scores_keep_native_rank_point_five_match(self):
        ship = rle("11110000", "00000000")
        weak = rle("11000000", "00000000")
        gt = gt_for(8, 2, (ship, {}))

        result = evaluate_pair(gt, block(weak, ship), block(ship, weak))

        self.assertEqual(result["point_five"]["both_matched"], 1)
        for strict in ("0.75", "0.90"):
            transition = result["strict"][strict]
            self.assertEqual(transition["matched_below"]["strict"], 1)
            self.assertEqual(transition["unmatched"]["strict"], 0)
        metrics = result["common_pair_geometry"]
        self.assertEqual(metrics["native"]["fn_pixels"], 2)
        self.assertEqual(metrics["center"]["fn_pixels"], 0)
        self.assertEqual(metrics["native"]["mean_dx_pixels"], -1.0)
        self.assertEqual(metrics["center_minus_native"]["mean_dx_pixels"], 1.0)


    def test_ties_use_later_gt_and_pair_across_different_native_ranks(self):
        left = rle("11110000", "00000000")
        right = rle("00001111", "00000000")
        wide = rle("11111111", "00000000")
        ignored = rle("00000000", "11110000")
        empty = rle("00000000", "00000000")
        gt = gt_for(
            8, 2, (left, {}), (right, {}), (ignored, {"ignore": 1}),
            (ignored, {"iscrowd": 1, "ignore": 1}), (empty, {"iscrowd": 1}),
        )

        result = evaluate_pair(gt, block(empty, wide, left), block(right))

        self.assertEqual(result["point_five"], {
            "both_unmatched": 0, "native_only": 1, "center_only": 0, "both_matched": 1,
        })
        self.assertEqual(result["exclusions"], {"empty": 1, "crowd": 1, "ignore": 1})
        self.assertEqual(result["candidates"]["native"], {"mapped": 3, "empty": 1})
        for matrix in result["strict"].values():
            self.assertEqual(matrix["matched_below"]["strict"], 1)
            self.assertEqual(matrix["strict"]["unmatched"], 1)
        geometry = result["common_pair_geometry"]
        self.assertEqual(geometry["native"]["fp_pixels"], 4)
        self.assertEqual(geometry["native"]["mean_dx_pixels"], -2.0)
        self.assertEqual(geometry["center_minus_native"]["fp_pixels"], -4)

    def test_exact_strict_thresholds_and_absent_common_metrics(self):
        ship = rle("111111111111111111110000")
        exact_75 = rle("111111111111111000000000")
        exact_90 = rle("111111111111111111000000")
        gt = gt_for(24, 1, (ship, {}))

        result = evaluate_pair(gt, block(exact_90), block(exact_75))

        self.assertEqual(result["strict"]["0.75"]["strict"]["strict"], 1)
        self.assertEqual(result["strict"]["0.90"]["strict"]["matched_below"], 1)
        no_common = evaluate_pair(gt, block(), block(exact_90))
        self.assertEqual(no_common["point_five"]["center_only"], 1)
        self.assertEqual(no_common["strict"]["0.90"]["unmatched"]["strict"], 1)
        metrics = no_common["common_pair_geometry"]
        self.assertEqual(metrics["count"], 0)
        self.assertIsNone(metrics["center_minus_native"]["mean_centroid_distance_pixels"])

    def test_malformed_stream_is_rejected_without_reranking(self):
        ship = rle("11110000")
        gt = gt_for(8, 1, (ship, {}))
        bad_rank = block(ship, ship)
        bad_rank[0]["native_ranks"].reverse()
        with self.assertRaisesRegex(ValueError, "native rank order"):
            evaluate_pair(gt, bad_rank, block())
        bad_score = block(ship, ship)
        bad_score[0]["coco"][0]["score"] = 0.0
        with self.assertRaisesRegex(ValueError, "native rank order"):
            evaluate_pair(gt, bad_score, block())
        bad_accounting = block(ship)
        bad_accounting[0]["unmapped_slot1_count"] = 0
        with self.assertRaisesRegex(ValueError, "top-100 row accounting"):
            evaluate_pair(gt, bad_accounting, block())


if __name__ == "__main__":
    unittest.main()
