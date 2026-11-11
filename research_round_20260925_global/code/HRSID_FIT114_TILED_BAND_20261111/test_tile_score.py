import unittest
from collections import Counter

import numpy as np

import tile_score


class TileScoreTests(unittest.TestCase):
    def test_ranked_edits_are_per_side_and_break_ties_by_pixel(self):
        inside = np.array([True, True, True, False, False, False, False])
        pixels = np.array([1, 2, 3, 4, 5, 6, 7])
        logits = np.array([0.7, -0.4, -0.4, 0.4, 0.9, 0.9, -0.3])
        order = tile_score.equal_edits(np.where(inside, -logits, logits), inside, pixels)
        np.testing.assert_array_equal(order, [4, 5, 1, 2])
        np.testing.assert_array_equal(
            tile_score.equal_edits([1.0], np.array([True]), np.array([7])), [0])

    def test_scoring_reconstructs_exact_margins_and_threshold_crossing(self):
        counts = Counter()
        inside = np.array([True, True, False, False])
        truth = np.array([True, False, True, False])
        tile_score.score_candidate(counts, np.array([1, 2]), inside, truth, 8, 10)
        self.assertEqual(counts["margin_gain_75"], 7)
        self.assertEqual(counts["margin_gain_90"], 19)
        self.assertEqual(counts["recover_90"], 1)
        self.assertEqual(counts["recover_75"], 0)
        self.assertEqual(counts["matched_correct_add"], 1)
        self.assertEqual(counts["matched_correct_delete"], 1)
        tile_score.score_candidate(counts, np.array([0, 3]), inside, None)
        self.assertEqual(counts["edited_unmatched"], 1)
        self.assertEqual(counts["unmatched_add"], 1)
        self.assertEqual(counts["unmatched_delete"], 1)

    def test_geometry_includes_in_canvas_edge_band_and_refuses_overflow(self):
        mask = np.zeros((800, 800), dtype=bool)
        mask[0, 0] = True
        crop, points = tile_score.candidate_geometry(mask)
        self.assertEqual(crop, [0, 0, 400, 400])
        self.assertEqual(points.tolist(), [[y, x] for y in range(3) for x in range(3)])
        mask[-1, -1] = True
        self.assertIsNone(tile_score.candidate_geometry(mask))

    def test_oracle_maximizes_iou_at_fixed_quota(self):
        inside = np.array([True, True, True, False, False, False])
        truth = np.array([False, True, False, True, False, True])
        pixel = np.arange(len(inside))
        ordered = tile_score.oracle_edits(inside, truth, pixel)
        np.testing.assert_array_equal(ordered, [3, 5, 0, 2])

    def test_unmatched_tile_remains_unmatched_after_full_roster_matching(self):
        class Prior:
            def original_matches(self, predictions, gt):
                self.observed = (predictions, gt)
                return {0: gt[0]}

        prior = Prior()
        coords = [[1, 1], [1, 2], [2, 1], [2, 2]]
        inside = [True, True, False, False]
        identity = dict(image_id=1, group="synthetic", encoded_sha256="a"*64, crop_xyxy=[0, 0, 400, 400],
                        candidate=dict(native_rank=2, query_id=3, score=0.8, category_id=1, segmentation={}))
        selected = dict(identity, index=1, predictions=["native_first", "tile"],
                        points=np.array(coords), inside=np.array(inside), areas=[10, 10])
        row = dict(identity, points=[
            dict(y=y, x=x, inside=side, full=1.0 if side else -1.0,
                 native_crop=-1.0 if side else 1.0, lowpass_crop=1.0 if side else -1.0)
            for (y, x), side in zip(coords, inside)
        ])
        result = tile_score.score_tiles(prior, [selected], [row], {1: ["same_gt"]})
        self.assertEqual(prior.observed, (["native_first", "tile"], ["same_gt"]))
        self.assertEqual(result["census"]["selected_fixed_matches"], 0)
        self.assertEqual(result["primary"]["native_crop"]["pixels_changed_unmatched"], 4)
        self.assertEqual(result["primary"]["gt_oracle"]["pixels_changed"], 0)
        self.assertEqual(result["secondary_direct_sign"]["lowpass_crop"]["pixels_changed"], 0)
        row["points"].reverse()
        with self.assertRaisesRegex(ValueError, "coordinate order"):
            tile_score.score_tiles(prior, [selected], [row], {1: ["same_gt"]})


if __name__ == "__main__":
    unittest.main()
