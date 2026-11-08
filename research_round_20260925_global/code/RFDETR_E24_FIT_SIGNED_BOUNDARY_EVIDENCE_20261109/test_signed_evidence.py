import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import signed_evidence as s


class SignedEvidenceTests(unittest.TestCase):
    def test_exact_pair_auc_with_ties_and_missing_class(self):
        self.assertEqual(s.pair_auc(np.array([0, 1, 1, 2]), np.array([0, 1, 0, 1])), (3.5, 4))
        self.assertEqual(s.pair_auc(np.array([1, 2]), np.array([1, 1])), (0.0, 0))

    def test_deterministic_mixed_action_prefix(self):
        positions = np.array([19, 6, 15, 8])
        scores = np.array([2, 2, 1, 2])
        self.assertEqual(s.prefix(positions, scores, 3).tolist(), [1, 3, 0])
        self.assertEqual(s.prefix(positions, scores, 16).tolist(), [1, 3, 0, 2])

    def test_signed_margin_exact_arithmetic(self):
        self.assertEqual(s.tally(3, 2, +1), dict(selected=3, correct=2, wrong=1, net_correct=1, margin=11))
        self.assertEqual(s.tally(3, 2, -1), dict(selected=3, correct=2, wrong=1, net_correct=1, margin=8))

    def test_directional_contrast_sign_and_invalid_fallback(self):
        np.testing.assert_allclose(
            s.rank_disagreement(np.array([.6, -.4, 1., -1.]), np.array([True, True, False, True])),
            [-.6, .4, -2., 1.],
        )

    def test_image_paired_unit_weights(self):
        paired = s.paired_image_means([((.4, .6), (.3, .5)), ((1.,), (.7,)), ((), ())])
        self.assertEqual(paired["images"], 2)
        self.assertAlmostEqual(paired["delta"], .2)
        self.assertEqual(paired["positive_images"], 2)

    def test_applied_rank_ten_does_not_backfill_empty_bands(self):
        rows = [{"score": 1. - i / 100} for i in range(12)]
        sizes = np.array([0] + [1] * 11)
        self.assertEqual(s.applied_rows(rows, sizes), set(range(1, 10)))

    def test_unmatched_are_separate_from_errors(self):
        acc = s.ImageMetrics()
        acc.record_application("stock", 16, +1, "unmatched", 3)
        acc.record_application("stock", 16, +1, "matched_small", 2)
        self.assertEqual(acc.application[("stock", 16, +1)], {"unmatched": 3, "matched_small": 2})
        self.assertEqual(acc.pixel, {})


    def test_original_match_and_application_accounting(self):
        rows = [
            dict(score=.9, start=0, stop=4, status="matched", GT_annotation_id=11),
            dict(score=.8, start=4, stop=7, status="unmatched", GT_annotation_id=None),
            dict(score=.7, start=7, stop=7, status="matched", GT_annotation_id=12),
        ]
        arrays = dict(
            offsets=np.array([0, 4, 7, 7]), label_offsets=np.array([0, 4, 4, 4]),
            positions=np.array([12, 13, 16, 17, 21, 25, 23]),
            edit_sign=np.array([1, 1, -1, -1, -1, -1, 1]),
            valid=np.array([True, False, True, True, False, True, True]),
            y=np.array([True, False, False, True]),
        )
        scores = {arm: np.arange(7.0) for arm in s.ARMS}
        img = s.ImageMetrics()
        img.add(rows, arrays, {11: True, 12: False}, scores)
        self.assertEqual(img.population["correct_labeled_pixels"], 2)
        self.assertEqual(img.population["top10_nonempty"], 2)
        sc = s.summarize_scope([img], s.SCOPES[1])
        self.assertEqual(sc["addition"]["band"]["valid_hits"], 1)
        self.assertEqual(sc["addition"]["arms"]["stock"]["auc"]["pair_weighted"], 0)
        self.assertEqual(sc["removal"]["arms"]["stock"]["auc"]["pair_weighted"], 1)
        self.assertEqual(sc["addition"]["arms"]["stock"]["mixed_direction_prefix"]["16"]["margin"], 1)
        self.assertEqual(sc["removal"]["arms"]["stock"]["mixed_direction_prefix"]["16"]["margin"], -1)
        app = s.summarize_application([img])
        self.assertEqual(app["candidates"], {"matched_small": 1, "unmatched": 1})
        self.assertEqual(app["arms"]["stock"]["16"]["removal"]["unmatched_fraction"], .5)
        self.assertEqual(app["arms"]["stock"]["16"]["total_selected"], 7)


if __name__ == "__main__":
    unittest.main()
