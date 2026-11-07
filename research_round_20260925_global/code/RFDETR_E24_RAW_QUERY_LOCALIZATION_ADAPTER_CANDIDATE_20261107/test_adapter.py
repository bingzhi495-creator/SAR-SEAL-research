"""Synthetic masks for the prelocked two-pixel localization geometry."""

import unittest

import numpy as np

from adapter import localize_pair


class LocalizationTests(unittest.TestCase):
    def test_two_pixel_band_partitions_directional_errors(self):
        gt = np.zeros((11, 11), dtype=bool)
        gt[2:7, 2:7] = True
        prediction = gt.copy()
        prediction[2, 2] = prediction[4, 4] = False
        prediction[1, 2] = prediction[10, 10] = True
        result = localize_pair(np, gt, gt, prediction)
        self.assertEqual(result["areas"], dict(GT_union=25, GT_disagreement=0, FG_band=24,
                                               FG_outside=1, BG_band=56, BG_outside=40))
        self.assertEqual(result["errors"], dict(FN=dict(band=1, outside=1), FP=dict(band=1, outside=1)))

    def test_finite_image_does_not_assume_outside_background(self):
        gt = np.ones((3, 3), dtype=bool)
        prediction = gt.copy()
        prediction[0, 0] = False
        result = localize_pair(np, gt, gt, prediction)
        self.assertEqual(result["areas"]["FG_band"], 0)
        self.assertEqual(result["errors"]["FN"], dict(band=0, outside=1))

    def test_disagreement_is_neither_shared_fg_nor_shared_bg(self):
        integer = np.zeros((7, 7), dtype=bool)
        integer[2:5, 2:5] = True
        floating = integer.copy()
        floating[2, 5] = True
        base = localize_pair(np, integer, floating, integer)
        alternate = localize_pair(np, integer, floating, floating)
        self.assertEqual(base, alternate)
        self.assertEqual(base["areas"]["GT_disagreement"], 1)
        self.assertEqual(base["errors"], dict(FN=dict(band=0, outside=0), FP=dict(band=0, outside=0)))

    def test_exposure_and_signed_errors_use_distinct_denominators(self):
        from adapter import summarize_localization

        gt = np.zeros((11, 11), dtype=bool)
        gt[2:7, 2:7] = True
        prediction = gt.copy()
        prediction[2, 2] = prediction[4, 4] = False
        prediction[1, 2] = True
        summary = summarize_localization([localize_pair(np, gt, gt, prediction)])
        self.assertEqual(summary["FN_minus_FP"]["band"], dict(pixels=0, instance_mean_pp=0.0, pooled_pp=0.0))
        self.assertEqual(summary["FN_minus_FP"]["total"], dict(pixels=1, instance_mean_pp=4.0, pooled_pp=4.0))
        fn = summary["exposure_and_concentration"]["FN"]
        self.assertEqual(fn["band_exposure_fraction"], .96)
        self.assertEqual(fn["error_band_fraction"], .5)
        self.assertAlmostEqual(fn["concentration_minus_exposure_pp"], -46)

    def test_integer_best_query_uses_ascending_natural_query_tie(self):
        from adapter import integer_best_query

        values = [.4, .85, .85, *([.1] * 97)]
        self.assertEqual(integer_best_query(values, .85), 1)
        with self.assertRaisesRegex(ValueError, "accepted integer raw maximum"):
            integer_best_query(values, .84)
