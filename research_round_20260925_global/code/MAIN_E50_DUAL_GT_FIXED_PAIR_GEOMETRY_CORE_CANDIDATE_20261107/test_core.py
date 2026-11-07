"""Synthetic fixtures for the fixed-pair dual-GT geometry core."""

import unittest

from core import analyze_pair


def mask(*rows):
    return (len(rows), len(rows[0])), bytes(int(pixel) for row in rows for pixel in row)


class FixedPairTests(unittest.TestCase):
    def test_shared_regions_and_directional_disagreement_errors(self):
        integer = mask("110", "100")
        floating = mask("101", "100")
        native = mask("111", "010")
        result = analyze_pair(integer, floating, native, integer, floating)
        self.assertEqual(result["shape"], [2, 3])
        self.assertEqual(result["regions"], {
            "shared_foreground": 2,
            "shared_background": 2,
            "integer_only": 1,
            "float_only": 1,
            "GT_union": 4,
        })
        self.assertEqual(result["methods"]["native"]["errors"], {
            "shared_foreground_FN": 1,
            "shared_background_FP": 1,
            "integer_band_FN": 0,
            "integer_band_FP": 1,
            "float_band_FN": 0,
            "float_band_FP": 1,
        })

    def test_signed_error_changes_keep_the_common_denominator(self):
        integer = mask("110", "100")
        floating = mask("101", "100")
        native = mask("111", "010")
        result = analyze_pair(integer, floating, native, integer, floating)
        self.assertEqual(result["regions"]["GT_union"], 4)
        contrasts = result["contrasts"]
        self.assertEqual(contrasts["PLAIN_minus_native"]["errors"], {
            "shared_foreground_FN": -1, "shared_background_FP": -1,
            "integer_band_FN": 0, "integer_band_FP": -1,
            "float_band_FN": 1, "float_band_FP": 0,
        })
        self.assertEqual(contrasts["FULL_minus_native"]["errors"]["integer_band_FN"], 1)
        self.assertEqual(contrasts["FULL_minus_PLAIN"]["errors"], {
            "shared_foreground_FN": 0, "shared_background_FP": 0,
            "integer_band_FN": 1, "integer_band_FP": 1,
            "float_band_FN": -1, "float_band_FP": -1,
        })

    def test_exact_iou75_and_directions_under_both_annotations(self):
        integer = mask("11110000")
        floating = mask("11101000")
        native = mask("11100000")
        result = analyze_pair(integer, floating, native, integer, floating)
        exact = dict(intersection=3, union=4, at_least_0_75=True)
        self.assertEqual(result["methods"]["native"]["iou"], {"integer": exact, "float": exact})
        self.assertEqual(result["methods"]["PLAIN"]["iou"]["float"],
                         dict(intersection=3, union=5, at_least_0_75=False))
        contrasts = result["contrasts"]
        self.assertEqual(contrasts["PLAIN_minus_native"]["iou75_crossing"], {"integer": 0, "float": -1})
        self.assertEqual(contrasts["FULL_minus_native"]["iou75_crossing"], {"integer": -1, "float": 0})
        self.assertEqual(contrasts["FULL_minus_PLAIN"]["iou75_crossing"], {"integer": -1, "float": 1})

    def test_invalid_grid_shape_and_nonbinary_bytes_are_rejected(self):
        valid = mask("10", "01")
        invalid = (
            (mask("1001"), "grid shape"),
            (((2, 2), b"\x01\x00\x02\x00"), "0/1 byte"),
            (((2, 2), b"\x01\x00\x01"), "0/1 byte"),
            (((True, 4), b"\x01\x00\x01\x00"), "positive integer"),
        )
        for candidate, message in invalid:
            with self.subTest(candidate=candidate), self.assertRaisesRegex(ValueError, message):
                analyze_pair(valid, valid, valid, candidate, valid)

    def test_empty_prediction_is_retained_and_each_gt_must_be_nonempty(self):
        integer, floating, empty = mask("10"), mask("01"), mask("00")
        result = analyze_pair(integer, floating, empty, integer, floating)
        self.assertEqual(result["regions"]["shared_foreground"], 0)
        self.assertEqual(result["regions"]["GT_union"], 2)
        self.assertEqual(result["methods"]["native"]["iou"],
                         {kind: dict(intersection=0, union=1, at_least_0_75=False)
                          for kind in ("integer", "float")})
        for gi, gf in ((empty, floating), (integer, empty)):
            with self.subTest(gi=gi, gf=gf), self.assertRaisesRegex(ValueError, "GT mask must be nonempty"):
                analyze_pair(gi, gf, empty, integer, floating)
