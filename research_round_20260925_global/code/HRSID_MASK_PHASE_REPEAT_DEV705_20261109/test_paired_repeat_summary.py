from decimal import Decimal
import unittest

from paired_repeat_summary import summarize


def pair(seed, native, center):
    return dict(seed=seed, native_ap_fraction=native, center_ap_fraction=center)


class PairedRepeatSummaryTest(unittest.TestCase):
    def test_exact_one_point_passes_on_the_three_seed_native_mean(self):
        result = summarize([pair(5203, 0.56, 0.57), pair(5201, "0.56", "0.57"), pair(5202, 0.56, 0.57)])

        self.assertEqual(result["decision"], "MEETS_PRELOCKED_DEV_UTILITY_GATE")
        self.assertEqual(result["all_seeds"]["stronger_reference"], "native_mean")
        self.assertEqual(Decimal(result["all_seeds"]["mean_center_minus_stronger_reference_ap_points"]), Decimal(1))
        self.assertEqual(Decimal(result["all_seeds"]["sample_sd_paired_difference_ap_points"]), Decimal(0))
        self.assertEqual([row["seed"] for row in result["paired_results"]], [5201, 5202, 5203])
        self.assertEqual([row["role"] for row in result["paired_results"]], ["screening", "confirmation", "confirmation"])


    def test_exact_one_point_against_historical_reference(self):
        result = summarize(
            pair(seed, "0.49", "0.5142744492075145") for seed in (5201, 5202, 5203)
        )

        self.assertEqual(result["decision"], "MEETS_PRELOCKED_DEV_UTILITY_GATE")
        self.assertEqual(result["all_seeds"]["stronger_reference"], "historical")
        self.assertEqual(result["historical_ap_fraction"], "0.5042744492075145")
        self.assertEqual(Decimal(result["all_seeds"]["mean_center_minus_stronger_reference_ap_points"]), Decimal(1))

    def test_benchmark_uses_the_native_mean_not_seedwise_maxima(self):
        result = summarize([pair(5201, .49, .50), pair(5202, .54, .55), pair(5203, .51, .52)])

        self.assertEqual(result["decision"], "MEETS_PRELOCKED_DEV_UTILITY_GATE")
        self.assertEqual(result["all_seeds"]["stronger_reference"], "native_mean")
        self.assertEqual(Decimal(result["all_seeds"]["mean_center_minus_stronger_reference_ap_points"]), Decimal(1))

    def test_tiny_below_gate_fails_despite_every_pair_being_positive(self):
        result = summarize(
            [pair(5201, "0.56", "0.57"), pair(5202, "0.56", "0.5699999999999999"), pair(5203, "0.56", "0.57")]
        )

        self.assertEqual(result["decision"], "DOES_NOT_MEET_PRELOCKED_DEV_UTILITY_GATE")
        self.assertTrue(result["all_seeds"]["each_center_above_native"])
        self.assertFalse(result["all_seeds"]["passes_ap_margin"])
        self.assertLess(Decimal(result["all_seeds"]["mean_center_minus_stronger_reference_ap_points"]), Decimal(1))

    def test_tied_or_negative_pair_fails_even_when_the_mean_margin_passes(self):
        for last_center in (.56, .55):
            with self.subTest(last_center=last_center):
                result = summarize([pair(5201, .56, .60), pair(5202, .56, .60), pair(5203, .56, last_center)])
                self.assertEqual(result["decision"], "DOES_NOT_MEET_PRELOCKED_DEV_UTILITY_GATE")
                self.assertFalse(result["all_seeds"]["each_center_above_native"])
                self.assertTrue(result["all_seeds"]["passes_ap_margin"])

    def test_historical_reference_can_block_three_positive_pairs(self):
        result = summarize(pair(seed, "0.49", "0.50") for seed in (5201, 5202, 5203))

        self.assertEqual(result["decision"], "DOES_NOT_MEET_PRELOCKED_DEV_UTILITY_GATE")
        self.assertEqual(result["all_seeds"]["stronger_reference"], "historical")
        self.assertTrue(result["all_seeds"]["each_center_above_native"])
        self.assertFalse(result["all_seeds"]["passes_ap_margin"])

    def test_confirmation_only_and_sample_sd_expose_screening_effect(self):
        result = summarize([pair(5201, .56, .61), pair(5202, .56, .57), pair(5203, .56, .59)])

        self.assertEqual(result["confirmation_only"]["seeds"], [5202, 5203])
        self.assertEqual(Decimal(result["confirmation_only"]["mean_paired_difference_ap_points"]), Decimal(2))
        self.assertEqual(Decimal(result["all_seeds"]["mean_paired_difference_ap_points"]), Decimal(3))
        self.assertEqual(Decimal(result["all_seeds"]["sample_sd_paired_difference_ap_points"]), Decimal(2))

    def test_missing_duplicate_wrong_and_noninteger_seeds_are_rejected(self):
        good = [pair(5201, .56, .57), pair(5202, .56, .57)]
        for rows in (good, good + [pair(5202, .56, .57)], good + [pair(5204, .56, .57)],
                     good + [pair(5203.0, .56, .57)]):
            with self.subTest(rows=rows):
                with self.assertRaisesRegex(ValueError, "Exactly one pair"):
                    summarize(rows)

    def test_bad_ap_fractions_are_rejected(self):
        good = [pair(5201, .56, .57), pair(5202, .56, .57)]
        for value in (False, float("nan"), float("inf"), "NaN", "0_5", "-0.01", "1.01"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "numeric AP fraction"):
                    summarize(good + [pair(5203, .56, value)])


if __name__ == "__main__":
    unittest.main()
