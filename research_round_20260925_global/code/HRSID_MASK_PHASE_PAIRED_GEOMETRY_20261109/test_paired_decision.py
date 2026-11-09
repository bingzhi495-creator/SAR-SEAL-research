from decimal import Decimal
import math
import unittest

from paired_decision import decide


class PairedDecisionTest(unittest.TestCase):
    def test_exact_one_point_reaches_gate(self):
        outcome = decide(0.56, 0.57)

        self.assertEqual(outcome["decision"], "ADVANCE_PRELOCKED_REPEATS")
        self.assertEqual(Decimal(outcome["center_minus_stronger_reference_ap_points"]), Decimal(1))
        self.assertEqual(outcome["stronger_reference"], "native")


    def test_tiny_below_one_point_allows_one_pair(self):
        center = math.nextafter(0.57, 0.0)
        outcome = decide(0.56, center)

        self.assertEqual(outcome["decision"], "ALLOW_ONE_PRELOCKED_PAIR")
        self.assertLess(Decimal(outcome["center_minus_stronger_reference_ap_points"]), Decimal(1))

    def test_tiny_positive_gain_is_not_rounded_to_zero(self):
        center = math.nextafter(0.56, 1.0)
        outcome = decide(0.56, center)

        self.assertEqual(outcome["decision"], "ALLOW_ONE_PRELOCKED_PAIR")
        self.assertGreater(Decimal(outcome["center_minus_stronger_reference_ap_points"]), Decimal(0))

    def test_zero_gain_stops_and_exact_reference_tie_is_retained(self):
        outcome = decide(0.5042744492075145, 0.5042744492075145)

        self.assertEqual(outcome["decision"], "STOP_CURRENT_RECIPE")
        self.assertEqual(outcome["stronger_reference"], "both")
        self.assertEqual(Decimal(outcome["center_minus_stronger_reference_ap_points"]), Decimal(0))

    def test_historical_reference_can_block_a_positive_new_arm_contrast(self):
        outcome = decide(0.49, 0.50)

        self.assertEqual(outcome["decision"], "STOP_CURRENT_RECIPE")
        self.assertEqual(outcome["stronger_reference"], "historical")
        self.assertEqual(outcome["historical_ap_fraction"], "0.5042744492075145")
        self.assertEqual(
            outcome["historical_reference_sha256"],
            "91e7af64d6401e6a6635e94375cf8d524fa572e8a4c0a43c45979ddac77ee42d",
        )
        self.assertEqual(Decimal(outcome["center_minus_native_ap_points"]), Decimal(1))
        self.assertLess(Decimal(outcome["center_minus_historical_ap_points"]), Decimal(0))

    def test_invalid_ap_fractions_are_rejected(self):
        for value in (float("nan"), float("inf"), -0.001, 1.001, True, "0.5"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite numeric AP fraction"):
                    decide(value, 0.5)
                with self.assertRaisesRegex(ValueError, "finite numeric AP fraction"):
                    decide(0.5, value)


if __name__ == "__main__":
    unittest.main()
