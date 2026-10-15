"""Synthetic-only policy tests. No SAR data, project imports, or Torch/NumPy."""

import unittest

from N20_TWO_SEED_PRELOCKED_RESULT_INTERPRETER_UNBOUND_20261016_01 import (
    STRONG_AP,
    interpret_two_seed_main,
)


SEED_201 = 202610150201
SEED_202 = 202610150202
SEED_203 = 202610150203


def row(seed, local, center):
    return {
        "seed": seed,
        "epoch": 24,
        "mode": "atmost",
        "AP_unit": "percent",
        "LOCAL_AP": local,
        "CENTER_AP": center,
        "strong_AP": STRONG_AP,
    }


class TwoSeedDecision(unittest.TestCase):
    def test_both_pairs_win_and_mean_delta_reaches_one_allows_only_203(self):
        result = interpret_two_seed_main(
            [row(SEED_202, STRONG_AP + 1.5, 68.0),
             row(SEED_201, STRONG_AP + 0.5, 68.0)]
        )
        self.assertEqual(result["decision"], "ALLOW_SEED203")
        self.assertEqual(result["allowed_next_seeds"], [SEED_203])
        self.assertEqual(result["seeds"], [SEED_201, SEED_202])
        self.assertEqual(result["Delta_AP_points_by_seed"], {SEED_201: 0.5, SEED_202: 1.5})
        self.assertEqual(result["mean_Delta_AP_points"], 1.0)

    def test_mean_delta_can_pass_while_one_pair_loses_and_must_stop(self):
        result = interpret_two_seed_main(
            [row(SEED_201, 71.0, 72.0),
             row(SEED_202, STRONG_AP + 3.0, 68.0)]
        )
        self.assertEqual(result["mean_Delta_AP_points"], 1.0)
        self.assertEqual(result["decision"], "STOP_CURRENT_RECIPE")
        self.assertEqual(result["allowed_next_seeds"], [])
        self.assertLess(result["paired_LOCAL_minus_CENTER_AP_points_by_seed"][SEED_201], 0)

    def test_average_then_max_cannot_rescue_insufficient_mean_delta(self):
        result = interpret_two_seed_main(
            [row(SEED_201, 70.4, 70.0),
             row(SEED_202, STRONG_AP + 0.4, 60.0)]
        )
        self.assertTrue(all(gain > 0 for gain in result["paired_LOCAL_minus_CENTER_AP_points_by_seed"].values()))
        self.assertLess(result["mean_Delta_AP_points"], 1.0)
        self.assertGreater(result["mean_LOCAL_minus_max_meanCENTER_strong_AP_points"], 1.0)
        self.assertIs(result["diagnostic_aggregate_used_for_decision"], False)
        self.assertEqual(result["decision"], "STOP_CURRENT_RECIPE")

    def test_exactly_two_distinct_prelocked_seeds_are_required(self):
        valid = [row(SEED_201, 69.0, 68.0), row(SEED_202, 70.0, 68.0)]
        invalid_inputs = [
            valid[:1],
            valid + [row(SEED_203, 71.0, 68.0)],
            [valid[0], valid[0].copy()],
            [valid[0], row(SEED_203, 70.0, 68.0)],
            tuple(),
            {"results": valid},
        ]
        for value in invalid_inputs:
            with self.subTest(value=value), self.assertRaises(ValueError):
                interpret_two_seed_main(value)

    def test_only_exact_e24_atmost_percent_keys_and_strong_reference_are_accepted(self):
        mutations = [
            ("epoch", 8), ("epoch", True), ("mode", "exact"),
            ("AP_unit", "fraction"), ("strong_AP", STRONG_AP + 0.1),
            ("E8_AP", 99.0),
        ]
        for key, value in mutations:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                first = row(SEED_201, 69.0, 68.0)
                second = row(SEED_202, 70.0, 68.0)
                first[key] = value
                interpret_two_seed_main([first, second])

    def test_both_ap_values_must_be_defined_finite_percent_numbers(self):
        for key in ("LOCAL_AP", "CENTER_AP"):
            for value in (-1.0, -0.01, 100.01, float("nan"), float("inf"), True, "68.0"):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    first = row(SEED_201, 69.0, 68.0)
                    first[key] = value
                    interpret_two_seed_main([first, row(SEED_202, 70.0, 68.0)])

    def test_decision_never_certifies_results_or_unlocks_rescue_variants(self):
        result = interpret_two_seed_main(
            [row(SEED_201, STRONG_AP + 0.5, 68.0),
             row(SEED_202, STRONG_AP + 1.5, 68.0)]
        )
        for key in (
            "qualification_performed", "task_efficacy_established",
            "innovation_established", "TGRS_readiness_established",
            "additional_budget_or_fusion_allowed", "secondary_results_may_replace_main",
        ):
            with self.subTest(key=key):
                self.assertIs(result[key], False)

    def test_result_carries_explicit_protocol_and_copies_sorted_input(self):
        first = row(SEED_201, 69.0, 68.0)
        second = row(SEED_202, 70.0, 68.0)
        result = interpret_two_seed_main([second, first])
        self.assertEqual(result["main_epoch"], 24)
        self.assertEqual(result["main_mode"], "atmost")
        self.assertEqual(result["AP_unit"], "percent")
        self.assertEqual(result["strong_AP_percent"], STRONG_AP)
        self.assertEqual(result["main_results"], [first, second])
        first["LOCAL_AP"] = 99.0
        self.assertEqual(result["main_results"][0]["LOCAL_AP"], 69.0)
        self.assertIs(result["source_authentication_performed"], False)


if __name__ == "__main__":
    unittest.main()
