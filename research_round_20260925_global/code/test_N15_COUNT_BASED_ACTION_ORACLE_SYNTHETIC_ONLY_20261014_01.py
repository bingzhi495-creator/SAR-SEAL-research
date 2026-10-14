"""SYNTHETIC_ONLY: stdlib unit tests; never read actual SAR artifacts."""
import importlib.util
from fractions import Fraction
from itertools import product
import json
from pathlib import Path
import unittest

SOURCE = Path(__file__).with_name("N15_COUNT_BASED_ACTION_ORACLE_UNBOUND_DRAFT_20261014_01.py")
SPEC = importlib.util.spec_from_file_location("_N15_synthetic_count_oracle", SOURCE)
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)

class CountOracleTests(unittest.TestCase):
    def test_helpful_add_before_helpful_remove(self):
        # Native IoU=2/7. Add two FN:4/7; remove two FP:2/5.
        result = HELPER.oracle_counts(2, 2, 3, dict(a=2, b=3, c=1, d=2, z=1), 2, "atmost")
        self.assertEqual(result["selected_counts"], dict(a=0, b=2, c=0, d=0, z=0))
        self.assertEqual(result["selected_support"], 2)
        self.assertEqual(result["actual_flips"], 2)
        self.assertEqual(result["iou_fraction"], dict(numerator=4, denominator=7))
        self.assertFalse(result["empty_union"])

    def test_exact_budget_uses_only_real_noops_after_helpful_actions(self):
        result = HELPER.oracle_counts(2, 1, 1, dict(a=1, b=1, c=1, d=1, z=2), 3, "exact")
        self.assertEqual(result["selected_counts"], dict(a=1, b=1, c=0, d=0, z=1))
        self.assertEqual(result["selected_support"], 3)
        self.assertEqual(result["actual_flips"], 2)
        self.assertEqual(result["iou_fraction"], dict(numerator=1, denominator=1))

    def test_forced_harm_adds_false_positives_before_removing_true_positives(self):
        # Native perfect3/3. Noop + 2 TN->FP gives3/5; noop + 2 TP->FN gives1/3.
        result = HELPER.oracle_counts(3, 0, 0, dict(a=0, b=0, c=2, d=2, z=1), 3, "exact")
        self.assertEqual(result["selected_counts"], dict(a=0, b=0, c=0, d=2, z=1))
        self.assertEqual(result["selected_support"], 3)
        self.assertEqual(result["harmful_flips"], 2)
        self.assertEqual(result["iou_fraction"], dict(numerator=3, denominator=5))

    def test_atmost_can_leave_budget_unused_and_exact_has_no_fictitious_noop(self):
        classes = dict(a=1, b=0, c=1, d=0, z=0)
        atmost = HELPER.oracle_counts(1, 1, 0, classes, 2, "atmost")
        exact = HELPER.oracle_counts(1, 1, 0, classes, 2, "exact")
        self.assertEqual(atmost["selected_support"], 1)
        self.assertEqual(atmost["iou_fraction"], dict(numerator=1, denominator=1))
        self.assertEqual(exact["selected_support"], 2)
        self.assertEqual(exact["selected_noops"], 0)
        self.assertEqual(exact["iou_fraction"], dict(numerator=0, denominator=1))

    def test_empty_union_is_explicitly_one(self):
        removed = HELPER.oracle_counts(0, 1, 0, dict(a=1, b=0, c=0, d=0, z=2), 1, "exact")
        native_empty = HELPER.oracle_counts(0, 0, 0, dict(a=0, b=0, c=0, d=1, z=1), 0, "atmost")
        forced_added = HELPER.oracle_counts(0, 0, 0, dict(a=0, b=0, c=0, d=1, z=0), 1, "exact")
        for result in (removed, native_empty):
            self.assertTrue(result["empty_union"])
            self.assertEqual(result["union"], 0)
            self.assertEqual(result["iou_fraction"], dict(numerator=1, denominator=1))
        self.assertFalse(forced_added["empty_union"])
        self.assertEqual(forced_added["iou_fraction"], dict(numerator=0, denominator=1))

    def test_invalid_counts_and_unrealizable_budget_are_rejected(self):
        legal = dict(a=1, b=1, c=1, d=1, z=0)
        cases = [(-1, 1, 1, legal, 1, "exact"),
                 (True, 1, 1, legal, 1, "exact"),
                 (1, 1, 1, dict(a=2, b=1, c=1, d=1, z=0), 1, "atmost"),
                 (1, 1, 1, dict(a=1, b=2, c=1, d=1, z=0), 1, "atmost"),
                 (1, 1, 1, dict(a=1, b=1, c=2, d=1, z=0), 1, "atmost"),
                 (1, 1, 1, dict(a=1, b=1, c=1, d=True, z=0), 1, "exact"),
                 (1, 1, 1, dict(a=1, b=1, c=1, d=1), 1, "exact"),
                 (1, 1, 1, legal, 5, "exact"),
                 (1, 1, 1, legal, 5, "atmost"),
                 (1, 1, 1, legal, 1.0, "exact"),
                 (1, 1, 1, legal, 1, "wrong")]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                HELPER.oracle_counts(*args)

    def test_exact_fraction_retains_information_for_large_integer_counts(self):
        large = 10**400
        result = HELPER.oracle_counts(large, 1, 1, dict(a=1, b=1, c=0, d=0, z=0), 1, "atmost")
        self.assertEqual(result["iou_fraction"], dict(numerator=large + 1, denominator=large + 2))

    def test_independent_small_count_enumeration(self):
        # Enumerate ALL feasible selected-count vectors; never use greedy order.
        instances = candidate_vectors = comparisons = 0
        keys = ("a", "b", "c", "d", "z")
        for tp, fp, fn in product(range(3), repeat=3):
            for a, b, c, d, z in product(range(fp + 1), range(fn + 1), range(tp + 1), range(3), range(3)):
                classes = dict(zip(keys, (a, b, c, d, z)))
                pool = a + b + c + d + z
                exact_best = [None] * (pool + 1)
                for aa, bb, cc, dd, zz in product(*(range(value + 1) for value in (a, b, c, d, z))):
                    cost = aa + bb + cc + dd + zz
                    after = (tp + bb - cc, fp - aa + dd, fn - bb + cc)
                    union = sum(after)
                    value = Fraction(after[0], union) if union else Fraction(1)
                    if exact_best[cost] is None or value > exact_best[cost]:
                        exact_best[cost] = value
                    candidate_vectors += 1
                atmost_best = Fraction(0)
                for K, exact_value in enumerate(exact_best):
                    self.assertIsNotNone(exact_value)
                    atmost_best = max(atmost_best, exact_value)
                    for mode, expected in (("exact", exact_value), ("atmost", atmost_best)):
                        result = HELPER.oracle_counts(tp, fp, fn, classes, K, mode)
                        fraction = result["iou_fraction"]
                        actual = Fraction(fraction["numerator"], fraction["denominator"])
                        self.assertEqual(actual, expected, (tp, fp, fn, classes, K, mode))
                        selected = result["selected_counts"]
                        self.assertTrue(all(0 <= selected[key] <= classes[key] for key in keys))
                        self.assertEqual(sum(selected.values()), result["selected_support"])
                        self.assertEqual(sum(selected[key] for key in keys if key != "z"), result["actual_flips"])
                        self.assertEqual(result["selected_support"], K if mode == "exact" else min(K, a + b))
                        comparisons += 1
                instances += 1
        print(json.dumps(dict(SYNTHETIC_ONLY=True, small_count_instances=instances,
                              feasible_selected_vectors=candidate_vectors, oracle_comparisons=comparisons)))

if __name__ == "__main__":
    unittest.main()
