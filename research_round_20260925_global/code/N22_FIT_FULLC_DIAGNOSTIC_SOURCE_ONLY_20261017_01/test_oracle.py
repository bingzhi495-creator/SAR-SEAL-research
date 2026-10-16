"""UNIT_FIXTURE-only counts. No real data, model, gate or AP computation."""
import importlib.util
from pathlib import Path
import sys
import unittest
from fractions import Fraction

ORACLE = Path(__file__).with_name('oracle.py')


def load_oracle():
    module_spec = importlib.util.spec_from_file_location('N22_UNIT_FIXTURE_oracle', ORACLE)
    module = importlib.util.module_from_spec(module_spec)
    before_imports = set(sys.modules)
    module_spec.loader.exec_module(module)
    assert not ({'torch', 'numpy'} & (set(sys.modules)-before_imports))
    return module


class ActionOracleContract(unittest.TestCase):
    def test_beneficial_oracle_global_fixed_budget_counterexample_to_FP_first(self):
        result = load_oracle().beneficial_oracle(
            2, 9, 2, 3, 4, native_fn=3, native_fp=4)
        self.assertEqual(result['iou'], Fraction(4, 7))
        self.assertEqual((result['intersection'], result['union']), (4, 7))
        self.assertEqual((result['add_FN'], result['remove_FP'], result['flips']), (2, 2, 4))
        self.assertGreater(result['iou'], Fraction(3, 6))

    def test_same_pair_router_fixed_PR_and_ideal_candidate_comparison(self):
        oracle = load_oracle()
        fixed = oracle.beneficial_oracle(4, 15, 3, 2, 4, native_fn=6, native_fp=5)
        ideal = oracle.ideal_candidate_oracle(4, 15, 4, 3, 4, native_fn=6, native_fp=5)
        routed = oracle.router_outcome(
            4, 15, 4, native_fn=6, native_fp=5,
            candidate_fn=4, candidate_fp=3, candidate_tp=2, candidate_tn=7,
            b_FN=1, b_FP=1, h_TP=1, h_TN=1, selected_count=4)
        self.assertEqual(fixed['iou'], Fraction(1, 2))
        self.assertEqual(ideal['iou'], Fraction(8, 15))
        self.assertEqual(routed['iou'], Fraction(4, 15))
        self.assertEqual((routed['intersection'], routed['union']), (4, 15))
        self.assertEqual(oracle.threshold_crossings(routed['iou'], fixed['iou'])['up'],
                         (Fraction(1, 2),))
        self.assertEqual(oracle.threshold_crossings(fixed['iou'], ideal['iou'])['up'], ())

    def test_small_exhaustive_all_feasible_budgeted_action_pairs(self):
        oracle = load_oracle()
        cases = 0
        for intersection in range(4):
            for native_fn in range(4):
                if intersection + native_fn == 0:
                    continue
                for native_fp in range(4):
                    union = intersection + native_fn + native_fp
                    for available_fn in range(native_fn + 1):
                        for available_fp in range(native_fp + 1):
                            for budget in range(native_fn + native_fp + 3):
                                result = oracle.beneficial_oracle(
                                    intersection, union, available_fn, available_fp, budget,
                                    native_fn=native_fn, native_fp=native_fp)
                                feasible = [Fraction(intersection+a, union-r)
                                    for a in range(available_fn + 1)
                                    for r in range(available_fp + 1) if a+r <= budget]
                                self.assertEqual(result['iou'], max(feasible))
                                self.assertLessEqual(result['flips'], budget)
                                cases += 1
        print('UNIT_FIXTURE_EXHAUSTIVE_CASES=' + str(cases))

    def test_zero_budget_empty_union_and_available_FP_without_FN(self):
        oracle = load_oracle()
        kept = oracle.beneficial_oracle(2, 7, 2, 2, 0, native_fn=3, native_fp=2)
        self.assertEqual((kept['intersection'], kept['union'], kept['iou']), (2, 7, Fraction(2, 7)))
        self.assertEqual(kept['flips'], 0)
        empty = oracle.beneficial_oracle(0, 0, 0, 0, 5, native_fn=0, native_fp=0)
        self.assertEqual(empty['status'], 'NA_EMPTY_NATIVE_UNION')
        self.assertIsNone(empty['iou'])
        self.assertIsNone(oracle.threshold_crossings(None, Fraction(1))['up'])
        zero = oracle.beneficial_oracle(0, 3, 0, 2, 2, native_fn=1, native_fp=2)
        self.assertEqual((zero['intersection'], zero['union'], zero['iou']), (0, 1, Fraction(0)))

    def test_invalid_native_and_available_action_counts_are_rejected(self):
        oracle = load_oracle()
        invalid = [
            (2, 1, 0, 0, 1, 0, 0),
            (1, 4, 0, 0, 1, 1, 1),
            (0, 3, 0, 1, 1, 0, 3),
            (1, 3, 2, 0, 1, 1, 1),
            (1, 3, 0, 2, 1, 1, 1),
            (1, 3, 0, 0, -1, 1, 1),
            (1, 3, 0, 0, True, 1, 1),
            (1, 3, 0.0, 0, 1, 1, 1),
        ]
        for i, u, a, r, k, fn, fp in invalid:
            with self.subTest(counts=(i, u, a, r, k, fn, fp)):
                with self.assertRaises(ValueError):
                    oracle.beneficial_oracle(i, u, a, r, k, native_fn=fn, native_fp=fp)

    def test_router_nonzero_changes_and_selected_noop_slots(self):
        oracle = load_oracle()
        outcome = oracle.router_outcome(4, 15, 6, native_fn=6, native_fp=5,
            candidate_fn=4, candidate_fp=3, candidate_tp=2, candidate_tn=7,
            b_FN=2, b_FP=1, h_TP=1, h_TN=2, selected_count=6)
        self.assertEqual((outcome['intersection'], outcome['union'], outcome['iou']),
                         (5, 16, Fraction(5, 16)))
        noop = oracle.router_outcome(4, 15, 4, native_fn=6, native_fp=5,
            candidate_fn=4, candidate_fp=3, candidate_tp=2, candidate_tn=7,
            b_FN=1, b_FP=0, h_TP=0, h_TN=0, selected_count=3)
        self.assertEqual((noop['flips'], noop['unchanged_selected'], noop['unused_budget']), (1, 2, 1))

    def test_router_budget_categories_and_selection_counts_are_rejected(self):
        oracle = load_oracle()
        base = dict(native_fn=6, native_fp=5, candidate_fn=4, candidate_fp=3,
            candidate_tp=2, candidate_tn=7, b_FN=1, b_FP=1, h_TP=1, h_TN=1,
            selected_count=4)
        for change in [dict(selected_count=3), dict(selected_count=5),
                       dict(candidate_fn=7), dict(candidate_fp=6), dict(candidate_tp=5),
                       dict(b_FN=5), dict(b_FP=4), dict(h_TP=3), dict(h_TN=8),
                       dict(h_TN=-1), dict(selected_count=True)]:
            values = dict(base, **change)
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    oracle.router_outcome(4, 15, 4, **values)

    def test_ten_exact_thresholds_and_equality_boundaries(self):
        oracle = load_oracle()
        result = oracle.threshold_crossings(Fraction(1, 2), Fraction(19, 20))
        self.assertEqual(result['thresholds'], tuple(Fraction(n, 20) for n in range(10, 20)))
        self.assertEqual(result['up'], tuple(Fraction(n, 20) for n in range(11, 20)))
        reverse = oracle.threshold_crossings(Fraction(19, 20), Fraction(1, 2))
        self.assertEqual(reverse['down'], result['up'])
        with self.assertRaises(ValueError):
            oracle.threshold_crossings(0.5, Fraction(1))


if __name__ == '__main__':
    print('UNIT_FIXTURE_ONLY: pure integer/Fraction actions, not AP or a scientific run.')
    unittest.main(verbosity=2)
