"""Synthetic verification of the exact fixed-pair oracle."""
from fractions import Fraction
import unittest

from signed_edit_oracle import signed_edit_oracle


class SignedEditOracleTests(unittest.TestCase):
    def test_combined_edits_can_reach_both_thresholds_where_one_sign_cannot(self):
        result = signed_edit_oracle((81, 100), (80, 100), 5, 6)
        self.assertEqual(result['minimum_edits'], {
            'fn_only': {'integer': None, 'float': None, 'dual': None},
            'fp_only': {'integer': None, 'float': None, 'dual': None},
            'joint': {'integer': 10, 'float': 11, 'dual': 11},
        })
        curve = {row['budget']: row['joint'] for row in result['curve']}
        self.assertEqual(tuple(curve), (0, 1, 2, 4, 8, 16, 32))
        self.assertEqual(curve[8], {
            'additions': 5, 'removals': 3, 'integer_iou': (86, 97), 'float_iou': (85, 97),
        })
        self.assertEqual(curve[16]['float_iou'], (85, 94))


    def test_exhaustive_small_counts_match_enumerated_edits(self):
        # All count/capacity cases realized by one prediction and two nonempty GTs
        # on a four-pixel grid, including partial capacities within a band.
        masks = [{pixel for pixel in range(4) if bits & (1 << pixel)} for bits in range(16)]
        cases = set()
        for prediction in masks:
            for integer_gt in masks[1:]:
                for float_gt in masks[1:]:
                    integer = (len(prediction & integer_gt), len(prediction | integer_gt))
                    floating = (len(prediction & float_gt), len(prediction | float_gt))
                    fn = len((integer_gt & float_gt) - prediction)
                    fp = len(prediction - (integer_gt | float_gt))
                    cases.update((integer, floating, a, b) for a in range(fn + 1) for b in range(fp + 1))
        threshold = Fraction(9, 10)
        for integer, floating, fn, fp in sorted(cases):
            with self.subTest(integer=integer, floating=floating, fn=fn, fp=fp):
                result = signed_edit_oracle(integer, floating, fn, fp)
                for mode, caps in (('fn_only', (fn, 0)), ('fp_only', (0, fp)), ('joint', (fn, fp))):
                    possibilities = {
                        (a, b): (Fraction(integer[0] + a, integer[1] - b),
                                 Fraction(floating[0] + a, floating[1] - b))
                        for a in range(caps[0] + 1) for b in range(caps[1] + 1)
                    }
                    expected = {
                        name: min((sum(edits) for edits, values in possibilities.items()
                                   if selected(values) >= threshold), default=None)
                        for name, selected in (('integer', lambda values: values[0]),
                                               ('float', lambda values: values[1]), ('dual', min))
                    }
                    self.assertEqual(result['minimum_edits'][mode], expected)
                    for row in result['curve']:
                        budget = row['budget']
                        answer = row[mode]
                        edits = (answer['additions'], answer['removals'])
                        self.assertLessEqual(sum(edits), budget)
                        actual = (Fraction(*answer['integer_iou']), Fraction(*answer['float_iou']))
                        self.assertEqual(actual, possibilities[edits])
                        allowed = [values for cost, values in possibilities.items() if sum(cost) <= budget]
                        self.assertEqual(actual, (max(pair[0] for pair in allowed),
                                                  max(pair[1] for pair in allowed)))

    def test_threshold_comparison_remains_exact_for_large_counts(self):
        base = 10**30
        result = signed_edit_oracle((9 * base, 10 * base), (9 * base, 10 * base + 1), 1, 0)
        self.assertEqual(result['minimum_edits']['joint'], {'integer': 0, 'float': 1, 'dual': 1})

    def test_rejects_invalid_or_unrealizable_common_counts(self):
        for args in (
            ((2, 1), (1, 1), 0, 0),
            ((1, 1), (1, 1), True, 0),
            ((8, 10), (1, 3), 0, 0),
            ((0, 1), (0, 1), 0, 1),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                signed_edit_oracle(*args)


if __name__ == '__main__':
    unittest.main()
