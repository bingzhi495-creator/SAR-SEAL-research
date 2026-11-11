"""Behavioral tests for continuous-IoU edit oracle."""

import unittest
from fractions import Fraction
from itertools import combinations

import numpy as np

from area_flex_math import solve_edit_oracles


def brute_force(pred_area, truth_area, intersection, occupancy, inside, max_pairs):
    outcomes = []
    for flips in range(min(2 * max_pairs, len(occupancy)) + 1):
        for edit_set in combinations(range(len(occupancy)), flips):
            adds = tuple(point for point in edit_set if not inside[point])
            deletes = tuple(point for point in edit_set if inside[point])
            area = pred_area + len(adds) - len(deletes)
            overlap = intersection + sum(occupancy[p] for p in adds) - sum(occupancy[p] for p in deletes)
            iou = overlap / (area + truth_area - overlap)
            outcomes.append((iou, adds, deletes))

    def rank(outcome):
        iou, adds, deletes = outcome
        return -iou, len(adds) + len(deletes), len(adds), adds, deletes

    balanced = min((x for x in outcomes if len(x[1]) == len(x[2])), key=rank)
    flexible = min(outcomes, key=rank)
    balanced_flips = len(balanced[1]) + len(balanced[2])
    same_count = min((x for x in outcomes if len(x[1]) + len(x[2]) == balanced_flips), key=rank)
    return balanced, flexible, same_count


class EditOracleTests(unittest.TestCase):
    def test_same_cap_allows_two_additions(self):
        occupancy = np.array([0.75, 0.75, 0.75])
        inside = np.array([False, True, False])

        balanced, flexible, same_count = solve_edit_oracles(1, 3, 0.75, occupancy, inside, 1)

        self.assertEqual(balanced.flips, 0)
        self.assertEqual(flexible.adds, (0, 2))
        self.assertEqual(flexible.deletes, ())
        self.assertEqual(flexible.iou, 0.6)
        self.assertEqual(same_count, balanced)

    def test_zero_cap_and_empty_support_keep_original(self):
        for occupancy, inside, cap in [
            (np.array([0.5, 0.25]), np.array([False, True]), 0),
            (np.array([]), np.array([], dtype=bool), 4),
        ]:
            for output in solve_edit_oracles(2, 3, 1, occupancy, inside, cap):
                self.assertEqual((output.flips, output.iou, output.intersection, output.pred_area), (0, 0.25, 1, 2))

    def test_fractional_ties_use_first_support_index(self):
        occupancy = np.array([0.375, 0.375, 0.125, 0.125])
        inside = np.array([False, False, True, True])

        balanced, flexible, same_count = solve_edit_oracles(2, 1, 0.25, occupancy, inside, 1)

        self.assertEqual((balanced.adds, balanced.deletes), ((0,), (2,)))
        self.assertEqual(flexible, same_count)
        self.assertEqual((same_count.adds, same_count.deletes), ((0, 1), ()))

        balanced, flexible, same_count = solve_edit_oracles(
            1, 2.5, 0.5, np.array([0.25, 0.25]), np.array([False, False]), 1,
        )
        self.assertEqual(flexible.adds, (0, 1))
        self.assertEqual(balanced.flips, same_count.flips)

    def test_equal_iou_prefers_fewer_flips_then_fewer_additions(self):
        _, flexible, _ = solve_edit_oracles(
            1, 1, 0.5, np.array([0.25]), np.array([False]), 1,
        )
        self.assertEqual(flexible.flips, 0)

        occupancy = np.array([0.75, 0.5, 0.125])
        inside = np.array([False, False, True])
        _, flexible, same_count = solve_edit_oracles(2, 2, 0.625, occupancy, inside, 1)
        for choice in (flexible, same_count):
            self.assertEqual(choice.adds, (0,))
            self.assertEqual(choice.deletes, (2,))

    def test_exhaustive_512_locked_support_cases(self):
        rng = np.random.default_rng(1127)
        for case in range(512):
            with self.subTest(case=case):
                n = rng.integers(1, 9)
                weights = rng.integers(0, 5, size=n)
                occupancy = tuple(Fraction(int(w), 4) for w in weights)
                inside = rng.choice([False, True], size=n)
                max_pairs = rng.integers(0, 5)
                fixed_overlap = Fraction(int(rng.integers(0, 5)), 4)
                pred_area = 1 + int(np.count_nonzero(inside))
                truth_area = 1 + sum(occupancy)
                overlap = fixed_overlap + sum(occupancy[i] for i in range(n) if inside[i])

                expected = brute_force(pred_area, truth_area, overlap, occupancy, inside, max_pairs)
                actual = solve_edit_oracles(pred_area, truth_area, overlap, occupancy, inside, max_pairs)
                for result, (iou, adds, deletes) in zip(actual, expected):
                    self.assertEqual(result.iou, float(iou))
                    self.assertEqual(tuple(sorted(result.adds)), adds)
                    self.assertEqual(tuple(sorted(result.deletes)), deletes)


if __name__ == "__main__":
    unittest.main()

