from fractions import Fraction
from itertools import combinations
import unittest

import numpy as np

from geometry import analyze_pair


def iou(counts):
    return counts["tp"] / max(1, counts["tp"] + counts["fp"] + counts["fn"])


class GeometryTests(unittest.TestCase):
    def test_equal_budget_and_local_dominance(self):
        pred = np.zeros((28, 30), dtype=bool)
        pred[6:15, 7:25] = True
        truth = pred.copy()
        truth[6, 7:12] = False
        truth[15, 20:25] = True
        result = analyze_pair(pred, truth)
        self.assertEqual(result["budget"], 16)
        self.assertEqual(result["local_repair_available"], 10)
        self.assertEqual(result["methods"]["local_band"], dict(tp=162, fp=0, fn=0, edits=10))
        self.assertGreaterEqual(iou(result["methods"]["local_band"]), iou(result["methods"]["joint"]))
        for name, counts in result["methods"].items():
            if name != "joint_free":
                self.assertLessEqual(counts["edits"], result["budget"])


    def test_true_border_translation_does_not_crop_foreground(self):
        pred = np.zeros((18, 32), dtype=bool)
        pred[7:11, :20] = True
        truth = pred.copy()
        truth[:, 19] = False
        result = analyze_pair(pred, truth)
        self.assertEqual(result["budget"], 8)
        self.assertEqual(result["methods"]["translation"], result["methods"]["baseline"])
        self.assertEqual(result["methods"]["local_band"], dict(tp=76, fp=0, fn=0, edits=4))

    def test_interior_crop_equals_same_pair_on_larger_canvas(self):
        pred = np.zeros((18, 19), dtype=bool)
        pred[5:11, 6:12] = True
        truth = np.roll(pred, (0, 1), axis=(0, 1))
        expected = analyze_pair(pred, truth)
        self.assertEqual(expected["methods"]["joint_free"], dict(tp=36, fp=0, fn=0, edits=12))
        padding = ((7, 11), (13, 9))
        self.assertEqual(analyze_pair(np.pad(pred, padding), np.pad(truth, padding)), expected)

    def test_local_oracles_equal_brute_force_for_small_edit_pool(self):
        pred = np.zeros((13, 13), dtype=bool)
        pred[2:6, 2:7] = True
        truth = pred.copy()
        for row, col in ((2, 2), (2, 3), (3, 2)):
            truth[row, col] = False
        truth[6, 2] = truth[10, 10] = True
        result = analyze_pair(pred, truth)
        self.assertEqual(result["budget"], 2)
        near = [(2, 2), (2, 3), (3, 2), (6, 2)]
        for name, points in (("local_band", near), ("local_any", near + [(10, 10)])):
            possibilities = []
            for size in range(3):
                for combination in combinations(points, size):
                    edited = pred.copy()
                    for point in combination:
                        edited[point] = truth[point]
                    tp = int(np.count_nonzero(edited & truth))
                    fp = int(np.count_nonzero(edited & ~truth))
                    fn = int(np.count_nonzero(~edited & truth))
                    possibilities.append((Fraction(tp, tp + fp + fn), -size))
            optimum = result["methods"][name]
            tp, fp, fn = optimum["tp"], optimum["fp"], optimum["fn"]
            self.assertEqual((Fraction(tp, tp + fp + fn), -optimum["edits"]), max(possibilities))

    def test_empty_prediction_has_no_local_band(self):
        pred = np.zeros((12, 12), dtype=bool)
        truth = pred.copy()
        truth[3:8, 3:8] = True
        result = analyze_pair(pred, truth)
        self.assertEqual(result["local_repair_available"], 0)
        self.assertEqual(result["bulk_admissible_count"], 0)
        self.assertEqual(result["methods"]["local_band"], dict(tp=0, fp=0, fn=25, edits=0))
        self.assertEqual(result["methods"]["local_any"], dict(tp=2, fp=0, fn=23, edits=2))

if __name__ == "__main__":
    unittest.main()
