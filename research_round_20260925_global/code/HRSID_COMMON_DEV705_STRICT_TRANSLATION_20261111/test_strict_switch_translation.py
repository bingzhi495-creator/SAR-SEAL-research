import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("p312", Path(__file__).with_name("strict_switch_translation.py"))
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class ExactShiftTest(unittest.TestCase):
    def test_out_of_frame_foreground_still_counts_against_iou(self):
        gt = np.zeros((5, 5), dtype=bool)
        gt[2, 0] = True
        pred = gt.copy()
        pred[2, 1] = True
        row = worker.translate_best(pred, gt)
        self.assertEqual(row["pred_area"], 2)
        self.assertEqual(row["overlaps"]["horizontal"], 1)
        self.assertFalse(worker.pass_090(row, "horizontal"))
        self.assertTrue(row["area_impossible"])


    def test_joint_only_recovery_and_exact_threshold(self):
        gt = np.zeros((8, 8), dtype=bool)
        gt[1:3, 1:6] = True
        pred = np.zeros_like(gt)
        pred[2:4, 2:7] = True
        pred[3, 6] = False
        row = worker.translate_best(pred, gt)
        self.assertFalse(row["area_impossible"])
        self.assertFalse(worker.pass_090(row, "raw"))
        self.assertFalse(worker.pass_090(row, "vertical"))
        self.assertFalse(worker.pass_090(row, "horizontal"))
        self.assertTrue(worker.pass_090(row, "joint"))
        self.assertEqual(worker.iou(row, "joint"), .90)
        result = worker.summarize([row])
        self.assertEqual(result["joint_only_recovered"], 1)
        self.assertEqual(result["by_family"]["joint"]["recovered"], 1)
        self.assertEqual(result["by_family"]["joint"]["feasible_not_recovered"], 0)


if __name__ == "__main__":
    unittest.main()
