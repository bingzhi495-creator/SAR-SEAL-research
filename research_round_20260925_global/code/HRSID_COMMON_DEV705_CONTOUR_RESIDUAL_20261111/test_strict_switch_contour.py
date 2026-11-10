import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("p313", Path(__file__).with_name("strict_switch_contour.py"))
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class ContourSupportTest(unittest.TestCase):
    def test_erosion_band_preserves_interior_and_off_canvas_fp(self):
        pred = np.zeros((9, 9), dtype=bool)
        pred[2:7, 2:7] = True
        gt = pred.copy()
        gt[4, 4] = gt[3, 2] = False
        inside = np.ones_like(pred)
        inside[:, :3] = False
        gt &= inside
        row = worker.measure_bands(pred, gt, inside)
        self.assertEqual(row["off_canvas_fp"], 5)
        self.assertEqual(row["fp"], 6)
        self.assertEqual(row["bands"][1]["editable_fp"], 0)
        self.assertEqual(row["bands"][2]["editable_fp"], 0)
        self.assertEqual(row["bands"][4]["editable_fp"], 1)
        self.assertGreater(row["bands"][1]["fp_near_gt"], 0)
        self.assertIsNone(row["bands"][4]["min_edits"])


    def test_unchanged_edit_budget_restricts_full_canvas_and_band(self):
        pred = np.zeros((7, 18), dtype=bool)
        gt = pred.copy()
        gt[3, 1:16] = True
        pred[3, 1:13] = True
        record = worker.measure_prediction(pred, gt)
        self.assertEqual(record["shift"], (0, 0))
        self.assertEqual(record["edit_budget"], 1)
        self.assertEqual(record["contour"]["raw"]["bands"][2]["min_edits"], 2)
        summary = worker.contour_summary([record])
        for scope in ("raw", "best_translation_remaining"):
            self.assertEqual(summary[scope]["targets"], 1)
            self.assertEqual(summary[scope]["unrestricted_in_canvas_reachable"], 1)
            self.assertEqual(summary[scope]["same_K_unrestricted_in_canvas_recoverable"], 0)
            self.assertEqual(summary[scope]["by_radius"]["2"]["recoverable"], 1)
            self.assertEqual(summary[scope]["by_radius"]["2"]["same_K_recoverable"], 0)

    def test_tied_best_translation_uses_minimum_movement_then_signed_axes(self):
        pred = np.zeros((7, 7), dtype=bool)
        gt = pred.copy()
        pred[2, 3] = pred[4, 3] = True
        gt[3, 3] = True
        record = worker.measure_prediction(pred, gt)
        self.assertEqual(record["shift"], (-1, 0))
        self.assertEqual(record["overlaps"], {"raw": 0, "vertical": 1, "horizontal": 0, "joint": 1})
        self.assertEqual(record["translation_xor_pixels"], 4)


if __name__ == "__main__":
    unittest.main()
