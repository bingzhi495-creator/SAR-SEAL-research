import unittest

import numpy as np

from policy import THRESHOLDS, account_image, calibrate, check_dev_gate, merge_accounts, select_image


class FitPolicyTests(unittest.TestCase):
    def test_forced_application_keeps_unmatched_and_empty_rows(self):
        rows = [
            {"start": 0, "stop": 3, "status": "matched", "mask_empty": False},
            {"start": 3, "stop": 5, "status": "unmatched", "mask_empty": False},
            {"start": 5, "stop": 5, "status": "ignored", "mask_empty": True},
        ]
        positions = np.array([1, 3, 5, 0, 5])
        scores = np.array([0.51, 0.85, 0.90, 0.91, 0.50])
        edits = select_image(rows, positions, scores, threshold=None)
        self.assertEqual([selected.tolist() for selected in edits], [[5, 3, 1], [0, 5], []])
        counts = account_image(
            rows, positions, np.array([0, 3, 3, 3]), np.array([True, False, True]), edits,
        )
        self.assertEqual(counts["helpful_edits"], 2)
        self.assertEqual(counts["harmful_edits"], 1)
        self.assertEqual(counts["mean_net_correct"], 1.0)
        self.assertEqual(counts["cohorts"]["unmatched"]["edits"], 2)
        self.assertEqual(counts["cohorts"]["ignored"]["empty_masks"], 1)


    def test_calibration_ties_prefer_fewer_edits_then_stricter(self):
        rows = [
            {"start": 0, "stop": 3, "status": "matched", "mask_empty": False},
            {"start": 3, "stop": 4, "status": "unmatched", "mask_empty": False},
        ]
        calibration_image = {
            "match_rows": rows,
            "positions": np.array([1, 2, 3, 1]),
            "label_offsets": np.array([0, 3, 3]),
            "y": np.array([True, True, False]),
            "scores": np.array([0.92, 0.59, 0.58, 0.99]),
        }

        result = calibrate([calibration_image])
        self.assertEqual([point["threshold"] for point in result["curve"]], list(THRESHOLDS))
        self.assertEqual(result["threshold"], 0.90)
        by_threshold = {point["threshold"]: point for point in result["curve"]}
        self.assertEqual(by_threshold[0.50]["net_correct"], by_threshold[0.90]["net_correct"])
        self.assertEqual(by_threshold[0.50]["total_edits"], 4)
        self.assertEqual(by_threshold[0.90]["total_edits"], 2)
        self.assertEqual(by_threshold[1.00]["total_edits"], 0)
        with self.assertRaises(ValueError):
            select_image(rows, calibration_image["positions"], calibration_image["scores"], threshold=0.51)

    def test_gate_requires_each_locked_condition(self):
        rows = [{"start": 0, "stop": 17, "status": "matched", "mask_empty": False}]
        positions = np.arange(17)
        label_offsets = np.array([0, 17])
        labels = positions < 16
        rank = account_image(rows, positions, label_offsets, labels, (positions[:16],))
        control = account_image(rows, positions, label_offsets, labels, (positions[1:],))
        forced = {"rank": merge_accounts([rank, rank]), "mean": merge_accounts([control] * 2),
                  "uncertainty": merge_accounts([control] * 2)}
        metrics = {
            "no_edit": {"AP": 0.700, "AP75": 0.810},
            "uncertainty": {"AP": 0.700, "AP75": 0.811},
            "mean": {"AP": 0.699, "AP75": 0.810},
            "rank": {"AP": 0.701, "AP75": 0.810},
        }

        self.assertTrue(check_dev_gate(forced, metrics)["eligible_for_dev"])
        metrics["rank"]["AP75"] = 0.809
        self.assertFalse(check_dev_gate(forced, metrics)["eligible_for_dev"])
        metrics["rank"]["AP75"] = 0.810
        metrics["rank"]["AP"] = 0.7009
        self.assertFalse(check_dev_gate(forced, metrics)["eligible_for_dev"])
        metrics["rank"]["AP"] = 0.701
        forced["mean"] = forced["rank"]
        self.assertFalse(check_dev_gate(forced, metrics)["eligible_for_dev"])


    def test_calibration_curve_equals_application_and_keeps_strict_threshold(self):
        rows = [
            {"start": 0, "stop": 18, "status": "matched", "mask_empty": False},
            {"start": 18, "stop": 20, "status": "ignored", "mask_empty": False},
        ]
        positions = np.r_[np.arange(18), 1, 3]
        scores = np.r_[np.linspace(0.5, 1.0, 18), 0.50, 0.95]
        labels = np.arange(18) % 2 == 0
        offsets = np.array([0, 18, 18])
        image = dict(match_rows=rows, positions=positions, scores=scores, y=labels, label_offsets=offsets)

        calibration = calibrate([image])
        for point in calibration["curve"]:
            edits = select_image(rows, positions, scores, threshold=point["threshold"])
            accounting = account_image(rows, positions, offsets, labels, edits)
            for key in ("helpful_edits", "harmful_edits", "total_edits", "net_correct"):
                self.assertEqual(point[key], accounting[key])
        self.assertEqual(select_image(rows, positions, scores, threshold=0.50)[1].tolist(), [3])
        with self.assertRaises(ValueError):
            account_image(rows, positions, offsets, labels, (np.array([1]), np.array([2])))


if __name__ == "__main__":
    unittest.main()


