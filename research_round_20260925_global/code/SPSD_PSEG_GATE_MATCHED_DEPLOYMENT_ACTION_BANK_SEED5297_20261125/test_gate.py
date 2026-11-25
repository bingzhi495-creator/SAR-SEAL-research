from pathlib import Path
from runpy import run_path
import unittest

import numpy as np


ROOT = Path(__file__).parent
RENDER = run_path(str(ROOT / "render_gate.py"))
SCORE = run_path(str(ROOT / "score_gate.py"))


class GateScoringTest(unittest.TestCase):
    def test_archive_is_not_the_matched_previous_control(self):
        self.assertEqual(RENDER["ANCHORS"]["archive_robust"], "action_robust")
        self.assertNotIn("previous_robust", RENDER["ANCHORS"])
        self.assertNotEqual(RENDER["REFIT_PARAMETERS"], RENDER["ARCHIVED_PARAMETERS"])

    def test_separate_group_paired_delta_paths(self):
        arms = RENDER["ARMS"]
        means = np.arange(len(arms), dtype=float)
        values = np.tile(means, (2,1))
        metrics = {arm: {"AP": means[i]} for i, arm in enumerate(arms)}
        joint, matched, archive = SCORE["contrasts"](arms, values, metrics, np.array([[0,1],[1,1]]))
        self.assertEqual(joint["previous_robust"]["iou_points"], 3.0)
        self.assertEqual(matched["mean"]["ap_points"], 3.0)
        self.assertEqual(archive["robust"]["percentile95_points"], [-6.0, -6.0])

    def test_gate_checks_both_references_and_same_head_controls(self):
        safe = {"iou_points": 0.16, "percentile95_points": [0.05, 0.23], "ap_points": -0.05}
        rivals = ("previous_robust", "deployment_mean", "deployment_cell_pessimistic", "static")
        stats = {r: {"deployment_robust_minus": {key: safe.copy() for key in rivals}} for r in ("pseg","spsd")}
        exposure = {"changed_common_hulls": 20, "changed_common_groups": 8}
        self.assertTrue(SCORE["gate"](stats, exposure)["advances"])
        stats["spsd"]["deployment_robust_minus"]["deployment_mean"]["iou_points"] = -0.001
        self.assertFalse(SCORE["gate"](stats, exposure)["advances"])
        exposure["changed_common_groups"] = 7
        self.assertFalse(SCORE["gate"](stats, exposure)["criteria"]["exposure"])


if __name__ == "__main__":
    unittest.main()
