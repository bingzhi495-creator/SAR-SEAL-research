"""Synthetic pre-score checks for the frozen menu and group pairwise rule."""
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace
import unittest

import numpy as np


ROOT=Path(__file__).parent
M=run_path(str(ROOT/"common_menu.py"))
S=run_path(str(ROOT/"score_gate.py"))


class MenuTest(unittest.TestCase):
    def test_preferences_use_identical_choices_true_unequal_and_half_credit(self):
        gains=np.array([0,1,-2,0])
        estimates=np.array([[0,0],[1,0],[-2,1],[0,0]])
        n,score=M["preference"](gains,estimates)
        self.assertEqual(n,5)
        np.testing.assert_allclose(score, [1.,.2])

    def test_selected_minimum_checks_both_labels_and_retains_exact_static(self):
        data=np.array([[[1,0],[0,1]],[[0,1],[1,0]]])
        actions=np.array([[[0],[0]],[[1],[0]],[[0],[1]],[[1],[1]]])
        truth=M["differences"](data,actions)
        np.testing.assert_array_equal(truth,[[0,0],[-1,1],[-1,1],[-2,2]])
        estimate=np.stack([truth.min(axis=1),truth.min(axis=1)],axis=1)/2
        outcome=M["measure"](data,actions,estimate,[0,3],k=1)
        np.testing.assert_array_equal(outcome["minimum_raw"],[0,-2])
        np.testing.assert_array_equal(outcome["harmful"],[False,True])
        self.assertEqual(outcome["full_pairs"],5)
        self.assertEqual(outcome["direct_pairs"],1)

    def test_advancement_checks_matched_and_same_head_in_both_references(self):
        row={"iou_points":.16,"percentile95_points":[.01,.3],"ap_points":-.09}
        names=("regression_robust","ranking_mean","ranking_cell_pessimistic","static")
        refs={ref:{"ranking_robust_minus":{name:row.copy() for name in names}} for ref in ("pseg","spsd")}
        exposure={"changed_common_hulls":20,"changed_common_groups":8}
        self.assertTrue(S["gate"](refs,exposure)["advances"])
        refs["spsd"]["ranking_robust_minus"]["ranking_mean"]["iou_points"]=-.01
        self.assertFalse(S["gate"](refs,exposure)["advances"])


if __name__=="__main__":
    unittest.main()
