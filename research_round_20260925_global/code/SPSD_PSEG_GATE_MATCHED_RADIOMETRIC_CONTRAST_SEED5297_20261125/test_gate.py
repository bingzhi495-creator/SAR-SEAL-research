"""Synthetic tests for gain-component evaluation and separate paired criteria."""
from pathlib import Path
from runpy import run_path
import unittest

import numpy as np


ROOT=Path(__file__).parent
M=run_path(str(ROOT/"common_menu.py"))
S=run_path(str(ROOT/"score_gate.py"))


class CalibrationMenuTest(unittest.TestCase):
    def test_minimum_equals_midpoint_minus_half_gap(self):
        gains=np.array([[3.,1.],[-2.,4.],[-2.,-7.],[0.,0.]])
        m,h=M["components"](gains)
        np.testing.assert_array_equal(m,np.array([2.,1.,-4.5,0.]))
        np.testing.assert_array_equal(h,np.array([1.,3.,2.5,0.]))
        np.testing.assert_array_equal(m-h,gains.min(axis=-1))

    def test_truth_swaps_retain_own_menu_without_replacing_deployed_choice(self):
        truth=np.array([[[1,0],[0,1]],[[0,1],[1,0]]])
        actions=np.array([[[0],[0]],[[1],[0]],[[0],[1]],[[1],[1]]])
        pred=np.repeat(np.array([[[0.,0.]],[[3.,1.]],[[4.,.5]],[[5.,4.]]]),3,axis=1)
        own=[[0,1,2],[0,2,3],[0,3]]
        result=M["measure"](truth,actions,pred,[1,3,3],own,k=1)
        np.testing.assert_array_equal(result["selected_minimum_raw"],[-1,-2,-2])
        np.testing.assert_array_equal(result["swaps"]["truth_both"]["indices"],[0,0,0])
        np.testing.assert_array_equal(result["swaps"]["predicted"]["indices"],[1,3,3])
        self.assertEqual(result["direct_actions"],3)
        self.assertGreater(result["direct_midpoint_mae"][0],0)
        self.assertTrue(all(np.isfinite(result["full_gap_mae"])))

    def test_advancement_requires_specificity_separately_from_main(self):
        row={"iou_points":.16,"percentile95_points":[.01,.3],"ap_points":-.09}
        names=("regression_robust","per_side_robust","joint_mean","joint_cell_pessimistic","static")
        suites={ref:{"joint_robust_minus":{name:row.copy() for name in names}} for ref in ("pseg","spsd")}
        exposure=dict(changed_common_hulls=20,changed_common_groups=8,changed_common_hulls_vs_per_side=20,
                      changed_common_groups_vs_per_side=8)
        self.assertTrue(S["gate"](suites,exposure)["advances"])
        suites["spsd"]["joint_robust_minus"]["per_side_robust"]["percentile95_points"]=[-.01,.3]
        result=S["gate"](suites,exposure)
        self.assertTrue(result["main_pass"])
        self.assertFalse(result["specificity_pass"])
        self.assertFalse(result["advances"])


if __name__=="__main__":
    unittest.main()
