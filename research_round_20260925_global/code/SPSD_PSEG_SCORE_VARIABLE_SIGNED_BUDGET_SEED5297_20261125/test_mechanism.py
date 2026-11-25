from pathlib import Path
from runpy import run_path
import unittest

import numpy as np

ROOT = Path(__file__).parent
M = run_path(str(ROOT/"mechanism.py"))


class VariableGeometryTest(unittest.TestCase):
    def test_two_terms_are_exact_and_change_sign_in_reverse(self):
        f = M["iou_terms"]
        delta, overlap, area = f(16,7,14,8,10)
        self.assertAlmostEqual(delta,8/16-7/19)
        self.assertAlmostEqual(delta,overlap+area)
        self.assertGreater(overlap,0)
        self.assertGreater(area,0)
        np.testing.assert_allclose(f(14,8,16,7,10),[-delta,-overlap,-area],rtol=0,atol=1e-15)
        self.assertAlmostEqual(f(16,7,14,7,10)[1],0)
        self.assertAlmostEqual(f(16,7,16,8,10)[2],0)

    def test_signed_budget_requires_actual_fixed_support_and_flip_count(self):
        old=np.array([[0,0,1,1],[1,1,0,0]],dtype=bool)
        new=np.array([[1,1,0,1],[1,1,1,0]],dtype=bool)
        points=np.array([[0,2],[1,3],[6,4]],dtype=int)
        self.assertEqual(M["signed_changes"](old,new,points,2),(3,1))
        with self.assertRaisesRegex(ValueError,"support"):
            M["signed_changes"](old,new,np.array([[0,2],[1,3],[7,4]]),2)
        with self.assertRaisesRegex(ValueError,"flip budget"):
            M["signed_changes"](old,new,points,1)


    def test_variable_and_static_can_differ_by_more_overlap_than_two_k(self):
        f=M["validate_pairwise_gain"]
        self.assertIsNone(f(np.array([3.,-1.]),5,3,2))
        with self.assertRaisesRegex(ValueError,"set-difference"):
            f(np.array([3.01,0.]),5,3,2)
        with self.assertRaisesRegex(ValueError,"set-difference"):
            f(np.array([0.,-1.01]),5,3,2)

    def test_different_reference_misses_force_no_advancement(self):
        row=dict(iou_points=.18,percentile95_points=[.01,.3],ap_points=-.08)
        names=("bce_robust","fixed_iou","bce_mean","static")
        refs={key:{"variable_iou_minus":{k:row.copy() for k in names}} for key in ("pseg","spsd")}
        exposure={"bce_robust":(25,9),"fixed_iou":(21,8)}
        self.assertTrue(M["gate"](refs,exposure)["advances"])
        refs["pseg"]["variable_iou_minus"]["fixed_iou"]["percentile95_points"]=[-.01,.3]
        self.assertFalse(M["gate"](refs,exposure)["advances"])


if __name__=="__main__":
    unittest.main()
