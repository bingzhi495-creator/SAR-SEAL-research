from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from signed_ledger import audit

class SignedLedgerTest(unittest.TestCase):
    def test_all_four_membership_parts_and_author_iou(self):
        p=np.array([[.8,.3],[.2,.8],[.6,.4],[.1,.7]])
        y=np.array([[0,1],[1,1],[1,0],[0,0]])
        f=((0,),(0,))
        v=((1,2),(1,))
        with self.assertRaises(ValueError):
            audit(p,y,f,v,8,9,6)
        v=((1,2),())
        data=audit(p,y,f,v,8,9,6)
        self.assertEqual(data['outside_plus_count'],2)
        self.assertEqual(data['outside_minus_count'],1)
        self.assertEqual(data['inside_plus_count'],1)
        self.assertEqual(data['inside_minus_count'],0)
        self.assertEqual(data['true_intersection_gain'],3)
        self.assertEqual(data['area_delta'],2)
        self.assertAlmostEqual(data['true_iou_delta'],8/11-5/12)
        self.assertAlmostEqual(data['score_error'],data['overlay_error']+data['slope_error'])

    def test_equal_quota_changed_sites_have_zero_area_slope_contribution(self):
        p=np.full((3,2),.5)
        y=np.array([[1,1],[0,0],[1,0]])
        data=audit(p,y,((0,),(0,)),((1,),(1,)),7,9,5)
        self.assertEqual(data['area_delta'],0)
        self.assertEqual(data['slope_error'],0)
        self.assertEqual(data['inside_minus_count'],1)
        self.assertAlmostEqual(data['pred_margin'],0)

if __name__=='__main__':
    unittest.main()
