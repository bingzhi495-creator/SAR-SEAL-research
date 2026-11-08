import unittest
import numpy as np
from selective_math import edited_mask, risk_threshold, select_equal_count


class SelectiveActionTest(unittest.TestCase):
    def test_area_preserving_mask_edit_and_abstention(self):
        mask=np.array([[True,True,False,False]],dtype=bool)
        unchanged=edited_mask(mask,[],[])
        changed=edited_mask(mask,[2],[0])
        np.testing.assert_array_equal(unchanged,mask)
        np.testing.assert_array_equal(changed,[[False,True,True,False]])
        self.assertEqual(int(changed.sum()),int(mask.sum()))

    def test_budget_alternatives_have_same_count(self):
        score=np.array([.1,.4,.4,.2])
        self.assertEqual(select_equal_count(score,[2,3,1,4],2).tolist(),[2,1])

    def test_risk_gate_rejects_a_tied_harmful_action(self):
        score=np.array([.9,.8,.8,.7])
        useful=np.array([True,True,False,True])
        strict_lost=np.array([False,False,True,False])
        matched=np.ones(4,dtype=bool)
        margin=np.array([38,19,-19,19])
        result=risk_threshold(score,useful,strict_lost,matched,margin,["a","b","b","c"],1,1)
        self.assertEqual(result["count"],1)
        self.assertEqual(result["threshold"],.9)

    def test_rejected_calibration_abstains(self):
        result=risk_threshold([.9],[False],[True],[True],[-19],["a"],1,1)
        self.assertEqual(result["count"],0)
        self.assertIsNone(result["threshold"])


if __name__=="__main__":
    unittest.main()

