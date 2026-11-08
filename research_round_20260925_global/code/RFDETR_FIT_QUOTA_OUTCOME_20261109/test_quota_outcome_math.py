import unittest
import numpy as np
from quota_outcome_math import realized, action_features, fit_ridge


class QuotaOutcomeTest(unittest.TestCase):
    def test_realized_iou_and_balanced_anchor(self):
        y=realized(100,8,8,np.ones(8),np.ones(8))
        self.assertEqual(y["k"].tolist(),list(range(9)))
        self.assertEqual(y["fp"][4],4)
        self.assertEqual(y["fn"][4],4)
        self.assertAlmostEqual(y["iou"][4],96/104)
        self.assertEqual(y["multi"][4],0)
        self.assertEqual(y["margin"][4],0)

    def test_anchor_and_finite_ridge(self):
        a=np.arange(8,dtype=float)
        k,x=action_features(a,a,a/8,a/8,64,.8,3)
        np.testing.assert_array_equal(x[k==4],np.zeros((1,x.shape[1])))
        y=np.column_stack((k-4,4-k))
        model=fit_ridge(x,y,np.ones(len(k))/len(k))
        self.assertTrue(np.isfinite(model(x)).all())
        np.testing.assert_array_equal(model(x[k==4]),np.zeros((1,2)))


if __name__=="__main__":
    unittest.main()

