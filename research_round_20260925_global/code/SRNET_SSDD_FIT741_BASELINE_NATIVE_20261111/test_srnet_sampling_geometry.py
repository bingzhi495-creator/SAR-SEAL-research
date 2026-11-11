import unittest

import numpy as np

from srnet_sampling_geometry import analytic_nodes, oracle_metric


class GridTest(unittest.TestCase):
    def test_analytic_displacement_rotates_and_remains_symmetric(self):
        n=28
        box=np.array([[20.,30.,12.,4.,np.pi/2.]])
        train,test,delta=analytic_nodes(box,n)
        np.testing.assert_allclose(train-test,delta,atol=1e-13)
        np.testing.assert_allclose(delta[0,0],[-4*27/(2*28*29),12*27/(2*28*29)],atol=1e-13)
        np.testing.assert_allclose(delta[0,0],-delta[0,-1],atol=1e-13)
        self.assertEqual(train.shape,(1,n*n,2))

    def test_oracle_counts_false_positive_false_negative_and_strict(self):
        truth=np.array([[True,True,False],[False,False,False]])
        estimate=np.array([[True,False,True],[False,False,False]])
        item=oracle_metric(estimate,truth)
        self.assertEqual((item["intersection"],item["union"],item["fp"],item["fn"]),(1,3,1,1))
        self.assertAlmostEqual(item["iou"],1/3)
        self.assertFalse(item["strict_090"])
        self.assertTrue(oracle_metric(truth,truth)["strict_090"])


if __name__ == "__main__":
    unittest.main()
