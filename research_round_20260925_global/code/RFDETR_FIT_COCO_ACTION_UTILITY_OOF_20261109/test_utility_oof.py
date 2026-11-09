import unittest

import numpy as np

from utility_oof import case_thresholds, paired_difference


class UtilityOOFTest(unittest.TestCase):
    def test_iou_threshold_equality_and_eight_edit_change(self):
        np.testing.assert_array_equal(case_thresholds(42,98,4),[1,1,1,1,1,1,0,0,0,0])
        self.assertFalse(case_thresholds(42,98,3)[5])
        self.assertTrue(case_thresholds(42,98,5)[5])

    def test_micro_mean_and_equal_content_group_interval(self):
        old=np.zeros(4,dtype=int)
        new=np.array([3,3,3,-1])
        groups=["a","a","a","b"]
        result=paired_difference(groups,new,old)
        self.assertEqual(result["micro"],2.)
        self.assertEqual(result["group_mean"],1.)
        self.assertEqual(result["groups"],2)
        self.assertEqual(result["group_percentile95"],[-1.,3.])


if __name__=="__main__":
    unittest.main()

