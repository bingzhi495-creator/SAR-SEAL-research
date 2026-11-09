import unittest
import numpy as np
from rank_quota_oracle import perfect_order


class PointOracle(unittest.TestCase):
    def test_correct_first_and_stable_pixel_tie(self):
        pixel=np.array([9,4,6,1,3,8])
        inside=np.array([False,False,False,True,True,True])
        correct=np.array([False,True,True,False,True,False])
        a,d=perfect_order(pixel,inside,correct)
        self.assertEqual(a.tolist(),[1,2,0])
        self.assertEqual(d.tolist(),[4,3,5])


if __name__=="__main__":
    unittest.main()

