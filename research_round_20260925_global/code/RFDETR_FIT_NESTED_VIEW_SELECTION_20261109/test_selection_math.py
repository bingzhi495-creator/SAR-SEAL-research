import unittest

import numpy as np

from selection_math import fit_ridge, geometric_choice


class SelectionMathTest(unittest.TestCase):
    def test_equal_content_group_weight_with_repeated_rows(self):
        x=np.array([[0,1],[2,0],[1,3]],dtype=float)
        y=np.array([1.,0.,-1.])
        a=fit_ridge(x,y,["a","b","c"])
        b=fit_ridge(x[[0,0,0,1,2]],y[[0,0,0,1,2]],["a","a","a","b","c"])
        np.testing.assert_allclose(a(x),b(x),atol=1e-12)

    def test_both_actions_lose_the_last_strict_case(self):
        native=np.ones((3,10),dtype=bool)
        raw=native.copy()
        flip=native.copy()
        raw[[0,2],8:]=False
        flip[[1,2],8:]=False
        q=geometric_choice(native,raw,flip)
        self.assertEqual(q["minimum_native_ap90_losses"],1)
        self.assertEqual(q["maximum_passed"][8],2)
        self.assertEqual(q["gained_over_raw"][8],1)


if __name__=="__main__":
    unittest.main()

