import unittest
import numpy as np
from local_math import boundary_samples, signed_prefix


class SignedLocalEvidence(unittest.TestCase):
    def test_inward_normal_on_both_sides(self):
        mask=np.zeros((8, 10), dtype=bool)
        mask[:, 4:]=True
        xy,mid,normal,distance,inside=boundary_samples(mask,32,5201)
        self.assertEqual(len(xy),32)
        np.testing.assert_array_equal(normal,np.tile([1,0],(32,1)))
        self.assertTrue(np.all(mid[:,0]==3.5))
        self.assertEqual(set(distance),{1,2})
        self.assertEqual(int(inside.sum()),16)

    def test_equal_budget_flip_cost_and_stable_tie(self):
        p=np.array([0,1,4,6,7])
        inside=np.array([False,False,False,True,True])
        truth=np.array([True,False,True,True,False])
        scores=np.array([0.4,0.9,0.9,0.3,0.6])
        x=signed_prefix(p,inside,truth,scores,1)
        self.assertEqual(x['selected'].tolist(),[1,4])
        self.assertEqual((x['add_margin'],x['delete_margin']),(-9,9))
        self.assertEqual((x['fp_delta'],x['fn_delta']),(0,0))

    def test_empty_and_constant_masks_have_no_support(self):
        for value in (False,True):
            self.assertEqual(len(boundary_samples(np.full((5,6),value),4,1)[0]),0)


if __name__=="__main__":
    unittest.main()

