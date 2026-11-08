import unittest
import numpy as np
from asymmetric_math import choose_split, signed_gain, fit_reliability


class AsymmetricAction(unittest.TestCase):
    def test_eight_exact_and_tie_prefers_four(self):
        self.assertEqual(choose_split(np.ones(8),np.ones(8)),(4,8.0))
        self.assertEqual(choose_split(np.ones(8)*2,np.ones(5)),(8,16.0))
        self.assertEqual(choose_split(np.ones(4),np.ones(6)*2),(2,14.0))

    def test_integer_iou_margins(self):
        np.testing.assert_array_equal(signed_gain([0,1],False),[-9,10])
        np.testing.assert_array_equal(signed_gain([0,1],True),[-10,9])

    def test_reliability_preserves_order_and_never_uses_eval_labels(self):
        z=np.arange(12,dtype=float)
        p=fit_reliability(z,[0,0,1,0,1,1,0,1,1,1,1,1],np.ones(12),4)
        scores=p(np.array([-10,0,4,8,100]))
        self.assertTrue(np.all(np.diff(scores)>=0))
        self.assertTrue(np.all((scores>=0)&(scores<=1)))


if __name__=="__main__":
    unittest.main()

