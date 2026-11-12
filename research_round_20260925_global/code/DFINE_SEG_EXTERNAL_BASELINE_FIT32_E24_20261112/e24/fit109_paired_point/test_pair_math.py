import unittest

import numpy as np

from pair_math import pair_assignment, choose_pairs, contexts


class PairMath(unittest.TestCase):
    def test_exact_distance_assignment(self):
        a=np.array([12,35,58])
        b=np.array([11,37,59])
        x=pair_assignment(a,b,10)
        np.testing.assert_array_equal(x,np.array([[12,11],[35,37],[58,59]]))

    def test_zero_edit_and_same_count(self):
        self.assertEqual(choose_pairs(np.array([-.1,0.,-.01]),2),[])
        self.assertEqual(choose_pairs(np.array([.1,.1,.01]),2),[0,1])
        self.assertEqual(choose_pairs(np.array([-.3,-.1]),1,forced=1),[1])

    def test_true_reversal_and_shuffle_preserve_side_layers(self):
        f=np.arange(4*2*14,dtype=np.float32).reshape(4,2,14)
        layers=np.array([[1,1],[1,2],[2,1],[2,2]])
        self.assertTrue(np.array_equal(contexts(f,layers,"P1",2)["paired"],f[:,::-1]))
        p=contexts(f,layers,"P1",2)["shuffled"]
        np.testing.assert_array_equal(p[:,:,1:9],f[:,::-1,1:9])
        for side in (0,1):
            for layer in (1,2):
                sel=layers[:,1-side]==layer
                self.assertEqual(sorted(p[sel,side,0].tolist()),sorted(f[sel,1-side,0].tolist()))


if __name__=="__main__":
    unittest.main()