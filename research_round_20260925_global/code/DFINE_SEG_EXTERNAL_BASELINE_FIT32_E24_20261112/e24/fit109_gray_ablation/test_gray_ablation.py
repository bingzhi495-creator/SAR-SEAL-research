import unittest

import numpy as np

from gray_ablation import shuffle_gray


class GrayControl(unittest.TestCase):
    def test_keeps_probability_geometry_and_joint_gray(self):
        x=np.arange(10*2*14,dtype=np.float32).reshape(10,2,14)
        x[:5,:,2]=1
        x[5:,:,2]=2
        y=shuffle_gray(x,"P1",4)
        np.testing.assert_array_equal(x[:,:,:9],y[:,:,:9])
        self.assertTrue(np.any(x[:,:,9:]!=y[:,:,9:]))
        for sign in (0,1):
            for depth in (1,2):
                mask=x[:,sign,2]==depth
                self.assertEqual(sorted(map(tuple,x[mask,sign,9:])),sorted(map(tuple,y[mask,sign,9:])))
        np.testing.assert_array_equal(y,shuffle_gray(x,"P1",4))


if __name__=="__main__":
    unittest.main()