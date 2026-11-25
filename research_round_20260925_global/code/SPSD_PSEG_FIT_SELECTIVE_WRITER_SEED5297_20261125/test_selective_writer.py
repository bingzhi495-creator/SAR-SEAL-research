from pathlib import Path
from runpy import run_path
import unittest

import numpy as np


SOURCE = Path(__file__).with_name("selective_writer.py")


class SelectiveWriterTests(unittest.TestCase):
    def test_signed_action_features_and_static_zero(self):
        scope = run_path(str(SOURCE))
        raw = np.zeros((3,2,14),dtype=np.float32)
        raw[...,6] = 5
        raw[...,8] = .5
        raw[...,0] = [[.2,.7],[.8,.3],[.4,.5]]
        posterior = np.array([[[.2,.1],[.7,.9]], [[.8,.5],[.3,.2]], [[.4,.6],[.5,.4]]])
        static = ((0,),(0,))
        proposal = ((1,),(1,))
        features, margin = scope["features"](raw,posterior,proposal,static,1)
        np.testing.assert_allclose(features, [.525,.025,.5,5,.5,.125,3/32,1,1],atol=1e-7)
        self.assertAlmostEqual(margin,.5)
        z, zero = scope["features"](raw,posterior,static,static,1)
        self.assertEqual(zero,0.)
        np.testing.assert_array_equal(z[[0,1,2,7,8]],0.)


    def test_group_image_candidate_balancing_and_degenerate_scale(self):
        m = run_path(str(SOURCE))
        g = ["A","A","A","A","B"]
        i = [1,1,1,2,3]
        np.testing.assert_allclose(m["group_weights"](g,i),[1/12,1/12,1/12,1/4,1/2])
        x = np.zeros((5,9),dtype=np.float64)
        y = np.array([1.,1.,1.,-1.,-1.])
        ridge = m["fit"](x,y,g,i)
        self.assertEqual(ridge["alpha"],.02)
        np.testing.assert_array_equal(ridge["scale"],np.ones(9))
        np.testing.assert_allclose(m["score"](ridge,x),-.5)

    def test_binary_reference_gain_is_an_exact_signed_integer(self):
        m = run_path(str(SOURCE))
        target = np.array([[[1,0],[1,1]],[[1,1],[0,1]]],dtype=np.uint8)
        static, after = ((0,),(0,)), ((1,),(1,))
        np.testing.assert_array_equal(m["true_numerators"](target,after,static,1),[1,1])
        np.testing.assert_array_equal(m["true_numerators"](target,static,static,1),[0,0])
        with self.assertRaisesRegex(ValueError,"quota"):
            m["true_numerators"](target,((0,0),(0,1)),static,2)

if __name__ == "__main__":
    unittest.main()
