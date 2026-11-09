import unittest

import numpy as np

from flip_evidence import signed_scores, shuffled


class FlipEvidenceTest(unittest.TestCase):
    def test_signed_delta_and_fixed_probability_mean(self):
        p0=np.array([.1,.8,.5,.2],dtype=np.float32)
        p1=np.array([.9,.1,.5,.3],dtype=np.float32)
        side=np.array([False,True,False,False])
        base,delta,mean=signed_scores(p0,p1,side)
        np.testing.assert_allclose(base,[-.4,-.3,0,-.3],atol=1e-7)
        np.testing.assert_allclose(delta,[.8,.7,0,.1],atol=1e-7)
        np.testing.assert_allclose(mean,(1-2*side)*((p0+p1)/2-.5),atol=1e-7)

    def test_shuffle_is_deterministic_and_preserves_side_and_bins(self):
        p0=np.array([.42,.44,.41,.55,.58,.53,.2,.22,.77,.78],dtype=np.float32)
        side=p0>.5
        d=np.arange(len(p0),dtype=np.float32)
        a=shuffled(d,p0,side,5201)
        np.testing.assert_array_equal(a,shuffled(d,p0,side,5201))
        self.assertTrue((a!=d).any())
        bands=np.minimum((np.abs(p0-.5)*8).astype(int),3)
        for flag in (False,True):
            for b in range(4):
                sel=(side==flag)&(bands==b)
                np.testing.assert_array_equal(np.sort(a[sel]),np.sort(d[sel]))


if __name__=="__main__":
    unittest.main()

