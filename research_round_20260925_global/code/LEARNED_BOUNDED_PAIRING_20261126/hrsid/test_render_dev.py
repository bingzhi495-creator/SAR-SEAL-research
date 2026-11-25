"""Synthetic parity test for the archived finite SF and independent writer."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from render_dev import old_arms


class OriginalArms(unittest.TestCase):
    def test_realized_fixed_count_and_historical_pair_ordinal_ties(self):
        xy=np.array([[10,11],[12,13],[14,15]],np.int32)
        q=np.array([[.875,.75],[.25,.125],[.125,.125]],np.float32)
        sf,ind,ia,di=old_arms(xy,q,2)
        np.testing.assert_array_equal(sf,[0,1])
        np.testing.assert_array_equal(ind,[[10,13],[12,15]])
        np.testing.assert_array_equal(ia,[0,1])
        np.testing.assert_array_equal(di,[1,2])


if __name__=="__main__":
    unittest.main()
