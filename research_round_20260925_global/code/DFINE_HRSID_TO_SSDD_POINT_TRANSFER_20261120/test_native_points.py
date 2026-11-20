"""Native canvas interpolation must preserve exact 800-square features."""
import unittest
from types import SimpleNamespace

import numpy as np

from native_points import make_native


class NativeTest(unittest.TestCase):
    def test_800_square_is_exact_original(self):
        from pathlib import Path
        from native_points import load_old
        old,rank=load_old(Path(PAIR),Path(RANK))
        native,adapted=make_native(old,rank,800,800)
        plane=np.random.default_rng(1).random((160,160),dtype=np.float32)
        coords=np.array([0,1,799,800,10000,639999],dtype=np.int64)
        np.testing.assert_array_equal(adapted.sample_probability(plane,coords,800),
                                      rank.sample_probability(plane,coords,800))
        stock=np.zeros((800,800),dtype=bool)
        stock[31:36,71:80]=True
        box=[60.2,20.1,90.8,45.4]
        features=old.image_features(np.zeros((800,800,3),dtype=np.uint8))
        original=old.paired_points(stock,box,int(stock.sum()),.82,plane,features,rank)
        changed=native.paired_points(stock,box,int(stock.sum()),.82,plane,features,adapted)
        for x,y in zip(original,changed,strict=True):
            np.testing.assert_array_equal(x,y)

    def test_native_endpoints_and_interpolation_bounds(self):
        from pathlib import Path
        from native_points import load_old
        old,rank=load_old(Path(PAIR),Path(RANK))
        native,adapted=make_native(old,rank,193,214)
        plane=np.full((160,160),.25,dtype=np.float32)
        np.testing.assert_array_equal(adapted.sample_probability(plane,np.array([0,214*193-1]),214),
                                      np.array([.25,.25],dtype=np.float16))
        self.assertEqual(native.box_pixels([-1,-1,500,500]),193*214)


if __name__=="__main__":
    import os
    PAIR=os.environ["POINT_PAIR_SOURCE"]
    RANK=os.environ["POINT_RANK_SOURCE"]
    unittest.main()
