import unittest

import numpy as np

from collar_math import collar_region, choose, max_iou


class Collar(unittest.TestCase):
    def test_exact_half_open_integer_support_with_border_clipping(self):
        _,coords=collar_region((.2,1.,2.,3.4),(5,5))
        self.assertIn(0,coords)
        self.assertNotIn(6,coords)
        self.assertIn(7,coords)
        self.assertNotIn(12-1,coords)
        self.assertTrue(all(0<=x<25 for x in coords))

    def test_connectivity_and_equal_count_are_independent_controls(self):
        stock=np.zeros((8,8),dtype=bool)
        stock[3,3]=True
        box=(3.,3.,5.,5.)
        region,coords=collar_region(box,stock.shape)
        values=np.zeros(len(coords))
        for y,x,v in ((2,3,.60),(1,3,.50),(6,6,.95)):
            i=coords.tolist().index(y*8+x)
            values[i]=v
        r=choose(stock,region,coords,values)
        self.assertEqual(r["ordinary"],[11,19,54])
        self.assertEqual(r["connected"],[11,19])
        self.assertEqual(r["equal_count"],[54,19])

    def test_exact_truth_oracle_keeps_raw_and_avoids_pure_false_adds(self):
        self.assertEqual(max_iou(10.,12.,10.,np.array([0.,1.,1.])),1.)
        self.assertEqual(max_iou(10.,12.,10.,np.array([0.,0.])),10/12)


if __name__=="__main__":
    unittest.main()