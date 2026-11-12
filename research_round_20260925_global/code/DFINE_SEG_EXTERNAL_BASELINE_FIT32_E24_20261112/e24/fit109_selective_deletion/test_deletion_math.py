import unittest

import numpy as np

from deletion_math import choose_prefix, continuous_after_drop, ordered_deletions, ridge


class DeletionMath(unittest.TestCase):
    def test_predicted_area_recomputed(self):
        values=continuous_after_drop(100.,100.,75.,np.array([0.,.1,1.]))
        self.assertAlmostEqual(values[0,0],60.)
        self.assertGreater(values[1,0],values[0,0])
        self.assertGreater(values[2,0],values[1,0])
        self.assertLess(values[3,0],values[2,0])
        self.assertEqual(values.shape,(4,3))

    def test_order_uses_native_row_major_ties(self):
        coords=np.array([33,21,12])
        values=np.array([.51,.51,.55])
        self.assertEqual(ordered_deletions(coords,values,2),[21,33])


    def test_reject_and_shorter_tie(self):
        self.assertEqual(choose_prefix(np.array([.1,.05]),100),1)
        self.assertEqual(choose_prefix(np.array([-.1,0.]),100),0)

    def test_unpenalized_intercept_and_constant_columns(self):
        x=np.array([[4.,7.],[4.,7.],[4.,7.]])
        y=np.array([.1,.2,.3])
        model=ridge(x,y,np.array([1.,1.,2.]))
        np.testing.assert_allclose(model(np.array([[4.,7.],[4.,7.]])),[.225,.225])

if __name__=="__main__":
    unittest.main()