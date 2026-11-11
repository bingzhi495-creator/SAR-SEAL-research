import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from nested_lattice import evaluate_axis, sample_nested, prolongate


class NestedTests(unittest.TestCase):
    def test_inserted_midpoints_retain_old_knots_and_clamps(self):
        grid=np.array([[2.,6.,-2.]])
        result=prolongate(grid,1)
        np.testing.assert_array_equal(result,[[2.,4.,6.,2.,-2.]])
        old=np.array([[2.,3.,5.,4.,0.,-2.]])
        fine=evaluate_axis(result,6,axis=1,parent=3)
        np.testing.assert_array_equal(fine,old)

    def test_sampling_old_knots_matches_ordinary_centers(self):
        image=np.arange(6,dtype=float)[None]
        refined=sample_nested(image,3,axis=1)
        np.testing.assert_array_equal(refined,[[.5,1.5,2.5,3.5,4.5]])
        np.testing.assert_array_equal(refined[:,::2],[[.5,2.5,4.5]])


if __name__=="__main__":
    unittest.main()
