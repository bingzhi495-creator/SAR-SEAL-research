from pathlib import Path
from runpy import run_path
import unittest
import numpy as np

M=run_path(str(Path(__file__).with_name("aggregate.py")))

class AggregateTest(unittest.TestCase):
    def test_images_then_groups_are_equal(self):
        rows={(1,"a"):[np.array([4.]),np.array([8.])],(2,"a"):[np.array([0.])],
              (3,"b"):[np.array([10.])]}
        actual=M["group_table"](rows,{1:"a",2:"a",3:"b",4:"c"},("a","b","c"),1)
        np.testing.assert_allclose(actual.ravel(),[3.,10.,0.])

    def test_subgroup_rate_is_exposure_weighted_and_undefined_at_zero(self):
        val,ci=M["rate"](np.array([[.5,1.,0.],[.6,2.,1.],[0.,0.,0.]]),
                         np.array([[0,0,0],[2,2,2],[0,1,2]]))
        self.assertAlmostEqual(val,(1.1-1)/3)
        self.assertIsNotNone(ci)
        val,ci=M["rate"](np.zeros((3,3)),np.array([[0,1,2]]))
        self.assertIsNone(val)
        self.assertIsNone(ci)

if __name__=="__main__":
    unittest.main()
