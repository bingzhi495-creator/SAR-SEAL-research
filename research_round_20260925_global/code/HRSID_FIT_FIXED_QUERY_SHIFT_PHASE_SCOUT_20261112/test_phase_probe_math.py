"""Fixtures for sign, fitted scaling and exactly paired pixel actions."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

path=Path(__file__).with_name("phase_probe_math.py")
spec=importlib.util.spec_from_file_location("_phase_probe_math",path)
probe=importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class Tests(unittest.TestCase):
    def test_registered_pair_direction(self):
        logit=np.array([[4.,3.,5.,6.,2.],[0.,0.,2.,-1.,1.]])
        normal=np.array([[1.,0.],[0.,1.]])
        data=probe.shift_features(logit,np.array([-1.,1.]),normal)
        np.testing.assert_array_equal(data["normal"][:,4], [1.,1.])
        np.testing.assert_array_equal(data["tangent"][:,4], [-2.,-1.])
        np.testing.assert_array_equal(data["generic"][:,4:], [[1.,2.],[1.,1.]])
        for mode in ("generic","normal","tangent"):
            self.assertEqual(data[mode].shape,(2,6))

    def test_train_standardization_is_external_to_predict(self):
        x=np.arange(24,dtype=float).reshape(4,6)
        fit=probe.fit_weighted(x,np.array([0.,.25,.75,1.]),np.ones(4))
        actual=probe.predict(x,fit)
        self.assertTrue(np.all(np.diff(actual)>0))
        self.assertTrue(np.all((0<=actual)&(actual<=1)))

    def test_equal_count_and_stable_ties(self):
        inside=[False,True,False,True]
        add,remove=probe.edits([.5,.3,.5,.3],inside,1)
        np.testing.assert_array_equal(add,[0])
        np.testing.assert_array_equal(remove,[1])
        with self.assertRaises(ValueError):
            probe.edits([.5,.3,.5,.3],inside,3)


if __name__=="__main__":
    unittest.main()

