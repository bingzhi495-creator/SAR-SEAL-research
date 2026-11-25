import importlib.util
import os
import unittest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from quota import panel, features, fit, choose

spec = importlib.util.spec_from_file_location('writer',os.environ['SAR_SEAL_VARIABLE_WRITER'])
writer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(writer)

class QuotaTest(unittest.TestCase):
    def test_frozen_panel_and_fixed_first_tie(self):
        p = np.array([[[.7,.8],[.2,.1]],[[.6,.7],[.5,.4]],[[.2,.1],[.8,.9]]])
        static = ((0,), (0,))
        rows = panel(writer,p,static,1,5)
        old = writer.select(p,static,1,5)
        self.assertEqual(rows[0]['action'],old['fixed_iou']['action'])
        self.assertEqual(max(rows,key=lambda r:r['score'])['action'],old['variable_iou']['action'])
        self.assertTrue(all(sum(map(len,r['action']))==2 for r in rows))
        raw,x = features(rows,1,5,False)
        self.assertEqual(choose(rows,raw,x,dict(scale=[1]*4,coefficients=[-1,0,0,0]))['action'],
                         old['fixed_iou']['action'])

    def test_sign_reversal_leaves_the_raw_offset_untouched(self):
        rows = [dict(adds=a, score=s) for a,s in [(2,0.1),(1,0.2),(3,0.4),(0,0.8),(4,0.9)]]
        original, raw_features = features(rows,2,8,False)
        offset, control = features(rows,2,8,True)
        np.testing.assert_array_equal(original,offset)
        np.testing.assert_allclose(control[:,0], original[[0,3,4,1,2]])
        np.testing.assert_allclose(control[:,1:], raw_features[:,1:])
        model=fit(control,np.zeros(5),np.full(5,.2))
        np.testing.assert_array_equal(model['coefficients'], np.zeros(4))

if __name__ == '__main__':
    unittest.main()
