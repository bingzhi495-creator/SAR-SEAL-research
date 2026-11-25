"""Small regression fixtures for same-budget native ROI addition."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

s=importlib.util.spec_from_file_location('p',Path(__file__).with_name('native_roi.py'))
p=importlib.util.module_from_spec(s)
s.loader.exec_module(p)


class Add(unittest.TestCase):
    def test_exact_budget_and_ties(self):
        candidates=np.array([17,11,19, 7])
        np.testing.assert_array_equal(p.select(candidates,np.ones(4),2),[7,11])
        self.assertEqual(p.budget(50,np.array([1., -.2, 0.])),2)
        self.assertEqual(p.budget(900,np.ones(25)),8)

    def test_sham_preserves_each_shell_raw_block(self):
        cells=np.arange(12)
        shell=np.repeat([1,2],6)
        raw=np.arange(12,dtype=float)
        original=np.arange(12,dtype=float)*2
        moved=p.shuffle(original,cells,shell,raw,(103,17,90))
        for lo,hi in [(0,4),(4,6),(6,10),(10,12)]:
            self.assertEqual(sorted(moved[lo:hi]),sorted(original[lo:hi]))
        np.testing.assert_array_equal(moved,p.shuffle(original,cells,shell,raw,(103,17,90)))


if __name__=='__main__':
    unittest.main()
