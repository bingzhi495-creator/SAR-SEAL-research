"""Regression tests for fixed-image counts and aligned cue ties."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

s=importlib.util.spec_from_file_location('p',Path(__file__).with_name('activation.py'))
p=importlib.util.module_from_spec(s)
s.loader.exec_module(p)


class Activation(unittest.TestCase):
    def test_rank_normalization_with_averaged_ties(self):
        np.testing.assert_allclose(p.midranks(np.array([2.,2.,4.,1.])),[.5,.5,1.,0.])
        np.testing.assert_allclose(p.midranks(np.array([1.])),[.5])

    def test_ceil_counts_are_identical_under_distinct_scores(self):
        scores=[dict(pos=i,rank=i+1,query_id=i,k=3,q_rf=float(i),q_a=float(4-i),detector=float(i)) for i in range(5)]
        arms=p.keep_group(291,scores)
        self.assertEqual({len(row) for row in arms.values()},{3})
        self.assertEqual(arms['joint'],{0,1,2})
        self.assertEqual(arms['rf_only'],{2,3,4})
        self.assertEqual(p.keep_group(291,scores),arms)


if __name__=='__main__':
    unittest.main()
