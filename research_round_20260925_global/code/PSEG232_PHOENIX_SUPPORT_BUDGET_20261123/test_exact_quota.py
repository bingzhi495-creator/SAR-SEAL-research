"""Run with the published 74-chip reference worker path as the only argument."""
from fractions import Fraction
from itertools import combinations
from pathlib import Path
import runpy
import sys
import unittest

import numpy as np


class ExactQuotaTest(unittest.TestCase):
    def test_exact_dyadic_objective_and_tie_breakers(self):
        native=np.array([[False,False,False],[True,True,True]])
        coords=np.array([[1,4],[0,3],[2,5]])
        tiny=np.nextafter(np.float32(0),np.float32(1))
        examples = ([1,1,0,2.**-54,1,1], [1,1,0,tiny,1,1],
                    [1,1,0,0,1,1], [.5,.5,.5,.5,.5,.5], [.875,.5,.125,.75,.125,.5])
        half=Fraction(1,2)
        for vals in examples:
            p=np.array(vals,dtype=np.float32).reshape(native.shape)
            for k in (1,2,3):
                observed=ENGINE.policies(native,coords,p,p>.5,k)[0]
                if k==1 and vals[3]==2.**-54:
                    self.assertEqual(ENGINE.last_allocation_diagnostic["legacy_rounded_split_differs"],1)
                    self.assertEqual(ENGINE.last_allocation_diagnostic["legacy_rank_violation"],1)
                if k==1 and vals[3]==tiny:
                    self.assertEqual(ENGINE.last_allocation_diagnostic["legacy_true_objective_inferior"],1)
                    self.assertEqual(ENGINE.last_allocation_diagnostic["legacy_rank_violation"],0)
                for arm in ("nc","wc"):
                    ranked=[]
                    for a in range(4):
                        d=2*k-a
                        if not 0<=d<=3:
                            continue
                        for add in combinations(range(3),a):
                            for drop in combinations(range(3),d):
                                u=sum((Fraction(float(p.flat[coords[i,0]]))-half for i in add),Fraction(0))
                                u+=sum((half-Fraction(float(p.flat[coords[i,1]])) for i in drop),Fraction(0))
                                ranked.append((-u,abs(a-d),a,add,drop))
                    _,_,_,add,drop=min(ranked)
                    expected=native.copy()
                    for i in add: expected.flat[coords[i,0]]=True
                    for i in drop: expected.flat[coords[i,1]]=False
                    np.testing.assert_array_equal(observed[arm],expected)


if __name__=="__main__":
    wrapper=runpy.run_path(str(Path(__file__).with_name("infer_factorial.py")))
    ENGINE=wrapper["load_engine"](Path(sys.argv.pop(1)))
    unittest.main()
