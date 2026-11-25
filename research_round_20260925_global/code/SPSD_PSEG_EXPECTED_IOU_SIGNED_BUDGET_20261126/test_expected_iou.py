from itertools import product
import importlib.util
import os
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from expected_iou import reciprocal, select

spec=importlib.util.spec_from_file_location("writer",os.environ["SAR_SEAL_VARIABLE_WRITER"])
writer=importlib.util.module_from_spec(spec)
spec.loader.exec_module(writer)

def brute_reciprocal(t,c):
    return sum(np.prod(np.where(z,t,1-t))/(c+sum(z)) for z in product((False,True),repeat=len(t)))

class ExpectedIoUTest(unittest.TestCase):
    def test_poisson_binomial_and_degenerate(self):
        for t in (np.array([]),np.array([.1,1.,.8,0.]),np.full(5,.31)):
            self.assertAlmostEqual(reciprocal(t,4),brute_reciprocal(t,4),places=14)

    def test_certified_queue_matches_exhaustive_finite_menu(self):
        random=np.random.default_rng(491)
        for trial in range(16):
            p=random.uniform(.03,.97,size=(4,2,2))
            base=((0,), (0,))
            choices,audit=select(writer,p,base,1,8)
            oa,od=writer.ranking(p[:,0],True),writer.ranking(p[:,1],False)
            def utility(a,d):
                outside=p[(tuple(i for i in range(4) if i not in a)),0,:]
                outside=np.concatenate((outside,p[np.asarray(d,dtype=int),1,:]),axis=0)
                i=4+p.sum((0,1))-outside.sum(axis=0)
                return i*np.array([brute_reciprocal(col,8+len(a)-len(d)) for col in outside.T])
            static=utility(*base)
            queue=[]
            for a in (1,0,2):
                for a_set in writer.side_sets(oa,a,base[0]):
                    for d_set in writer.side_sets(od,2-a,base[1]):
                        queue.append((float((utility(a_set,d_set)-static).min()),(a_set,d_set)))
            for arm,rows in (("expectation_exact",queue),("expectation_fixed",[r for r in queue if len(r[1][0])==1])):
                best=max(rows,key=lambda r:r[0])
                self.assertAlmostEqual(choices[arm]["score"],best[0],places=13)
                self.assertEqual(choices[arm]["action"],best[1])
            self.assertLessEqual(audit["evaluated"],len(queue))

if __name__=="__main__":
    unittest.main()
