"""Validate bounds against exhaustive fixed-count actions on small supports."""
from itertools import combinations, product
from pathlib import Path
from runpy import run_path
import unittest

import numpy as np

api=run_path(str(Path(__file__).with_name("bound_retry.py")))
gain,upper,refine=(api[item] for item in ("gain","support_upper","refine"))


class TestUpper(unittest.TestCase):
    def test_continuous_support_and_integer_exact(self):
        rng=np.random.default_rng(5297)
        for count in range(2,6):
            for _ in range(7):
                k=int(rng.integers(1,count+1))
                q=rng.random((count,2,2))
                inc=(tuple(range(k)),tuple(range(k)))
                base=gain(q,inc)
                feasible=list(combinations(range(count),k))
                exact=max(float(np.min(gain(q,pair)-base)) for pair in product(feasible,repeat=2))
                top=upper(q,k,inc)
                self.assertGreaterEqual(top+1e-9,exact)
                a,b,status=refine(q,k,inc,0.0,top)
                self.assertEqual(status,"0")
                self.assertAlmostEqual(a,exact,places=7)
                self.assertAlmostEqual(b,exact,places=7)


    def test_no_quota_group_has_zero_group_weighted_gain(self):
        a=dict(group="A",image=1,k=2,finite=1.0,lower=1.0,upper=1.2)
        result=api["summarize"]([a],[("A",1),("B",2)])
        self.assertAlmostEqual(result["group_means"][0],.5)
        self.assertAlmostEqual(result["group_means"][2],.1)
        self.assertAlmostEqual(result["candidate_means"][0],1.0)


if __name__=="__main__":
    unittest.main()
