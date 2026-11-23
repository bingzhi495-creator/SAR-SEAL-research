"""Differential check against all signed subsets on small endpoint rosters."""
from itertools import combinations
from collections import Counter
from math import fsum
from pathlib import Path
import runpy
import unittest

import numpy as np
from pycocotools import mask as codec


class QuotaTest(unittest.TestCase):
    def test_budgets_ties_and_forced_negative_edits(self):
        writer = runpy.run_path(str(Path(__file__).with_name("write_ablation.py")))
        select = writer["quota_choices"]
        examples = (
            [[.5,.5]]*4, [[.75,.75],[1.,.5],[.75,1.],[1.,.5]],
            [[1.,1.]]*4, [[0.,0.]]*4, [[.25,.75]]*4,
            [[.5,.75],[.875,.25],[0.,1.],[.875,.5]],
        )
        for values in examples:
            q=np.array(values,dtype=float)
            n=len(q)
            for k in range(1,n+1):
                actual=select(q,k)
                for arm,limit in (("balance",0),("relaxed",2),("free",2*n)):
                    trials=[]
                    for a in range(n+1):
                        d=2*k-a
                        if not 0<=d<=n or abs(a-d)>limit:
                            continue
                        for adds in combinations(range(n),a):
                            for drops in combinations(range(n),d):
                                utility=fsum([q[x,0]-.5 for x in adds]+[.5-q[x,1] for x in drops])
                                trials.append((-utility,abs(a-d),a,adds,drops))
                    neg,_,_,adds,drops=min(trials)
                    selected_add,selected_drop,gain=actual[arm]
                    self.assertEqual((tuple(sorted(selected_add)),tuple(sorted(selected_drop)),gain),(adds,drops,-neg))

        def rle(mask):
            entry=codec.encode(np.asfortranarray(mask,dtype=np.uint8))
            return {"counts":entry["counts"].decode("ascii"),"size":entry["size"]}
        stock=np.zeros((512,512),dtype=bool)
        stock.ravel()[:40]=True
        # Coordinates deliberately disagree with ordinal order. This exercises exact tied-score parity.
        coords=[[99,3],[101,0],[78,7],[98,1]]
        candidate=dict(position=1,k=2,segmentation=rle(stock),coords=coords)
        scores=dict(position=1,probabilities=[[1.,1.]]*4)
        balanced=stock.copy()
        balanced.flat[[99,101]],balanced.flat[[3,0]]=True,False
        counts={arm:Counter() for arm in writer["ARMS"]}
        rows=writer["render"](candidate,scores,{"phoenix":{"swaps":2,"rle":rle(balanced)}},counts)
        self.assertEqual(rows["balance"]["rle"],rle(balanced))
        self.assertEqual([(counts[a]["adds"],counts[a]["drops"]) for a in writer["ARMS"]],[(2,2),(3,1),(4,0)])
        for arm in writer["ARMS"]:
            self.assertEqual(np.count_nonzero(codec.decode(rows[arm]["rle"]) != stock),4)


if __name__ == "__main__":
    unittest.main()
