"""Synthetic-only behavior checks for the FIT association risk verifier."""

import unittest

import numpy as np
from pycocotools import mask

from assoc_risk import risk_for_image


def rle(binary):
    payload=mask.encode(np.asfortranarray(binary,dtype=np.uint8))
    return dict(size=payload["size"],counts=payload["counts"].decode("ascii"))


def candidates(*planes):
    return [dict(rank=rank,score=1-rank/20,area=int(plane.sum()),rle=rle(plane))
            for rank,plane in enumerate(planes,1)]


class RiskTests(unittest.TestCase):
    def test_distinct_crowded_truth_separate_original_matches_and_unknown_triplets(self):
        a=np.zeros((20,20),np.uint8)
        b=np.zeros_like(a)
        a[2:6,0:4]=1
        b[2:6,5:9]=1
        union=a|b
        truth=[dict(id=1,area=16,segmentation=rle(a),iscrowd=0,ignore=0),
               dict(id=2,area=16,segmentation=rle(b),iscrowd=0,ignore=0)]
        pools={"5201":candidates(a,b,union), "5202":candidates(a,b,union,a), "5203":candidates(a,b,union)}
        structural=dict(methods={"probe":{"pairs":{"1":[1,1],"2":[1,3],"3":[1,1]}}},
                        reps={"90":{"5201":{},"5202":{"2":1,"3":1,"4":1},"5203":{}},
                              "85":{"5201":{},"5202":{},"5203":{}}})
        result=risk_for_image(pools,structural,truth,(20,20))
        self.assertEqual(result["methods"]["probe"],dict(triplets=3,all_known_same=1,known_different=1,unresolved=1,unique_known_same_GT=1))
        self.assertEqual(result["suppression"]["90"]["5202"],
                         dict(suppressed=3,same_known=1,distinct_known=1,unresolved=1,distinct_known_near_2px=1,
                              stock_distinct_matched=1,stock_distinct_near_2px=1,distinct_known_stock_distinct=1))
        self.assertEqual(result["suppression"]["85"]["5202"]["suppressed"],0)


if __name__=="__main__":
    unittest.main()

