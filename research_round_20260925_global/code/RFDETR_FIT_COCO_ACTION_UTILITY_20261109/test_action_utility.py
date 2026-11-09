from itertools import combinations,product
import unittest

import torch

from action_utility import iou_table, regrets


class ActionUtilityTest(unittest.TestCase):
    def test_integer_thresholds_include_equality(self):
        utility=iou_table(torch.tensor([42]),torch.tensor([98]))[0]
        for hits in range(9):
            i=42+hits-4
            n=sum(20*i>=m*(98-i) for m in range(10,20))
            self.assertEqual(int(utility[hits]),n)
        self.assertEqual(int(utility[4]),6)

    def test_both_objectives_match_explicit_action_search(self):
        z=torch.tensor([[[.5,1.1,-.6,.7,-1.,2.],[.7,-.4,1.8,.2,-.3,0.]]],requires_grad=True)
        y=torch.tensor([[[True,True,False,True,False,False],[True,False,False,True,True,False]]])
        valid=torch.ones_like(y)
        valid[0,1,5]=False
        utility=iou_table(torch.tensor([37]),torch.tensor([84]))
        result=regrets(z,y,valid,utility)
        choices=[list(combinations(torch.nonzero(a).flatten().tolist(),4)) for a in valid[0]]
        actions=[]
        for a,d in product(*choices):
            a,d=list(a),list(d)
            h=int(y[0,0,a].sum()+y[0,1,d].sum())
            s=float(z[0,0,a].sum()+z[0,1,d].sum())
            actions.append((h,s))
        lo=min(h for h,_ in actions)
        hi=max(h for h,_ in actions)
        span=int(utility[0,hi]-utility[0,lo])
        cost=[
            lambda h:int(utility[0,hi]-utility[0,h]),
            lambda h:span*(hi-h)/(hi-lo),
        ]
        self.assertGreater(span,0)
        for value,c in zip(result,cost,strict=True):
            want=(max(s+c(h) for h,s in actions)-max(s for h,s in actions if c(h)==0))/8
            self.assertAlmostEqual(float(value[0]),want,places=6)
        sum(x.sum() for x in result).backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_inactive_candidates_have_exact_zero_gradient(self):
        z=torch.randn((2,2,6),requires_grad=True)
        y=torch.zeros_like(z,dtype=torch.bool)
        y[0,:,:3]=True
        valid=torch.ones_like(y)
        table=iou_table(torch.tensor([500,37]),torch.tensor([1200,84]))
        losses=regrets(z,y,valid,table)
        for loss in losses:
            self.assertTrue(torch.equal(loss,torch.zeros_like(loss)))
        sum(x.sum() for x in losses).backward()
        self.assertTrue(torch.equal(z.grad,torch.zeros_like(z)))


if __name__=="__main__":
    unittest.main()

