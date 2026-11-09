import itertools
import unittest
import torch
from prefix_losses import precision_ramp, uniform_pairwise


class PrefixLossTest(unittest.TestCase):
    def test_ramp_matches_exhaustive_loss_augmented_selection(self):
        z=torch.tensor([[1.0,-1.0,2.0,3.0,-2.0,.3],[-1.,.5,2.,0.,-3.,0.]],requires_grad=True)
        y=torch.tensor([[True,True,False,False,True,False],[False,True,False,True,False,False]])
        v=torch.tensor([[True]*6,[True,True,True,False,False,False]])
        loss=precision_ramp(z,y,v)
        for i in range(len(z)):
            k=min(4,int(v[i].sum()))
            entries=[]
            for ix in itertools.combinations(torch.nonzero(v[i]).flatten().tolist(),k):
                ix=list(ix)
                entries.append((int(y[i,ix].sum()),float(z[i,ix].sum())))
            most=max(n for n,_ in entries)
            reference=(max(s+most-n for n,s in entries)-max(s for n,s in entries if n==most))/k
            self.assertAlmostEqual(float(loss[i]),reference,places=6)
        loss.sum().backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_pairwise_respects_label_order_and_inert_groups(self):
        v=torch.ones((2,4),dtype=torch.bool)
        y=torch.tensor([[True,True,False,False],[True,True,True,True]])
        good=torch.tensor([[2.,2.,-2.,-2.],[1.,2.,3.,4.]])
        self.assertLess(float(uniform_pairwise(good,y,v)[0]),float(uniform_pairwise(-good,y,v)[0]))
        self.assertEqual(float(uniform_pairwise(good,y,v)[1]),0.)


    def test_full_selection_has_zero_loss_and_gradient(self):
        z=torch.tensor([[2.0605785846710205,1.4295438528060913,-.14620758593082428,.38464006781578064]],
                       requires_grad=True)
        y=torch.tensor([[True,False,False,False]])
        loss=precision_ramp(z,y,torch.ones_like(y))
        loss.sum().backward()
        self.assertEqual(float(loss[0]),0.)
        self.assertTrue(torch.equal(z.grad,torch.zeros_like(z)))


if __name__=="__main__":
    unittest.main()

