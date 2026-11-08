from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F

from risk_criterion import pixel_center_bce


def point_sample(logits, coords, **kwargs):
    return F.grid_sample(logits, (2*coords-1)[:, :, None], **kwargs)[:, :, :, 0]


class StockCriterion:
    def __init__(self, api):
        self.api=api

    def loss_masks(self, outputs, targets, matches, normalizer):
        logits=outputs["logits"]
        if not len(matches[0][1]):
            return {"loss_mask_ce":logits.sum()*0,"loss_mask_dice":logits.sum()*0}
        self.api.get_uncertain_point_coords_with_randomness(logits,None,4,3,.75)
        self.dice=logits.sum()*3
        return {"loss_mask_ce":logits.sum()*2,"loss_mask_dice":self.dice}


class RiskCriterionTests(unittest.TestCase):
    def test_uniform_loss_changes_only_bce_and_keeps_gradient(self):
        sampler=lambda logits,*args: torch.zeros((len(logits),4,2))
        api=SimpleNamespace(get_uncertain_point_coords_with_randomness=sampler)
        criterion=StockCriterion(api)
        generator=torch.Generator().manual_seed(101)
        target=torch.ones((1,1,1),dtype=torch.bool)
        z=torch.tensor([[[[1.25]]]],requires_grad=True)
        matches=[(torch.tensor([0]),torch.tensor([0]))]
        with pixel_center_bce(criterion,api,point_sample,"B",generator):
            observed=criterion.loss_masks({"logits":z},[{"masks":target}],matches,torch.tensor(2.0))
            self.assertIs(observed["loss_mask_dice"],criterion.dice)
            torch.testing.assert_close(observed["loss_mask_ce"],F.softplus(-z).sum()/2)
            observed["loss_mask_ce"].backward()
            torch.testing.assert_close(z.grad,-torch.sigmoid(-z.detach())/2)
        self.assertIs(api.get_uncertain_point_coords_with_randomness,sampler)
        self.assertNotIn("loss_masks",vars(criterion))

    def test_empty_pairs_keep_stock_zero_and_rng(self):
        sampler=lambda *args: None
        api=SimpleNamespace(get_uncertain_point_coords_with_randomness=sampler)
        criterion=StockCriterion(api)
        generator=torch.Generator().manual_seed(9)
        state=generator.get_state()
        idx=torch.tensor([],dtype=torch.int64)
        with pixel_center_bce(criterion,api,point_sample,"D",generator):
            losses=criterion.loss_masks({"logits":torch.ones((1,1,1,1),requires_grad=True)},
                [{"masks":torch.zeros((0,2,2),dtype=torch.bool)}],[(idx,idx)],torch.tensor(1.))
            self.assertTrue(all(x.requires_grad and x.item()==0 for x in losses.values()))
            self.assertTrue(torch.equal(generator.get_state(),state))


if __name__=="__main__":
    unittest.main()
