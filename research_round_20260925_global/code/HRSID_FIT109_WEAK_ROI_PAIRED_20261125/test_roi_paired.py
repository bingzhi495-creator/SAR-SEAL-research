"""Synthetic tests for paired ROI weights, inference gate, and exterior pixels."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch

spec=importlib.util.spec_from_file_location('roi',Path(__file__).with_name('roi_paired.py'))
worker=importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class PairedRoi(unittest.TestCase):
    def test_loss_reweights_exact_weak_only_with_sum_normalizer(self):
        logit=torch.tensor([[[[1.,2.]]], [[[-2.,2.]]]])
        target=torch.tensor([[[[1.,1.]]], [[[0.,0.]]]])
        per=worker.per_instance_loss(logit,target)
        self.assertAlmostEqual(float(worker.objective(logit,target,torch.tensor([1.,1.]))),float(per.mean()))
        self.assertAlmostEqual(float(worker.objective(logit,target,torch.tensor([3.,1.]))),float((3*per[0]+per[1])/4))

    def test_label_free_inside_occupancy_and_strict_tie(self):
        stock=np.zeros((8,8),dtype=bool)
        crop=[0,0,4,5]
        stock[:3,:4]=True
        self.assertTrue(worker.gate(stock,crop))
        stock[3,0]=True
        self.assertFalse(worker.gate(stock,crop))
        self.assertFalse(worker.gate(stock,[0,0,0,5]))

    def test_splice_and_threshold_preserve_exterior(self):
        stock=np.ones((4,4),dtype=bool)
        outputs=worker.render(stock,[1,1,3,3],torch.full((1,1,48,48),-1.))
        self.assertEqual(int(outputs.sum()),12)
        self.assertTrue(outputs[0].all() and outputs[:,-1].all())
        self.assertTrue(worker.render(stock,[1,1,3,3],torch.zeros(1,1,48,48)).all())


if __name__=='__main__':
    unittest.main()
