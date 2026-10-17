"""Synthetic real-nn contract; GREEN is pending a Root-owned actual CPU task.

The implementation import intentionally precedes Torch. Before the factory
exists, the recorded RED can therefore occur without importing Torch at all.
This fixture never instantiates or executes the original full UNet++.
"""
import unittest

import full_instance_adapter as adapter_module

import torch


class FullInstanceAdapterContract(unittest.TestCase):
    def test_one_image_encoding_serves_grouped_individual_and_permuted_queries(self):
        class TinyRealBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.feature=torch.nn.Sequential(torch.nn.Conv2d(3,4,3,padding=1),torch.nn.ReLU())
                self.semantic_output=torch.nn.Conv2d(4,1,1)
                self.calls=0

            def forward(self,image):
                self.calls+=1
                return self.semantic_output(self.feature(image))

        torch.manual_seed(202610173101)
        backbone=TinyRealBackbone().eval()
        backbone.feature[0].bias.requires_grad_(False)
        original_flags=[parameter.requires_grad for parameter in backbone.parameters()]
        adapter=adapter_module.make_adapter(torch,backbone,'feature',4,query_channels=8,hidden=6)
        self.assertFalse(backbone.training)
        self.assertEqual([parameter.requires_grad for parameter in backbone.parameters()],original_flags)
        # Avoid the trivially equal all-zero endpoint: exercise genuine nonzero
        # readout after the factory's required zero-initialized final layer.
        with torch.no_grad():
            for parameter in adapter.head.parameters():parameter.fill_(.125)
        image=torch.randn(1,3,8,12)
        embeddings=torch.randn(5,8)
        coarse=torch.randn(5,200,200)
        encoded=adapter.encode(image)
        self.assertEqual(backbone.calls,1)
        whole=adapter.predict(encoded,embeddings,coarse)
        individual=torch.cat([adapter.predict(encoded,embeddings[q:q+1],coarse[q:q+1])
                              for q in range(5)],dim=0)
        grouped=torch.cat([adapter.predict(encoded,embeddings[first:first+2],coarse[first:first+2])
                           for first in range(0,5,2)],dim=0)
        order=torch.tensor([3,0,4,1,2],dtype=torch.int64)
        permuted=adapter.predict(encoded,embeddings.index_select(0,order),coarse.index_select(0,order))
        self.assertEqual(tuple(whole.shape),(5,1,8,12))
        self.assertGreater(float(whole.abs().sum()),0.)
        torch.testing.assert_close(individual,whole,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(grouped,whole,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(permuted,whole.index_select(0,order),rtol=1e-5,atol=1e-6)
        self.assertEqual(backbone.calls,1)
        backbone.calls=0
        forwarded=adapter(image,embeddings,coarse,query_batch_size=2)
        self.assertEqual(backbone.calls,1)
        torch.testing.assert_close(forwarded,whole,rtol=1e-5,atol=1e-6)


if __name__=='__main__':unittest.main()
