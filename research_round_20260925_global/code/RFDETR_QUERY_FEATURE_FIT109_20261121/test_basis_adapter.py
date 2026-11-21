"""Pure CPU Torch witnesses, with no RF-DETR import, checkpoint or dataset."""
import importlib.util
from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent


def adapter():
    spec = importlib.util.spec_from_file_location("basis_adapter", HERE / "basis_adapter.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class EndpointSamplingTests(unittest.TestCase):
    def test_four_corners_and_interior_keep_add_drop_order(self):
        module = adapter()
        spatial = torch.arange(256*3*5, dtype=torch.float32).reshape(1,256,3,5)/31
        height, width = 11, 13
        pairs = torch.tensor([[0,width-1],[(height-1)*width,height*width-1],
                              [5*width+7,3*width+2],[5*width+7,0]])
        before = pairs.clone()
        result = module.sample_sealed_endpoints(spatial, pairs, (height,width))
        reference = F.interpolate(spatial, size=(height,width), mode="bilinear", align_corners=False)
        reference = reference[0].flatten(1)[:,pairs.reshape(-1)].T.reshape(len(pairs),2,256)
        self.assertEqual(result.shape,(4,2,256))
        self.assertTrue(result.is_contiguous())
        self.assertTrue(torch.equal(pairs,before))
        torch.testing.assert_close(result,reference,atol=module.REFERENCE_ATOL,rtol=module.REFERENCE_RTOL)

    def test_native_144_to_800_four_corners_and_interior_against_dense_channels(self):
        module=adapter()
        with torch.random.fork_rng():
            torch.manual_seed(5218)
            spatial=torch.randn(256,144,144)
        pairs=torch.tensor([[0,799],[799*800,800*800-1],[400*800+400,12345],[400,319999]])
        actual=module.sample_sealed_endpoints(spatial,pairs,(800,800))
        channels=torch.tensor([0,1,127,255])
        dense=F.interpolate(spatial[None,channels],size=(800,800),mode="bilinear",align_corners=False)
        expected=dense[0].flatten(1)[:,pairs.reshape(-1)].T.reshape(4,2,4)
        torch.testing.assert_close(actual[:,:,channels],expected,atol=module.REFERENCE_ATOL,rtol=module.REFERENCE_RTOL)
        torch.testing.assert_close(actual[0,0],spatial[:,0,0],atol=0,rtol=0)
        torch.testing.assert_close(actual[1,1],spatial[:,-1,-1],atol=0,rtol=0)

    def test_empty_support_and_invalid_coordinates(self):
        module=adapter()
        spatial=torch.zeros(256,3,5)
        self.assertEqual(module.sample_sealed_endpoints(spatial,torch.empty((0,2),dtype=torch.long),(11,13)).shape,
                         (0,2,256))
        for pairs in (torch.tensor([[-1,0]]),torch.tensor([[0,143]]),torch.tensor([[0.,1.]]),
                      torch.tensor([[True,False]]),torch.tensor([0,1])):
            with self.subTest(pairs=pairs),self.assertRaises(ValueError):
                module.sample_sealed_endpoints(spatial,pairs,(11,13))


class ToyHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_features_proj = torch.nn.Conv2d(256,256,1)
        self.query_features_proj = torch.nn.Linear(7,256)
        self.bias = torch.nn.Parameter(torch.tensor([.13]))
        self.calls = 0

    def forward(self, spatial, query_inputs):
        self.calls += 1
        spatial = self.spatial_features_proj(spatial)
        projected = self.query_features_proj(query_inputs)
        return torch.einsum("bchw,bnc->bnhw",spatial,projected)+self.bias


class ToyCapture:
    """Injected context protocol witness, not an RF-DETR model replacement."""
    def __init__(self,head):
        self.head=head
        self.spatial=self.queries=None
        self.spatial_hits=self.query_hits=0
        self.closed=False

    def spatial_hook(self,_module,_inputs,output):
        self.spatial_hits+=1
        self.spatial=output

    def query_hook(self,_module,inputs):
        self.query_hits+=1
        self.queries,=inputs

    def seal(self,_native):
        if not self.closed or (self.spatial_hits,self.query_hits)!=(1,1):
            raise ValueError("One closed final-only capture")
        return SimpleNamespace(spatial=self.spatial.detach().clone(), queries=self.queries.detach().clone(),
                               bias=self.head.bias.detach().clone())


@contextmanager
def toy_capture_final_query(head):
    observation=ToyCapture(head)
    spatial=head.spatial_features_proj.register_forward_hook(observation.spatial_hook)
    query=head.query_features_proj.register_forward_pre_hook(observation.query_hook)
    try:
        yield observation
        observation.closed=True
    finally:
        spatial.remove()
        query.remove()


class BasisCaptureTests(unittest.TestCase):
    def test_inference_forward_yields_detached_ordinary_feature_tensors(self):
        module=adapter()
        head=ToyHead().eval()
        with torch.inference_mode():
            with module.capture_frozen_basis(head,toy_capture_final_query) as observed:
                native=head(torch.zeros(1,256,3,5),torch.zeros(1,164,7))
            basis=observed.seal(native)
            sampled=module.sample_sealed_endpoints(basis.spatial,[[0,10]],(11,13))
            selected=module.select_projected_queries(basis,[7])
        for value in (basis.spatial,basis.projected_queries,basis.bias,sampled,selected):
            self.assertFalse(value.requires_grad)
            self.assertFalse(torch.is_inference(value))

    def test_injected_hook_projects_whole_bank_and_keeps_query_selection(self):
        module=adapter()
        with torch.random.fork_rng():
            torch.manual_seed(5217)
            head=ToyHead().eval()
            spatial=torch.randn(1,256,3,5)
            queries=torch.randn(1,164,7)
            before={k:v.clone() for k,v in head.state_dict().items()}
            seen=[]
            counter=head.query_features_proj.register_forward_pre_hook(
                lambda _m,inputs:seen.append(tuple(inputs[0].shape)))
            with torch.no_grad():
                with module.capture_frozen_basis(head,toy_capture_final_query) as observed:
                    native=head(spatial,queries)
                basis=observed.seal(native)
            counter.remove()
            self.assertEqual(seen,[(1,164,7),(1,164,7)])
            self.assertEqual(head.calls,1)
            self.assertEqual(basis.projected_queries.shape,(1,164,256))
            selected=module.select_projected_queries(basis,[5,1,5])
            with torch.no_grad():
                expected=head.query_features_proj(queries)[0,[5,1,5]]
            torch.testing.assert_close(selected,expected,atol=0,rtol=0)
            self.assertTrue(all(torch.equal(before[k],v) for k,v in head.state_dict().items()))
            self.assertFalse(basis.spatial.requires_grad)
            self.assertFalse(basis.projected_queries.requires_grad)
            self.assertFalse(head.spatial_features_proj._forward_hooks)
            self.assertFalse(head.query_features_proj._forward_pre_hooks)

    def test_corrupt_native_logits_are_rejected(self):
        module=adapter()
        with torch.random.fork_rng(),torch.no_grad():
            torch.manual_seed(5219)
            head=ToyHead().eval()
            with module.capture_frozen_basis(head,toy_capture_final_query) as observed:
                native=head(torch.randn(1,256,3,5),torch.randn(1,164,7))
            wrong=native.clone()
            wrong[0,0,0,0]+=1
            with self.assertRaisesRegex(ValueError,"Exact full native contraction"):
                observed.seal(wrong)

    def test_early_seal_repeated_forward_and_invalid_query_ids(self):
        module=adapter()
        head=ToyHead().eval()
        with torch.no_grad():
            with module.capture_frozen_basis(head,toy_capture_final_query) as observed:
                with self.assertRaisesRegex(ValueError,"closed"):
                    observed.seal(None)
                native=head(torch.zeros(1,256,3,5),torch.zeros(1,164,7))
            basis=observed.seal(native)
            with self.assertRaisesRegex(ValueError,"once"):
                observed.seal(native)
            for ids in ([164],[-1],[1.0],[False],[[0]]):
                with self.subTest(ids=ids),self.assertRaises(ValueError):
                    module.select_projected_queries(basis,ids)
            with module.capture_frozen_basis(head,toy_capture_final_query) as repeated:
                head(torch.zeros(1,256,3,5),torch.zeros(1,164,7))
                raw=head(torch.zeros(1,256,3,5),torch.zeros(1,164,7))
            with self.assertRaisesRegex(ValueError,"closed final-only"):
                repeated.seal(raw)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
