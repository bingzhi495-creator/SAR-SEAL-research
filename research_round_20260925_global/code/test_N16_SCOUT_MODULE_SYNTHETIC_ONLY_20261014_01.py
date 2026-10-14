"""SYNTHETIC_ONLY: CPU random fixtures; no actual SAR artifacts or CUDA work."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
import importlib.util
from pathlib import Path
import unittest
import torch

torch.set_num_threads(2)
torch.set_num_interop_threads(2)
if torch.__version__ != "2.0.1+cu118" or torch.cuda.is_initialized():
    raise RuntimeError("Use the qualified seal_gpu runtime with cold CUDA")
SOURCE = Path(__file__).with_name("N16_SCOUT_MODULE_UNBOUND_DRAFT_20261014_01.py")
SPEC = importlib.util.spec_from_file_location("_N16_synthetic_scout", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

class ScoutTests(unittest.TestCase):
    @staticmethod
    def fixture(n=18):
        generator = torch.Generator(device="cpu").manual_seed(4567)
        return (torch.randn(n,256,generator=generator),
                torch.randn(n,256,generator=generator),
                8*torch.randn(n,generator=generator))

    def test_private_same_seed_initialization_preserves_cpu_rng_and_cold_cuda(self):
        before = torch.get_rng_state().clone()
        first = MODULE.make_scout(torch, 202610160701)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        second = MODULE.make_scout(torch, 202610160701)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertFalse(torch.cuda.is_initialized())
        self.assertEqual(sum(p.numel() for p in first.parameters()), 16547)
        self.assertEqual(len(first.state_dict()), 4)
        self.assertEqual(tuple(first.buffers()), ())
        self.assertTrue(all(torch.equal(value, second.state_dict()[key]) for key,value in first.state_dict().items()))
        self.assertTrue(all(bool((value != 0).any()) for value in first.state_dict().values()))

    def test_manual_formula_matches_nn_on_random_constant_and_empty_inputs(self):
        head = MODULE.make_scout(torch, 202610160701)
        cases = [self.fixture(18), self.fixture(0),
                 (torch.zeros(5,256),torch.ones(5,256),torch.tensor([-32.,-8.,0.,8.,32.]))]
        for inputs in cases:
            before = torch.get_rng_state().clone()
            with torch.no_grad():
                expected = head(*inputs)
                actual = MODULE.independent_logits(torch,head.state_dict(),*inputs)
            self.assertEqual(actual.shape,(inputs[0].shape[0],3))
            self.assertEqual(actual.dtype,torch.float32)
            self.assertTrue(bool(torch.isfinite(actual).all()))
            self.assertTrue(torch.allclose(actual,expected,atol=1e-6,rtol=1e-5))
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_different_random_fixture_labels_update_real_losses_and_gradients(self):
        inputs = self.fixture()
        utility = MODULE.make_scout(torch, 202610160701)
        error = MODULE.make_scout(torch, 202610160701)
        initial_state = {key:value.clone() for key,value in utility.state_dict().items()}
        targets = (torch.tensor([0,1,2]*6), torch.tensor([2,0,1]*6))
        losses = []
        for head, target in zip((utility,error), targets):
            optimizer = torch.optim.AdamW(head.parameters(),lr=.02,weight_decay=0)
            first_loss = float(torch.nn.functional.cross_entropy(head(*inputs),target).item())
            first_gradient_norm = 0.
            for step in range(8):
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(head(*inputs),target)
                self.assertTrue(bool(torch.isfinite(loss)))
                loss.backward()
                self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters()))
                if step == 0:
                    first_gradient_norm = sum(float(p.grad.abs().sum().item()) for p in head.parameters())
                optimizer.step()
            final_loss = float(torch.nn.functional.cross_entropy(head(*inputs),target).item())
            self.assertGreater(first_gradient_norm, 0.)
            self.assertLess(final_loss, first_loss - .01)
            losses.append((first_loss,final_loss))
            self.assertTrue(any(not torch.equal(value,initial_state[key]) for key,value in head.state_dict().items()))
        self.assertTrue(any(not torch.equal(value,error.state_dict()[key]) for key,value in utility.state_dict().items()))
        self.assertFalse(torch.cuda.is_initialized())
        print("SYNTHETIC_ONLY loss updates",losses)

    def test_invalid_dtype_device_shape_and_nonfinite_inputs_are_rejected(self):
        head = MODULE.make_scout(torch, 202610160701)
        F,e,L = self.fixture(3)
        nan_F = F.clone()
        nan_F[0,0] = float("nan")
        inf_e = e.clone()
        inf_e[0,0] = float("inf")
        invalid = [(nan_F,e,L), (F,inf_e,L), (F,e,L*float("nan")),
                   (F[:,:255],e,L), (F,e[:2],L), (F,e,L[:,None]),
                   (F,e.double(),L), (F,torch.empty(3,256,device="meta"),L)]
        for index, args in enumerate(invalid):
            with self.subTest(case=index), self.assertRaises(ValueError):
                head(*args)
        self.assertFalse(torch.cuda.is_initialized())

if __name__ == "__main__":
    unittest.main()
