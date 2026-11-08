"""CPU checks for the one-window grid-sample backward substitution."""
from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid_backward_probe import GridBackwardTrace, cpu_backward_grid_sample, routed_grid_sample


class GridBackwardTest(unittest.TestCase):
    def test_same_forward_and_gradient_at_boundaries(self):
        base = torch.arange(24, dtype=torch.float64).reshape(1, 2, 3, 4) / 13
        coords = torch.tensor([[[[-1.4, -.8], [-1., 1.], [.13, -.27], [1.4, .9]]]], dtype=torch.float64)
        gradient = torch.arange(1, 9, dtype=torch.float64).reshape(1, 2, 1, 4) / 7
        for padding in ("zeros", "border"):
            for align in (False, True, None):
                with self.subTest(padding=padding, align=align):
                    data = base.clone().requires_grad_()
                    grid = coords.clone().requires_grad_()
                    expected = F.grid_sample(data, grid, padding_mode=padding, align_corners=align)
                    wanted = torch.autograd.grad(expected, (data, grid), gradient)
                    trace = GridBackwardTrace()
                    actual = cpu_backward_grid_sample(data, grid, padding, align, trace)
                    actual_grads = torch.autograd.grad(actual, (data, grid), gradient)
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual_grads, wanted, strict=True)))
                    stats = trace.summary()
                    self.assertEqual(stats["routed_forward_calls"], 1)
                    self.assertEqual(stats["cpu_backward_calls"], 1)
                    self.assertEqual(stats["observed_shapes"][0]["effective_dtype"], "torch.float64")


    def test_repeated_backward_and_promoted_effective_dtype(self):
        tensors = []
        for _ in range(2):
            data = (torch.arange(20).view(1, 1, 4, 5) / 9).to(torch.bfloat16).requires_grad_()
            grid = torch.tensor([[[[.12, .45], [.12, .45], [-.3, .95], [1.31, -.95]]]]).requires_grad_()
            trace = GridBackwardTrace()
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                expected = F.grid_sample(data, grid, padding_mode="border", align_corners=False)
                actual = cpu_backward_grid_sample(data, grid, "border", False, trace)
            wanted = torch.autograd.grad(expected.sum(), (data, grid))
            found = torch.autograd.grad(actual.sum(), (data, grid))
            self.assertEqual(actual.dtype, torch.float32)
            self.assertEqual(found[0].dtype, torch.bfloat16)
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(wanted, found, strict=True)))
            self.assertEqual(trace.summary()["observed_shapes"][0]["effective_dtype"], "torch.float32")
            tensors.append(found)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(*tensors, strict=True)))

    def test_cpu_calls_outside_probe_keep_native_path(self):
        data = torch.tensor([[[[1., 2.], [3., 4.]]]], requires_grad=True)
        grid = torch.tensor([[[[.25, -.5]]]])
        trace = GridBackwardTrace()
        expected = F.grid_sample(data, grid, mode="nearest", align_corners=False)
        actual = routed_grid_sample(trace)(data, grid, mode="nearest", align_corners=False)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(trace.summary()["unchanged_calls"], 1)
        self.assertEqual(trace.summary()["routed_forward_calls"], 0)


if __name__ == "__main__":
    unittest.main()
