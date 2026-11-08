"""CPU checks for preserving native bilinear interpolation and its derivative semantics."""
from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from combined_sampling_backward_probe import (ResizeBackwardTrace, cpu_backward_interpolate, load_grid_probe,
                                                 routed_interpolate)


class ResizeBackwardTest(unittest.TestCase):
    def test_native_size_and_scale_semantics(self):
        initial = torch.arange(60, dtype=torch.float64).reshape(1, 3, 4, 5) / 17
        cases = (
            dict(size=(3, 7)),
            dict(size=7, align_corners=True),
            dict(scale_factor=(1.35, .8), recompute_scale_factor=None),
            dict(scale_factor=(1.35, .8), recompute_scale_factor=False),
            dict(scale_factor=(1.35, .8), recompute_scale_factor=True),
            dict(scale_factor=1.6, align_corners=True, recompute_scale_factor=False),
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                data = initial.clone().requires_grad_()
                trace = ResizeBackwardTrace()
                stock = F.interpolate(data, mode="bilinear", **kwargs)
                candidate = cpu_backward_interpolate(data, trace=trace, **kwargs)
                grad = torch.arange(stock.numel(), dtype=stock.dtype).reshape_as(stock) / 31
                native_grad, = torch.autograd.grad(stock, data, grad)
                tested_grad, = torch.autograd.grad(candidate, data, grad)
                self.assertTrue(torch.equal(candidate, stock))
                self.assertTrue(torch.equal(tested_grad, native_grad))
                self.assertEqual(trace.summary()["cpu_backward_calls"], 1)


    def test_exact_repeated_bfloat_backward(self):
        gradients = []
        for _ in range(2):
            data = torch.arange(20).reshape(1, 1, 4, 5).to(torch.bfloat16).requires_grad_()
            trace = ResizeBackwardTrace()
            actual = cpu_backward_interpolate(data, scale_factor=(1.65, 1.3), trace=trace)
            expected = F.interpolate(data, scale_factor=(1.65, 1.3), mode="bilinear")
            adjoint = torch.arange(actual.numel()).reshape_as(actual).to(actual.dtype) / 35
            actual_grad, = torch.autograd.grad(actual, data, adjoint)
            expected_grad, = torch.autograd.grad(expected, data, adjoint)
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.equal(actual_grad, expected_grad))
            self.assertEqual(actual_grad.dtype, torch.bfloat16)
            self.assertEqual(trace.summary()["observed_shapes"][0]["effective_dtype"], "torch.bfloat16")
            gradients.append(actual_grad)
        self.assertTrue(torch.equal(*gradients))

    def test_out_of_scope_interpolation_remains_stock(self):
        data = torch.arange(20.).reshape(1, 1, 4, 5).requires_grad_()
        trace = ResizeBackwardTrace()
        for kwargs in (dict(mode="nearest"), dict(mode="bilinear", antialias=True)):
            expected = F.interpolate(data, size=(3, 4), **kwargs)
            actual = routed_interpolate(trace)(data, size=(3, 4), **kwargs)
            self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(trace.summary()["unchanged_calls"], 2)
        self.assertEqual(trace.summary()["routed_forward_calls"], 0)

    def test_exact_grid_and_replay_load_without_work(self):
        grid = load_grid_probe()
        replay = grid.load_replay()
        self.assertTrue(callable(replay.trace_first_window))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
