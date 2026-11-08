"""Behavior checks for passive repeated-backward localization."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from same_graph_backward_node_trace import trace_backwards


class SameGraphTraceTest(unittest.TestCase):
    def test_native_cpu_convolution_records_equal_local_and_leaf_gradients(self):
        conv = torch.nn.Conv2d(1, 2, 3, bias=False)
        with torch.no_grad():
            conv.weight.fill_(.25)
        x = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5).requires_grad_()
        loss = conv(x).square().mean()
        trace = trace_backwards(loss, {"x": x, "w": conv.weight}, torch.device("cpu"))
        self.assertTrue(trace["leaf_gradient_exact"])
        self.assertEqual(trace["local_difference_count"], 0)
        self.assertGreaterEqual(trace["selected_node_count"], 1)
        self.assertTrue(any(row["operator"] == "ConvolutionBackward0" and row["exact_backward_inputs"]
                            and row["exact_backward_outputs"] for row in trace["node_comparisons"]))
        torch.testing.assert_close(x.grad, torch.autograd.grad(loss, x, retain_graph=True)[0])


    def test_local_difference_uses_received_and_emitted_gradient_direction(self):
        class ConvolutionBackwardProbe(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                ctx.calls = 0
                return value.clone()

            @staticmethod
            def backward(ctx, received):
                ctx.calls += 1
                return received * ctx.calls

        x = torch.ones(2, requires_grad=True)
        trace = trace_backwards(ConvolutionBackwardProbe.apply(x).sum(), {"x": x}, torch.device("cpu"))
        self.assertFalse(trace["leaf_gradient_exact"])
        self.assertEqual(trace["local_difference_count"], 1)
        row, = trace["candidate_local_differences"]
        self.assertTrue(row["exact_backward_inputs"])
        self.assertFalse(row["exact_backward_outputs"])

if __name__ == "__main__":
    unittest.main()
