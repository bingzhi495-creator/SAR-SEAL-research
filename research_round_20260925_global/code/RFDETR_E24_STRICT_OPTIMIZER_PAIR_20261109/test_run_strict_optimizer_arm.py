from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_strict_optimizer_arm import BackwardControl, REFERENCE_FIRST_GRADIENT_SHA256, check_first_step


class FirstStepGateTests(unittest.TestCase):
    def test_passes_exact_first_window(self):
        check_first_step(REFERENCE_FIRST_GRADIENT_SHA256, (4, 36, 8))

    def test_rejects_different_gradient_before_step(self):
        with self.assertRaisesRegex(ValueError, "first clipped gradient"):
            check_first_step("0" * 64, (4, 36, 8))

    def test_rejects_wrong_route_counts(self):
        for counts in (None, (3, 36, 8), (4, 35, 8), (4, 36, 7)):
            with self.assertRaisesRegex(ValueError, "first-window routes"):
                check_first_step(REFERENCE_FIRST_GRADIENT_SHA256, counts)

    def test_tensor_backward_receives_loss_and_original_method_is_restored(self):
        import torch

        old = torch.Tensor.backward

        class RecordingBackward:
            calls = 1487
            seconds = 0.0

            def __call__(self, tensor, *args, **kwargs):
                self.calls += 1
                return old(tensor, *args, **kwargs)

        f = torch.nn.functional
        control = BackwardControl.__new__(BackwardControl)
        control.torch = torch
        control.strict = RecordingBackward()
        control.grid = SimpleNamespace(routed_grid_sample=lambda _trace: f.grid_sample)
        control.combined = SimpleNamespace(routed_interpolate=lambda _trace: f.interpolate)
        control.grid_trace = control.resize_trace = None
        control.first_counts = (4, 36, 8)
        control.routes = lambda: ({}, {})
        x = torch.tensor(3.0, requires_grad=True)

        def stock():
            (2 * x).backward()
            return {"microbatches": 1488}

        result = control.train(stock)
        self.assertEqual(x.grad.item(), 2.0)
        self.assertEqual(result["backward_control"]["strict_calls"], 1488)
        self.assertIs(torch.Tensor.backward, old)


if __name__ == "__main__":
    unittest.main()

