import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.optim.swa_utils import AveragedModel

from state_preservation import ContinuationEMA, FirstStepTrace, compare_first
from compare_state_pair import load_gradient, numeric_comparison


class FirstUpdateTest(unittest.TestCase):
    def test_history_changes_actual_first_update_without_changing_inputs_or_gradient(self):
        traces = {}
        for arm in ("S", "F"):
            model = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                model.weight.fill_(1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, fused=False)
            if arm == "S":
                optimizer.state[model.weight] = dict(step=torch.tensor(4464.0),
                    exp_avg=torch.full_like(model.weight, 0.20), exp_avg_sq=torch.full_like(model.weight, 0.50))
            trace = FirstStepTrace(model, optimizer, arm, initial_step=4464)
            for index in range(4):
                sample = type("Sample", (), {"tensors": torch.ones(1, 1), "mask": torch.zeros(1, 1, dtype=torch.bool)})()
                trace.observe_input(sample, [{"labels": torch.tensor([1])}], index)
            model.weight.grad = torch.full_like(model.weight, 0.05)
            optimizer.step()
            traces[arm] = trace.result()
        match = compare_first(traces["S"], traces["F"])
        self.assertTrue(match["same_initial_forward_conditions"])
        self.assertTrue(match["same_clipped_gradient"])
        self.assertNotEqual(traces["S"]["raw_update_l2"], traces["F"]["raw_update_l2"])

    def test_private_gradient_archive_roundtrip_and_numeric_comparison(self):
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, fused=False)
        with TemporaryDirectory() as temp:
            arm = Path(temp).resolve()
            trace = FirstStepTrace(model, optimizer, "F", initial_step=4464,
                                   gradient_output=arm / "FIRST_CLIPPED_GRAD_PRIVATE.pt")
            for index in range(4):
                sample = type("Sample", (), {"tensors": torch.ones(1, 1), "mask": None})()
                trace.observe_input(sample, [{"labels": torch.tensor([1]), "masks": torch.empty((0, 2, 2), dtype=torch.bool)}], index)
            model.weight.grad = torch.full_like(model.weight, 0.05)
            optimizer.step()
            weights = load_gradient(arm, trace.result())
            exact = numeric_comparison(weights, weights)
            self.assertEqual(exact["changed_elements"], 0)
            other = {key: value + 0.001 for key, value in weights.items()}
            perturbed = numeric_comparison(weights, other)
            self.assertAlmostEqual(perturbed["max_abs_difference"], 0.001, places=7)
            self.assertEqual(perturbed["changed_elements"], 1)

    def test_relative_count_proxy_does_not_reset_historical_average(self):
        model = torch.nn.Linear(1, 1, bias=False)
        observed = []
        def averaging(averages, live, count):
            observed.append(int(count))
            averages[0].mul_(0.5).add_(live[0], alpha=0.5)
        avg = AveragedModel(model, multi_avg_fn=averaging, use_buffers=True)
        with torch.no_grad():
            avg.module.weight.fill_(4)
            model.weight.fill_(8)
            avg.n_averaged.fill_(4464)
        relative = ContinuationEMA(avg, 4464)
        self.assertEqual(int(relative.n_averaged), 0)
        relative.update_parameters(model)
        self.assertEqual(observed, [4464])
        self.assertEqual(int(relative.n_averaged), 1)
        self.assertEqual(avg.module.weight.item(), 6)


if __name__ == "__main__":
    unittest.main()
