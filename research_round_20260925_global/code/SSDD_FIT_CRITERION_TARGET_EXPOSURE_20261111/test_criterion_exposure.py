import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

import criterion_exposure as worker
from criterion_exposure import assert_assignment, losses, uniform_expected


class ReadoutTest(unittest.TestCase):
    def test_same_zero_logit_bce_and_different_dice(self):
        logits = torch.zeros((1, 2))
        bce_a, dice_a = losses(torch, logits, torch.tensor([[0., 0.]]))
        bce_b, dice_b = losses(torch, logits, torch.tensor([[0., 1.]]))
        torch.testing.assert_close(bce_a, bce_b)
        torch.testing.assert_close(dice_a, torch.tensor([.5]))
        torch.testing.assert_close(dice_b, torch.tensor([1. / 3.]))

    def test_uniform_expectation_is_per_original_target_and_draw(self):
        a = torch.tensor([[[False, False], [False, False]]])
        b = torch.tensor([[[False, True], [False, True]]])
        self.assertEqual(uniform_expected(a, b, 8).tolist(), [4.])

    def test_one_to_one_assignment_within_every_group(self):
        src = torch.tensor([0, 1, 2, 3])
        dst = torch.tensor([1, 0, 0, 1])
        assert_assignment(torch, src, dst, 2, 2, 2)
        with self.assertRaises(ValueError):
            assert_assignment(torch, src, torch.tensor([1, 1, 0, 0]), 2, 2, 2)


    def test_cli_passes_through_pinned_interpreter(self):
        args = ["criterion_exposure.py", "prior", "geometry", "source.py",
                "anchor.json", "output", "--expected-python", "/exact/venv/bin/python"]
        with (patch.object(sys, "argv", args), patch.dict(os.environ, CUDA_VISIBLE_DEVICES="-1"),
              patch.object(os, "sched_getaffinity", return_value={2, 3}, create=True),
              patch.object(sys, "addaudithook"), patch.object(worker, "run") as run):
            worker.main()
        run.assert_called_once_with(*(Path(value) for value in args[1:6]), Path(args[-1]))

if __name__ == "__main__":
    unittest.main()
