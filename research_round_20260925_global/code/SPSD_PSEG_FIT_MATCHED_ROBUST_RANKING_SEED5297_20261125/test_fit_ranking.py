from pathlib import Path
from runpy import run_path
import unittest

import torch


MODULE = Path(__file__).with_name("fit_ranking.py")


class MatchedRankingTests(unittest.TestCase):
    def test_robust_rank_prefers_safe_action_and_treats_static_as_zero(self):
        f = run_path(str(MODULE))
        # The largest mean belongs to the harmful action. True ties are omitted.
        truth = torch.tensor([[[4., -2.], [1., 1.], [2., 0.]]])
        valid = torch.tensor([[True, True, True]])
        prepared = f["rank_targets"](truth, valid)
        safe = torch.tensor([[[.8, -.3], [.2, .2], [.4, 0.]]], requires_grad=True)
        wrong = torch.tensor([[[.8, .3], [.2, -.2], [.4, 0.]]])
        safe_loss = f["rank_penalty"](safe, prepared)
        wrong_loss = f["rank_penalty"](wrong, prepared)
        self.assertLess(safe_loss.detach().item(), float(wrong_loss))
        self.assertEqual(prepared.valid.sum().item(), 5)
        self.assertEqual(prepared.possible.sum().item(), 6)
        safe_loss.backward()
        self.assertTrue(torch.isfinite(safe.grad).all().item())


    def test_zero_supervision_produces_zero_finite_gradient(self):
        f = run_path(str(MODULE))
        targets = torch.zeros((1, 2, 2))
        valid = torch.tensor([[True, False]])
        pairs = f["rank_targets"](targets, valid)
        predicted = torch.tensor([[[.2, .1], [99., 99.]]], requires_grad=True)
        loss = f["rank_penalty"](predicted, pairs)
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(pairs.possible.sum().item(), 1)
        loss.backward()
        torch.testing.assert_close(predicted.grad, torch.zeros_like(predicted))

    def test_candidate_average_is_independent_of_how_many_valid_pairs(self):
        f = run_path(str(MODULE))
        truth = torch.tensor([[[1., 1.], [7., 7.]], [[1., 1.], [-2., -1.]]])
        valid = torch.tensor([[True, False], [True, True]])
        predicted = torch.tensor([[[.3, -.1], [99., 99.]], [[.05, .05], [.4, .4]]])
        batch = f["rank_penalty"](predicted, f["rank_targets"](truth, valid))
        separate = [f["rank_penalty"](predicted[i:i+1], f["rank_targets"](truth[i:i+1], valid[i:i+1]))
                    for i in range(2)]
        torch.testing.assert_close(batch, torch.stack(separate).mean())


if __name__ == "__main__":
    unittest.main()
