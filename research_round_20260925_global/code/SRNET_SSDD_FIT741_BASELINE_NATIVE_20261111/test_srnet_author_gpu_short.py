import unittest

import torch

from srnet_author_gpu_short import LOSS_KEYS, training_update


class FullLossUpdateTest(unittest.TestCase):
    def test_all_author_loss_groups_backpropagate_and_sgd_changes_parameter(self):
        class SyntheticCarrier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.coarse_head = torch.nn.Linear(1, 1, bias=False)
                self.refine_head = torch.nn.Linear(1, 1, bias=False)
                self.mask_head = torch.nn.Linear(1, 1, bias=False)
                for parameter in self.parameters():
                    torch.nn.init.constant_(parameter, .5)

            def forward_train(self, value):
                coarse = self.coarse_head(value).square().mean()
                refined = self.refine_head(value).square().mean()
                mask = self.mask_head(value).square().mean()
                return dict(zip(LOSS_KEYS, ([coarse], [coarse], [refined], [refined], [mask])))

            def _parse_losses(self, losses):
                values = {key: sum(value) for key, value in losses.items()}
                total = sum(values.values())
                return total, {key: float(value.detach()) for key, value in values.items()}

        model = SyntheticCarrier()
        optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9, weight_decay=.0001)
        state = {}
        actual = training_update(model, optimizer, (torch.ones(1, 1),), {}, torch, state)
        self.assertEqual(set(actual["losses"]), set(LOSS_KEYS))
        self.assertEqual(actual["gradient_tensors"], 3)
        self.assertTrue(all(value > 0 for value in actual["gradient_norms"].values()))
        self.assertTrue(all(parameter.item() < .5 for parameter in model.parameters()))
        self.assertEqual(state["stage"], "optimizer_step_complete")
        self.assertEqual(state["model_forward_calls"], 1)
        self.assertEqual(state["completed_forward_calls"], 1)
        self.assertEqual(state["completed_backward_calls"], 1)
        self.assertEqual(state["optimizer_updates"], 1)


if __name__ == "__main__":
    unittest.main()
