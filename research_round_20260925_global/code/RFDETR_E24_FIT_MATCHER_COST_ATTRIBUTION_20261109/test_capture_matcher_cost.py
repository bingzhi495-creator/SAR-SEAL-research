import sys
from pathlib import Path
import unittest

import numpy as np
import torch
from rfdetr.models import matcher as native

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_matcher_cost import capture_final, make_sampler, view_plan


class CaptureTest(unittest.TestCase):
    def test_real_native_matcher_on_small_synthetic_fixture(self):
        gen = torch.Generator().manual_seed(773)
        outputs = dict(pred_logits=torch.randn(1, 4, 2, generator=gen),
                       pred_boxes=torch.rand(1, 4, 4, generator=gen).clamp(.1, .9),
                       pred_masks=dict(spatial_features=torch.randn(1, 3, 8, 8, generator=gen),
                                       query_features=torch.randn(1, 4, 3, generator=gen),
                                       bias=torch.tensor(.1)))
        masks = torch.zeros(2, 32, 32, dtype=torch.bool)
        masks[0, 3:8, 4:10] = True
        masks[1, 15:25, 14:25] = True
        targets = [dict(boxes=torch.tensor([[.25, .25, .2, .2], [.7, .7, .25, .25]]),
                        labels=torch.zeros(2, dtype=torch.long), masks=masks)]
        matcher = native.HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2, cost_mask_ce=5, cost_mask_dice=5)
        with torch.no_grad():
            packet = capture_final(matcher, native, outputs, targets, width=2, groups=2)
        sample = make_sampler(packet, "cpu")
        logits, labels = sample(packet["uniform_draw"].numpy())
        np.testing.assert_array_equal(logits, packet["native_pred_at_draw"].numpy())
        np.testing.assert_array_equal(labels, packet["native_target_at_draw"].numpy())
        native_cost = make_sampler(packet, "cpu", native_cost=True)
        ce, dice = native_cost(packet["uniform_draw"].numpy())
        np.testing.assert_array_equal(ce, packet["native_mask_ce"].numpy())
        np.testing.assert_array_equal(dice, packet["native_mask_dice"].numpy())
        self.assertEqual(packet["native_total"].shape, (4, 2))
        self.assertEqual(packet["native_src"].numel(), 4)

    def test_tranche_is_fixed_unique_and_order_independent(self):
        items = [dict(dataset_index=i, image_sha256=f"{i:064x}", partition="head_fit") for i in range(100)]
        a = view_plan(items, "cohort32")
        self.assertEqual(a, view_plan(items[::-1], "cohort32"))
        self.assertEqual(len(a), 32)
        self.assertEqual(len({item["image_sha256"] for item, repeat in a}), 32)
        self.assertTrue(all(repeat == 0 for _, repeat in a))
        self.assertEqual(len(view_plan(items, "smoke")), 24)


if __name__ == "__main__":
    unittest.main()
