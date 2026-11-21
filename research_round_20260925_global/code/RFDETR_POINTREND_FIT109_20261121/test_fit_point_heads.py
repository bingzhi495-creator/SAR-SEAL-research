import importlib.util
from pathlib import Path
import unittest

import torch

from fit_point_heads import train_oof_heads


def adapter_module():
    here = Path(__file__).resolve().parent
    paths = (here / "roi_point_head.py",
             here.parent / "RFDETR_POINTREND_SUBDIVISION_20261121" / "roi_point_head.py")
    path = next(p for p in paths if p.is_file())
    spec = importlib.util.spec_from_file_location("synthetic_roi_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestFitPointHeads(unittest.TestCase):
    def setUp(self):
        self.folds = {f"source{i}": i for i in range(5)}
        self.rows = [dict(group=g, image_id=i + 1, rank=1, query_id=i, matched_gt_id=100 + i, k=0)
                     for g, i in self.folds.items()]
        self.spatial = torch.linspace(-1, 1, 256 * 4 * 4).reshape(256, 4, 4)
        self.raw = torch.linspace(-2, 2, 16).reshape(4, 4)
        self.gt = torch.arange(8).repeat(8, 1) > 3
        self.adapter = adapter_module()

    def load(self, row):
        return self.spatial, self.raw, torch.tensor([0., 0., 1., 1.]), self.gt

    def test_real_head_oof_no_held_source_zero_budget_and_reproducibility(self):
        initial_rng = torch.get_rng_state().clone()
        initial_threads = torch.get_num_threads()
        heads, audits = train_oof_heads(self.rows, self.folds, self.load, adapter=self.adapter, steps=2)
        heads2, audits2 = train_oof_heads(list(reversed(self.rows)), self.folds, self.load,
                                          adapter=self.adapter, steps=2)
        self.assertEqual(audits, audits2)
        self.assertEqual(len(heads), 5)
        self.assertTrue(torch.equal(initial_rng, torch.get_rng_state()))
        self.assertEqual(initial_threads, torch.get_num_threads())
        for fold, audit in audits.items():
            self.assertEqual(audit["supervised_points"], 392)
            self.assertEqual(audit["zero_budget_updates"], 2)
            self.assertEqual(audit["held_positive_rows"], 1)
            self.assertGreater(audit["parameter_update_l2"], 0)
            self.assertFalse(heads[fold].training)
            self.assertEqual(len(audit["private_schedule"]), 2)
            for group, image_id, rank, query_id, gt_id in audit["private_schedule"]:
                self.assertNotEqual(self.folds[group], fold)
                self.assertEqual(gt_id, image_id + 99)
            for a, b in zip(heads[fold].parameters(), heads2[fold].parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_split_image_or_reused_truth_rejected_before_tensor_loading(self):
        def forbidden(row):
            self.fail("invalid manifest reached ROI loader")

        for extra in (dict(self.rows[0], group="source2", rank=2, matched_gt_id=900),
                      dict(self.rows[0], rank=2, query_id=4)):
            with self.assertRaises(ValueError):
                train_oof_heads(self.rows + [extra], self.folds, forbidden, adapter=self.adapter, steps=1)

    def test_invalid_box_cannot_train_and_thread_setting_is_restored(self):
        initial_threads = torch.get_num_threads()
        def invalid_box(row):
            return self.spatial, self.raw, torch.tensor([0., 0., 1.1, 1.]), self.gt

        with self.assertRaisesRegex(ValueError, "clipped"):
            train_oof_heads(self.rows, self.folds, invalid_box, adapter=self.adapter, steps=1)
        self.assertEqual(torch.get_num_threads(), initial_threads)


if __name__ == "__main__":
    unittest.main()
