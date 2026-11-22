import importlib.util
import os
from pathlib import Path
import unittest

import numpy as np
import torch


def load_test_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FullFitRecipeTest(unittest.TestCase):
    def test_single_recipe_keeps_all_five_source_folds_eligible(self):
        local = Path(__file__).resolve().parent
        worker = load_test_module(local / "refit_pointrend.py")
        code = Path(os.environ["SEAL_POINTREND_TEST_CODE"])
        legacy = load_test_module(code / "RFDETR_POINTREND_FIT109_20261121/fit_point_heads.py")
        adapter = load_test_module(code / "RFDETR_POINTREND_SUBDIVISION_20261121/roi_point_head.py")
        folds = {f"group{i}": i for i in range(5)}
        rows = [dict(group=group, image_id=i, rank=1, query_id=0, matched_gt_id=i, k=0)
                for i, group in enumerate(folds)]
        spatial = np.ones((256, 8, 8), dtype=np.float32)
        logits = np.ones((8, 8), dtype=np.float32)
        box = np.array([0., 0., 1., 1.], dtype=np.float32)
        truth = np.zeros((8, 8), dtype=bool)
        truth[:4] = True
        def payload(row):
            return spatial, logits, box, truth
        head, audit = worker.fit_full_recipe(legacy, adapter, rows, folds, payload, steps=2, cpu_threads=1)
        self.assertEqual(audit["train_positive_rows"], 5)
        self.assertEqual(audit["held_positive_rows"], 0)
        self.assertEqual(audit["train_positive_sources"], 5)
        self.assertEqual(audit["held_sources"], 0)
        self.assertEqual(audit["schedule_seed"], 152170)
        self.assertEqual(audit["point_seed"], 252170)
        self.assertEqual(audit["supervised_points"], 392)
        self.assertGreater(audit["parameter_update_l2"], 0)
        self.assertTrue(any(entry[0] == "group0" for entry in audit["private_schedule"]))
        self.assertIsInstance(head, adapter.BinaryPointHead)
        self.assertFalse(head.training)


if __name__ == "__main__":
    unittest.main()
