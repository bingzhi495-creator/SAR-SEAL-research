import unittest

import numpy as np
import torch

from pathlib import Path
from runpy import run_path

functions = run_path(str(Path(__file__).with_name("train_core.py")))
PointHead = functions["PointHead"]
predict = functions["predict"]
train_matched = functions["train_matched"]


class CoreTrainTest(unittest.TestCase):
    def test_exact_zero_initial_residual_and_param_capacity(self):
        features = np.zeros((2, 2, 14), dtype=np.float32)
        features[..., 0] = [[0., .2], [.7, 1.]]
        center = np.zeros(14, dtype=np.float32)
        scale = np.ones(14, dtype=np.float32)
        head = PointHead()
        expected = torch.from_numpy(features[..., 0]).clamp(1e-4, 1 - 1e-4).logit().sigmoid().numpy()
        expected = np.repeat(expected[..., None], 2, axis=-1)
        np.testing.assert_array_equal(predict(head, features, center, scale), expected)
        self.assertEqual(sum(t.numel() for t in head.parameters()), 1346)

    def test_full_paired_schedule_balances_groups_and_separates_four_modes(self):
        features = np.zeros((4, 2, 14), dtype=np.float32)
        features[..., 0] = np.array([.2, .2, .6, .8], dtype=np.float32)[:, None]
        target = np.zeros((len(features), 2, 2), dtype=np.uint8)
        target[..., 0] = 1
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        random_state = torch.random.get_rng_state().clone()
        try:
            arms, center, scale, audit = train_matched(
                features, target, ["g0", "g0", "g0", "g1"], [0, 0, 1, 2], seed=5297)
        finally:
            torch.set_num_threads(previous_threads)
        self.assertTrue(torch.equal(random_state, torch.random.get_rng_state()))
        self.assertEqual((audit["fit_groups"], audit["fit_images"], audit["fit_pairs"]), (2, 3, 4))
        self.assertAlmostEqual(float(center[0]), .6, places=6)
        self.assertAlmostEqual(float(scale[0]), np.sqrt(.06), places=6)
        np.testing.assert_array_equal(scale[1:], np.ones(13, dtype=np.float32))
        for entry in audit["heads"].values():
            self.assertEqual((entry["parameters"], entry["optimizer_updates"],
                              entry["observed_pair_examples"], entry["observed_point_examples"]),
                             (1346, 1000, 64_000, 128_000))
            self.assertEqual(entry["initial_tensor_sha256"], audit["initial_tensor_sha256"])
            self.assertEqual(entry["batch_sha256"], audit["batch_sha256"])
        predicted = {mode: predict(model, features, center, scale) for mode, model in arms.items()}
        self.assertTrue(np.all(predicted["dual"][..., 0] > .9))
        self.assertTrue(np.all(predicted["dual"][..., 1] < .10))
        self.assertTrue(np.all(predicted["pseg_repeat"] > .9))
        self.assertTrue(np.all(predicted["spsd_repeat"] < .10))
        self.assertTrue(np.all(abs(predicted["mean_repeat"] - .5) < .08))

    def test_rejects_mixed_assignments_and_unlocked_seed(self):
        features = np.full((2, 2, 14), .5, dtype=np.float32)
        target = np.zeros((2, 2, 2), dtype=np.uint8)
        with self.assertRaises(ValueError):
            train_matched(features, target, [0, 1], [0, 0])
        with self.assertRaises(ValueError):
            train_matched(features, target, [0, 1], [0, 1], seed=5300)


if __name__ == "__main__":
    unittest.main()

