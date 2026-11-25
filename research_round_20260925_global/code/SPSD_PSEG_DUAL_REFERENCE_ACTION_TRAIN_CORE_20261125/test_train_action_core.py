import os
import unittest
from pathlib import Path
from runpy import run_path

import numpy as np
import torch


functions = run_path(str(Path(__file__).with_name("train_action_core.py")))
signed_action_deltas = functions["signed_action_deltas"]


class ActionTrainCoreTest(unittest.TestCase):
    def test_changed_add_and_drop_endpoints_have_exact_signed_gain_in_both_references(self):
        t = torch.tensor([[[1., 0.], [1., 0.]], [[0., 1.], [0., 1.]]])
        actions = torch.tensor([[[-1., 1.], [1., -1.]],
                                [[-1., 0.], [1., 0.]],
                                [[0., 1.], [0., -1.]]])
        expected = torch.tensor([[0., 0.], [-.5, .5], [.5, -.5]])
        torch.testing.assert_close(signed_action_deltas(t, actions, torch.tensor(1.)), expected)

    def test_two_full_fits_share_draws_initialization_and_balanced_ragged_packets(self):
        new_packet = functions["CandidatePacket"]
        train = functions["train_matched_action"]
        packets = []
        for index, (group, image, p, n) in enumerate(([0, 0, .2, 2], [0, 0, .4, 4],
                                                       [0, 1, .6, 3], [1, 2, .8, 2])):
            f = np.zeros((n, 2, 14), dtype=np.float32)
            f[..., 0] = p
            f[:, :, 1] = np.arange(n)[:, None] * 0.2
            t = np.zeros((n, 2, 2), dtype=np.uint8)
            t[0, :, :] = [[1, 0], [0, 1]]
            t[1:, :, 1] = 1
            w = np.zeros((1 + (index % 2), n, 2), dtype=np.int8)
            w[0, :2] = [[-1, 1], [1, -1]]
            if len(w) > 1:
                w[1, [0, n - 1], 0] = [-1, 1]
            packets.append(new_packet(f, t, w, 1, group, image))
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        rng_state = torch.random.get_rng_state().clone()
        try:
            models, center, scale, audit = train(packets, source_core=Path(os.environ["SPSD_ACTION_SOURCE_CORE"]))
        finally:
            torch.set_num_threads(old_threads)
        self.assertTrue(torch.equal(rng_state, torch.random.get_rng_state()))
        self.assertEqual((audit["fit_groups"], audit["fit_images"], audit["fit_candidates"]), (2, 3, 4))
        self.assertAlmostEqual(float(center[0]), .625, places=6)
        self.assertAlmostEqual(float(scale[0]), np.sqrt(.044375), places=6)
        self.assertEqual(set(models), {"bce", "bce_signed_action"})
        initial, batches, final = set(), set(), set()
        for name, stats in audit["heads"].items():
            self.assertEqual((stats["parameters"], stats["optimizer_updates"], stats["observed_candidates"]),
                             (1346, 1_000, 64_000))
            self.assertTrue(np.isfinite(stats["last_total_loss"]))
            initial.add(stats["initial_tensor_sha256"])
            batches.add(stats["batch_sha256"])
            final.add(stats["final_tensor_sha256"])
            self.assertFalse(models[name].training)
        self.assertEqual((len(initial), len(batches), len(final)), (1, 1, 2))
        a, b = audit["heads"]["bce"], audit["heads"]["bce_signed_action"]
        self.assertEqual((a["observed_endpoints"], a["observed_actions"]), (b["observed_endpoints"], b["observed_actions"]))
        self.assertEqual(a["last_total_loss"], a["last_bce_loss"])
        self.assertAlmostEqual(b["last_total_loss"], b["last_bce_loss"] + b["last_signed_action_loss"], places=6)


    def test_zero_change_action_leaves_both_models_identical(self):
        packet = functions["CandidatePacket"](np.full((2, 2, 14), .4, dtype=np.float32),
                                                np.ones((2, 2, 2), dtype=np.uint8),
                                                np.zeros((1, 2, 2), dtype=np.int8), 1, "one", 0)
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            _, _, _, audit = functions["train_matched_action"](
                [packet], source_core=Path(os.environ["SPSD_ACTION_SOURCE_CORE"]))
        finally:
            torch.set_num_threads(old_threads)
        baseline, treatment = audit["heads"]["bce"], audit["heads"]["bce_signed_action"]
        self.assertEqual(treatment["last_signed_action_loss"], 0.0)
        self.assertEqual(baseline["final_tensor_sha256"], treatment["final_tensor_sha256"])

    def test_inconsistent_static_actions_fail_before_updates(self):
        weights = np.array([[[-1, 0], [1, 0]], [[1, 0], [-1, 0]]], dtype=np.int8)
        packet = functions["CandidatePacket"](np.full((2, 2, 14), .4, dtype=np.float32),
                                                np.ones((2, 2, 2), dtype=np.uint8), weights, 1, "one", 0)
        with self.assertRaisesRegex(ValueError, "one common static incumbent"):
            functions["train_matched_action"]([packet], source_core=Path(os.environ["SPSD_ACTION_SOURCE_CORE"]))


if __name__ == "__main__":
    unittest.main()
