"""Synthetic proof that the fixed-point target ablation changes only target values."""

from types import SimpleNamespace
from unittest import TestCase

import numpy as np

from run_target_value_ablation import freeze_target_banks, held_comparison, selected_training


class TargetBankTests(TestCase):
    def test_all_targets_share_exact_original_read_only_indices(self):
        chosen = np.frombuffer(np.array([0, 2, 3], dtype=np.int64).tobytes(), dtype=np.int64)
        occupancy = np.frombuffer(np.array([.1, .7, .8], dtype=np.float64).tobytes(), dtype=np.float64)
        bank = SimpleNamespace(indices=chosen, targets=occupancy)
        args = (np.array([0, 0, 1, 0]), np.array([1, 0, 0, 0]), np.array([.1, .2, .7, .8]),
                np.array([.2, .1, -.1, .3]), np.array([0, 1, 2, 3]), 7)

        def original(*received):
            self.assertEqual(tuple(map(id, received)), tuple(map(id, args)))
            return {"random": bank}

        arms = freeze_target_banks(original, np, *args)
        self.assertEqual(tuple(arms), ("stepwise_hard", "composed_hard", "raster_occupancy"))
        for arm in arms.values():
            self.assertIs(arm.indices, chosen)
            self.assertFalse(arm.targets.flags.writeable)
        np.testing.assert_array_equal(arms["stepwise_hard"].targets, [1, 0, 0])
        np.testing.assert_array_equal(arms["composed_hard"].targets, [0, 1, 0])
        self.assertIs(arms["raster_occupancy"], bank)

    def test_selected_census_distinguishes_additions_removals_and_fractional_labels(self):
        rows = []
        for length in [16] * 30 + [7]:
            indices = np.frombuffer(np.arange(length, dtype=np.int64).tobytes(), dtype=np.int64)
            raster = np.frombuffer(np.full(length, .5).tobytes(), dtype=np.float64)
            random = SimpleNamespace(indices=indices, targets=raster)
            b, c = np.zeros(length), np.zeros(length)
            b[0], c[1] = 1, 1
            banks = freeze_target_banks(lambda *_: {"random": random}, np, c, b, raster, b, b, 1)
            rows.append({"banks": banks})
        evidence = selected_training([dict(cursor=2, rows=rows)], np)
        self.assertEqual((evidence["points"], evidence["stepwise0_composed1"], evidence["stepwise1_composed0"],
                          evidence["hard_target_changed"], evidence["fractional_raster_targets"]),
                         (487, 31, 31, 62, 487))
        self.assertEqual(len(set(evidence["selected_target_stream_sha256s"].values())), 3)
        with self.assertRaisesRegex(ValueError, "487 historical"):
            selected_training([dict(cursor=2, rows=rows[:-1])], np)

    def test_crop_gate_retains_continuous_control_in_center_guard(self):
        def row(polygon=.8, center=.8):
            return dict(polygon_iou=polygon, center_iou=center, raster_iou=.8, ring_bce=.2)

        scores = dict(anchor=[row()] * 8, stepwise_hard=[row()] * 8, raster_occupancy=[row(.81)] * 8,
                      composed_hard=[row(.81)] * 8)
        self.assertTrue(held_comparison(scores, primary=True)["descriptive_gate"])
        scores["raster_occupancy"] = [row(.81, .805)] * 8
        guarded = held_comparison(scores, primary=True)
        self.assertFalse(guarded["descriptive_gate"])
        self.assertFalse(guarded["descriptive_gate_checks"]["center_iou_guard_at_least_minus_002"])
        self.assertNotIn("descriptive_gate", held_comparison(scores, primary=False))

