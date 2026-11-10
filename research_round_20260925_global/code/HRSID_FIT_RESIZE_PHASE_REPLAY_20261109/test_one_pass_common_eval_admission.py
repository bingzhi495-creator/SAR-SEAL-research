"""Synthetic closed-pair and unchanged DEV evaluator guard tests."""

from hashlib import sha256
from copy import deepcopy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase

import one_pass_common_eval_admission as subject


CODE = Path(os.environ.get("SAR_SEAL_ORDINARY_CODE_ROOT", Path(__file__).resolve().parent.parent))
ORIGINAL = CODE / "HRSID_FIT_RESIZE_PHASE_REPLAY_20261109/common_fit_eval_admission.py"
PHASE = CODE / "HRSID_MASK_PHASE_E24_DEV705_20261109/hrsid_mask_phase_dev705.py"
FILTERED = CODE / "HRSID_COCO_E24_FILTERED_DEV705_NATIVE_EVAL_20261109/hrsid_native_dev705_filtered.py"
BASE = CODE / "HRSID_COCO_E24_DEV705_NATIVE_EVAL_20261109/hrsid_native_dev705.py"
TRAINING_PROTOCOL, IMPLEMENTATION_COMMIT, QUALIFICATION_SHA = "a" * 40, "b" * 40, "c" * 64


class AdmissionTests(TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.original = subject.load_original(ORIGINAL)
        self.phase = self.original.load_phase(PHASE)
        self.identity = dict(
            training_protocol=TRAINING_PROTOCOL, implementation_commit=IMPLEMENTATION_COMMIT,
            training_seed=5202, initialization_seed=5201, initial_expanded_sha256=self.phase.INITIAL_SHA,
            fit_index_sha256=self.original.FIT_INDEX_SHA, source_manifest_sha256=subject.SOURCE_MANIFEST_SHA256,
            qualification_pair_sha256=QUALIFICATION_SHA,
        )
        self.arms = {}
        self.pair = dict(
            self.identity, status=subject.PAIR_STATUS, policy=subject.POLICY, stage="e24",
            completed_epochs=24, visits_per_arm=70560, optimizer_updates_per_arm=17640, ema_updates_per_arm=17640,
            verified_selected_mask_visits_per_arm=70560, consumed_direct_visits=45000, consumed_crop_visits=25560,
            changed_crop_visits=20000, ordered_common_fit_visits_match=True,
            shared_consumed_ledger_sha256="d" * 64, FIT=True, DEV=False, TEST=False, AP=False,
        )
        for mode, digit, mask, checkpoint in (("stepwise_center", "1", "3", "5"), ("one_pass", "2", "4", "6")):
            stream_sha = mask * 64
            arm = dict(
                status=subject.ARM_STATUS, original_os_wait=True, original_os_exit=0, resource_stop=False,
                completed_epochs=24, optimizer_updates=17640, ema_updates=17640, target_filter_sha256=self.phase.FILTER_SHA,
                filtered_empty_rows_total=0, checkpoint_sha256=checkpoint * 64,
                common_consumed_ledger_sha256=self.pair["shared_consumed_ledger_sha256"],
                verified_selected_mask_visits=70560, verified_selected_mask_stream_sha256=stream_sha,
                common_support=dict(self.identity, policy=subject.POLICY, stage="e24", mode=mode),
                phase=dict(self.phase.expected_phase(mode), protocol=TRAINING_PROTOCOL,
                           seed=5202, phase_trainer_sha256=subject.TRAINER_SHA256),
                **{key: digit * 64 for key in self.original.ARM_DIGESTS},
            )
            arm["visit_ledger_sha256"] = "e" * 64
            self.arms[mode] = arm
            self.pair[f"{mode}_checkpoint_sha256"] = checkpoint * 64
            self.pair[f"{mode}_selected_mask_stream_sha256"] = stream_sha

    def write(self, name, value):
        raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        path = self.root / (name + ".json")
        path.write_bytes(raw)
        return path, sha256(raw).hexdigest()

    def read(self, **overrides):
        pair = self.pair.copy()
        paths = {}
        for mode, arm in self.arms.items():
            path, digest = self.write(mode, arm)
            paths[f"{mode}_path"] = path
            pair[f"{mode}_arm_receipt_sha256"] = digest
        path, digest = self.write("pair", pair)
        locks = dict(training_protocol=TRAINING_PROTOCOL, implementation_commit=IMPLEMENTATION_COMMIT,
                     qualification_pair_sha256=QUALIFICATION_SHA)
        return subject.read_pair(self.original, self.phase, path, digest, **paths, **dict(locks, **overrides))

    def test_exact_pair_authorizes_both_b_c_endpoints(self):
        pair, arms = self.read()
        self.assertEqual(arms, self.arms)
        self.assertEqual(pair["stepwise_center_checkpoint_sha256"], "5" * 64)
        self.assertEqual(pair["one_pass_checkpoint_sha256"], "6" * 64)
        self.assertEqual(self.phase.TRAINING_PROTOCOL, TRAINING_PROTOCOL)

    def test_old_or_mixed_support_pair_refuses_admission(self):
        original = deepcopy(self.pair)
        for changes in (
            dict(status="HRSID_MASK_COMMON_FIT_E24_PAIR_READY_V1"),
            dict(policy="ordered_common_nonempty_prepared_rows_v1"),
            dict(one_pass_checkpoint_sha256="7" * 64),
            dict(one_pass_selected_mask_stream_sha256="8" * 64),
            dict(source_manifest_sha256="9" * 64),
            dict(qualification_pair_sha256="a" * 64),
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.pair = dict(original, **changes)
                self.read()
        self.pair = original
        self.arms["one_pass"]["common_support"]["mode"] = "stepwise_center"
        with self.assertRaisesRegex(ValueError, "fixed B/C support"):
            self.read()
        self.assertEqual(self.phase.TRAINING_PROTOCOL, self.original.FIRST_PROTOCOL)

    def test_bound_original_endpoint_rejects_other_mode_seed_and_checkpoint_sha(self):
        from unittest.mock import Mock

        _, arms = self.read()
        for mode in subject.MODES:
            with self.subTest(mode=mode):
                arm = arms[mode]
                filtered, base = self.phase.load_evaluator(FILTERED, BASE)
                base.run_export, base.run_score = Mock(), Mock()
                original_export, original_score = base.run_export, base.run_score
                subject.bind_evaluator(self.original, self.phase, base, filtered, arm)
                checkpoint = {
                    "pytorch-lightning_version": "2.6.6", "epoch": 23, "global_step": base.STEPS,
                    "args": {"seed": 5202}, self.original.COMMON_MARKER: arm["common_support"],
                    self.phase.PHASE_MARKER: arm["phase"],
                    base.ENDPOINT_KEY: dict(
                        protocol=TRAINING_PROTOCOL, epoch=24, actual_updates=base.STEPS,
                        native_index_sha256=base.FIT_INDEX_SHA, coco_sha256=base.COCO_SHA,
                        original_recipe_sha256=base.RECIPE_SHA, initial_expanded_sha256=base.INITIAL_SHA,
                        FIT_only=True, exact_restart_trajectory_verified=False, rng_state={},
                        target_filter_sha256=self.phase.FILTER_SHA, filtered_empty_rows_total=0,
                    ),
                }
                base.check_endpoint(checkpoint, TRAINING_PROTOCOL)
                for key, changes in (
                    ("args", {"seed": 5201}),
                    (self.original.COMMON_MARKER, {"policy": "ordered_common_nonempty_prepared_rows_v1"}),
                    (self.phase.PHASE_MARKER, {"mask_resize_mode": "center"}),
                ):
                    with self.assertRaises(ValueError):
                        base.check_endpoint(dict(checkpoint, **{key: dict(checkpoint[key], **changes)}),
                                            TRAINING_PROTOCOL)
                args = dict(training_protocol=TRAINING_PROTOCOL, checkpoint_sha256=arm["checkpoint_sha256"],
                            checkpoint=self.root / "checkpoint_23.ckpt")
                with self.assertRaisesRegex(ValueError, "checkpoint_23"):
                    base.run_export(SimpleNamespace(**dict(args, checkpoint=self.root / "checkpoint_7.ckpt")))
                for operation in (base.run_export, base.run_score):
                    with self.assertRaisesRegex(ValueError, "independently accepted"):
                        operation(SimpleNamespace(**dict(args, checkpoint_sha256="f" * 64)))
                original_export.assert_not_called()
                original_score.assert_not_called()
                base.run_export(SimpleNamespace(**args))
                base.run_score(SimpleNamespace(**args))
                original_export.assert_called_once()
                original_score.assert_called_once()

