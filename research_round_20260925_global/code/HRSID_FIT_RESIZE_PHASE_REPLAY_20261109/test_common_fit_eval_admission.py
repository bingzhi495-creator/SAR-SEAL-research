from copy import deepcopy
from hashlib import sha256
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import common_fit_eval_admission as bridge


CODE = Path(__file__).resolve().parent.parent
P203 = CODE / "HRSID_MASK_PHASE_E24_DEV705_20261109/hrsid_mask_phase_dev705.py"
P153 = CODE / "HRSID_COCO_E24_FILTERED_DEV705_NATIVE_EVAL_20261109/hrsid_native_dev705_filtered.py"
P149 = CODE / "HRSID_COCO_E24_DEV705_NATIVE_EVAL_20261109/hrsid_native_dev705.py"
PROTOCOL = "a" * 40
INITIAL_SHA = "63ff93afdd977ba79b5b5a46ba8b4839ef6c40eb0722b29e4667d4932404fbc2"
FIT_SHA = "e95cf26cc60a2c6fb651e907541380a256f2df7b1e153244a6c0f803b0b19f8f"
FILTER_SHA = "3b164cbf8c104539eabee8775c8e1f297d9a1607e37473c4d7a4a9b561f93471"
IDENTITY = dict(
    training_protocol=PROTOCOL, implementation_commit="b" * 40, training_seed=5202, initialization_seed=5201,
    initial_expanded_sha256=INITIAL_SHA, fit_index_sha256=FIT_SHA,
    source_manifest_sha256="c" * 64, qualification_pair_sha256="d" * 64,
)


def write(path, value):
    raw = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    path.write_bytes(raw)
    return sha256(raw).hexdigest()


def arm(mode):
    center = mode == "center"
    return dict(
        status="HRSID_MASK_COMMON_FIT_E24_ARM_ACCEPTED_V1",
        phase=dict(
            protocol=PROTOCOL, mask_resize_mode=mode, stage="e24", seed=5202,
            initial_expanded_sha256=INITIAL_SHA, phase_trainer_sha256="e" * 64,
            phase_patch_sha256="811eccf2b42319272aea8b3f50a71bc029734d006bba43260f66117e9421948e",
            native_trainer_sha256="11d17cab878d62576b8d19e6231caab90c9990c03909cb01e1d04468add452a6",
        ),
        common_support=dict(**IDENTITY, policy="ordered_common_nonempty_prepared_rows_v1", mode=mode, stage="e24"),
        original_os_wait=True, original_os_exit=0, resource_stop=False,
        completed_epochs=24, optimizer_updates=17640, ema_updates=17640,
        target_filter_sha256=FILTER_SHA, filtered_empty_rows_total=0,
        checkpoint_sha256=("1" if center else "2") * 64, visit_ledger_sha256="3" * 64,
        result_sha256="4" * 64, stdout_sha256="5" * 64, terminal_sha256=("6" if center else "7") * 64,
        phase_provenance_sha256="8" * 64, repeat_seed_witness_sha256=("9" if center else "a") * 64,
        common_audit_sha256=("b" if center else "c") * 64, common_consumed_ledger_sha256="f" * 64,
    )


class CommonFitDevAdmissionTest(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.pair_path, self.native_path, self.center_path = (
            self.root / name for name in ("pair.json", "native.json", "center.json")
        )
        self.native, self.center = arm("native"), arm("center")
        self.phase = bridge.load_phase(P203)

    def seal(self, **changes):
        pair = dict(
            **IDENTITY, status="HRSID_MASK_COMMON_FIT_E24_PAIR_READY_V1",
            ordered_common_fit_visits_match=True, completed_epochs=24, visits_per_arm=70560,
            optimizer_updates_per_arm=17640, ema_updates_per_arm=17640,
            native_arm_receipt_sha256=write(self.native_path, self.native),
            center_arm_receipt_sha256=write(self.center_path, self.center),
            native_checkpoint_sha256=self.native["checkpoint_sha256"],
            center_checkpoint_sha256=self.center["checkpoint_sha256"],
            shared_consumed_ledger_sha256="f" * 64, changed_mask_visits=29,
            FIT=True, DEV=False, TEST=False, AP=False,
        )
        pair.update(changes)
        return write(self.pair_path, pair)

    def read(self, digest, *, mode="center", training_protocol=PROTOCOL):
        return bridge.read_pair(self.phase, self.pair_path, digest, native_path=self.native_path,
                                center_path=self.center_path, mode=mode, training_protocol=training_protocol)

    def checkpoint(self, base, arm):
        checkpoint = dict(epoch=23, global_step=17640, args={"seed": 5202}, **{"pytorch-lightning_version": "2.6.6"})
        checkpoint[self.phase.PHASE_MARKER] = deepcopy(arm["phase"])
        checkpoint["sar_seal_hrsid_mask_common_support_v1"] = deepcopy(arm["common_support"])
        checkpoint[base.ENDPOINT_KEY] = dict(
            protocol=PROTOCOL, epoch=24, actual_updates=17640, native_index_sha256=base.FIT_INDEX_SHA,
            coco_sha256=base.COCO_SHA, original_recipe_sha256=base.RECIPE_SHA, initial_expanded_sha256=INITIAL_SHA,
            FIT_only=True, exact_restart_trajectory_verified=False, rng_state={},
            target_filter_sha256=FILTER_SHA, filtered_empty_rows_total=0,
        )
        return checkpoint

    def test_same_seed_closed_pair_preserves_original_endpoint_and_gt(self):
        selected = self.read(self.seal())
        self.assertEqual(selected, self.center)
        filtered, base = self.phase.load_evaluator(P153, P149)
        bridge.bind_evaluator(self.phase, base, filtered, selected)
        checkpoint = self.checkpoint(base, selected)
        before = deepcopy(checkpoint)

        base.check_endpoint(checkpoint, PROTOCOL)

        self.assertEqual(checkpoint, before)
        self.assertEqual(base.DEV_IMAGES, 705)
        self.assertEqual(base.GT_COUNT, 1782)
        self.assertEqual(base.EMPTY_COUNT, 4)


    def test_pair_and_arm_protocol_seed_and_population_are_exact(self):
        for mutation in (
            {"training_seed": 5203},
            {"training_seed": 5202.0},
            {"initialization_seed": 5202},
            {"FIT": 1},
            {"DEV": True},
            {"ordered_common_fit_visits_match": False},
        ):
            with self.subTest(mutation=mutation):
                with self.assertRaisesRegex(ValueError, "common FIT E24 pair"):
                    self.read(self.seal(**mutation))
        with self.assertRaisesRegex(ValueError, "externally locked"):
            self.read(self.seal(), training_protocol="a49361b92977a605267025b1e7798420936cbe53")
        with self.assertRaisesRegex(ValueError, "actual mask changes"):
            self.read(self.seal(changed_mask_visits=0))

    def test_peer_receipt_and_shared_ledger_must_match_authorized_pair(self):
        digest = self.seal()
        self.center_path.write_bytes(self.center_path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.read(digest, mode="native")
        self.center["common_consumed_ledger_sha256"] = "9" * 64
        with self.assertRaisesRegex(ValueError, "common FIT E24 arm"):
            self.read(self.seal(), mode="native")

    def test_common_marker_binds_source_qualification_and_mode(self):
        for field, wrong in (
            ("source_manifest_sha256", "0" * 64),
            ("qualification_pair_sha256", "0" * 64),
            ("mode", "native"),
            ("policy", "different_support"),
        ):
            original = self.center["common_support"][field]
            self.center["common_support"][field] = wrong
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "common-support policy"):
                self.read(self.seal(), mode="native")
            self.center["common_support"][field] = original

    def test_checkpoint_rejects_wrong_saved_seed_common_marker_and_filter(self):
        selected = self.read(self.seal())
        filtered, base = self.phase.load_evaluator(P153, P149)
        bridge.bind_evaluator(self.phase, base, filtered, selected)
        pristine = self.checkpoint(base, selected)

        wrong = deepcopy(pristine)
        wrong["args"]["seed"] = 5203
        with self.assertRaisesRegex(ValueError, "checkpoint args.seed"):
            base.check_endpoint(wrong, PROTOCOL)
        wrong = deepcopy(pristine)
        wrong["sar_seal_hrsid_mask_common_support_v1"]["qualification_pair_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "common-support checkpoint marker"):
            base.check_endpoint(wrong, PROTOCOL)
        wrong = deepcopy(pristine)
        wrong[base.ENDPOINT_KEY]["filtered_empty_rows_total"] = 1
        with self.assertRaisesRegex(ValueError, "filter SHA-256 and independent row count"):
            base.check_endpoint(wrong, PROTOCOL)

    def test_export_exact_name_and_digest_are_checked_before_original_worker(self):
        selected = self.read(self.seal())
        filtered, base = self.phase.load_evaluator(P153, P149)
        calls = []
        base.run_export = calls.append
        base.run_score = calls.append
        bridge.bind_evaluator(self.phase, base, filtered, selected)
        args = SimpleNamespace(checkpoint=self.root / "checkpoint_7.ckpt",
                               checkpoint_sha256=selected["checkpoint_sha256"], training_protocol=PROTOCOL)
        with self.assertRaisesRegex(ValueError, "checkpoint_23.ckpt filename"):
            base.run_export(args)
        args.checkpoint = self.root / "checkpoint_23.ckpt"
        args.checkpoint_sha256 = "0" * 64
        for action in (base.run_export, base.run_score):
            with self.assertRaisesRegex(ValueError, "independently accepted arm archive SHA-256"):
                action(args)
        self.assertEqual(calls, [])
        args.checkpoint_sha256 = selected["checkpoint_sha256"]
        base.run_export(args)
        base.run_score(args)
        self.assertEqual(calls, [args, args])

    def test_binds_exact_original_gt_and_original_phase_source(self):
        changed = self.root / "modified.py"
        changed.write_bytes(P203.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "immutable P203"):
            bridge.load_phase(changed)
        selected = self.read(self.seal())
        filtered, base = self.phase.load_evaluator(P153, P149)
        base.FILES["DEV_ORIGINAL_GT_PRIVATE.json"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "original DEV705 population, ground truth"):
            bridge.bind_evaluator(self.phase, base, filtered, selected)


    def test_cli_forwards_training_protocol_and_rejects_wrong_checkpoint_before_dev_access(self):
        digest = self.seal()
        command = [
            sys.executable, "-I", "-B", "-X", "utf8", str(Path(bridge.__file__).resolve()),
            "--phase-source", str(P203), "--filtered-source", str(P153), "--base-source", str(P149),
            "--pair-receipt", str(self.pair_path), "--pair-receipt-sha256", digest,
            "--native-arm-receipt", str(self.native_path), "--center-arm-receipt", str(self.center_path),
            "--arm", "center", "export", "--protocol", "f" * 40, "--training-protocol", PROTOCOL,
            "--checkpoint-sha256", self.center["checkpoint_sha256"], "--output", str(self.root / "fresh"),
            "--archive", str(self.root / "absent.zip"), "--checkpoint", str(self.root / "checkpoint_7.ckpt"),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=20,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("checkpoint_23.ckpt filename", result.stderr)
        self.assertFalse((self.root / "fresh").exists())


if __name__ == "__main__":
    unittest.main()
