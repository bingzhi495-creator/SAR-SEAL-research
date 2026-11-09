"""Synthetic tests for seed-qualified DEV705 admission, with no real DEV inputs."""
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


HERE = Path(__file__).resolve().parent
ENTRY = HERE / "hrsid_mask_phase_repeat_dev705.py"
CODE = HERE.parent
P203 = CODE / "HRSID_MASK_PHASE_E24_DEV705_20261109/hrsid_mask_phase_dev705.py"
P153 = CODE / "HRSID_COCO_E24_FILTERED_DEV705_NATIVE_EVAL_20261109/hrsid_native_dev705_filtered.py"
P149 = CODE / "HRSID_COCO_E24_DEV705_NATIVE_EVAL_20261109/hrsid_native_dev705.py"
REPEAT_PROTOCOL = "a49361b92977a605267025b1e7798420936cbe53"
REPEAT_TRAINER_SHA = "42a174573a04c529dfbae943bcc88ef4ed5d793d7eda6766511396034fda321f"
FILTER_SHA = "3b164cbf8c104539eabee8775c8e1f297d9a1607e37473c4d7a4a9b561f93471"
INITIAL_SHA = "63ff93afdd977ba79b5b5a46ba8b4839ef6c40eb0722b29e4667d4932404fbc2"

SPEC = importlib.util.spec_from_file_location("hrsid_mask_phase_repeat_dev705", ENTRY)
BRIDGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BRIDGE)


def write_json(path, obj):
    raw = json.dumps(obj).encode()
    path.write_bytes(raw)
    return sha256(raw).hexdigest()


def arm(mode, seed):
    return dict(
        status="HRSID_MASK_PHASE_E24_ARM_ACCEPTED_V1",
        phase=dict(protocol=REPEAT_PROTOCOL, mask_resize_mode=mode, stage="e24", seed=seed,
                   phase_trainer_sha256=REPEAT_TRAINER_SHA, initial_expanded_sha256=INITIAL_SHA,
                   phase_patch_sha256="811eccf2b42319272aea8b3f50a71bc029734d006bba43260f66117e9421948e",
                   native_trainer_sha256="11d17cab878d62576b8d19e6231caab90c9990c03909cb01e1d04468add452a6"),
        original_os_wait=True, original_os_exit=0, resource_stop=False,
        completed_epochs=24, optimizer_updates=17640, ema_updates=17640,
        target_filter_sha256=FILTER_SHA, filtered_empty_rows_total=2 if mode == "native" else 0,
        checkpoint_sha256=("1" if mode == "native" else "2") * 64,
        visit_ledger_sha256="3" * 64, result_sha256="4" * 64, stdout_sha256="5" * 64,
        terminal_sha256="6" * 64, phase_provenance_sha256="7" * 64,
        repeat_seed_witness_sha256=("8" if mode == "native" else "9") * 64,
    )


class RepeatDev705Test(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.pair_path, self.native_path, self.center_path = (self.root / name for name in (
            "pair.json", "native.json", "center.json"))
        self.native, self.center = arm("native", 5202), arm("center", 5202)

    def seal(self, **changes):
        pair = dict(
            status="HRSID_MASK_PHASE_E24_PAIR_READY_V1", training_protocol=REPEAT_PROTOCOL, seed=5202,
            ordered_fit_visits_match=True, completed_epochs=24, visits_per_arm=70560,
            native_arm_receipt_sha256=write_json(self.native_path, self.native),
            center_arm_receipt_sha256=write_json(self.center_path, self.center),
            native_checkpoint_sha256=self.native["checkpoint_sha256"],
            center_checkpoint_sha256=self.center["checkpoint_sha256"],
        )
        pair.update(changes)
        return write_json(self.pair_path, pair)

    def read(self, pair_sha, *, seed=5202, mode="center"):
        return BRIDGE.read_pair(BRIDGE.load_phase(P203), self.pair_path, pair_sha, native_path=self.native_path,
                                center_path=self.center_path, mode=mode, seed=seed)

    def test_exact_repeat_accepts_zero_filter_rows_and_rejects_cross_seed_peer(self):
        self.assertEqual(self.read(self.seal()), self.center)
        self.native["phase"]["seed"] = 5203
        with self.assertRaisesRegex(ValueError, "arm training phase"):
            self.read(self.seal())
        self.center["phase"]["seed"] = 5203
        self.assertEqual(self.read(self.seal(seed=5203), seed=5203, mode="native"), self.native)

    def test_rejects_old_receipts_and_unbound_witnesses(self):
        self.seal()
        pair = json.loads(self.pair_path.read_bytes())
        del pair["seed"]
        with self.assertRaisesRegex(ValueError, "repeat pair seed"):
            self.read(write_json(self.pair_path, pair))
        del self.native["repeat_seed_witness_sha256"]
        with self.assertRaisesRegex(ValueError, "witness SHA-256"):
            self.read(self.seal())
        self.native["repeat_seed_witness_sha256"] = self.center["repeat_seed_witness_sha256"]
        with self.assertRaisesRegex(ValueError, "Distinct original repeat seed witnesses"):
            self.read(self.seal())

    def test_requires_exact_source_and_pair_bytes(self):
        changed = self.root / "changed.py"
        changed.write_bytes(P203.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "immutable P203"):
            BRIDGE.load_phase(changed)
        digest = self.seal()
        self.center_path.write_bytes(self.center_path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.read(digest)

    def test_checkpoint_still_uses_original_filter_and_endpoint_checks(self):
        phase = BRIDGE.load_phase(P203)
        selected = self.read(self.seal())
        filtered, base = phase.load_evaluator(P153, P149)
        BRIDGE.bind_evaluator(phase, base, filtered, selected, seed=5202)
        checkpoint = dict(
            epoch=23, global_step=17640, args={"seed": 5202}, **{"pytorch-lightning_version": "2.6.6"},
        )
        checkpoint[phase.PHASE_MARKER] = deepcopy(selected["phase"])
        checkpoint[base.ENDPOINT_KEY] = dict(
            protocol=REPEAT_PROTOCOL, epoch=24, actual_updates=17640, native_index_sha256=base.FIT_INDEX_SHA,
            coco_sha256=base.COCO_SHA, original_recipe_sha256=base.RECIPE_SHA,
            initial_expanded_sha256=base.INITIAL_SHA, FIT_only=True, exact_restart_trajectory_verified=False,
            rng_state={}, target_filter_sha256=FILTER_SHA, filtered_empty_rows_total=0,
        )
        before = deepcopy(checkpoint)
        base.check_endpoint(checkpoint, REPEAT_PROTOCOL)
        self.assertEqual(checkpoint, before)
        with self.assertRaisesRegex(ValueError, "checkpoint args.seed"):
            base.check_endpoint(dict(checkpoint, args={"seed": 5203}), REPEAT_PROTOCOL)
        checkpoint[base.ENDPOINT_KEY]["filtered_empty_rows_total"] = True
        with self.assertRaisesRegex(ValueError, "filter SHA-256 and independent row count"):
            base.check_endpoint(checkpoint, REPEAT_PROTOCOL)

    def command(self, pair_sha, operation, *, seed=5202, base=P149, checkpoint_sha="1" * 64,
                training_protocol=REPEAT_PROTOCOL):
        missing = self.root / "missing"
        paths = (["--archive", str(missing), "--checkpoint", str(missing)] if operation == "export" else
                 ["--export", str(missing), "--export-report-sha256", "a" * 64])
        return subprocess.run(
            [sys.executable, "-I", "-B", str(ENTRY), "--phase-source", str(P203), "--filtered-source", str(P153),
             "--base-source", str(base), "--pair-receipt", str(self.pair_path), "--pair-receipt-sha256", pair_sha,
             "--native-arm-receipt", str(self.native_path), "--center-arm-receipt", str(self.center_path),
             "--arm", "center", "--training-seed", str(seed), operation, "--protocol", "a" * 40,
             "--training-protocol", training_protocol, "--checkpoint-sha256", checkpoint_sha,
             "--output", str(self.root / "fresh"), *paths],
            capture_output=True, text=True, timeout=30,
        )

    def test_export_and_score_fail_before_development_inputs(self):
        digest = self.seal()
        for operation in ("export", "score"):
            with self.subTest(operation=operation):
                wrong_arm = self.command(digest, operation)
                self.assertNotEqual(wrong_arm.returncode, 0)
                self.assertIn("independently accepted arm archive SHA-256", wrong_arm.stderr)
                wrong_seed = self.command(digest, operation, seed=5203, base=self.root / "missing-source")
                self.assertNotEqual(wrong_seed.returncode, 0)
                self.assertIn("repeat pair seed", wrong_seed.stderr)
                old_protocol = self.command(digest, operation, checkpoint_sha=self.center["checkpoint_sha256"],
                                            training_protocol="d66baf3c57fafdba4242ad1b55e08cf76bfd77f7")
                self.assertNotEqual(old_protocol.returncode, 0)
                self.assertIn("Exact paired protocol", old_protocol.stderr)
                self.assertFalse((self.root / "fresh").exists())


if __name__ == "__main__":
    unittest.main()
