"""Synthetic A/B admission tests. No scientific files are opened."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from hrsid_mask_phase_dev705 import (
    FILTER_SHA, INITIAL_SHA, PHASE_MARKER, TRAINING_PROTOCOL,
    bind_evaluator, load_evaluator, read_pair,
)

HERE = Path(__file__).resolve().parent
P149 = HERE.parent / "HRSID_COCO_E24_DEV705_NATIVE_EVAL_20261109/hrsid_native_dev705.py"
P153 = HERE.parent / "HRSID_COCO_E24_FILTERED_DEV705_NATIVE_EVAL_20261109/hrsid_native_dev705_filtered.py"


class PairedAdmissionTest(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.native_path = self.root / "native.json"
        self.center_path = self.root / "center.json"
        self.pair_path = self.root / "pair.json"
        self.native = self.arm("native", "1" * 64, 2)
        self.center = self.arm("center", "2" * 64, 0)

    def arm(self, mode, checkpoint, count):
        return dict(
            status="HRSID_MASK_PHASE_E24_ARM_ACCEPTED_V1",
            phase=dict(protocol=TRAINING_PROTOCOL, mask_resize_mode=mode, stage="e24", seed=5201,
                       phase_trainer_sha256="8550b43791a6cb6f53736dc7e64b9c5c6204723cf4c07c3b112491ba1bd74043",
                       phase_patch_sha256="811eccf2b42319272aea8b3f50a71bc029734d006bba43260f66117e9421948e",
                       native_trainer_sha256="11d17cab878d62576b8d19e6231caab90c9990c03909cb01e1d04468add452a6",
                       initial_expanded_sha256=INITIAL_SHA),
            original_os_wait=True, original_os_exit=0, resource_stop=False,
            completed_epochs=24, optimizer_updates=17640, ema_updates=17640,
            target_filter_sha256=FILTER_SHA, filtered_empty_rows_total=count,
            checkpoint_sha256=checkpoint, visit_ledger_sha256="3" * 64, result_sha256="4" * 64,
            stdout_sha256="5" * 64, terminal_sha256="6" * 64, phase_provenance_sha256="7" * 64,
        )

    def write(self, path, obj):
        raw = json.dumps(obj).encode()
        path.write_bytes(raw)
        return sha256(raw).hexdigest()

    def seal(self, **changes):
        pair = dict(
            status="HRSID_MASK_PHASE_E24_PAIR_READY_V1", training_protocol=TRAINING_PROTOCOL,
            ordered_fit_visits_match=True, completed_epochs=24, visits_per_arm=70560,
            native_arm_receipt_sha256=self.write(self.native_path, self.native),
            center_arm_receipt_sha256=self.write(self.center_path, self.center),
            native_checkpoint_sha256=self.native["checkpoint_sha256"],
            center_checkpoint_sha256=self.center["checkpoint_sha256"],
        )
        pair.update(changes)
        return self.write(self.pair_path, pair)

    def read(self, digest, mode="center"):
        return read_pair(self.pair_path, digest, native_path=self.native_path,
                         center_path=self.center_path, mode=mode)

    def test_accepts_center_zero_filter_rows_only_after_both_receipts(self):
        digest = self.seal()
        self.assertEqual(self.read(digest), self.center)
        self.assertEqual(self.read(digest, "native"), self.native)
        self.native_path.unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.read(digest)



    def test_exact_phase_marker_and_legacy_endpoint_at_zero_count(self):
        arm = self.read(self.seal())
        filtered, base = load_evaluator(P153, P149)
        original_endpoint = base.check_endpoint
        bind_evaluator(base, filtered, arm)
        checkpoint = {
            "pytorch-lightning_version": "2.6.6", "epoch": 23, "global_step": base.STEPS,
            PHASE_MARKER: deepcopy(arm["phase"]),
            base.ENDPOINT_KEY: dict(
                protocol=TRAINING_PROTOCOL, epoch=24, actual_updates=base.STEPS,
                native_index_sha256=base.FIT_INDEX_SHA, coco_sha256=base.COCO_SHA,
                original_recipe_sha256=base.RECIPE_SHA, initial_expanded_sha256=base.INITIAL_SHA,
                FIT_only=True, exact_restart_trajectory_verified=False, rng_state={},
                target_filter_sha256=FILTER_SHA, filtered_empty_rows_total=0,
            ),
        }
        before = deepcopy(checkpoint)
        base.check_endpoint(checkpoint, TRAINING_PROTOCOL)
        self.assertEqual(checkpoint, before)
        with self.assertRaisesRegex(ValueError, "endpoint lineage"):
            original_endpoint(checkpoint, TRAINING_PROTOCOL)

        wrong_mode = deepcopy(checkpoint)
        wrong_mode[PHASE_MARKER]["mask_resize_mode"] = "native"
        wrong_filter_count = deepcopy(checkpoint)
        wrong_filter_count[base.ENDPOINT_KEY]["filtered_empty_rows_total"] = True
        for invalid in (wrong_mode, wrong_filter_count, dict(checkpoint, global_step=735)):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    base.check_endpoint(invalid, TRAINING_PROTOCOL)

    def test_wrong_provenance_or_bad_peer_receipt_blocks_both_arms(self):
        original = deepcopy(self.native)
        for mutation in (
            dict(filtered_empty_rows_total=-1),
            dict(filtered_empty_rows_total=True),
            dict(original_os_exit=False),
            dict(ema_updates=17639),
            dict(phase=dict(original["phase"], phase_patch_sha256="e" * 64)),
            dict(phase=dict(original["phase"], mask_resize_mode="center")),
        ):
            with self.subTest(mutation=mutation):
                self.native = dict(original, **mutation)
                with self.assertRaises(ValueError):
                    self.read(self.seal(), "center")
        self.native = original
        with self.assertRaises(ValueError):
            self.read(self.seal(ordered_fit_visits_match=False))
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            digest = self.seal()
            self.native_path.write_bytes(self.native_path.read_bytes() + b" ")
            self.read(digest, "center")

    def test_export_and_score_reject_the_other_accepted_arm_checkpoint(self):
        digest = self.seal()
        for command, paths in (
            ("export", ["--checkpoint", str(self.root / "missing.ckpt"),
                        "--archive", str(self.root / "missing.zip")]),
            ("score", ["--export", str(self.root / "missing_export"), "--export-report-sha256", "f" * 64]),
        ):
            with self.subTest(command=command):
                output = self.root / "fresh"
                args = [
                    sys.executable, "-I", "-B", str(HERE / "hrsid_mask_phase_dev705.py"),
                    "--filtered-source", str(P153), "--base-source", str(P149),
                    "--pair-receipt", str(self.pair_path), "--pair-receipt-sha256", digest,
                    "--native-arm-receipt", str(self.native_path), "--center-arm-receipt", str(self.center_path),
                    "--arm", "center", command, "--protocol", "a" * 40, "--training-protocol", TRAINING_PROTOCOL,
                    "--checkpoint-sha256", self.native["checkpoint_sha256"], "--output", str(output), *paths,
                ]
                outcome = subprocess.run(args, capture_output=True, text=True, timeout=30)
                self.assertNotEqual(outcome.returncode, 0)
                self.assertIn("independently accepted arm archive SHA-256", outcome.stderr)
                self.assertFalse(output.exists())
if __name__ == "__main__":
    unittest.main()
