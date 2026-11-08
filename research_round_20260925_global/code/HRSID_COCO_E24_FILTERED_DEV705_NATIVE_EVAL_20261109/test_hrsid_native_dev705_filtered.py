"""Synthetic qualification, marker validation, and CLI archive binding."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from hrsid_native_dev705_filtered import (
    FILTER_SHA, TRAINING_PROTOCOL, check_filtered_endpoint, load_original_evaluator,
    read_acceptance,
)

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "HRSID_COCO_E24_DEV705_NATIVE_EVAL_20261109/hrsid_native_dev705.py"


class FilteredEndpointTest(unittest.TestCase):
    def setUp(self):
        self.base = load_original_evaluator(SOURCE)
        marker = dict(
            protocol=TRAINING_PROTOCOL, epoch=24, actual_updates=self.base.STEPS,
            native_index_sha256=self.base.FIT_INDEX_SHA, coco_sha256=self.base.COCO_SHA,
            original_recipe_sha256=self.base.RECIPE_SHA, initial_expanded_sha256=self.base.INITIAL_SHA,
            FIT_only=True, exact_restart_trajectory_verified=False, rng_state={},
            target_filter_sha256=FILTER_SHA, filtered_empty_rows_total=7,
        )
        self.checkpoint = {
            "pytorch-lightning_version": "2.6.6", "epoch": 23, "global_step": self.base.STEPS,
            self.base.ENDPOINT_KEY: marker,
        }
        self.accepted = {
            "status": "HRSID_FILTERED_E24_DEV705_READY_V1",
            "training_protocol": TRAINING_PROTOCOL,
            "original_os_wait": True, "original_os_exit": 0, "resource_stop": False,
            "completed_epochs": 24, "optimizer_updates": self.base.STEPS, "ema_updates": self.base.STEPS,
            "target_filter_sha256": FILTER_SHA, "filtered_empty_rows_total": 7,
            "checkpoint_sha256": "f" * 64,
        }
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def write_receipt(self, accepted):
        raw = json.dumps(accepted).encode()
        receipt = self.root / "qualified.json"
        receipt.write_bytes(raw)
        return receipt, sha256(raw).hexdigest()

    def check(self, checkpoint, protocol=TRAINING_PROTOCOL):
        qualification = read_acceptance(*self.write_receipt(self.accepted))
        check_filtered_endpoint(checkpoint, protocol, original_check=self.base.check_endpoint,
                                qualification=qualification)

    def test_accepts_only_exact_independently_qualified_filtered_endpoint(self):
        before = deepcopy(self.checkpoint)
        self.check(self.checkpoint)
        self.assertEqual(self.checkpoint, before)
        with self.assertRaisesRegex(ValueError, "endpoint lineage"):
            self.base.check_endpoint(self.checkpoint, TRAINING_PROTOCOL)

    def test_rejects_wrong_or_missing_filter_fields_and_other_schema_changes(self):
        invalid = (
            ("target_filter_sha256", "e" * 64),
            ("filtered_empty_rows_total", 8),
            ("filtered_empty_rows_total", "7"),
            ("target_filter_sha256", None),
            ("filtered_empty_rows_total", None),
            ("native_index_sha256", "e" * 64),
            ("unexpected_field", 0),
        )
        for key, value in invalid:
            with self.subTest(key=key, value=value):
                checkpoint = deepcopy(self.checkpoint)
                marker = checkpoint[self.base.ENDPOINT_KEY]
                if value is None:
                    del marker[key]
                else:
                    marker[key] = value
                with self.assertRaises(ValueError):
                    self.check(checkpoint)
        with self.assertRaisesRegex(ValueError, "training protocol"):
            self.check(self.checkpoint, "c" * 40)
        with self.assertRaisesRegex(ValueError, "fixed-budget archive"):
            self.check(dict(self.checkpoint, global_step=self.base.STEPS - 1))

    def test_rejects_bad_postterminal_receipts(self):
        invalid = (
            dict(training_protocol="c" * 40), dict(target_filter_sha256="e" * 64),
            dict(original_os_wait=1), dict(original_os_exit=False), dict(resource_stop=True),
            dict(completed_epochs=23), dict(optimizer_updates=self.base.STEPS - 1),
            dict(ema_updates=self.base.STEPS - 1), dict(filtered_empty_rows_total=True),
            dict(filtered_empty_rows_total=0), dict(checkpoint_sha256="bad"),
            dict(unexpected_field=0),
        )
        for update in invalid:
            with self.subTest(update=update):
                with self.assertRaises(ValueError):
                    read_acceptance(*self.write_receipt(dict(self.accepted, **update)))
        receipt, _ = self.write_receipt(self.accepted)
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            read_acceptance(receipt, "a" * 64)

    def test_export_and_score_reject_archive_other_than_accepted_digest(self):
        receipt, digest = self.write_receipt(self.accepted)
        common = [
            sys.executable, "-B", str(HERE / "hrsid_native_dev705_filtered.py"),
            "--base-source", str(SOURCE), "--acceptance", str(receipt),
            "--acceptance-sha256", digest,
        ]
        for mode, extra in (
            ("export", ["--checkpoint", str(self.root / "missing.ckpt"), "--archive", str(self.root / "missing.zip")]),
            ("score", ["--export", str(self.root / "missing"), "--export-report-sha256", "c" * 64]),
        ):
            with self.subTest(mode=mode):
                result = subprocess.run(
                    common + [mode, "--protocol", "a" * 40, "--training-protocol", TRAINING_PROTOCOL,
                              "--checkpoint-sha256", "e" * 64, "--output", str(self.root / "fresh")] + extra,
                    capture_output=True, text=True, timeout=30,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("independently accepted archive SHA-256", result.stderr)
                self.assertFalse((self.root / "fresh").exists())


if __name__ == "__main__":
    unittest.main()

