"""Prospective guards for B/C long training, without FIT or Torch access."""

from hashlib import sha256
import json
from os import environ
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import one_pass_common_e24 as subject


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def reference(path):
    raw = path.read_bytes()
    return dict(path=path.as_posix(), bytes=len(raw), sha256=sha256(raw).hexdigest())


def make_inputs(directory, mode="one_pass"):
    root = Path(environ["P256_TEST_COMMON_SOURCE"]).resolve(strict=True)
    engine_source = root / "common_fit_train.py"
    short = root / "one_pass_common_qualification.py"
    engine, qualifier, short_pins = subject.configured_engine(engine_source)
    sources = {name: dict(path=(directory / name).as_posix(), bytes=1, sha256=digest)
               for name, digest in short_pins.items()}
    native = Path(sources[engine.NATIVE]["path"])
    sources[engine_source.name] = reference(engine_source)
    sources[short.name] = reference(short)
    sources[subject.SELF] = reference(Path(subject.__file__).resolve(strict=True))
    short_protocol, short_commit, e24_protocol, e24_commit = ("1" * 40, "2" * 40, "3" * 40, "4" * 40)
    short_manifest = sha256(json.dumps(short_pins, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    pair = dict(status="HRSID_ONE_PASS_COMMON_FIT_SHORT_PAIR_ACCEPTED_V1", stage="qualification",
                policy=qualifier.POLICY, training_protocol=short_protocol, implementation_commit=short_commit,
                training_seed=5202, initialization_seed=5201, initial_expanded_sha256=engine.INITIAL_EXPANDED_SHA,
                fit_index_sha256=engine.FIT_INDEX_SHA, source_manifest_sha256=short_manifest,
                ordered_common_fit_visits_match=True, visits_per_arm=32, optimizer_updates_per_arm=8,
                ema_updates_per_arm=8, verified_selected_mask_visits_per_arm=32, changed_mask_visits=7,
                consumed_direct_visits=20, consumed_crop_visits=12, changed_crop_visits=7,
                stepwise_center_arm_receipt_sha256="1" * 64, one_pass_arm_receipt_sha256="2" * 64,
                stepwise_center_terminal_sha256="3" * 64, one_pass_terminal_sha256="4" * 64,
                stepwise_center_selected_mask_stream_sha256="5" * 64, one_pass_selected_mask_stream_sha256="6" * 64,
                shared_consumed_ledger_sha256="7" * 64, FIT=True, DEV=False, TEST=False, AP=False)
    pair_path = directory / "synthetic-pair.json"
    pair_path.write_bytes(encoded(pair))
    pair_ref = reference(pair_path)
    output, audit = directory / "native-output", directory / "audit-output"
    lock = dict(status="HRSID_ONE_PASS_COMMON_FIT_E24_LOCK_V1", training_protocol=e24_protocol, stage="e24",
                mode=mode, training_seed=5202, initialization_seed=5201,
                initial_expanded_sha256=engine.INITIAL_EXPANDED_SHA, fit_index_sha256=engine.FIT_INDEX_SHA,
                native_output=output.as_posix(), audit_output=audit.as_posix(), implementation_commit=e24_commit,
                sources=sources, qualification_pair_ref=pair_ref)
    lock_path = directory / "synthetic-lock.json"
    lock_path.write_bytes(encoded(lock))
    args = SimpleNamespace(engine_source=engine_source, native_source=native, stage="e24", mode=mode,
                           training_seed=5202, output=output, audit_output=audit, protocol=e24_protocol,
                           expected_initial_sha256=engine.INITIAL_EXPANDED_SHA, training_lock=lock_path,
                           training_lock_sha256=reference(lock_path)["sha256"], qualification_protocol=short_protocol,
                           qualification_implementation_commit=short_commit, qualification_pair_sha256=pair_ref["sha256"])
    return args, lock, pair


class OnePassE24AdmissionTest(unittest.TestCase):
    def test_exact_short_pair_admits_long_stage_without_importing_torch(self):
        for mode in ("stepwise_center", "one_pass"):
            with self.subTest(mode=mode), TemporaryDirectory() as directory:
                args, lock, _ = make_inputs(Path(directory).resolve(), mode)
                engine, qualifier, admitted, digest = subject.admit_training(args)
                self.assertEqual(admitted, lock)
                self.assertEqual(digest, args.qualification_pair_sha256)
                self.assertEqual(engine.SELF, subject.SELF)
                self.assertEqual(engine.POLICY, qualifier.POLICY)
                self.assertEqual(qualifier.SELF, "one_pass_common_qualification.py")
                self.assertNotIn("torch", sys.modules)


    def test_wrong_stage_mode_seed_or_short_digest_fails_before_runtime(self):
        cases = (
            ("stage", "qualification", "Only prospective"),
            ("mode", "native", "Only prospective"),
            ("training_seed", 5201, "Only prospective"),
            ("qualification_pair_sha256", "0" * 64, "preaccepted"),
        )
        for attribute, value, reason in cases:
            with self.subTest(attribute=attribute), TemporaryDirectory() as directory:
                args, _, _ = make_inputs(Path(directory).resolve())
                setattr(args, attribute, value)
                with patch.object(subject, "_run_training") as train:
                    with self.assertRaisesRegex(RuntimeError, reason):
                        subject.run_locked(args)
                    train.assert_not_called()
                self.assertNotIn("torch", sys.modules)

    def test_rehashed_short_receipts_still_need_exact_roster_and_real_branches(self):
        alterations = (
            ("source_manifest_sha256", "0" * 64, "source, arm"),
            ("consumed_direct_visits", 19, "direct and changed crop"),
            ("one_pass_selected_mask_stream_sha256", "5" * 64, "source, arm"),
            ("AP", True, "source, arm"),
        )
        for field, replacement, reason in alterations:
            with self.subTest(field=field), TemporaryDirectory() as directory:
                args, lock, pair = make_inputs(Path(directory).resolve())
                pair[field] = replacement
                pair_path = Path(lock["qualification_pair_ref"]["path"])
                pair_path.write_bytes(encoded(pair))
                lock["qualification_pair_ref"] = reference(pair_path)
                args.training_lock.write_bytes(encoded(lock))
                args.training_lock_sha256 = reference(args.training_lock)["sha256"]
                args.qualification_pair_sha256 = lock["qualification_pair_ref"]["sha256"]
                with self.assertRaisesRegex(RuntimeError, reason):
                    subject.admit_training(args)
                self.assertNotIn("torch", sys.modules)

    def test_short_training_protocol_cannot_be_reused_as_the_e24_protocol(self):
        with TemporaryDirectory() as directory:
            args, lock, _ = make_inputs(Path(directory).resolve())
            args.protocol = lock["training_protocol"] = args.qualification_protocol
            args.training_lock.write_bytes(encoded(lock))
            args.training_lock_sha256 = reference(args.training_lock)["sha256"]
            with self.assertRaisesRegex(RuntimeError, "distinct protocol and implementation"):
                subject.admit_training(args)

    def test_original_runtime_guard_and_fresh_output_gate_precede_training(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            args, _, _ = make_inputs(root)
            admitted = subject.admit_training(args)
            engine = admitted[0]
            engine.PRIVATE = root
            engine.RELEASE = root / "ordinary"
            engine.FIT_LAYOUT = root / "fit"
            with (patch.object(subject, "admit_training", return_value=admitted),
                  patch.object(subject, "_run_training", return_value="E24 dispatch") as train,
                  patch.dict(environ, CUDA_VISIBLE_DEVICES="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")):
                self.assertEqual(subject.run_locked(args), "E24 dispatch")
                train.assert_called_once_with(args, *admitted)
                train.reset_mock()
                args.output.mkdir()
                with self.assertRaisesRegex(RuntimeError, "Fresh canonical private output"):
                    subject.run_locked(args)
                train.assert_not_called()


if __name__ == "__main__":
    unittest.main()
