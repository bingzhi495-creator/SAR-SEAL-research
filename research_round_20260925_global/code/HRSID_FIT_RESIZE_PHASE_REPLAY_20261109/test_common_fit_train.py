from contextlib import contextmanager, redirect_stderr
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest


ENTRY = Path(__file__).with_name("common_fit_train.py")


class CommonFitTrainTests(unittest.TestCase):
    def test_unlocked_training_seed_is_refused_before_any_input_is_opened(self):
        entry = runpy.run_path(str(ENTRY))
        arguments = ["--stage", "qualification", "--mode", "native", "--training-seed", "5203",
                     "--training-lock", "missing.json", "--training-lock-sha256", "0" * 64,
                     "--output", "missing-native", "--audit-output", "missing-audit",
                     "--native-source", "missing-native.py", "--protocol", "1" * 40,
                     "--expected-initial-sha256", "2" * 64]
        errors = StringIO()
        with redirect_stderr(errors), self.assertRaises(SystemExit) as error:
            entry["main"](arguments)
        self.assertEqual(error.exception.code, 2)
        self.assertIn("invalid choice: 5203", errors.getvalue())

    def test_full_training_requires_prior_short_pair_before_source_loading(self):
        entry = runpy.run_path(str(ENTRY))
        with TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            lock = parent / "TRAIN_LOCK.json"
            output, audit = parent / "native", parent / "audit"
            document = dict(status="HRSID_MASK_COMMON_FIT_TRAIN_LOCK_V1", training_protocol="b" * 40,
                            implementation_commit="c" * 40, stage="e24", mode="native", training_seed=5202,
                            initialization_seed=5201, initial_expanded_sha256=entry["INITIAL_EXPANDED_SHA"],
                            fit_index_sha256=entry["FIT_INDEX_SHA"], native_output=output.as_posix(),
                            audit_output=audit.as_posix(), sources={}, qualification_pair_ref=None)
            raw = json.dumps(document).encode("utf-8")
            lock.write_bytes(raw)
            args = SimpleNamespace(training_lock=lock, training_lock_sha256=sha256(raw).hexdigest(),
                                   protocol=document["training_protocol"], stage="e24", mode="native", training_seed=5202,
                                   expected_initial_sha256=document["initial_expanded_sha256"], output=output,
                                   audit_output=audit)
            with self.assertRaisesRegex(RuntimeError, "independently accepted short pair"):
                entry["read_training_lock"](args)
            self.assertFalse(output.exists())
            self.assertFalse(audit.exists())

    def test_closed_native_e24_uses_original_fit_only_receipt_field(self):
        entry = runpy.run_path(str(ENTRY))
        with TemporaryDirectory() as temporary:
            output = Path(temporary).resolve()
            report = output / "NATIVE_HRSID_COCO_FIT_E24_SAFE.json"
            closed = dict(status="NATIVE_HRSID_COCO_FIT_E24_COMPLETE_PENDING_INDEPENDENT_ACCEPTANCE",
                          total_visits=70560, total_updates=17640, ema_updates=17640, completed_epochs=24,
                          filtered_empty_rows_total=0, FIT_only=True, DEV=False, TEST=False, AP=False,
                          protocol="a" * 40, native_index_sha256=entry["FIT_INDEX_SHA"],
                          expanded_online_sha256=entry["INITIAL_EXPANDED_SHA"])
            raw = json.dumps(closed).encode("utf-8")
            report.write_bytes(raw)
            actual = entry["verify_native_result"](SimpleNamespace(stage="e24", protocol=closed["protocol"]),
                                                    output, 70560, 17640, {})
            self.assertEqual(actual, sha256(raw).hexdigest())

    def test_short_and_full_keep_same_engine_with_only_full_checkpoint_markers(self):
        entry = runpy.run_path(str(ENTRY))
        for stage in ("qualification", "e24"):
            with self.subTest(stage=stage), TemporaryDirectory() as temporary:
                self.exercise_stage(entry, stage, Path(temporary).resolve())

    def exercise_stage(self, entry, stage, parent):
        class QualificationComplete(Exception):
            pass

        def encode(value):
            return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")

        def write(path, value):
            with path.open("xb") as stream:
                stream.write(encode(value))

        short = stage == "qualification"
        output, audit = parent / "native", parent / "audit"
        args = SimpleNamespace(stage=stage, mode="center", protocol="a" * 40, output=output, audit_output=audit,
                               expected_initial_sha256=entry["INITIAL_EXPANDED_SHA"], training_seed=5202)
        lock = dict(implementation_commit="b" * 40, sources={name: dict(sha256="c" * 64)
                                                             for name in entry["SOURCES"]})
        closed = dict(FIT=True, DEV=False, TEST=False, AP=False)
        phase_name = "sar_seal_hrsid_mask_resize_phase_v1"
        saved = {}
        observed = []

        def attach_short(observer, trainer, native, base, args, phase_marker):
            self.assertTrue(short)
            trainer.phase_marker = phase_marker

        def bind_seed(trainer, observer, train, args, seed_everything, torch, record_witness):
            self.assertEqual(args.training_seed, 5202)
            trainer.witness = lambda: record_witness(dict(training_seed=5202, initialization_seed=5201))

        class TestLedger:
            def __init__(self, stream):
                self.stream = stream

            def finish(self, count):
                self_outer.assertEqual(count, 32 if short else 70560)
                self.stream.write("{}\n")
                return dict(consumed_visits=count, fetched_visits=count, prefetched_unconsumed=0,
                            unconsumed_prefetch_sha256=None, common_targets_total=count,
                            common_empty_visits=0, changed_mask_visits=count)

        self_outer = self

        @contextmanager
        def scoped_dataset(module_data, detection_type, fit_root, fit_count, transforms, mode, ledger):
            self.assertEqual((fit_root, fit_count, mode), (entry["FIT_LAYOUT"] / "train", 2937, "center"))
            observed.append("dataset")
            yield

        def engine(base, output, protocol, initialization):
            self.assertEqual(protocol, args.protocol)
            self.assertEqual(initialization, args.expected_initial_sha256)
            output.mkdir()
            observer = SimpleNamespace(on_save_checkpoint=lambda active, model, checkpoint: checkpoint.update(old=True))
            trainer = trainer_api.build_trainer(None, None, extra_callbacks=[observer])
            trainer.witness()
            if short:
                self.assertEqual(trainer.phase_marker["seed"], 5202)
                write(output / "QUALIFICATION_SAFE.json", dict(
                    closed, status="NATIVE_HRSID_MASK_RESIZE_QUALIFICATION_COMPLETE", visits=32, optimizer_updates=8,
                    ema_updates=8, completed_epochs=0, removed_empty_rows=0, targets_after_filter=32,
                    checkpoint_saved=False, seed=5202, stage=args.stage, mask_resize_mode=args.mode,
                    protocol=args.protocol, initial_online_sha256=entry["INITIAL_EXPANDED_SHA"]))
                raise QualificationComplete
            observer.on_save_checkpoint(None, None, saved)
            full_scope = {"FIT_only": closed["FIT"], **{key: closed[key] for key in ("DEV", "TEST", "AP")}}
            write(output / "NATIVE_HRSID_COCO_FIT_E24_SAFE.json", dict(
                full_scope, status="NATIVE_HRSID_COCO_FIT_E24_COMPLETE_PENDING_INDEPENDENT_ACCEPTANCE",
                total_visits=70560, total_updates=17640, ema_updates=17640, completed_epochs=24,
                filtered_empty_rows_total=0, protocol=args.protocol, native_index_sha256=entry["FIT_INDEX_SHA"],
                expanded_online_sha256=entry["INITIAL_EXPANDED_SHA"]))

        trainer_api = SimpleNamespace(build_trainer=lambda *args, **kwargs: SimpleNamespace())
        phase = SimpleNamespace(WINDOW_UPDATES=8, PHASE_MARKER=phase_name, QualificationComplete=QualificationComplete,
                                attach_short_window=attach_short, exclusive_json=write)
        repeat = SimpleNamespace(INITIALIZATION_SEED=5201, INITIAL_EXPANDED_SHA=entry["INITIAL_EXPANDED_SHA"],
                                 WITNESS_EVENT="repeat_seed", WITNESS_NAME="REPEAT_SEED_WITNESS_SAFE.json",
                                 witness_digest=lambda value: sha256(encode(value)).hexdigest(), bind_repeat_training=bind_seed)
        binding = SimpleNamespace(ConsumedFitLedger=TestLedger, common_fit_dataset=scoped_dataset,
                                  bind_common_observer=lambda observer, ledger: observed.append("observer"))
        native = SimpleNamespace(SEED=5201, FIT=2937, VISITS=2940, STEPS=735, ENDPOINTS=(1, 8, 24), ACCUM=4,
                                 event=lambda name, **fields: observed.append(name), run=engine)
        base = SimpleNamespace(LAYOUT=entry["FIT_LAYOUT"], INDEX_SHA=entry["FIT_INDEX_SHA"])
        qualification_sha = None if short else "d" * 64
        entry["run_engine"](args, lock, qualification_sha, phase=phase, repeat=repeat, native=native, base=base,
                            binding=binding, module_data=object(), trainer_api=trainer_api,
                            coco=SimpleNamespace(CocoDetection=object), native_transforms=object(),
                            seed_everything=object(), torch=object())
        report = json.loads((audit / entry["AUDIT_RESULT"]).read_bytes())
        self.assertEqual(observed[:3], ["dataset", "observer", "repeat_seed"])
        self.assertEqual(report["consumed_ledger_sha256"], sha256((audit / entry["AUDIT_LEDGER"]).read_bytes()).hexdigest())
        self.assertEqual(report["common_support"]["qualification_pair_sha256"], qualification_sha)
        self.assertEqual(report["completed_epochs"], 0 if short else 24)
        self.assertEqual(report["optimizer_updates"], 8 if short else 17640)
        if short:
            self.assertFalse(saved)
            self.assertFalse((output / "PHASE_PROVENANCE_SAFE.json").exists())
        else:
            self.assertEqual(saved, {"old": True, phase_name: report["phase"], entry["COMMON_MARKER"]: report["common_support"]})
            self.assertEqual(observed[-1], "HRSID_MASK_PHASE_E24_OUTPUT_COMPLETE")


if __name__ == "__main__":
    unittest.main()
