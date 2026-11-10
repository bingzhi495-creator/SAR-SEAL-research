from contextlib import redirect_stderr
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent


class OnePassCommonQualificationTests(unittest.TestCase):
    def entry(self):
        return runpy.run_path(str(HERE / "one_pass_common_qualification.py"))

    def test_full_stage_is_refused_before_opening_inputs(self):
        entry = self.entry()
        errors = StringIO()
        with redirect_stderr(errors), self.assertRaises(SystemExit) as outcome:
            entry["main"]([
                "--stage", "e24", "--mode", "one_pass", "--training-seed", "5202",
                "--training-lock", "missing.json", "--training-lock-sha256", "0" * 64,
                "--output", "unused-native", "--audit-output", "unused-audit",
                "--native-source", "missing-native.py", "--engine-source", "missing-engine.py",
                "--protocol", "1" * 40, "--expected-initial-sha256", "2" * 64,
            ])
        self.assertEqual(outcome.exception.code, 2)
        self.assertIn("invalid choice: 'e24'", errors.getvalue())

    def test_new_short_lock_admits_only_the_new_policy_and_qualified_sources(self):
        entry = self.entry()
        engine = entry["configured_engine"](HERE / "common_fit_train.py")
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            path = root / "LOCK.json"
            args = SimpleNamespace(
                protocol="a" * 40, stage="qualification", mode="one_pass", training_seed=5202,
                expected_initial_sha256=engine.INITIAL_EXPANDED_SHA, training_lock=path,
                output=root / "native", audit_output=root / "audit",
            )
            pins = {name: dict(path=(root / name).as_posix(), bytes=1, sha256=engine.SOURCE_PINS.get(name, "b" * 64))
                    for name in engine.SOURCES}
            lock = dict(status=engine.LOCK_STATUS, stage=args.stage, mode=args.mode, training_protocol=args.protocol,
                        implementation_commit="c" * 40, training_seed=5202, initialization_seed=5201,
                        initial_expanded_sha256=engine.INITIAL_EXPANDED_SHA, fit_index_sha256=engine.FIT_INDEX_SHA,
                        native_output=args.output.as_posix(), audit_output=args.audit_output.as_posix(),
                        qualification_pair_ref=None, sources=pins)

            def check(document):
                raw = json.dumps(document, sort_keys=True).encode("utf8")
                path.write_bytes(raw)
                args.training_lock_sha256 = sha256(raw).hexdigest()
                return engine.read_training_lock(args)

            self.assertEqual(check(lock), lock)
            self.assertEqual(engine.POLICY, "one_pass_on_stepwise_common_nonempty_rows_v1")
            self.assertIn("one_pass_common_binding.py", pins)
            self.assertIn("one_pass_common_pair_transform.py", pins)
            with self.assertRaisesRegex(RuntimeError, "per-arm stage"):
                check(dict(lock, status="HRSID_MASK_COMMON_FIT_TRAIN_LOCK_V1"))
            with self.assertRaisesRegex(RuntimeError, "cannot carry"):
                check(dict(lock, qualification_pair_ref={"old": "result"}))
            with self.assertRaisesRegex(RuntimeError, "closed source pins"):
                wrong = dict(pins, **{"one_pass_common_binding.py": dict(pins["one_pass_common_binding.py"], sha256="0" * 64)})
                check(dict(lock, sources=wrong))
            self.assertFalse(args.output.exists() or args.audit_output.exists())


if __name__ == "__main__":
    unittest.main()
