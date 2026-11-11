from hashlib import sha256
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import ssdd_repeat5202_short as subject


CODE = Path(__file__).resolve().parents[1] / "github_release_20261011/research_round_20260925_global/code"


def identity():
    return subject.make_identity(training_protocol="a"*40, implementation_commit="b"*40,
        adapter_sha256=sha256(Path(subject.__file__).read_bytes()).hexdigest(),
        original_manifest_sha256=subject.WINDOWS_MANIFEST)


class RepeatShortTest(unittest.TestCase):
    def test_separated_seeds_preserve_original_model_and_require_shared_train_config(self):
        train = SimpleNamespace(seed=5201)
        model = object()
        carrier = SimpleNamespace(train_config=train)
        witness = dict(initialization_seed=5201, online_tensor_sha256=subject.Q100_SHA, state_tensors=545)
        actual = subject.repeat_initializer(lambda *args: (model, train, carrier, witness))("fit", "out", "coco")
        self.assertEqual(actual, (model, train, carrier, witness))
        self.assertEqual(actual[1].seed, 5202)
        self.assertEqual(actual[3]["initialization_seed"], 5201)
        train.seed = 5201
        carrier.train_config = SimpleNamespace(seed=5201)
        with self.assertRaisesRegex(ValueError, "shared original TrainConfig"):
            subject.repeat_initializer(lambda *args: (model, train, carrier, witness))("fit", "out", "coco")

    def test_program_binding_isolated_and_requires_unchanged_seed_sites(self):
        programs = subject.load_programs(CODE)
        self.assertEqual(programs.worker.SEED, 5201)
        self.assertEqual(programs.qualifier.SEED, 5201)
        source = b"def run_fit_arm():\n    return SEED\n"
        with self.assertRaisesRegex(ValueError, "seed sites"):
            subject.bind_module(source, "fixture.py", {"run_fit_arm": (7, 1)})
        self.assertEqual(len(sha256(Path(subject.__file__).read_bytes()).hexdigest()), 64)

    def test_two_arm_original_witnesses_enforce_split_seeds_and_budget(self):
        lock = identity()
        qualifier = subject.load_programs(CODE).qualifier
        with patch.dict(sys.modules, {"ssdd_phase_qualification": qualifier}):
            fixture = runpy.run_path(str(CODE / subject.SUBDIR / "test_ssdd_phase_qualification.py"))["fixture"]

        def arm(mode, seed=5202):
            original = fixture(mode)
            report = json.loads(original.report_bytes)
            report["phase"].update(training_protocol=lock["training_protocol"],
                source_manifest_sha256=lock["source_manifest_sha256"], seed=seed)
            report["seed_witness"].update(train_config_seed=seed, torch_cpu_initial_seed=seed,
                                          torch_cuda_initial_seed=seed, lightning_global_seed=str(seed))
            logs = [json.loads(line) for line in original.stdout_bytes.splitlines() if line.startswith(b"{")]
            logs[0]["phase"] = report["phase"]
            logs[1]["seed_witness"] = report["seed_witness"]
            logs[-1] = dict(event=logs[-1]["event"], **report)
            stdout = b"".join(subject.compact(line)+b"\n" for line in logs)
            return original._replace(stdout_bytes=stdout, report_bytes=subject.compact(report),
                closure={**original.closure, "stdout_sha256": subject.digest(stdout)})

        args = dict(code_root=CODE, identity=lock, native=arm("native"), expected_fit_ids=tuple(range(1, 742)))
        result = json.loads(subject.qualify_closed_pair(**args, center=arm("center")))
        self.assertEqual((result["seed"], result["initialization_seed"], result["visits_per_arm"],
                         result["optimizer_updates_per_arm"], result["ema_updates_per_arm"]), (5202, 5201, 32, 8, 8))
        self.assertEqual(result["status"], subject.PAIR_STATUS)
        self.assertEqual(result["changes"]["same_count_mask_only_visits"], 1)
        with self.assertRaisesRegex(ValueError, "prospective phase pins"):
            subject.qualify_closed_pair(**args, center=arm("center", 5201))

    def test_old_protocol_and_unreserved_owner_are_rejected(self):
        lock = identity()
        with self.assertRaisesRegex(ValueError, "Distinct prospective"):
            subject.make_identity(training_protocol=subject.OLD_WINDOWS, implementation_commit="b"*40,
                adapter_sha256=lock["adapter_sha256"], original_manifest_sha256=subject.WINDOWS_MANIFEST)
        with self.assertRaisesRegex(ValueError, "original owner"):
            subject.run_windows_owned_short(code_root=Path("unused"), identity=lock, mode="native",
                stock_scope_lock=Path("unused"), stock_scope_sha256="c"*64, owner_admit=lambda *args: False)


if __name__ == "__main__":
    unittest.main()
