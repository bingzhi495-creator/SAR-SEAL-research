"""Exercise generated 5203 evidence, including all 70,560 ledger rows, without scientific inputs."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("replica", HERE / "replicate.py")
replica = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replica)
CODE = Path(os.environ["SAR_SEAL_ORIGINAL_CODE"]).resolve(strict=True)
REPLAY = CODE / "HRSID_FIT_RESIZE_PHASE_REPLAY_20261109"
sys.path.insert(0, str(REPLAY))


def fixture(filename, data=None):
    path = REPLAY / filename
    namespace = dict(__file__=str(path), __name__="_generated_fixture")
    exec(compile(data if data is not None else path.read_bytes(), str(path), "exec"), namespace)
    return namespace


class ReplicaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = TemporaryDirectory(dir=HERE)
        cls.addClassCleanup(cls.temp.cleanup)
        original = (REPLAY / "common_fit_train.py").read_bytes()
        derived = replica.worker_bytes(original)
        path = Path(cls.temp.name) / "common_fit_train.py"
        path.write_bytes(derived)
        cls.worker = replica.pinned(path, replica.digest(derived), "common_fit_train")
        cls.pair = replica.pinned(REPLAY / "common_fit_pair_receipt.py", replica.PAIR, "common_fit_pair_receipt")
        cls.ledger = replica.pinned(REPLAY / "common_fit_e24_ledger.py", replica.LEDGER, "common_fit_e24_ledger")
        cls.accept = replica.pinned(REPLAY / "common_fit_e24_acceptance.py", replica.ACCEPT, "common_fit_e24_acceptance")
        cls.sha = replica.digest(derived)

    def test_single_literal_and_scope(self):
        self.assertEqual((self.worker.TRAINING_SEED, self.worker.INITIALIZATION_SEED), (5203, 5201))
        with self.assertRaisesRegex(ValueError, "original common5202"):
            replica.worker_bytes(b"TRAINING_SEED = 5202")
        with self.assertRaisesRegex(ValueError, "single center"):
            replica.arm_paths(SimpleNamespace(work=Path(self.temp.name)), "e24", "native")

    def test_owner_bound_api_proof_and_rejection_paths(self):
        commit = "a" * 40
        paths = (replica.SELF, replica.PRELOCK)
        files = [dict(path=replica.PUBLISHED_DIR + "/" + p.name, git_blob_sha=replica.git_blob_sha(p.read_bytes()),
                      sha256=replica.digest(p.read_bytes())) for p in paths]
        document = dict(status="GITHUB_API_COMMIT_TREE_AND_BLOBS_VERIFIED", source="github_api",
                        repository=replica.REPOSITORY, commit=commit, tree="b" * 40, files=files)
        proof = Path(self.temp.name) / "api-proof.json"
        mirror = Path(self.temp.name) / "no-git"
        def verify(value, expected=None, given_commit=commit):
            raw = replica.canonical(value)
            proof.write_bytes(raw)
            return replica.verify_git(mirror, given_commit, proof=proof, proof_sha=expected or replica.digest(raw))
        with patch.object(replica.subprocess, "check_output", side_effect=AssertionError("API proof invoked Git")):
            publication, raw = verify(document)
            self.assertEqual(publication["tree"], document["tree"])
            self.assertEqual(publication["proof_sha256"], replica.digest(raw))
            for key, value in (("repository", "other/repository"), ("source", "local_guess"),
                               ("tree", "bad"), ("commit", "c" * 40), ("status", "pending")):
                with self.assertRaises(ValueError):
                    verify(dict(document, **{key: value}))
            for key, value in (("path", "wrong/replicate.py"), ("git_blob_sha", "d" * 40), ("sha256", "e" * 64)):
                with self.assertRaises(ValueError):
                    verify(dict(document, files=[dict(files[0], **{key: value}), files[1]]))
            with self.assertRaises(ValueError):
                verify(dict(document, files=files[:1] * 2))
            with self.assertRaises(ValueError):
                verify(document, expected="0" * 64)

    def test_local_git_resolves_exact_file_paths(self):
        repo = Path(self.temp.name) / "synthetic_git"
        repo.mkdir()
        def git(*args):
            return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
        git("init", "--quiet")
        destination = repo / replica.PUBLISHED_DIR
        destination.mkdir(parents=True)
        for path in (replica.SELF, replica.PRELOCK):
            (destination / path.name).write_bytes(path.read_bytes())
        git("-c", "core.autocrlf=false", "add", "--", replica.PUBLISHED_DIR)
        git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "-c", "commit.gpgsign=false",
            "commit", "--quiet", "-m", "Generated fixture")
        commit = git("rev-parse", "HEAD")
        publication, raw = replica.verify_git(repo, commit)
        self.assertEqual((publication["source"], publication["tree"], raw), ("local_git", git("rev-parse", "HEAD^{tree}"), None))

    def test_historical_source_manifest_without_training_inputs(self):
        work = Path(os.environ["SAR_SEAL_RUNTIME_ROOT"]) / "RFDETR_COMMON5203_E24_SYNTHETIC"
        ctx = replica.context(work, needs_containment=False)
        self.assertEqual(ctx.hashes["common_fit_train.py"], self.sha)
        self.assertEqual(ctx.base.INITIALIZATION_SEED, 5201)
        self.assertFalse(work.exists())

    def test_original_qualification_accepts_5203_and_rejects_5202(self):
        path = REPLAY / "test_common_fit_pair_receipt.py"
        generated = fixture(path.name, path.read_bytes().replace(b"5202", b"5203"))
        value = generated["accept"]()
        self.assertEqual((value["training_seed"], value["initialization_seed"],
                          value["visits_per_arm"], value["optimizer_updates_per_arm"]), (5203, 5201, 32, 8))
        closed = Path(self.temp.name) / "qual.json"
        raw = self.pair.encode_receipt(value)
        closed.write_bytes(raw)
        lock = dict(training_protocol=generated["PROTOCOL"], implementation_commit=generated["IMPLEMENTATION"],
                    sources={name: dict(sha256=h) for name, h in generated["SOURCE_HASHES"].items()},
                    qualification_pair_ref=dict(path=closed.as_posix(), bytes=len(raw), sha256=replica.digest(raw)))
        self.assertEqual(self.worker.verify_qualification(lock), replica.digest(raw))
        wrong = fixture(path.name)
        with self.assertRaises(ValueError):
            wrong["accept"]()

    def test_real_validator_on_three_synthetic_checkpoints_and_full_ledger(self):
        gen = fixture("test_common_fit_e24_acceptance.py")
        arm, calls = gen["synthetic_arm"]("center")
        _, arm, _, rows = gen["synthetic_full_streams"](arm, arm)
        short = gen["SHORT"]
        identity = self.accept.short_pair_identity(gen["PROTOCOL"], gen["IMPLEMENTATION"], gen["SOURCE_SHAS"],
                                                   short, replica.digest(short))
        legacy = self.accept.scoped_repeat(gen["PROTOCOL"], gen["SOURCE_SHAS"]["common_fit_train.py"])
        safe = self.accept.checked_arm_audit(arm, "center", identity, legacy)
        own = replica.single_ledger(self.ledger, safe, rows, arm.visits)
        result = self.accept.accept_arm(arm, "center", safe, legacy, range(2937), own)
        self.assertEqual(result["common_support"]["training_seed"], 5203)
        self.assertEqual(result["optimizer_updates"], 17640)
        self.assertEqual(len(calls), 3)

    def test_windows_os_wait_on_generated_success_and_failure(self):
        if os.name != "nt":
            self.skipTest("Windows containment")
        import psutil
        # The private helper below is historically closed source. Its state functions are never called.
        root = Path(os.environ["SAR_SEAL_RUNTIME_ROOT"])
        owner = replica.pinned(root / "P206_ROOT_HRSID_MASK_PHASE_DEV705_COORDINATOR_PRIVATE_20261109.py",
                               "55cb9a4a18a6b58fa5eac28a259422c058831523ae10ca0448818ee856ef0b39", "_synthetic_job")
        runtime = SimpleNamespace(RUNTIME=Path(sys.executable), psutil=psutil, RAM_FLOOR=0, RSS_CAP=1024**3,
                                  windows_runtime=lambda scratch: os.environ.copy())
        ctx = SimpleNamespace(owner=owner, runtime=runtime)
        for exit_code in (0, 4):
            script = Path(self.temp.name) / ("fixture" + str(exit_code) + ".py")
            control = Path(self.temp.name) / ("os" + str(exit_code))
            script.write_text("raise SystemExit(" + str(exit_code) + ")\n", encoding="ascii")
            if exit_code:
                with self.assertRaisesRegex(ValueError, "successful full OS wait"):
                    replica.contained(ctx, script, [], control, 15)
            else:
                replica.contained(ctx, script, [], control, 15)
            actual = json.loads((control / "OWNER_TERMINAL_PRIVATE.json").read_bytes())
            self.assertEqual(actual["original_os_exit"], exit_code)
            self.assertTrue(actual["original_os_wait"])
            self.assertEqual(actual["job_active_after_wait"], 0)

    def test_nested_owner_uses_original_science_runtime_on_generated_worker(self):
        if os.name != "nt":
            self.skipTest("Windows containment")
        root = Path(os.environ["SAR_SEAL_RUNTIME_ROOT"])
        work = root / "RFDETR_COMMON5203_E24_SYNTHETIC"
        ctx = replica.context(work, needs_containment=True)
        temp = Path(self.temp.name)
        outer, inner = temp / "nested_owner", temp / "nested_worker"
        script = temp / "worker.py"
        script.write_text("import sys\nprint(sys.executable, flush=True)\n", encoding="utf8")
        owner_script = temp / "nested.py"
        owner_script.write_text(
            "from pathlib import Path\nimport runpy\n"
            + "api = runpy.run_path(" + repr(str(replica.SELF)) + ")\n"
            + "ctx = api['context'](Path(" + repr(str(work)) + "), needs_containment=True)\n"
            + "api['contained'](ctx, Path(" + repr(str(script)) + "), [], Path(" + repr(str(inner)) + "), 30)\n"
            + "print('Synthetic nested owner complete', flush=True)\n", encoding="utf8")
        replica.contained(ctx, owner_script, [], outer, 50, owner_process=True)
        self.assertEqual(Path((inner / "OWNER.stdout").read_text().strip()).resolve(),
                         ctx.base.PYTHON.resolve())
        helper = replica.p240(ctx)
        io = replica.pinned(helper.P220, helper.P220_SHA, "_synthetic_verify_os")
        for control, submitted in ((outer, owner_script), (inner, script)):
            ref = ctx.owner.ref(control / "OWNER_TERMINAL_PRIVATE.json")
            birth = json.loads((control / "OWNER_BIRTH_PRIVATE.json").read_bytes())
            self.assertFalse(ctx.owner.live(birth))
            terminal, _ = helper.verified_terminal(control, ref, birth, replica.digest(submitted.read_bytes()), io)
            self.assertEqual(terminal["original_os_exit"], 0)


if __name__ == "__main__":
    unittest.main()
