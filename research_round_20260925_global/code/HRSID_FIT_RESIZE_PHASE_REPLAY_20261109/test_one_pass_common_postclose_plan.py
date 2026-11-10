"""Synthetic original wait and artifact binding through the existing P260 verifier."""
from hashlib import sha256
import json
from os import environ
from pathlib import Path
import unittest
from unittest.mock import patch

import one_pass_common_postclose_plan as subject
import test_one_pass_common_score_pair as existing


CODE = Path(environ["SAR_SEAL_ORDINARY_CODE_ROOT"]).resolve(strict=True)
MODES = ("stepwise_center", "one_pass")


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


class PostOwnerCloseTests(unittest.TestCase):
    def setUp(self):
        fixtures = existing.ClosedBCGeometryTest(
            "test_independently_closed_original_score_and_prediction_reports_produce_bc_geometry",
        )
        self.addCleanup(fixtures.doCleanups)
        fixtures.setUp()
        self.original = fixtures
        self.root = fixtures.root
        self.work = self.root / "attempt"
        self.control = self.root / "control"
        self.work.mkdir()
        self.control.mkdir()
        self.owner = {"pid": 30, "creation_time": 100.0}
        self.coordinator = {"pid": 29, "creation_time": 90.0}
        self.lock = dict(
            status="HRSID_ONE_PASS_COMMON_FIT_DEV705_OWNER_LOCK_V1",
            evaluation_protocol=fixtures.protocol,
            **{name: fixtures.plan[name] for name in (
                "training_protocol", "implementation_commit", "qualification_pair_sha256", "training_pair",
                "stepwise_center_arm_receipt", "one_pass_arm_receipt", "gt",
            )},
            owner_source=self.put(self.control / "PINNED_BC_OWNER.py", b"# synthetic owner\n"),
            work=str(self.work),
        )
        self.lock_ref = self.put(self.control / "EVALUATION_LOCK_PRIVATE.json", encoded(self.lock))
        self.entries = []
        for operation in ("export", "score"):
            for mode in MODES:
                stage = f"{operation}_{mode}".upper()
                directory = self.work / stage
                directory.mkdir()
                worker = {"pid": 40 + len(self.entries), "creation_time": 110.0 + 10 * len(self.entries)}
                terminal = dict(
                    stage=stage, owner=self.owner, worker=worker, exact_worker_closed=True, original_os_wait=True,
                    original_os_exit=0, resource_stop=False, stop_reason=None,
                    worker_wall_seconds=2.0, peak_worker_rss_bytes=1000,
                    stdout_ref=self.put(directory / "WORKER.stdout", b"worker event\n"),
                    stderr_ref=self.put(directory / "WORKER.stderr", b""),
                )
                artifact = fixtures.plan["arms"][mode][operation]
                if operation == "export":
                    output = self.work / (stage + "_ACTUAL_PRIVATE")
                    output.mkdir()
                    entry = dict(stage=stage, terminal=self.put(directory / "TERMINAL_PRIVATE.json", encoded(terminal)),
                                 report=self.copy(artifact["report"], output / "DEV_NATIVE_EXPORT_SAFE.json"),
                                 predictions=self.copy(artifact["predictions"], output / "DEV_NATIVE_PREDICTIONS_PRIVATE.jsonl"))
                else:
                    exported = self.entries[MODES.index(mode)]
                    entry = dict(
                        stage=stage, terminal=self.put(directory / "TERMINAL_PRIVATE.json", encoded(terminal)),
                        report=self.copy(artifact["report"], self.work / (stage + "_SAFE.json")),
                        export_terminal_sha256=exported["terminal"]["sha256"],
                        export_report_sha256=exported["report"]["sha256"],
                        predictions_sha256=exported["predictions"]["sha256"],
                    )
                self.entries.append(entry)
        self.refresh_owner()

    def put(self, path, raw):
        path.write_bytes(raw)
        return dict(path=str(path), bytes=len(raw), sha256=sha256(raw).hexdigest())

    def copy(self, ref, path):
        return self.put(path, Path(ref["path"]).read_bytes())

    def refresh_owner(self, *, actual_owner=None):
        self.completion_ref = self.put(self.work / "OWNER_COMPLETION_PRIVATE.json", encoded(dict(
            status="HRSID_ONE_PASS_COMMON_FIT_DEV705_OWNER_WORKERS_CLOSED_PENDING_OWNER_WAIT_V1",
            owner=self.owner, coordinator=self.coordinator, lock_ref=self.lock_ref,
            owner_source_sha256=self.lock["owner_source"]["sha256"], stages=self.entries,
        )))
        event = dict(
            event="HRSID_ONE_PASS_COMMON_FIT_DEV705_WORKERS_CLOSED",
            lock_sha256=self.lock_ref["sha256"], completion_sha256=self.completion_ref["sha256"],
            owner_source_sha256=self.lock["owner_source"]["sha256"],
        )
        self.outer_terminal_ref = self.put(self.control / "OWNER_TERMINAL_PRIVATE.json", encoded(dict(
            owner=actual_owner or self.owner, original_os_wait=True, original_os_exit=0, job_assigned=True,
            job_active_before_drain=0, job_active_after_wait=0, stop_reason=None, wall_seconds=50.0,
            stdout_ref=self.put(self.control / "OWNER.stdout", (json.dumps(event) + "\n").encode()),
            stderr_ref=self.put(self.control / "OWNER.stderr", b""),
        )))

    def build(self, is_live=lambda _: False):
        p260 = subject.load_p260(CODE)
        support = p260.load_support(CODE)
        with (patch.object(subject, "load_p260", return_value=p260),
              patch.object(p260, "load_support", return_value=support),
              patch.multiple(support.driver, GT_SHA=self.lock["gt"]["sha256"], DEV_IMAGES=1,
                             GT_COUNT=2, EMPTY_COUNT=1, CANVAS=(8, 2))):
            plan = subject.build_plan(
                CODE, self.lock_ref, self.outer_terminal_ref, self.completion_ref,
                owner_birth=self.owner, coordinator_birth=self.coordinator, is_live=is_live,
            )
            return plan

    def test_closed_original_owner_and_all_four_workers_produce_exact_p260_plan(self):
        plan = self.build()
        self.assertEqual(plan["status"], "HRSID_ONE_PASS_COMMON_FIT_E24_CLOSED_EXPORT_SCORE_PAIR_V1")
        self.assertEqual(set(plan["arms"]), set(MODES))
        for index, mode in enumerate(MODES):
            export, score = plan["arms"][mode]["export"], plan["arms"][mode]["score"]
            self.assertTrue(export["owner_closed"] and score["worker_closed"])
            self.assertEqual(export["terminal_sha256"], self.entries[index]["terminal"]["sha256"])
            self.assertEqual(score["export_report_sha256"], export["report"]["sha256"])


    def test_live_owner_or_foreign_outer_wait_never_yields_a_plan(self):
        with self.assertRaisesRegex(ValueError, "owner is still live"):
            self.build(is_live=lambda candidate: candidate == self.owner)
        self.refresh_owner(actual_owner=dict(self.owner, creation_time=101.0))
        with self.assertRaisesRegex(ValueError, "owner OS wait"):
            self.build()

    def test_foreign_or_live_worker_is_refused_after_owner_wait(self):
        entry = self.entries[0]
        terminal_path = Path(entry["terminal"]["path"])
        original = json.loads(terminal_path.read_bytes())
        with self.assertRaisesRegex(ValueError, "Live or out-of-order"):
            self.build(is_live=lambda candidate: candidate == original["worker"])
        foreign = dict(original, owner=dict(self.owner, creation_time=101.0))
        entry["terminal"] = self.put(terminal_path, encoded(foreign))
        self.refresh_owner()
        with self.assertRaisesRegex(ValueError, "Foreign or unclosed"):
            self.build()

    def test_original_worker_stdout_is_rehashed_from_its_verified_terminal(self):
        terminal = json.loads(Path(self.entries[0]["terminal"]["path"]).read_bytes())
        path = Path(terminal["stdout_ref"]["path"])
        path.write_bytes(b"worker Event\n")
        with self.assertRaisesRegex(ValueError, "artifact digest"):
            self.build()

    def test_rehashed_owner_stdout_must_still_bind_the_actual_completion(self):
        path = self.control / "OWNER_TERMINAL_PRIVATE.json"
        terminal = json.loads(path.read_bytes())
        stdout = Path(terminal["stdout_ref"]["path"])
        event = json.loads(stdout.read_bytes())
        event["completion_sha256"] = "0" * 64
        terminal["stdout_ref"] = self.put(stdout, (json.dumps(event) + "\n").encode())
        self.outer_terminal_ref = self.put(path, encoded(terminal))
        with self.assertRaisesRegex(ValueError, "stdout must bind"):
            self.build()

    def test_reused_other_arm_score_is_refused_even_if_copied_and_rehashed(self):
        second = self.entries[3]
        report = Path(second["report"]["path"])
        second["report"] = self.copy(self.entries[2]["report"], report)
        self.refresh_owner()
        with self.assertRaisesRegex(ValueError, "closed COCO report bound"):
            self.build()

    def test_reordered_stages_and_worker_births_are_refused(self):
        self.entries[0], self.entries[1] = self.entries[1], self.entries[0]
        self.refresh_owner()
        with self.assertRaisesRegex(ValueError, "stage order"):
            self.build()
        self.entries[0], self.entries[1] = self.entries[1], self.entries[0]
        second = self.entries[1]
        path = Path(second["terminal"]["path"])
        raw = json.loads(path.read_bytes())
        raw["worker"]["creation_time"] = 105.0
        second["terminal"] = self.put(path, encoded(raw))
        self.refresh_owner()
        with self.assertRaisesRegex(ValueError, "out-of-order"):
            self.build()


    def test_generated_plan_enters_exact_original_geometry_with_bc_labels(self):
        plan = self.build()
        plan_ref = self.put(self.control / "P260_PLAN_PRIVATE.json", encoded(plan))
        p260 = subject.load_p260(CODE)
        support = p260.load_support(CODE)
        with (patch.object(p260, "load_support", return_value=support),
              patch.multiple(support.driver, GT_SHA=self.lock["gt"]["sha256"], DEV_IMAGES=1,
                             GT_COUNT=2, EMPTY_COUNT=1, CANVAS=(8, 2))):
            result = p260.score_pair(
                CODE, Path(plan_ref["path"]), plan_ref["sha256"],
                evaluation_protocol=self.lock["evaluation_protocol"],
                training_pair_sha256=self.lock["training_pair"]["sha256"],
            )
        self.assertEqual(set(result["arms"]), set(MODES))
        self.assertEqual(result["metrics"]["strict_transition_axes"], {"row": "stepwise_center", "column": "one_pass"})
        self.assertEqual(result["metrics"]["strict"]["0.90"]["matched_below"]["strict"], 1)
        self.assertNotIn(str(self.root), json.dumps(result))


if __name__ == "__main__":
    unittest.main()
