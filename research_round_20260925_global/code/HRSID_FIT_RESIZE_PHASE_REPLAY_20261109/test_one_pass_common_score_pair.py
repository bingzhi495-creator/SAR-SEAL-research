"""Synthetic closed B/C training, original score reports, and paired geometry."""
from copy import deepcopy
from hashlib import sha256
import json
from os import environ
from pathlib import Path
import unittest
from unittest.mock import patch

import one_pass_common_score_pair as subject
import test_one_pass_common_eval_admission as admission_fixture
from test_one_pass_common_eval_admission import IMPLEMENTATION_COMMIT, QUALIFICATION_SHA, TRAINING_PROTOCOL
import test_score_closed_pair as geometry_fixture


CODE = Path(environ["SAR_SEAL_ORDINARY_CODE_ROOT"]).resolve(strict=True)
BC = ("stepwise_center", "one_pass")
ALIASES = ("native", "center")


def population(annotations):
    all12 = [.51, .82, .59, .49, .62, .70, .12, .34, .54, .38, .60, .65]
    return dict(annotations=annotations, AP=all12[0], AP75=all12[2], AP90=.05, APs=all12[3], AR100=all12[8], all12=all12)


class ClosedBCGeometryTest(unittest.TestCase):
    def setUp(self):
        training = admission_fixture.AdmissionTests("test_exact_pair_authorizes_both_b_c_endpoints")
        legacy = geometry_fixture.ClosedPairTest("test_qualified_streams_score_original_order")
        self.addCleanup(training.doCleanups)
        self.addCleanup(legacy.doCleanups)
        training.setUp()
        legacy.setUp()
        self.legacy = legacy
        self.root = legacy.root
        self.protocol = legacy.protocol
        self.gt = self.ref(Path(legacy.gt["path"]))
        training_pair = deepcopy(training.pair)
        self.training_arms = {}
        for mode in BC:
            self.training_arms[mode] = self.write(f"{mode}_training_arm.json", training.arms[mode])
            training_pair[f"{mode}_arm_receipt_sha256"] = self.training_arms[mode]["sha256"]
        self.training_pair = self.write("bc_training_pair.json", training_pair)
        self.plan = dict(
            status="HRSID_ONE_PASS_COMMON_FIT_E24_CLOSED_EXPORT_SCORE_PAIR_V1", stage="e24", training_seed=5202,
            protocol=self.protocol, training_protocol=TRAINING_PROTOCOL, implementation_commit=IMPLEMENTATION_COMMIT,
            qualification_pair_sha256=QUALIFICATION_SHA, training_pair=self.training_pair,
            stepwise_center_arm_receipt=self.training_arms["stepwise_center"],
            one_pass_arm_receipt=self.training_arms["one_pass"], gt=self.gt, arms={},
        )
        for index, (mode, alias) in enumerate(zip(BC, ALIASES, strict=True)):
            checkpoint = training.arms[mode]["checkpoint_sha256"]
            prediction = self.ref(Path(legacy.pair["arms"][alias]["predictions"]["path"]))
            report = self.write(f"{alias}/DEV_NATIVE_EXPORT_SAFE.json", dict(
                legacy.report[alias], checkpoint_sha256=checkpoint, training_protocol=TRAINING_PROTOCOL,
            ))
            score = self.write(f"{alias}_SCORE_SAFE.json", dict(
                status="HRSID_DEV705_E24_NATIVE_COCO_COMPLETE", protocol=self.protocol,
                training_protocol=TRAINING_PROTOCOL, checkpoint_sha256=checkpoint,
                predictions_sha256=prediction["sha256"], initialization="official_COCO_only",
                original_dev_gt_sha256=self.gt["sha256"], images=1, excluded_empty_gt_sensitivity=1,
                max_dets=[1, 10, 100], units="fraction", official_TEST_opened=False, official_TEST_AP=False,
                primary=population(2), sensitivity=population(1),
            ))
            shared = dict(mode=mode, evaluation_protocol=self.protocol, training_protocol=TRAINING_PROTOCOL,
                          training_pair_sha256=self.training_pair["sha256"], checkpoint_sha256=checkpoint,
                          original_os_wait=True, original_os_exit=0, resource_stop=False,
                          owner_closed=True, worker_closed=True, stdout_sha256="e" * 64)
            export = dict(
                shared, status="HRSID_ONE_PASS_COMMON_FIT_DEV705_ORIGINAL_EXPORT_CLOSED_V1",
                terminal_sha256=str(7 + index) * 64, report=report, predictions=prediction,
            )
            closed_score = dict(
                shared, status="HRSID_ONE_PASS_COMMON_FIT_DEV705_ORIGINAL_SCORE_CLOSED_V1",
                terminal_sha256=("9", "a")[index] * 64, report=score, export_terminal_sha256=export["terminal_sha256"],
                export_report_sha256=report["sha256"], predictions_sha256=prediction["sha256"],
                original_dev_gt_sha256=self.gt["sha256"],
            )
            self.plan["arms"][mode] = dict(export=export, score=closed_score)

    def ref(self, path):
        raw = path.read_bytes()
        return dict(path=str(path), bytes=len(raw), sha256=sha256(raw).hexdigest())

    def write(self, name, value):
        path = self.root / name
        path.write_bytes((json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
        return self.ref(path)

    def score(self, evaluation_protocol=None):
        pin = self.write("bc_plan.json", self.plan)
        support = subject.load_support(CODE)
        with patch.object(subject, "load_support", return_value=support), patch.multiple(
            support.driver, GT_SHA=self.gt["sha256"], DEV_IMAGES=1, GT_COUNT=2, EMPTY_COUNT=1, CANVAS=(8, 2),
        ):
            return subject.score_pair(
                CODE, Path(pin["path"]), pin["sha256"], evaluation_protocol=evaluation_protocol or self.protocol,
                training_pair_sha256=self.training_pair["sha256"],
            )

    def test_independently_closed_original_score_and_prediction_reports_produce_bc_geometry(self):
        result = self.score()
        self.assertEqual(result["status"], "HRSID_ONE_PASS_COMMON_FIT_E24_DEV705_PAIRED_GEOMETRY_COMPLETE")
        self.assertEqual(set(result["arms"]), set(BC))
        self.assertEqual(result["arms"]["one_pass"]["checkpoint_sha256"], "6" * 64)
        self.assertEqual(result["score_report_sha256s"]["one_pass"], self.plan["arms"]["one_pass"]["score"]["report"]["sha256"])
        self.assertEqual(result["metrics"]["strict"]["0.90"]["matched_below"]["strict"], 1)
        self.assertIn("one_pass_minus_stepwise_center", result["metrics"]["common_pair_geometry"])
        self.assertNotIn(str(self.root), json.dumps(result))


    def test_swapped_whole_arm_or_rehashed_other_checkpoint_report_is_rejected(self):
        first, second = (self.plan["arms"][mode] for mode in BC)
        self.plan["arms"] = dict(zip(BC, (second, first), strict=True))
        with self.assertRaisesRegex(ValueError, "closed B/C export operation and lineage"):
            self.score()
        self.plan["arms"] = dict(zip(BC, (first, second), strict=True))
        artifact = Path(second["export"]["report"]["path"])
        value = json.loads(artifact.read_bytes())
        value["checkpoint_sha256"] = first["export"]["checkpoint_sha256"]
        second["export"]["report"] = self.write(artifact.relative_to(self.root), value)
        second["score"]["export_report_sha256"] = second["export"]["report"]["sha256"]
        with self.assertRaisesRegex(ValueError, "export report bound to its admitted B/C checkpoint"):
            self.score()

    def test_reused_terminal_across_stages_or_from_training_is_rejected(self):
        reused = self.plan["arms"]["stepwise_center"]["export"]["terminal_sha256"]
        second = self.plan["arms"]["one_pass"]["score"]
        second["terminal_sha256"] = reused
        with self.assertRaisesRegex(ValueError, "Replayed or invalid"):
            self.score()
        training = json.loads(Path(self.training_arms["one_pass"]["path"]).read_bytes())
        second["terminal_sha256"] = training["terminal_sha256"]
        with self.assertRaisesRegex(ValueError, "Replayed or invalid"):
            self.score()

    def test_closure_from_other_prospective_evaluation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "closed prospective B/C evaluation plan"):
            self.score(evaluation_protocol="e" * 40)

    def test_rehashed_score_on_wrong_gt_and_open_owner_are_rejected(self):
        closed = self.plan["arms"]["one_pass"]["score"]
        closed["owner_closed"] = False
        with self.assertRaisesRegex(ValueError, "closed B/C score operation and lineage"):
            self.score()
        closed["owner_closed"] = True
        artifact = Path(closed["report"]["path"])
        value = json.loads(artifact.read_bytes())
        value["original_dev_gt_sha256"] = "0" * 64
        closed["report"] = self.write(artifact.relative_to(self.root), value)
        with self.assertRaisesRegex(ValueError, "closed COCO report bound"):
            self.score()


if __name__ == "__main__":
    unittest.main()
