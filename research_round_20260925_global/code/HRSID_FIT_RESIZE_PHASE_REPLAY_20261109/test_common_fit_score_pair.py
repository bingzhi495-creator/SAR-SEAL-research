import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import common_fit_score_pair as bridge
from test_common_fit_eval_admission import IDENTITY, PROTOCOL, arm

GEOMETRY_DIR = Path(__file__).resolve().parent.parent / "HRSID_MASK_PHASE_PAIRED_GEOMETRY_20261109"
sys.path.insert(0, str(GEOMETRY_DIR))
import test_score_closed_pair as legacy_fixture


def score_population(annotations):
    stats = [.51, .82, .59, .49, .62, .70, .12, .34, .54, .38, .60, .65]
    return dict(annotations=annotations, AP=stats[0], AP75=stats[2], AP90=.05,
                APs=stats[3], AR100=stats[8], all12=stats)


class CommonFitScorePairTest(unittest.TestCase):
    def setUp(self):
        self.legacy = legacy_fixture.ClosedPairTest("test_qualified_streams_score_original_order")
        self.addCleanup(self.legacy.doCleanups)
        self.legacy.setUp()
        self.root = self.legacy.root
        self.pair = self.legacy.pair
        self.pair.update(status="HRSID_MASK_COMMON_FIT_E24_CLOSED_EXPORT_SCORE_PAIR_V1", training_seed=5202)
        self.training_pair = dict(
            **IDENTITY, status="HRSID_MASK_COMMON_FIT_E24_PAIR_READY_V1", ordered_common_fit_visits_match=True,
            completed_epochs=24, visits_per_arm=70560, optimizer_updates_per_arm=17640, ema_updates_per_arm=17640,
            shared_consumed_ledger_sha256="f" * 64, changed_mask_visits=29,
            FIT=True, DEV=False, TEST=False, AP=False,
        )
        self.training_arms = {}
        self.score_reports = {}
        for mode in ("native", "center"):
            accepted = arm(mode)
            pin = self.legacy.write(mode + "_training_arm.json", accepted)
            self.training_arms[mode] = Path(pin["path"])
            self.training_pair[f"{mode}_arm_receipt_sha256"] = pin["sha256"]
            self.training_pair[f"{mode}_checkpoint_sha256"] = accepted["checkpoint_sha256"]
            report = self.legacy.report[mode]
            report.update(checkpoint_sha256=accepted["checkpoint_sha256"], training_protocol=PROTOCOL)
            self.pair["arms"][mode]["report"] = self.legacy.write(mode + "/DEV_NATIVE_EXPORT_SAFE.json", report)
            self.score_reports[mode] = dict(
                status="HRSID_DEV705_E24_NATIVE_COCO_COMPLETE", protocol=self.pair["protocol"],
                checkpoint_sha256=accepted["checkpoint_sha256"], predictions_sha256=report["predictions_sha256"],
                training_protocol=PROTOCOL, initialization="official_COCO_only",
                original_dev_gt_sha256=self.legacy.gt["sha256"], images=1,
                excluded_empty_gt_sensitivity=1, max_dets=[1, 10, 100], units="fraction",
                official_TEST_opened=False, official_TEST_AP=False,
                primary=score_population(2), sensitivity=score_population(1),
            )
            self.pair["arms"][mode]["score_closure"] = dict(
                original_os_wait=True, original_os_exit=0, resource_stop=False, owner_closed=True, worker_closed=True,
            )
            self.write_score(mode)
        self.write_training_pair()

    def write_score(self, mode):
        self.pair["arms"][mode]["score_closure"]["report"] = self.legacy.write(
            mode + "_SCORE_SAFE.json", self.score_reports[mode],
        )

    def write_training_pair(self):
        self.pair["training_pair"] = self.legacy.write("training_pair.json", self.training_pair)

    def score(self, digest=None, *, training_protocol=PROTOCOL):
        driver = bridge.load_driver()
        pin = self.legacy.write("pair.json", self.pair)
        with patch.object(bridge, "load_driver", return_value=driver), patch.multiple(
            driver, GT_SHA=self.legacy.gt["sha256"], DEV_IMAGES=1, GT_COUNT=2, EMPTY_COUNT=1, CANVAS=(8, 2),
        ):
            return bridge.score_pair(
                Path(pin["path"]), digest or pin["sha256"], native_arm_receipt=self.training_arms["native"],
                center_arm_receipt=self.training_arms["center"], training_protocol=training_protocol,
            )

    def test_closed_common_training_and_two_coco_scores_unlock_original_geometry(self):
        result = self.score()

        self.assertEqual(result["status"], "HRSID_MASK_COMMON_FIT_E24_DEV705_PAIRED_GEOMETRY_COMPLETE")
        self.assertEqual(result["training_seed"], 5202)
        self.assertEqual(result["support_policy"], "ordered_common_nonempty_prepared_rows_v1")
        self.assertEqual(result["metrics"]["point_five"]["both_matched"], 1)
        self.assertEqual(result["metrics"]["strict"]["0.90"]["matched_below"]["strict"], 1)
        self.assertEqual(result["score_report_sha256s"]["native"],
                         self.pair["arms"]["native"]["score_closure"]["report"]["sha256"])
        self.assertNotIn(str(self.root), json.dumps(result))

    def test_both_original_score_processes_must_have_closed(self):
        closure = self.pair["arms"]["center"]["score_closure"]
        for key in ("owner_closed", "worker_closed", "original_os_wait", "resource_stop"):
            with self.subTest(key=key):
                closure[key] = not closure[key]
                try:
                    with self.assertRaisesRegex(ValueError, "score owner and worker"):
                        self.score()
                finally:
                    closure[key] = not closure[key]
        del closure["owner_closed"]
        with self.assertRaisesRegex(ValueError, "score owner and worker"):
            self.score()

    def test_validly_rehashed_but_misbound_coco_report_is_rejected(self):
        report = self.score_reports["center"]
        for key in ("checkpoint_sha256", "predictions_sha256", "original_dev_gt_sha256", "training_protocol"):
            with self.subTest(key=key):
                expected = report[key]
                report[key] = "e" * len(expected)
                self.write_score("center")
                try:
                    with self.assertRaisesRegex(ValueError, "closed COCO report bound"):
                        self.score()
                finally:
                    report[key] = expected

    def test_unpinned_score_bytes_and_unclosed_export_are_rejected(self):
        arm = self.pair["arms"]["native"]
        arm["original_os_wait"] = False
        with self.assertRaisesRegex(ValueError, "Original export"):
            self.score()
        arm["original_os_wait"] = True
        score_file = Path(arm["score_closure"]["report"]["path"])
        score_file.write_bytes(score_file.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.score()

    def test_authority_digest_and_both_common_training_receipts_are_required(self):
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.score("0" * 64)
        self.training_pair["center_arm_receipt_sha256"] = "0" * 64
        self.write_training_pair()
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.score()


if __name__ == "__main__":
    unittest.main()
