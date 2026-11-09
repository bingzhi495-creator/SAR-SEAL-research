import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import score_closed_pair as driver
from test_paired_geometry import block, gt_for, rle


class ClosedPairTest(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        ship = rle("11110000", "00000000")
        weak = rle("11000000", "00000000")
        empty = rle("00000000", "00000000")
        self.gt = self.write("gt.json", gt_for(8, 2, (ship, {}), (empty, {})))
        self.protocol = "f" * 40
        self.training_pair = dict(
            status="HRSID_MASK_PHASE_E24_PAIR_READY_V1",
            training_protocol="d66baf3c57fafdba4242ad1b55e08cf76bfd77f7",
            ordered_fit_visits_match=True, completed_epochs=24, visits_per_arm=70560,
            native_arm_receipt_sha256="a" * 64, center_arm_receipt_sha256="b" * 64,
            native_checkpoint_sha256="c" * 64, center_checkpoint_sha256="d" * 64,
        )
        self.pair = dict(
            status="HRSID_MASK_PHASE_E24_CLOSED_EXPORT_PAIR_V1", protocol=self.protocol,
            training_pair=self.write("train_pair.json", self.training_pair), gt=self.gt, arms={},
        )
        self.report = {}
        for arm, predictions in (("native", block(weak, ship)), ("center", block(ship, weak))):
            stream = self.write(arm + "/DEV_NATIVE_PREDICTIONS_PRIVATE.jsonl", predictions[0])
            report = dict(
                status="HRSID_DEV705_E24_NATIVE_EXPORT_COMPLETE", protocol=self.protocol,
                checkpoint_sha256=self.training_pair[arm + "_checkpoint_sha256"],
                checkpoint_ema_updates=17640, training_protocol=self.training_pair["training_protocol"],
                initialization="official_COCO_only", image_manifest_sha256=driver.IMAGE_MANIFEST_SHA,
                predictions_sha256=stream["sha256"], images=1, raw_queries=164, native_top_rows=100,
                mapped_rows=2, unmapped_slot1_rows=98, num_select=100, query_width=164,
                precision="FP32_TF32_disabled", peak_allocated=1024, elapsed_seconds=1.5,
                GT_opened=False, official_TEST_opened=False, AP=False,
            )
            self.report[arm] = report
            self.pair["arms"][arm] = dict(
                original_os_wait=True, original_os_exit=0, resource_stop=False,
                report=self.write(arm + "/DEV_NATIVE_EXPORT_SAFE.json", report), predictions=stream,
            )

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(exist_ok=True)
        data = (json.dumps(value, allow_nan=False) + "\n").encode("utf-8")
        path.write_bytes(data)
        return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest())

    def score(self, digest=None):
        pin = self.write("pair.json", self.pair)
        with patch.multiple(
            driver, GT_SHA=self.gt["sha256"], DEV_IMAGES=1, GT_COUNT=2, EMPTY_COUNT=1, CANVAS=(8, 2),
        ):
            return driver.score_pair(Path(pin["path"]), digest or pin["sha256"])

    def test_qualified_streams_score_original_order(self):
        result = self.score()
        self.assertEqual(result["status"], "HRSID_MASK_PHASE_E24_DEV705_PAIRED_GEOMETRY_COMPLETE")
        self.assertEqual(result["metrics"]["point_five"]["both_matched"], 1)
        self.assertEqual(result["metrics"]["strict"]["0.90"]["matched_below"]["strict"], 1)
        self.assertEqual(result["metrics"]["exclusions"]["empty"], 1)
        self.assertFalse(result["official_TEST_opened"])
        self.assertNotIn(str(self.root), json.dumps(result))


    def test_pair_digest_is_required_even_for_qualified_receipt(self):
        with self.assertRaisesRegex(ValueError, "Pinned input size or SHA-256 mismatch"):
            self.score("0" * 64)

    def test_pinned_export_bytes_cannot_be_changed(self):
        path = Path(self.pair["arms"]["native"]["predictions"]["path"])
        data = json.loads(path.read_bytes())
        data["coco"][0]["score"] = 0.9
        self.write("native/DEV_NATIVE_PREDICTIONS_PRIVATE.jsonl", data)
        with self.assertRaisesRegex(ValueError, "Pinned prediction SHA-256 mismatch"):
            self.score()

    def test_training_arm_cannot_be_swapped_inside_rehashed_export(self):
        self.report["native"]["checkpoint_sha256"] = self.training_pair["center_checkpoint_sha256"]
        self.pair["arms"]["native"]["report"] = self.write("native/DEV_NATIVE_EXPORT_SAFE.json", self.report["native"])
        with self.assertRaisesRegex(ValueError, "Exact closed native export report required"):
            self.score()

    def test_original_gt_pin_cannot_be_changed_in_rehashed_pair(self):
        self.pair["gt"] = self.write("new_gt.json", {"images": [], "annotations": []})
        with self.assertRaisesRegex(ValueError, "Fixed original DEV GT digest required"):
            self.score()

    def test_report_totals_must_reconcile_with_predictions(self):
        self.report["center"]["mapped_rows"] = 1
        self.report["center"]["unmapped_slot1_rows"] = 99
        self.pair["arms"]["center"]["report"] = self.write("center/DEV_NATIVE_EXPORT_SAFE.json", self.report["center"])
        with self.assertRaisesRegex(ValueError, "candidate counts differ"):
            self.score()

    def test_original_os_wait_is_required(self):
        self.pair["arms"]["center"]["original_os_wait"] = 1
        with self.assertRaisesRegex(ValueError, "must have closed cleanly"):
            self.score()

    def test_cli_refuses_existing_output_before_scoring(self):
        path = Path(self.write("kept.json", {"keep": True})["path"])
        argv = ["score_closed_pair.py", "--pair", str(path), "--pair-sha256", "0" * 64, "--output", str(path)]
        with patch("sys.argv", argv):
            with self.assertRaisesRegex(ValueError, "Fresh canonical output file required"):
                driver.main()
        self.assertEqual(json.loads(path.read_bytes()), {"keep": True})


if __name__ == "__main__":
    unittest.main()
