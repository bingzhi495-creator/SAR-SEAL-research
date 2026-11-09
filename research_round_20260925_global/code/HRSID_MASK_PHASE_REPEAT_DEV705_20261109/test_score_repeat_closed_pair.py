import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import score_repeat_closed_pair as repeat


GEOMETRY_DIR = Path(__file__).resolve().parent.parent / "HRSID_MASK_PHASE_PAIRED_GEOMETRY_20261109"
sys.path.insert(0, str(GEOMETRY_DIR))
import test_score_closed_pair as original_fixture


class RepeatClosedPairTest(unittest.TestCase):
    def setUp(self):
        self.original = original_fixture.ClosedPairTest("test_qualified_streams_score_original_order")
        self.addCleanup(self.original.doCleanups)
        self.original.setUp()
        self.seed = 5202
        self.original.pair["seed"] = self.seed
        self.original.training_pair.update(training_protocol=repeat.REPEAT_PROTOCOL, seed=self.seed)
        self.write_training_pair()
        for arm in ("native", "center"):
            self.original.report[arm]["training_protocol"] = repeat.REPEAT_PROTOCOL
            self.write_report(arm)

    def write_training_pair(self):
        self.original.pair["training_pair"] = self.original.write("train_pair.json", self.original.training_pair)

    def write_report(self, arm):
        self.original.pair["arms"][arm]["report"] = self.original.write(
            arm + "/DEV_NATIVE_EXPORT_SAFE.json", self.original.report[arm],
        )

    def score(self, seed=None, digest=None):
        driver = repeat.load_driver()
        pin = self.original.write("pair.json", self.original.pair)
        with patch.object(repeat, "load_driver", return_value=driver), patch.multiple(
            driver, GT_SHA=self.original.gt["sha256"], DEV_IMAGES=1, GT_COUNT=2, EMPTY_COUNT=1, CANVAS=(8, 2),
        ):
            return repeat.score_pair(Path(pin["path"]), digest or pin["sha256"], self.seed if seed is None else seed)

    def test_closed_repeat_scores_unchanged_geometry_and_binds_seed(self):
        result = self.score()
        self.assertEqual(result["seed"], self.seed)
        self.assertEqual(result["training_protocol"], repeat.REPEAT_PROTOCOL)
        self.assertEqual(result["status"], "HRSID_MASK_PHASE_E24_DEV705_PAIRED_GEOMETRY_COMPLETE")
        self.assertEqual(result["metrics"]["point_five"]["both_matched"], 1)
        self.assertEqual(result["metrics"]["strict"]["0.90"]["matched_below"]["strict"], 1)
        self.assertNotIn(str(self.original.root), json.dumps(result))


    def test_seed_5203_can_be_scored_in_an_independent_scope(self):
        self.original.pair["seed"] = 5203
        self.original.training_pair["seed"] = 5203
        self.write_training_pair()

        result = self.score(seed=5203)

        self.assertEqual(result["seed"], 5203)
        self.assertEqual(result["metrics"]["point_five"]["both_matched"], 1)

    def test_export_receipt_cannot_cross_seed_or_omit_its_seed(self):
        for seed in (5203, 5202.0, None):
            with self.subTest(seed=seed):
                if seed is None:
                    self.original.pair.pop("seed")
                else:
                    self.original.pair["seed"] = seed
                with self.assertRaisesRegex(ValueError, "closed export pair receipt.*repeat seed"):
                    self.score()

    def test_training_receipt_cannot_cross_seed_inside_rehashed_export_pair(self):
        self.original.training_pair["seed"] = 5203
        self.write_training_pair()
        with self.assertRaisesRegex(ValueError, "accepted E24 training pair.*repeat seed"):
            self.score()

    def test_same_seed_does_not_authorize_a_different_export_checkpoint_or_protocol(self):
        report = self.original.report["center"]
        for mutation in (
            {"checkpoint_sha256": "0" * 64},
            {"training_protocol": "d66baf3c57fafdba4242ad1b55e08cf76bfd77f7"},
        ):
            with self.subTest(mutation=mutation):
                original = report.copy()
                report.update(mutation)
                self.write_report("center")
                with self.assertRaisesRegex(ValueError, "Exact closed native export report"):
                    self.score()
                report.update(original)

    def test_pair_digest_and_original_os_wait_are_still_required(self):
        with self.assertRaisesRegex(ValueError, "Pinned input size or SHA-256 mismatch"):
            self.score(digest="0" * 64)
        self.original.pair["arms"]["native"]["original_os_wait"] = False
        with self.assertRaisesRegex(ValueError, "must have closed cleanly"):
            self.score()

    def test_original_driver_has_an_exact_source_pin(self):
        path = self.original.root / "changed_driver.py"
        path.write_bytes(repeat.DRIVER_SOURCE.read_bytes() + b"\n# Changed source\n")
        with patch.object(repeat, "DRIVER_SOURCE", path):
            with self.assertRaisesRegex(ValueError, "Exact immutable closed geometry driver"):
                repeat.load_driver()

    def test_unauthorized_seed_cannot_use_the_bridge(self):
        for seed in (5201, 5204, 5202.0):
            with self.subTest(seed=seed):
                with self.assertRaisesRegex(ValueError, "Exact prelocked repeat training seed"):
                    self.score(seed=seed)


if __name__ == "__main__":
    unittest.main()
