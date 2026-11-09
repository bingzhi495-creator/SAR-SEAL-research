from hashlib import sha256
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import dev705_group_admission as admission

GEOMETRY_DIR = Path(__file__).resolve().parent.parent / "HRSID_MASK_PHASE_PAIRED_GEOMETRY_20261109"
sys.path.insert(0, str(GEOMETRY_DIR))
import test_score_closed_pair as original_fixture


FIRST_TRAIN = "d66baf3c57fafdba4242ad1b55e08cf76bfd77f7"
FIRST_EVAL = "4e5ea5db93423d4f4c0a814ca18e8386313cb58a"
REPEAT_TRAIN = "a49361b92977a605267025b1e7798420936cbe53"
REPEAT_EVAL = "0410a3268ba9192735588003124f3e364a6b494d"


def score_population(annotations):
    return dict(annotations=annotations, AP=.5, AP75=.3, AP90=.2, APs=.4, AR100=.6,
                all12=[.5, .8, .3, .4, .6, -1., .1, .5, .6, .6, .6, -1.])


class GroupAdmissionTest(unittest.TestCase):
    def setUp(self):
        fixture = original_fixture.ClosedPairTest("test_qualified_streams_score_original_order")
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        self.fixture = fixture
        self.pair = fixture.pair
        self.pair["protocol"] = FIRST_EVAL
        name = "P2039_chip.png"
        gt = json.loads(Path(fixture.gt["path"]).read_bytes())
        gt["images"][0]["file_name"] = name
        self.pair["gt"] = fixture.write("gt.json", gt)
        self.manifest = fixture.write("manifest.json", [dict(image_id=7, file_name=name, width=8, height=2)])
        self.driver = admission.load_driver()
        self.driver.GT_SHA = self.pair["gt"]["sha256"]
        self.driver.IMAGE_MANIFEST_SHA = self.manifest["sha256"]
        self.driver.DEV_IMAGES, self.driver.GT_COUNT, self.driver.EMPTY_COUNT = 1, 2, 1
        self.driver.CANVAS = (8, 2)
        self.group_digest = sha256(b'[[7,"P2039"]]\n').hexdigest()
        self.reports = {}
        for mode in self.driver.ARMS:
            export = fixture.report[mode]
            export.update(protocol=FIRST_EVAL, image_manifest_sha256=self.manifest["sha256"])
            self.pair["arms"][mode]["report"] = fixture.write(mode + "/DEV_NATIVE_EXPORT_SAFE.json", export)
            self.reports[mode] = dict(
                status="HRSID_DEV705_E24_NATIVE_COCO_COMPLETE", protocol=FIRST_EVAL,
                training_protocol=FIRST_TRAIN, checkpoint_sha256=export["checkpoint_sha256"],
                predictions_sha256=export["predictions_sha256"], initialization="official_COCO_only",
                original_dev_gt_sha256=self.driver.GT_SHA, images=1, excluded_empty_gt_sensitivity=1,
                max_dets=[1, 10, 100], units="fraction", official_TEST_opened=False, official_TEST_AP=False,
                primary=score_population(2), sensitivity=score_population(1),
            )

    def load_population(self):
        with patch.multiple(admission, GROUP_MAP_SHA256=self.group_digest, GROUP_COUNT=1):
            return admission.load_population(self.driver, self.manifest, self.pair["gt"])

    def read_pair(self, seed=5201, training=FIRST_TRAIN, evaluation=FIRST_EVAL):
        return admission.read_closed_pair(self.driver, self.fixture.write("pair.json", self.pair),
                                          training_seed=seed, training_protocol=training,
                                          evaluation_protocol=evaluation)

    def read_arm(self, mode, images, training=FIRST_TRAIN, evaluation=FIRST_EVAL):
        score_pin = self.fixture.write(mode + "/SCORE_SAFE.json", self.reports[mode])
        return admission.read_arm(
            self.driver, self.pair["arms"][mode], score_pin, training_protocol=training,
            evaluation_protocol=evaluation, checkpoint_sha=self.fixture.report[mode]["checkpoint_sha256"],
            images=images,
        )

    def test_locked_population_and_pair_preserve_original_arm_order_and_scores(self):
        gt, images, groups = self.load_population()
        self.assertEqual(groups, {7: "P2039"})
        self.assertEqual(gt["images"], images)
        self.assertEqual(self.read_pair(), (self.pair, self.fixture.training_pair))
        rows, expected, pins = self.read_arm("native", images)
        self.assertEqual([row["score"] for row in rows], [1., .5])
        self.assertEqual(expected, {"AP": .5, "AP75": .3})
        self.assertEqual(pins["predictions_sha256"], self.pair["arms"]["native"]["predictions"]["sha256"])
        self.assertNotIn(str(self.fixture.root), json.dumps(pins))

    def test_repeat_seed_must_be_present_in_both_original_receipts(self):
        self.pair.update(seed=5202, protocol=REPEAT_EVAL)
        self.fixture.training_pair.update(seed=5202, training_protocol=REPEAT_TRAIN)
        self.pair["training_pair"] = self.fixture.write("train_pair.json", self.fixture.training_pair)
        self.assertEqual(self.read_pair(5202, REPEAT_TRAIN, REPEAT_EVAL), (self.pair, self.fixture.training_pair))
        self.fixture.training_pair["seed"] = 5203
        self.pair["training_pair"] = self.fixture.write("train_pair.json", self.fixture.training_pair)
        with self.assertRaisesRegex(ValueError, "seed-specific accepted E24 training pair"):
            self.read_pair(5202, REPEAT_TRAIN, REPEAT_EVAL)
        with self.assertRaisesRegex(ValueError, "prelocked 5201 or 5202"):
            self.read_pair(5203, REPEAT_TRAIN, REPEAT_EVAL)

    def test_manifest_population_cannot_be_changed_even_with_a_new_pin(self):
        forged = [dict(image_id=7, file_name="P9999_chip.png", width=8, height=2)]
        self.manifest = self.fixture.write("manifest.json", forged)
        with self.assertRaisesRegex(ValueError, "fixed|Fixed original"):
            self.load_population()
        self.driver.IMAGE_MANIFEST_SHA = self.manifest["sha256"]
        with self.assertRaisesRegex(ValueError, "image order and identities"):
            self.load_population()

    def test_group_membership_is_independently_pinned(self):
        self.group_digest = "0" * 64
        with self.assertRaisesRegex(ValueError, "fixed DEV705 P-prefix membership"):
            self.load_population()

    def test_score_report_must_match_both_prediction_and_checkpoint(self):
        _, images, _ = self.load_population()
        score = self.reports["native"]
        for key in ("predictions_sha256", "checkpoint_sha256", "original_dev_gt_sha256"):
            with self.subTest(key=key):
                value = score[key]
                score[key] = "0" * 64
                try:
                    with self.assertRaisesRegex(ValueError, "closed COCO report bound"):
                        self.read_arm("native", images)
                finally:
                    score[key] = value
        score["primary"]["AP75"] = .2
        with self.assertRaisesRegex(ValueError, "fractional AP and AP75"):
            self.read_arm("native", images)

    def test_native_rank_order_is_validated_without_reranking(self):
        _, images, _ = self.load_population()
        arm = self.pair["arms"]["native"]
        stream = Path(arm["predictions"]["path"])
        block = json.loads(stream.read_bytes())
        block["native_ranks"].reverse()
        arm["predictions"] = self.fixture.write("native/DEV_NATIVE_PREDICTIONS_PRIVATE.jsonl", block)
        sha = arm["predictions"]["sha256"]
        self.fixture.report["native"]["predictions_sha256"] = sha
        self.reports["native"]["predictions_sha256"] = sha
        arm["report"] = self.fixture.write("native/DEV_NATIVE_EXPORT_SAFE.json", self.fixture.report["native"])
        with self.assertRaisesRegex(ValueError, "mapped native ranks"):
            self.read_arm("native", images)


if __name__ == "__main__":
    unittest.main()
