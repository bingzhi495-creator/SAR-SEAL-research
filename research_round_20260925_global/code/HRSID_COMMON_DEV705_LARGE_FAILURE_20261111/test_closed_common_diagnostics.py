"""Synthetic input capture and paired geometry diagnostics for closed common DEV705."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import closed_common_diagnostics as diagnostic


class VerifiedInputTest(unittest.TestCase):
    def test_gate_drives_the_single_checked_gt_and_both_candidate_reads(self):
        pair_sha, gt_sha = "a" * 64, "b" * 64
        training_protocol, evaluation_protocol = "1" * 40, "2" * 40
        pair = Path("synthetic-pair.json")
        native_receipt, center_receipt = Path("a.json"), Path("b.json")
        gt = {"images": [{"id": 1}, {"id": 2}], "annotations": []}
        rows = {
            "native": [{"image_id": 1, "score": .9}, {"image_id": 1, "score": .4}],
            "center": [{"image_id": 2, "score": .7}],
        }
        arms = ("native", "center")
        checkpoint = dict(zip(arms, ("c" * 64, "d" * 64), strict=True))
        admitted = {
            mode: dict(checkpoint_sha256=checkpoint[mode], export_report_sha256="e" * 64,
                       predictions_sha256="f" * 64)
            for mode in arms
        }
        calls = []

        def read_json(path, digest, limit):
            self.assertEqual(digest, gt_sha)
            calls.append("gt")
            return gt

        def read_arm(arm, digest, protocol, images):
            self.assertEqual(digest, checkpoint[arm])
            self.assertEqual(protocol, evaluation_protocol)
            self.assertIs(images, gt["images"])
            calls.append(arm)
            blocks = [{"coco": [row for row in rows[arm] if row["image_id"] == image["id"]]} for image in images]
            return Path(arm), blocks

        driver = SimpleNamespace(MIB=2**20, GT_SHA=gt_sha, ARMS=arms, read_json=read_json, read_arm=read_arm)
        gate = SimpleNamespace(load_driver=lambda: driver, OUTPUT_STATUS="closed_geometry")

        def score_pair(path, digest, **kwargs):
            self.assertEqual((path, digest), (pair, pair_sha))
            self.assertEqual(kwargs, dict(native_arm_receipt=native_receipt, center_arm_receipt=center_receipt,
                                          training_protocol=training_protocol))
            admitted_driver = gate.load_driver()
            population = admitted_driver.read_json(Path("admitted-only-gt.json"), gt_sha, 4 * 2**20)
            for mode in arms:
                admitted_driver.read_arm(mode, checkpoint[mode], evaluation_protocol, population["images"])
            return dict(status=gate.OUTPUT_STATUS, protocol=evaluation_protocol, training_protocol=training_protocol,
                        training_seed=5202, support_policy="common", original_dev_gt_sha256=gt_sha,
                        training_pair_receipt_sha256="4" * 64, closed_export_pair_sha256=pair_sha,
                        arms=admitted, score_report_sha256s={mode: "5" * 64 for mode in arms},
                        official_TEST_opened=False)

        gate.score_pair = score_pair
        with patch.object(diagnostic, "load_gate", return_value=gate):
            loaded_gt, native, center, provenance = diagnostic.verified_inputs(
                Path("code"), pair, pair_sha, native_arm_receipt=native_receipt,
                center_arm_receipt=center_receipt, training_protocol=training_protocol,
            )
        self.assertEqual(calls, ["gt", "native", "center"])
        self.assertIs(loaded_gt, gt)
        self.assertEqual(native, rows["native"])
        self.assertEqual(center, rows["center"])
        self.assertEqual(provenance["arms"]["center"]["checkpoint_sha256"], checkpoint["center"])
        self.assertEqual(provenance["arms"]["native"]["score_report_sha256"], "5" * 64)
        self.assertNotIn("admitted-only-gt", json.dumps(provenance))


class PinnedRuntimeTest(unittest.TestCase):
    def test_stock_gate_refuses_unpinned_pair_before_other_inputs(self):
        code = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as folder:
            pair = Path(folder).resolve() / "pair.json"
            pair.write_text("{}", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "Pinned input size or SHA-256 mismatch"):
                diagnostic.verified_inputs(code, pair, "a" * 64, native_arm_receipt="missing-a",
                                           center_arm_receipt="missing-b", training_protocol="1" * 40)

    def test_cli_refuses_existing_output_before_admission(self):
        with TemporaryDirectory() as folder:
            output = Path(folder) / "prior.json"
            output.write_text("prior", encoding="ascii")
            with patch.object(diagnostic, "verified_inputs", side_effect=AssertionError("premature admission")):
                with self.assertRaisesRegex(ValueError, "Fresh output"):
                    diagnostic.main(["--code-root", "unused", "--pair", "unused", "--pair-sha256", "a" * 64,
                                     "--native-arm-receipt", "unused", "--center-arm-receipt", "unused",
                                     "--training-protocol", "1" * 40, "--output", str(output)])
            self.assertEqual(output.read_text(encoding="ascii"), "prior")

    def test_stock_apis_include_canonical_large_ap_and_full_large_support(self):
        from pycocotools import mask

        code = Path(__file__).resolve().parents[1]
        images = [dict(id=i, height=800, width=800) for i in range(1, 706)]
        annotations = []
        for image in images:
            count = 3 if image["id"] <= 372 else 2
            for j in range(count):
                if image["id"] == 1 and j < 2:
                    x, y, width = (0, 0, 100) if j == 0 else (200, 100, 100)
                else:
                    x, y, width = 400 + 20 * j, 300, 10
                vertices = [x, y, x + width, y, x + width, y + width, x, y + width]
                raster, = mask.frPyObjects([vertices], 800, 800)
                if image["id"] >= 704:
                    raster, = mask.frPyObjects([dict(size=[800, 800], counts=[640000])], 800, 800)
                raster["counts"] = raster["counts"].decode("ascii")
                annotations.append(dict(id=len(annotations) + 1, image_id=image["id"], category_id=1,
                                        iscrowd=0, area=width * width, bbox=[x, y, width, width],
                                        segmentation=raster))
        self.assertEqual(len(annotations), 1782)
        gt = dict(images=images, annotations=annotations, categories=[dict(id=1, name="ship")])
        rows = [
            dict(image_id=1, category_id=1, score=.9 if i < 2 else .8, segmentation=a["segmentation"])
            for i, a in enumerate(annotations[:3])
        ]
        result = diagnostic.analyze_diagnostics(code, gt, [rows[0], rows[2]], [rows[1], rows[2]])
        area = result["eligible_gt_match_support_by_stratum"]
        self.assertEqual(area["large_edge"], dict(eligible_gt=1, both=0, native_only=1, center_only=0, neither=0))
        self.assertEqual(area["large_interior"], dict(eligible_gt=1, both=0, native_only=0, center_only=1, neither=0))
        self.assertEqual(area["small_interior"]["both"], 1)
        self.assertEqual(sum(group["eligible_gt"] for group in area.values()), 1778)
        score = result["equal_prefix_score_template"]
        for mode in ("native", "center"):
            for donor in ("native", "center"):
                self.assertGreater(score["cells"][mode][donor]["APl"], 0)
        self.assertEqual(score["contrasts_points"]["geometry_with_native_scores"]["APl"], 0.)
        self.assertEqual(result["paired_candidate_coverage"]["equal_nonempty_prefix"]["rows_per_arm"], 2)


if __name__ == "__main__":
    unittest.main()
