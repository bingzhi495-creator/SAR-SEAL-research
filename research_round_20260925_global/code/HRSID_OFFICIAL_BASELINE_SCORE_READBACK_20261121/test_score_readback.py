"""Compare independent readback and official scoring on generated predictions."""

from copy import deepcopy
from hashlib import sha256
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import verify_score as reader


def module_from(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IndependentScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sibling = Path(__file__).resolve().parent.parent / "HRSID_OFFICIAL_BASELINE_SCORE_WORK_20261121_01"
        source_dir = Path(os.environ.get("TEST_ORIGINAL_SCORE_DIR", sibling))
        cls.original = module_from("_original_synthetic_score", source_dir / "score_native.py")
        with patch.dict(sys.modules, {"score_native": cls.original}):
            cls.fixture_source = module_from("_synthetic_score_fixture", source_dir / "test_score_native.py")
        cls.adapter = cls.original.adapter_from(Path(os.environ["NATIVE_SCORE_ADAPTER"]))

    def compare_fixture(self, gt, views, blocks):
        rows, digests = self.fixture_source.parse(blocks)
        before = deepcopy((gt, views, rows, digests))
        expected = self.original.evaluate_views(gt, views, rows, digests, self.adapter)
        self.assertEqual((gt, views, rows, digests), before)
        actual = reader.evaluate_views(gt, views, rows, digests)
        self.assertEqual((gt, views, rows, digests), before)
        self.assertEqual(actual, expected)
        reader.compare_views(actual, expected)
        self.assertEqual(set(actual), {"official", "empty_raster_sensitivity", "image_identity_sensitivity"})
        for name, view in views.items():
            expected_digest = sha256(b"".join(digests[i] for i in view["images"])).hexdigest()
            self.assertEqual(actual[name]["ordered_prediction_block_digests_sha256"], expected_digest)
            self.assertEqual(set(actual[name]["metrics"]), set(self.original.METRICS) | {"AP90"})
        return actual

    def test_original_fixture_all_metrics_all_views(self):
        measured = self.compare_fixture(*self.fixture_source.fixture())
        primary = measured["official"]
        self.assertEqual(primary["images"], 3)
        self.assertEqual(primary["images_without_annotations"], 1)
        self.assertEqual(primary["images_without_predictions"], 1)
        self.assertEqual(primary["mapped_predictions"], 3)
        self.assertGreater(measured["empty_raster_sensitivity"]["metrics"]["AP"], primary["metrics"]["AP"])
        self.assertGreater(measured["image_identity_sensitivity"]["metrics"]["AP"], primary["metrics"]["AP"])

    def test_globally_empty_predictions(self):
        gt, views, blocks = self.fixture_source.fixture()
        for b in blocks:
            b.update(native_ranks=[], unmapped_slot1_count=100, coco=[])
        measured = self.compare_fixture(gt, views, blocks)
        for view in measured.values():
            self.assertEqual(view["metrics"]["AP"], 0)
            self.assertEqual(view["metrics"]["AP90"], 0)
            self.assertEqual(view["images_without_predictions"], view["images"])
            self.assertEqual(view["mapped_predictions"], 0)

    def test_false_positive_on_unannotated_image_penalizes_primary(self):
        gt, views, blocks = self.fixture_source.fixture()
        negative = blocks[0]
        self.assertNotIn(negative["image_id"], {item["image_id"] for item in gt["annotations"]})
        with_fp = self.compare_fixture(gt, views, blocks)
        negative.update(native_ranks=[], unmapped_slot1_count=100, coco=[])
        without_fp = self.compare_fixture(gt, views, blocks)
        self.assertGreater(without_fp["official"]["metrics"]["AP"], with_fp["official"]["metrics"]["AP"])

    def test_reject_modified_metrics_block_digest_counts_or_views(self):
        actual = self.compare_fixture(*self.fixture_source.fixture())
        for change in ("AP90", "hash", "predictions", "drop_view", "drop_metric"):
            reported = deepcopy(actual)
            if change == "AP90":
                reported["official"]["metrics"]["AP90"] += 0.01
            elif change == "hash":
                reported["empty_raster_sensitivity"]["ordered_prediction_block_digests_sha256"] = "0" * 64
            elif change == "predictions":
                reported["image_identity_sensitivity"]["mapped_predictions"] += 1
            elif change == "drop_view":
                del reported["empty_raster_sensitivity"]
            else:
                del reported["image_identity_sensitivity"]["metrics"]["AP"]
            with self.subTest(change=change), self.assertRaises(ValueError):
                reader.compare_views(actual, reported)


if __name__ == "__main__":
    unittest.main()
