"""Synthetic integration of the frozen sparse verifier, matcher, and features."""
from hashlib import sha256
from io import BytesIO
import unittest

from materialize import FEATURE_COLUMNS, materialize_image
from capture_post import verify_capture
from features import extract_features
from fixed_match_labels import build_fixed_labels
from test_fixed_match_labels import COCOeval, annotation, capture, mask, np, synthetic_coco


def inputs():
    rows, arrays = capture([[2, 3, 5], [2, 3, 5], []])
    gray = np.arange(32, dtype=np.uint8).reshape(4, 8)
    rows["gray_sha256"] = sha256(gray.tobytes()).hexdigest()
    return rows, arrays, gray


def build(rows, arrays, gray, annotations):
    return materialize_image(
        np, mask, COCOeval, integer_coco=synthetic_coco(annotations),
        rows_document=rows, arrays=arrays, gray=gray, verify_capture=verify_capture,
        build_fixed_labels=build_fixed_labels, extract_features=extract_features,
    )


class MaterializeTest(unittest.TestCase):
    def test_duplicate_and_empty_native_rows_survive_while_only_positive_is_labeled(self):
        rows, arrays, gray = inputs()
        output = build(rows, arrays, gray, [annotation()])
        self.assertEqual(output["rows_document"], rows)
        self.assertEqual(FEATURE_COLUMNS, ("abs_logit", "mean_contrast", "rank"))
        self.assertEqual([r["status"] for r in output["match_rows"]],
                         ["matched", "unmatched", "unmatched"])
        self.assertTrue(all("flip_correct" not in r for r in output["match_rows"]))
        a = output["arrays"]
        for key in ("positions", "offsets", "logits"):
            np.testing.assert_array_equal(a[key], arrays[key], strict=True)
        self.assertEqual(a["features"].shape, (64, 3))
        self.assertEqual(a["features"].dtype, np.float32)
        self.assertEqual(a["edit_sign"].dtype, np.int8)
        self.assertEqual(a["valid"].dtype, np.bool_)
        np.testing.assert_array_equal(a["features"][:32], a["features"][32:])
        np.testing.assert_array_equal(a["label_offsets"], np.array([0, 32, 32, 32], dtype=np.int64), strict=True)
        self.assertEqual(a["y"].dtype, np.bool_)
        self.assertEqual(a["positions"][:32][a["y"]].tolist(), [12, 13])


    def test_gt_presence_does_not_change_the_feature_population(self):
        rows, arrays, gray = inputs()
        stream = BytesIO()
        np.savez(stream, **arrays)
        stream.seek(0)
        with np.load(stream, allow_pickle=False) as archive:
            without_gt = build(rows, archive, gray, [])
        with_gt = build(rows, arrays, gray, [annotation()])
        self.assertEqual(without_gt["rows_document"], with_gt["rows_document"])
        for key in ("positions", "offsets", "logits", "features", "edit_sign", "valid"):
            np.testing.assert_array_equal(without_gt["arrays"][key], with_gt["arrays"][key], strict=True)
        self.assertTrue(all(r["status"] == "unmatched" for r in without_gt["match_rows"]))
        np.testing.assert_array_equal(without_gt["arrays"]["label_offsets"], np.zeros(4, dtype=np.int64), strict=True)
        self.assertEqual(without_gt["arrays"]["y"].size, 0)

    def test_gray_pixel_change_rejects_before_gt_or_feature_work(self):
        rows, arrays, gray = inputs()
        gray[0, 0] ^= np.uint8(1)
        with self.assertRaisesRegex(ValueError, "capture digest"):
            materialize_image(
                np, None, None, integer_coco=None, rows_document=rows, arrays=arrays, gray=gray,
                verify_capture=None, build_fixed_labels=None, extract_features=None,
            )

    def test_empty_capture_retains_a_zero_row_feature_matrix(self):
        rows, arrays, gray = inputs()
        rows["records"] = []
        arrays["offsets"] = arrays["offsets"][:1]
        arrays["positions"] = arrays["positions"][:0]
        arrays["logits"] = arrays["logits"][:0]
        output = build(rows, arrays, gray, [annotation()])
        self.assertEqual(output["arrays"]["features"].shape, (0, 3))
        self.assertEqual(output["arrays"]["label_offsets"].tolist(), [0])
        self.assertEqual(output["match_rows"], [])
        self.assertEqual(output["summary"]["unmatched_GT"], 1)


if __name__ == "__main__":
    unittest.main()
