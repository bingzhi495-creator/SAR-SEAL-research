"""Synthetic NumPy-only contract tests for one-image sparse capture."""
import unittest

import numpy as np

from capture_post import verify_capture


class SyntheticCodec:
    """Small reversible binary codec implementing the COCO mask API shape."""

    @staticmethod
    def encode(mask):
        assert mask.flags.f_contiguous
        counts = bytes(ord("0") + int(value) for value in mask.ravel(order="F"))
        return {"size": list(mask.shape), "counts": counts}

    @staticmethod
    def decode(rle):
        values = [value - ord("0") for value in rle["counts"]]
        return np.asarray(values, dtype=np.uint8).reshape(rle["size"], order="F")


def fixture():
    rows = {
        "image_id": 42, "image_sha256": "a" * 64, "gray_sha256": "b" * 64,
        "original_hw": [1, 7],
        "records": [
            {"native_rank": 2, "query_id": 1, "score": 0.75,
             "segmentation": {"size": [1, 7], "counts": "0001000"}},
            {"native_rank": 4, "query_id": 8, "score": 0.75,
             "segmentation": {"size": [1, 7], "counts": "1111111"}},
        ],
    }
    arrays = {
        "offsets": np.array([0, 5, 5], dtype=np.int64),
        "positions": np.array([1, 2, 3, 4, 5], dtype=np.uint32),
        "logits": np.array([-1, 0, 2, -2, -0.0], dtype=np.float32),
    }
    return rows, arrays


class VerifyCaptureTest(unittest.TestCase):
    def test_clipped_band_and_valid_stock_rank_gaps(self):
        rows, arrays = fixture()
        self.assertEqual(verify_capture(np, SyntheticCodec, rows, arrays), {
            "images": 1, "retained": 2, "empty_bands": 1, "support_pixels": 5,
            "positive_logits": 1, "negative_logits": 2, "zero_logits": 2,
        })


    def test_rejects_wrong_positions_and_array_contract(self):
        for key, value, message in (
            ("positions", np.array([1, 3, 2, 4, 5], dtype=np.uint32), "sorted clipped"),
            ("positions", np.array([1, 2, 3, 4, 6], dtype=np.uint32), "sorted clipped"),
            ("positions", np.array([1, 2, 3, 4, 5], dtype=np.int64), "uint32 positions"),
            ("offsets", np.array([0, 5, 4], dtype=np.int64), "partition"),
            ("offsets", np.array([0, 4, 5], dtype=np.int64), "sorted clipped"),
            ("logits", np.array([-1, 0, 0, -2, -0.0], dtype=np.float32), "positive sparse"),
            ("logits", np.array([-1, 0, 2, -2, np.nan], dtype=np.float32), "finite"),
        ):
            with self.subTest(key=key, message=message):
                rows, arrays = fixture()
                arrays[key] = value
                with self.assertRaisesRegex(ValueError, message):
                    verify_capture(np, SyntheticCodec, rows, arrays)

    def test_rejects_noncanonical_stock_rows_and_rle(self):
        for changes, message in (
            ({"query_id": 0}, "canonical native rank"),
            ({"query_id": 1}, "canonical native rank"),
            ({"native_rank": 2}, "canonical native rank"),
            ({"score": 0.875}, "canonical native rank"),
            ({"score": 0.1}, "canonical native rank"),
            ({"segmentation": {"size": [7, 1], "counts": "1111111"}}, "original-grid"),
            ({"segmentation": {"size": [1, 7], "counts": "1111112"}}, "canonical binary"),
        ):
            with self.subTest(changes=changes):
                rows, arrays = fixture()
                rows["records"][1].update(changes)
                with self.assertRaisesRegex(ValueError, message):
                    verify_capture(np, SyntheticCodec, rows, arrays)

    def test_asymmetric_two_dimensional_band_is_row_major(self):
        rows, _ = fixture()
        rows["original_hw"] = [4, 6]
        # A single foreground corner at y=0,x=5 gives the clipped rectangle
        # rows 0..2, columns 3..5. Positions here are C-order, RLE is F-order.
        rows["records"] = [{
            "native_rank": 3, "query_id": 2, "score": 0.5,
            "segmentation": {"size": [4, 6], "counts": "0" * 20 + "1000"},
        }]
        arrays = {
            "offsets": np.array([0, 9], dtype=np.int64),
            "positions": np.array([3, 4, 5, 9, 10, 11, 15, 16, 17], dtype=np.uint32),
            "logits": np.array([-1, -1, 1, -1, -1, -1, -1, -1, -1], dtype=np.float32),
        }
        result = verify_capture(np, SyntheticCodec, rows, arrays)
        self.assertEqual((result["support_pixels"], result["positive_logits"]), (9, 1))

    def test_all_native_rows_can_be_filtered(self):
        rows, arrays = fixture()
        rows["records"] = []
        arrays["offsets"] = np.array([0], dtype=np.int64)
        arrays["positions"] = arrays["positions"][:0]
        arrays["logits"] = arrays["logits"][:0]
        self.assertEqual(verify_capture(np, SyntheticCodec, rows, arrays)["retained"], 0)


if __name__ == "__main__":
    unittest.main()
