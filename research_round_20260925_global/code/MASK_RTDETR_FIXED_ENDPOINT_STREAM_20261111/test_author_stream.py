import tempfile
import unittest
from pathlib import Path

from author_stream import read_mask_stream, write_author_stream


def author_block(image_id):
    return {
        "bbox": [
            {"image_id": image_id, "category_id": 1, "score": .8, "bbox": [0, 0, 2, 2]},
            {"image_id": image_id, "category_id": 1, "score": .8, "bbox": [1, 1, 2, 2]},
        ],
        "mask": [
            {"image_id": image_id, "category_id": 1, "score": .8, "segmentation": {"size": [4, 4], "counts": "037"}},
            {"image_id": image_id, "category_id": 1, "score": .8, "segmentation": {"size": [4, 4], "counts": "a0"}},
        ],
    }


class AuthorStreamTest(unittest.TestCase):
    def test_write_and_read_complete_paired_stream_preserves_ties_and_masks(self):
        roster = {19: (4, 4), 3: (4, 4)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            digest = write_author_stream(path, roster, (author_block(key) for key in roster), per_image=2)
            result = read_mask_stream(path, roster, digest, per_image=2)
            self.assertEqual(result, author_block(19)["mask"] + author_block(3)["mask"])


    def test_rejects_incomplete_or_reordered_native_image_streams(self):
        roster = {19: (4, 4), 3: (4, 4)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            with self.assertRaises(ValueError):
                write_author_stream(path, roster, [author_block(19)], per_image=2)
            self.assertFalse(path.exists())
            path = Path(directory) / "reordered.jsonl"
            with self.assertRaises(ValueError):
                write_author_stream(path, roster, [author_block(3), author_block(19)], per_image=2)
            self.assertFalse(path.exists())

    def test_rejects_changed_digest_before_providing_predictions(self):
        roster = {19: (4, 4)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            write_author_stream(path, roster, [author_block(19)], per_image=2)
            with self.assertRaisesRegex(ValueError, "digest"):
                read_mask_stream(path, roster, "0" * 64, per_image=2)


if __name__ == "__main__":
    unittest.main()
