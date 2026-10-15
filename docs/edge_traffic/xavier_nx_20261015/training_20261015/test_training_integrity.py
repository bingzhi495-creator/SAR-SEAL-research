"""Protocol check: identical images must never cross experiment splits."""
import tempfile
from pathlib import Path
import unittest

from training_integrity import audit_split_leakage


class IntegrityCheck(unittest.TestCase):
    def test_rejects_same_image_bytes_with_different_names_across_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            a = Path(directory) / "train.jpg"
            b = Path(directory) / "unrelated_test_name.jpg"
            a.write_bytes(b"identical image bytes")
            b.write_bytes(a.read_bytes())
            with self.assertRaisesRegex(ValueError, "cross-split"):
                audit_split_leakage({"train": [a], "test": [b]})


if __name__ == "__main__":
    unittest.main()
