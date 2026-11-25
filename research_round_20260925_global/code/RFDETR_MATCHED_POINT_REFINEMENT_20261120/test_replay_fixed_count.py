"""Public example and regression checks for the fixed-count mask writer."""
import json
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from pycocotools import mask as codec


def rle(mask):
    encoded = codec.encode(np.asfortranarray(mask, dtype=np.uint8))
    return {"size": encoded["size"], "counts": encoded["counts"].decode("ascii")}


class WriterTest(unittest.TestCase):
    def test_cli_preserves_float32_pair_ties_and_candidate_metadata(self):
        with TemporaryDirectory() as root:
            folder = Path(root)
            input_path = folder / "candidates.jsonl"
            result = folder / "replay"
            native = np.zeros((5, 8), dtype=np.uint8)
            native.flat[:20] = 1
            candidate = {"image_id": 9, "category_id": 1, "score": 0.41,
                         "bbox": [0, 0, 8, 5], "rank": 2, "segmentation": rle(native)}
            row = {"candidate": candidate, "pairs": [[20, 0], [21, 1]],
                   "occupancy": [[0.75, 2**-25], [0.75, 0.0]], "cap": 1}
            input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            entry = Path(__file__).with_name("replay_fixed_count.py")
            with patch.object(sys, "argv", [str(entry), "--input", str(input_path), "--output", str(result)]):
                runpy.run_path(str(entry), run_name="__main__")
            saved_fixed = json.loads((result / "fixed.json").read_text())[0]
            saved_independent = json.loads((result / "independent.json").read_text())[0]
            fixed = codec.decode(saved_fixed.pop("segmentation"))
            independent = codec.decode(saved_independent.pop("segmentation"))
            expected_metadata = {k: v for k, v in candidate.items() if k != "segmentation"}
            self.assertEqual(saved_fixed, expected_metadata)
            self.assertEqual(saved_independent, expected_metadata)
            expected_fixed = native.copy()
            expected_fixed.flat[[20, 0]] = [1, 0]
            expected_independent = native.copy()
            expected_independent.flat[[20, 1]] = [1, 0]
            np.testing.assert_array_equal(fixed, expected_fixed)
            np.testing.assert_array_equal(independent, expected_independent)
            self.assertEqual(json.loads((result / "summary.json").read_text())["swaps_per_arm"], 1)



    def test_bad_candidate_does_not_publish_partial_results(self):
        with TemporaryDirectory() as root:
            folder = Path(root)
            input_path = folder / "candidates.jsonl"
            result = folder / "replay"
            native = np.zeros((5, 8), dtype=np.uint8)
            native.flat[:20] = 1
            candidate = {"image_id": 9, "category_id": 1, "score": 0.41, "segmentation": rle(native)}
            valid = {"candidate": candidate, "pairs": [], "occupancy": []}
            invalid = {"candidate": candidate, "pairs": [[0, 20]], "occupancy": [[0.7, 0.3]]}
            input_path.write_text("\n".join(map(json.dumps, (valid, invalid))) + "\n", encoding="utf-8")
            entry = Path(__file__).with_name("replay_fixed_count.py")
            with patch.object(sys, "argv", [str(entry), "--input", str(input_path), "--output", str(result)]):
                with self.assertRaisesRegex(ValueError, "input line 2"):
                    runpy.run_path(str(entry), run_name="__main__")
            self.assertFalse(result.exists())


if __name__ == "__main__":
    unittest.main()

