from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from coarse_fp_pixel_oracle import load_original_band, measure_pair, minimum_correct_flips


class MinimumFlipsTest(unittest.TestCase):
    def test_short_integer_search_reproduces_oracle(self):
        for union in range(1, 24):
            for intersection in range(union + 1):
                for fn in range(union - intersection + 1):
                    for fp in range(union - intersection - fn + 1):
                        expected = min(
                            (a + b for a in range(fn + 1) for b in range(fp + 1)
                             if 10 * (intersection + a) >= 9 * (union - b)),
                            default=None,
                        )
                        self.assertEqual(minimum_correct_flips(intersection, union, fn, fp), expected)


    def test_frozen_clipped_band_can_lack_corrections(self):
        pred = np.zeros((8, 8), dtype=bool)
        pred[:4, :4] = True
        truth = pred.copy()
        truth[4, 3] = truth[7, 6] = truth[7, 7] = True
        result = measure_pair(pred, truth, load_original_band())
        self.assertEqual(result["full_FN"], 3)
        self.assertEqual(result["band_FN"], 1)
        self.assertEqual(result["full_min_edits"], 2)
        self.assertIsNone(result["band_min_edits"])

if __name__ == "__main__":
    unittest.main()

