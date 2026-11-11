from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_cv5_mask_compare import pack, unpack, record_comparison


class ComparisonTests(unittest.TestCase):
    def test_roundtrip_preserves_row_edges_and_zero_instances(self):
        for masks in (np.empty((0, 2, 9), np.uint8),
                      np.arange(36, dtype=np.uint8).reshape(2, 2, 9) % 2):
            self.assertTrue(np.array_equal(unpack(pack(masks), masks.shape), masks))

    def test_distinguishes_tie_and_non_tie_pixel_differences(self):
        native = np.array([[1, 0, 0]], np.uint8)
        center = np.array([[0, 0, 1]], np.uint8)
        exact = np.array([[1, 1, 1]], np.uint8)
        out = record_comparison(native, center, native, exact, (1, 2))
        self.assertEqual(out, (0, 2, 1, 1))


if __name__ == "__main__":
    unittest.main()
