import unittest

import numpy as np
from pycocotools import mask
from fixed_shift_ap import translate


def rle(bitmap):
    value = mask.encode(np.asfortranarray(bitmap, dtype=np.uint8))
    return dict(size=value["size"], counts=value["counts"].decode("ascii"))


class FixedShiftTest(unittest.TestCase):
    def test_shift_agrees_with_dense_oracle_and_keeps_edge_rows(self):
        for y in range(4):
            source = np.zeros((4, 5), dtype=np.uint8)
            source[y, [0, 2, 4]] = 1
            original = rle(source)
            bbox = mask.toBbox(original)
            for dy in (-1, 1):
                with self.subTest(y=y, dy=dy):
                    shifted, status = translate(original, bbox, dy)
                    expected = np.zeros_like(source)
                    if 0 <= y + dy < 4:
                        expected[y + dy, [0, 2, 4]] = 1
                        self.assertEqual(status, "shifted")
                    else:
                        expected = source
                        self.assertEqual(status, "edge_unchanged")
                    np.testing.assert_array_equal(mask.decode(shifted), expected)
                    self.assertEqual(mask.area(shifted), mask.area(original))


    def test_long_runs_and_empty_passthrough(self):
        rng = np.random.default_rng(365)
        source = np.zeros((96, 73), dtype=np.uint8)
        source[10:-10] = rng.random((76, 73)) > .995
        original = rle(source)
        for dy in (-1, 1):
            shifted, status = translate(original, mask.toBbox(original), dy)
            self.assertEqual(status, "shifted")
            np.testing.assert_array_equal(mask.decode(shifted), np.roll(source, dy, axis=0))
        empty = rle(np.zeros_like(source))
        for dy in (-1, 1):
            self.assertEqual(translate(empty, mask.toBbox(empty), dy), (empty, "empty_unchanged"))


if __name__ == "__main__":
    unittest.main()
