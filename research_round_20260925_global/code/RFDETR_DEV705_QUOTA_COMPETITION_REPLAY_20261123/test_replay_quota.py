"""Verify sparse endpoint indexing across the COCO Fortran-order raster."""

import unittest

import numpy as np

from replay_quota import edit_for_action


class ReplayQuotaTest(unittest.TestCase):
    def test_source_prefix_and_signs_preserve_row_major_coordinates(self):
        stock = np.asfortranarray([[False, True, False], [True, True, False]])
        coords = np.asarray([[0, 3], [2, 1], [5, 4]], dtype=np.int32)
        q = np.asarray([[.9, .05], [.7, .10], [.45, .15]], dtype=np.float32)
        order = np.array([0, 1, 2]), np.array([0, 1, 2])
        expected = stock.copy(order="C")
        expected.reshape(-1)[[0, 2, 5, 3]] = [True, True, True, False]
        sparse = edit_for_action(stock, coords, q, order, adds=3, drops=1)
        self.assertEqual(sparse["flipped"], [0, 2, 5, 3])
        self.assertEqual((sparse["edits"], sparse["adds"], sparse["drops"], sparse["area_delta"]),
                         (4, 3, 1, 2))
        self.assertEqual(sparse["nonpositive_flips"], 1)
        reconstructed = stock.copy(order="C")
        reconstructed.reshape(-1)[sparse["flipped"]] ^= True
        np.testing.assert_array_equal(expected, reconstructed)
        with self.assertRaisesRegex(ValueError, "endpoint signs"):
            edit_for_action(stock, coords[:, ::-1], q, order, adds=3, drops=1)


if __name__ == "__main__":
    unittest.main()
