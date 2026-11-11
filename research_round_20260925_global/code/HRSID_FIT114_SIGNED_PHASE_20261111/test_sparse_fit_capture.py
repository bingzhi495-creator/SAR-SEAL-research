import unittest

import numpy as np

from sparse_fit_capture import band_points, parity_counts


class CaptureTest(unittest.TestCase):
    def test_one_pixel_both_sides_exact_sign_edge_and_order(self):
        z = np.full((10, 10), -2., dtype=np.float32)
        z[:6, :6] = 3.
        p = z > 0
        result = band_points(p, z, "11"*32, 3)
        self.assertEqual(result, band_points(p, z, "11"*32, 3))
        by_coord = {(x["y"], x["x"]): x for x in result}
        self.assertIn((0, 0), by_coord)
        self.assertNotIn((3, 3), by_coord)
        self.assertIn((6, 6), by_coord)
        self.assertNotIn((7, 7), by_coord)
        self.assertEqual(by_coord[0, 0]["logits"], [3.] * 9)
        self.assertEqual(by_coord[6, 6]["logits"][1], 3.)
        self.assertEqual(sum(row["inside"] for row in result), 20)
        self.assertEqual(sum(not row["inside"] for row in result), 13)
        self.assertEqual(band_points(np.zeros_like(p), z, "11"*32, 3), [])

    def test_parity_keeps_cpu_only_old_only_and_values_apart(self):
        old = [
            dict(query_id=2, native_rank=1, score=.8, segmentation=dict(counts="old")),
            dict(query_id=7, native_rank=2, score=.5, segmentation=dict(counts="unique")),
        ]
        cpu = [
            dict(query_id=2, native_rank=2, score=.7, segmentation=dict(counts="new")),
            dict(query_id=5, native_rank=3, score=.1, segmentation=dict(counts="unique")),
        ]
        result = parity_counts(old, cpu)
        self.assertEqual((result["missing_old_rows"], result["new_cpu_rows"], result["common_rows"]), (1, 1, 1))
        self.assertEqual((result["rank_different"], result["score_different"], result["rle_different"]), (1, 1, 1))
        self.assertEqual(result["ordered_query_list_exact"], 0)


if __name__ == "__main__":
    unittest.main()
