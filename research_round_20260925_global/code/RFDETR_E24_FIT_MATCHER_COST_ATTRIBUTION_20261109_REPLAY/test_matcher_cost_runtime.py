from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matcher_cost_runtime as r
import matcher_cost_math as m


class RuntimeTests(unittest.TestCase):
    def test_image_cluster_before_pair_resampling(self):
        paired = r.image_pair([(0.75, .25), (.25, .25), (0., .25)])
        self.assertEqual(paired["images"], 3)
        self.assertAlmostEqual(paired["iid_minus_jitter"], 1/12)
        self.assertEqual(paired["positive_images"], 1)
        self.assertEqual(paired["negative_images"], 1)
        self.assertEqual(paired["tied_images"], 1)

    def test_missing_reference_support_keeps_metrics_empty(self):
        paired = r.image_pair([])
        self.assertIsNone(paired["iid_minus_jitter"])
        self.assertIsNone(paired["percentile95"])

    def test_exact_native_totals_callback_used_by_both_arms(self):
        calls = []
        def sample(points):
            return np.zeros((4, len(points)), np.float32), np.ones((1, len(points)), np.float32)
        def native_total(points):
            calls.append(points.copy())
            return np.array([[0.], [1.], [0.], [1.]], dtype=np.float32)
        draw = np.zeros((m.K, 2), np.float32)
        result = m.diagnose_view(np.zeros((4, 1), np.float32), sample, np.array([True]),
                                 (5., 5.), [draw], width=2, groups=2, side=2,
                                 native_sampled_total=native_total)
        self.assertEqual(result["sampled_cost_source"], "native_total")
        np.testing.assert_array_equal(calls[0], draw)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(np.unique(calls[1], axis=0)), m.K)
        self.assertEqual(result["reference"]["iou75"]["small"]["alternate_dense_good"], 0)


if __name__ == "__main__":
    unittest.main()
