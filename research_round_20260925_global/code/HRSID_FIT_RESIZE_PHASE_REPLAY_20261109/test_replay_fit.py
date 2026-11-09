import unittest

import numpy as np

from replay_fit import paired_measurements, summarize


class ReplayMetricsTests(unittest.TestCase):
    def test_positive_downward_centroid_is_expressed_in_original_units(self):
        ref = np.zeros((2, 7, 7), dtype=bool)
        ref[0, 1:3, 2:4] = True
        legacy = ref.copy()
        legacy[0] = np.roll(ref[0], 1, axis=0)
        center = ref.copy()
        # The second reference row is empty and excluded from all displacement comparisons.
        legacy[1, 4, 4] = center[1, 4, 4] = True
        result = paired_measurements(legacy, center, ref, np.array([True, True]), (2, -3))
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["small"]["legacy_y"], 2.)
        self.assertEqual(result["small"]["center_y"], 0.)
        self.assertEqual(result["small"]["legacy_iou"], 1 / 3)
        self.assertEqual(result["small"]["center_iou"], 1.)
        self.assertEqual(result["reference_empty_for_common_survivors"], 1)


if __name__ == "__main__":
    unittest.main()
