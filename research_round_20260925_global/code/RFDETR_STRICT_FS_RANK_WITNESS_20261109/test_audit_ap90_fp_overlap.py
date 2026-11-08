from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_ap90_fp_overlap import classify_prefix


class PrefixTest(unittest.TestCase):
    def test_fixed_prefix_and_inclusive_overlap_edges(self):
        matches = np.array([0, 2, 0, 0, 3, 0])
        max_iou = np.array([0.90, 0.96, 0.50, 0.499, 0.91, 0.92])
        self.assertEqual(classify_prefix(matches, max_iou, 2),
                         {"required_TP": 2, "rank_of_required_TP": 5,
                          "FP_before_required_TP": 3, "FP_max_IoU_ge_090": 1,
                          "FP_max_IoU_ge_050_lt_090": 1, "FP_max_IoU_lt_050": 1})


if __name__ == "__main__":
    unittest.main()
