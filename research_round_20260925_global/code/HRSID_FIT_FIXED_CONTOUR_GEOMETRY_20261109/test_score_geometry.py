import unittest

import numpy as np

from score_geometry import rank_pairs, summarize


def row(budget=5):
    base = dict(tp=8, fp=1, fn=1, edits=0)
    fixed = dict(tp=9, fp=1, fn=0, edits=1)
    methods = dict.fromkeys(("baseline", "translation", "thickness", "joint", "joint_free"), base)
    methods.update(local_band=fixed, local_any=fixed)
    return dict(budget=budget, bulk_admissible_count=0, local_repair_available=2,
                methods=methods, small=True)


class FixedGeometryTests(unittest.TestCase):
    def test_ranked_one_to_one_breaks_iou_ties_by_later_truth(self):
        matrix = np.array([[.80, .80, 0], [.93, 0, .75], [0, .92, 0]])
        self.assertEqual(rank_pairs(matrix, .75), [(0, 1), (1, 0)])
        self.assertEqual(rank_pairs(matrix, .90), [(1, 0), (2, 1)])

    def test_pair_gate_retains_empty_groups(self):
        result = summarize([[row()] for _ in range(16)] + [[] for _ in range(8)], main=True)
        self.assertEqual(result["groups"], 24)
        self.assertEqual(result["pairs"], 16)
        self.assertEqual(result["methods"]["local_band"]["recovered"], 16)
        self.assertEqual(result["contrasts"]["local_vs_joint_free"]["interval"], [1., 1.])
        self.assertEqual(result["decision"], "ADVANCE_LOCAL_PREDICTIVE_GEOMETRY")


if __name__ == "__main__":
    unittest.main()
