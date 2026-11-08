from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from fit_edit_allocation_oracle import allocation_summary, prefix_repair, select_representatives
import numpy as np


class AllocationTest(unittest.TestCase):
    def test_pooling_can_help_without_extra_capacity(self):
        summary=allocation_summary([[(16, 2), (16, 30)], [(16, None)]])
        self.assertEqual(summary["fixed"]["eligible"],1)
        self.assertEqual(summary["per_image_pool"]["eligible"],2)
        self.assertEqual(summary["global_pool"]["eligible"],2)
        self.assertEqual(summary["capacity"],48)
        self.assertEqual(summary["per_image_pool"]["spent"],32)

    def test_first_prefix_and_forced_k_are_distinct_when_edits_can_harm(self):
        self.assertEqual(prefix_repair(8, 10, [10, -9], 2), (1, False))
        self.assertEqual(prefix_repair(8, 10, [10, -9], 1), (1, True))
        self.assertEqual(prefix_repair(8, 10, [-9, 10], 2), (None, False))


    def test_select_first_unique_coarse_and_skip_gt_with_any_strict_witness(self):
        overlaps=np.array([[.91,.51,.01,.01,.01], [.05,.56,.01,.01,.01],
                           [.06,.85,.01,.01,.01], [.06,.51,.54,.01,.01],
                           [.01,.01,.01,.20,.55]])
        pairs, coverage=select_representatives(overlaps,[True,True,True,True,False])
        self.assertEqual(pairs,[(1,1)])
        self.assertEqual(coverage,{"already_strict_gt":1,"unique_coarse_representatives":1,
                                  "ambiguous_only_gt":1,"no_candidate_ge050_gt":1})


if __name__=="__main__":
    unittest.main()

