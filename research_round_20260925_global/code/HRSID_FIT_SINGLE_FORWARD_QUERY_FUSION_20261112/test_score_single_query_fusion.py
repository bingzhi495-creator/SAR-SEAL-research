import unittest
import numpy as np
from score_single_query_fusion import suppression_groups


class SuppressionGroups(unittest.TestCase):
    def test_direct_first_suppression_cannot_reuse_or_transitively_join_donor(self):
        overlaps = np.eye(6)
        for i, j, v in ((0, 1, .85), (0, 2, .84), (1, 2, .95), (2, 3, .92), (3, 4, .99)):
            overlaps[i, j] = overlaps[j, i] = v
        self.assertEqual(suppression_groups(overlaps), [(0, [1]), (2, [3]), (4, []), (5, [])])
