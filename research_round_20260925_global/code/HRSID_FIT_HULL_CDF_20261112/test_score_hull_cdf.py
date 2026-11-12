import unittest
import numpy as np
from score_hull_cdf import paired_effect

class GroupEffectTest(unittest.TestCase):
    def test_group_weights_and_partitions_use_source_not_rows(self):
        rows=[(0, {'test': 100., 'control': 90.}),
              (0, {'test': 100., 'control': 90.}),
              (1, {'test': 80., 'control': 90.}),
              (2, {'test': 80., 'control': 80.})]
        draws=np.array([[0,1,2],[0,0,0],[1,1,1]])
        actual=paired_effect(rows,'test','control',draws)
        self.assertAlmostEqual(actual['group_mean_points'],0.)
        self.assertEqual(actual['positive_partitions'],1)
        self.assertEqual(actual['contributing_groups'],3)
