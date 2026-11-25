import unittest
import numpy as np

from score_rank import conditional_iou, bootstrap, fixed_signed_slots


class HeldRankScore(unittest.TestCase):
    def test_same_cardinality_and_signed_ordinal_ties(self):
        q = np.array([[.5,.4],[.6,.4],[.6,.2],[.3,.9]],dtype=np.float32)
        self.assertEqual(fixed_signed_slots(q,2).tolist(), [[1,2],[2,0]])
        self.assertEqual(fixed_signed_slots(q,0).shape,(0,2))

    def test_fractional_iou_uses_signed_mass_not_count(self):
        truth = np.array([[.8,.4],[.9,.02],[.3,.9]],dtype=np.float64)
        edits = np.array([[1,0]])
        self.assertAlmostEqual(conditional_iou(3.,5.,6.,truth,edits),
                               3.5/7.5, places=14)
        self.assertAlmostEqual(conditional_iou(3.,5.,6.,truth,edits[:0]),3./8.)

    def test_seed_mean_bootstrap_uses_paired_prefix_resamples(self):
        seed_by_group = np.array([[1.,-1.],[-1.,1.],[0.,0.]])
        self.assertEqual(bootstrap(seed_by_group.mean(axis=0)),[0.,0.])


if __name__ == "__main__":
    unittest.main()

