import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from bilateral_quality_scout import fit_quality, quality_targets, pair_concordance, candidate_features


class QualityScoutTest(unittest.TestCase):
    def test_targets_are_existing_coco_threshold_survival(self):
        self.assertTrue(np.allclose(quality_targets(np.array([.49, .50, .90, .95, 1.])),
                                    [0., .1, .9, 1., 1.]))

    def test_cues_share_support_and_empty_candidates_are_preserved(self):
        a = dict(features=np.array([[.2, .1, .4], [.8, -.9, -.9], [.5, .3, .6]]),
                 valid=np.array([True, False, True]))
        common, mean, rank = candidate_features(.7, 9, a, 0, 3)
        self.assertEqual(common[-1], 2 / 3)
        self.assertAlmostEqual(mean, .2)
        self.assertAlmostEqual(rank, .5)
        empty, mean, rank = candidate_features(.1, 0, a, 0, 0)
        self.assertEqual(empty[-1], 0.)
        self.assertEqual((mean, rank), (0., 0.))

    def test_pair_equal_ties_and_fit_predict(self):
        self.assertEqual(pair_concordance(np.array([.5, .8]), np.array([.5, .9])), .375)
        x = np.arange(12, dtype=float).reshape(-1, 1)
        y = np.linspace(0, 1, 12)
        pred = fit_quality(x, y, x)
        self.assertTrue(np.all(np.diff(pred) > 0))
        self.assertTrue(np.all((pred > 0) & (pred < 1)))


if __name__ == "__main__":
    unittest.main()

