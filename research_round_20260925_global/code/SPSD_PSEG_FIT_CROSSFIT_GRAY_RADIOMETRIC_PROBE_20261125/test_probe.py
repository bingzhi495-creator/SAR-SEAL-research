import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe import midrank, ridge, vectors


class RankProbeTest(unittest.TestCase):
    def test_shared_rank_ties_and_positive_affine(self):
        gray = np.array([[0, 1], [1, 2]])
        np.testing.assert_array_equal(midrank(gray), [[-.375, 0], [0, .375]])
        np.testing.assert_array_equal(midrank(gray), midrank(gray * 2 + 1))
        np.testing.assert_array_equal(midrank(gray, axis=0), [[-.25, -.25], [.25, .25]])

    def test_static_stays_zero_and_regression_retains_sign(self):
        features = np.zeros((2, 2, 14))
        features[..., 0] = [[.2, .4], [.5, .1]]
        packet = SimpleNamespace(features=features, k=1, targets=np.ones((2, 2, 2)),
                                 action_weights=np.array([[[0, 0], [0, 0]],
                                                          [[-1, 0], [1, 0]]]))
        x, y = vectors(packet)
        np.testing.assert_array_equal(x[0], 0)
        np.testing.assert_array_equal(y, 0)
        pred = ridge(np.array([[1.], [-1.]]), np.array([1., -1.]), np.array([1., 1.]))
        self.assertGreater(float(pred[0]), 0)
        self.assertLess(float(pred[0]), 1)


if __name__ == "__main__":
    unittest.main()
