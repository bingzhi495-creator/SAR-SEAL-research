import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attribute import decision_indices


class DecisionObjectiveTest(unittest.TestCase):

    def test_changes_only_the_requested_factor(self):
        pred = np.array([[0, 0], [3, 3], [3, 1], [7, 1], [1.5, .5]])
        true = np.array([[0, 0], [-1, -1], [16, -4], [1, 1], [3.5, 2.5]])
        self.assertEqual(decision_indices(pred, true), (1, 2, 3, 4))

    def test_keeps_original_first_maximum_under_ties(self):
        pred = np.zeros((2, 3, 2))
        true = np.zeros_like(pred)
        self.assertEqual(decision_indices(pred, true), (0, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
