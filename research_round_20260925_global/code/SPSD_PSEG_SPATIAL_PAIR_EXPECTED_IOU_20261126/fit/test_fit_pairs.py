"""Synthetic check for equal decoded-group, candidate and pair weights."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_pairs import weighted_pair_rows


class PairWeights(unittest.TestCase):
    def test_different_group_sizes_and_candidate_lengths_are_balanced(self):
        group_id, fold, weight = weighted_pair_rows(
            ["one", "one", "two"], [0, 0, 1], [1, 3, 2], {"one": 0, "two": 1})
        np.testing.assert_array_equal(group_id,[0,0,0,0,1,1])
        np.testing.assert_array_equal(fold,[0,0,0,0,1,1])
        np.testing.assert_allclose(weight,[.5,1/6,1/6,1/6,.5,.5])
        np.testing.assert_allclose(np.bincount(group_id,weights=weight),[1,1])


if __name__=="__main__":
    unittest.main()
