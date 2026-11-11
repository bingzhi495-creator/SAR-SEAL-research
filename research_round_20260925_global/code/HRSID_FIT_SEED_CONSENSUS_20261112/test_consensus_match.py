import unittest
import numpy as np
from consensus_match import reciprocal_matches

class ReciprocalMatches(unittest.TestCase):
    def test_disjoint_and_reciprocal(self):
        pairs, state = reciprocal_matches(np.array([[.88,.17],[.76,.72],[.04,.03]]))
        self.assertEqual(pairs, {0:0})
        self.assertEqual(state, ('accepted','not_reciprocal','below_iou'))

    def test_single_endpoint_and_ambiguity(self):
        self.assertEqual(reciprocal_matches(np.array([[.5]])), ({0:0}, ('accepted',)))
        self.assertEqual(reciprocal_matches(np.array([[.65,.61]])), ({}, ('ambiguous',)))
        self.assertEqual(reciprocal_matches(np.array([[.65],[.61]])), ({}, ('ambiguous','not_reciprocal')))

    def test_empty_and_invalid(self):
        self.assertEqual(reciprocal_matches(np.empty((2,0))), ({}, ('no_donor','no_donor')))
        self.assertEqual(reciprocal_matches(np.empty((0,2))), ({}, ()))
        with self.assertRaises(ValueError):
            reciprocal_matches(np.array([[np.nan]]))

if __name__ == '__main__':
    unittest.main()
