import unittest

import numpy as np

from matching_factors import THRESHOLD, mix_edges


class MatchingFactorsTest(unittest.TestCase):
    def test_eligibility_from_first_priority_and_ties_from_second(self):
        eligible=np.array([[.75,.96,.7,.79],[.95,.74,.80,.85]])
        priority=np.array([[.40,.93,.99,.93],[.1,.99,.1,.2]])
        mixed=mix_edges(eligible,priority)
        np.testing.assert_array_equal(mixed>=THRESHOLD,eligible>=THRESHOLD)
        self.assertLess(mixed[0,0],mixed[0,1])
        self.assertEqual(mixed[0,1],mixed[0,3])
        self.assertEqual(mixed[1,0],mixed[1,2])
        self.assertGreater(mixed[1,3],mixed[1,2])

    def test_empty_input_and_shape_validation(self):
        self.assertEqual(mix_edges([],[]).size,0)
        with self.assertRaises(ValueError):
            mix_edges(np.ones((2,3)),np.ones((3,2)))


if __name__=="__main__":
    unittest.main()

