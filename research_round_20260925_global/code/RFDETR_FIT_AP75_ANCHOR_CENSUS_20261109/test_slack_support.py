import unittest

import numpy as np

from slack_support import bounds, needs_hits, slack_after


class AP75SlackTest(unittest.TestCase):
    def test_integer_threshold_matches_all_possible_flip_counts(self):
        for start in range(-35,36):
            r=needs_hits(start)
            for hit in range(9):
                self.assertEqual(hit>=r,slack_after(start,hit)>=0)
        self.assertEqual(needs_hits(-1),5)
        self.assertEqual(needs_hits(0),4)
        self.assertEqual(needs_hits(27),1)
        self.assertEqual(needs_hits(28),0)

    def test_feasible_counts_include_forced_hits_and_short_side(self):
        y=np.array([1,1,1,0,0,1,1,1,1,1],dtype=bool)
        inside=np.array([0,0,0,0,0,1,1,1,1,1],dtype=bool)
        self.assertEqual(bounds(y,inside),(6,7))
        self.assertIsNone(bounds(y[:8],inside[:8]))


if __name__=="__main__":
    unittest.main()

