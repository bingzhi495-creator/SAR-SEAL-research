"""Synthetic checks for optional and exact-count free endpoints."""
import unittest

from free_pairing import choose_free


class FreePairingTests(unittest.TestCase):
    def test_natural_does_not_force_negative_marginal(self):
        pairs=[[.9,.8],[.2,.1]]
        self.assertEqual(choose_free(pairs,2),[(0,1)])
        self.assertEqual(choose_free(pairs,2,fixed_count=2),[(0,1),(1,0)])

    def test_exact_ties_follow_endpoint_ordinal(self):
        self.assertEqual(choose_free([[.8,.2],[.8,.2]],1),[(0,0)])
        self.assertEqual(choose_free([[.4,.4]],1),[])


if __name__=="__main__":
    unittest.main()
