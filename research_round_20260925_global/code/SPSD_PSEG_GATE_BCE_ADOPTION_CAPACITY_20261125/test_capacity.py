import itertools
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capacity import bounds


class CapacityTest(unittest.TestCase):

    def test_stratified_disagreement_matches_exhaustive_choices(self):
        k = [1,1,1,2,2,3,3]
        out = bounds(k)
        logical = [set(i) for i in itertools.combinations(range(len(k)), 2)
                   if sorted([k[j] for j in i]) == [1,2]]
        best = max(len(a ^ b) for a in logical for b in logical)
        self.assertEqual(out["maximum_membership_disagreement"], best)
        self.assertEqual(out["quota_matched_per_sign"], 3)

    def test_zero_learned_adoptions(self):
        self.assertEqual(bounds([6,6])["adopted"], 0)
        self.assertEqual(bounds([6,6])["maximum_membership_disagreement"], 0)


if __name__ == "__main__":
    unittest.main()
