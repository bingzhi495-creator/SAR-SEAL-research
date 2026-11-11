import unittest
import numpy as np
from score_structure import first_exclusive, greedy_representatives

class StructureTest(unittest.TestCase):
    def test_representative_is_first_survivor_nontransitive(self):
        overlap = np.array([[1,.92,.65],[.92,1,.92],[.65,.92,1]])
        kept, lookup = greedy_representatives(overlap,.90)
        self.assertEqual(kept,[0,2])
        self.assertEqual(lookup,{0:0,1:0,2:2})

    def test_rank_first_requires_two_before_consuming(self):
        left = np.array([[.8,0],[.8,.7],[0,.9]])
        right = np.array([[0,0],[.65,0],[.80,.75]])
        self.assertEqual(first_exclusive(left,right),{1:(0,0),2:(1,1)})

if __name__ == '__main__':
    unittest.main()
