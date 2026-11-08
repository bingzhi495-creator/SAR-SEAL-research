import unittest
import numpy as np
from teacher_boundary_v2 import assignment_flags

class ExactAssignmentFlags(unittest.TestCase):
    def test_half_inclusive_and_raw_competition_are_separate(self):
        inter=np.array([[50,49,90],[49,0,89],[0,0,0]])
        union=np.array([[100,100,100],[100,0,100],[0,0,0]])
        half,raw=assignment_flags(inter,union,np.array([0,0,1]))
        self.assertEqual(half.tolist(),[True,False,False])
        self.assertEqual(raw.tolist(),[True,False,False])

if __name__=="__main__":
    unittest.main()
