import unittest
import numpy as np
from fit_support import bounds
from inside_menus import menu


class PairSupportTest(unittest.TestCase):
    def test_rematching_can_recover_same_endpoints(self):
        q = np.array([[1.,1.],[0.,0.]])
        r = bounds(20,20.,18.,q,menu)
        self.assertEqual(r["paired"],r["raw"])
        self.assertGreater(r["rematched"],.90)
        self.assertEqual(r["rematched"],r["unrestricted"])

    def test_unequal_counts_can_help_overfull_mask(self):
        q = np.array([[0.,0.],[0.,0.]])
        r = bounds(22,20.,20.,q,menu)
        self.assertEqual(r["rematched"],r["raw"])
        self.assertEqual(r["unrestricted"],1.)


if __name__ == "__main__":
    unittest.main()
