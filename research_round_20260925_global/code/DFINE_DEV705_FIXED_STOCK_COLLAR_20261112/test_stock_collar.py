import unittest
import numpy as np
from stock_collar import action_bonuses


class MarginTest(unittest.TestCase):
    def test_actions_disjoint_and_outside_truth_deletion_counted(self):
        q = np.array([[1., .2, 1., .8, .1]])
        stock = np.array([[False, True, False, False, False]])
        inside = np.array([[False, True, True, True, False]])
        expanded = np.ones((1, 5), dtype=bool)
        a,d,n = action_bonuses(q, stock, inside, expanded, 2, .90)
        self.assertAlmostEqual(a, 1.)
        self.assertAlmostEqual(d, 1.42)
        self.assertAlmostEqual(n, 1.62)

    def test_outside_expansion_is_inaccessible(self):
        q = np.array([[1., 1.]])
        stock = np.array([[True, False]])
        inside = np.array([[True, False]])
        a,d,n = action_bonuses(q, stock, inside, inside, 1, .90)
        self.assertEqual((a,d,n),(0.,0.,0.))


if __name__ == "__main__":
    unittest.main()
