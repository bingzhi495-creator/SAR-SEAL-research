import unittest
import numpy as np
from add_nested_oracle import donor_domain


class NestedDomain(unittest.TestCase):
    def test_includes_one_and_two_but_excludes_neither_and_outside(self):
        a = np.array([0, 0, 1, 1, 0], dtype=bool)
        b = np.array([1, 0, 0, 1, 1], dtype=bool)
        c = np.array([1, 1, 1, 1, 1], dtype=bool)
        band = np.array([1, 1, 1, 1, 0], dtype=bool)
        self.assertEqual(donor_domain(a, b, c, band).tolist(), [True, True, True, False, False])
