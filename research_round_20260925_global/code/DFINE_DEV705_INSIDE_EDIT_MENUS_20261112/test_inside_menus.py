import unittest
import numpy as np
from inside_menus import menu


class MenusTest(unittest.TestCase):
    def test_both_signs_can_be_required_at_equal_total_cap(self):
        r = menu(20, 20., 18., np.array([1.]), np.array([0.]), 2)
        self.assertLess(r["add_only"][0], .90)
        self.assertLess(r["drop_only"][0], .90)
        self.assertGreater(r["balanced"][0], .90)
        self.assertEqual(r["unrestricted"], r["balanced"])
        self.assertEqual(r["unrestricted"][1:], (1, 1))

    def test_area_bias_can_require_unbalanced_actions(self):
        r = menu(22, 20., 20., np.array([0., 0.]), np.array([0., 0.]), 2)
        self.assertEqual(r["unrestricted"], (1., 0, 2))
        self.assertEqual(r["balanced"], r["raw"])


if __name__ == "__main__":
    unittest.main()
