import itertools
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from equal_edit_allocator import exact_allocate, prefix_expected_iou


class AllocatorTest(unittest.TestCase):
    def test_exact_budget_matches_brute_force_and_caps(self):
        curves = [np.array([0., .2, .8]), np.array([0., .4, .45]), np.array([0., -.1])]
        for budget in range(6):
            choices = [p for p in itertools.product(*(range(len(u)) for u in curves)) if sum(p) == budget]
            pick = exact_allocate(curves, budget)
            self.assertEqual(sum(pick), budget)
            self.assertAlmostEqual(sum(u[k] for u, k in zip(curves, pick)),
                                   max(sum(u[k] for u, k in zip(curves, p)) for p in choices))
        with self.assertRaises(ValueError):
            exact_allocate(curves, 6)

    def test_expected_geometry_uses_correct_signed_union(self):
        # One correct removal and one correct addition transform a three-pixel mask.
        q = prefix_expected_iou(3, np.array([1., 1.]), np.array([-1, 1]), np.array([0, 1]))
        self.assertTrue(np.allclose(q, [0., 2/3 - 1/2, 1 - 1/2]))
        q = prefix_expected_iou(3, np.array([0., 0.]), np.array([-1, 1]), np.array([0, 1]))
        self.assertTrue(np.allclose(q, [0., 2/3 - 1, .5 - 1]))


if __name__ == "__main__":
    unittest.main()

