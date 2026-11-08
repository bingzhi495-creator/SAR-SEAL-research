import itertools
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from equal_edit_allocator import BASE_SCOUT, BASE_SCOUT_SHA, MATERIAL_API, exact_allocate, load_module, prefix_expected_iou


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

    def test_release_helper_locator_does_not_rebase_from_imported_module(self):
        base = load_module(BASE_SCOUT, BASE_SCOUT_SHA, "_published_bilateral_helper_test")
        api = load_module(MATERIAL_API, base.API_SHA, "_published_materializer_test")
        self.assertTrue(callable(api.original_gt))
        self.assertTrue(callable(api.runtime))

    def test_expected_geometry_uses_correct_signed_union(self):
        # One correct removal and one correct addition transform a three-pixel mask.
        q = prefix_expected_iou(3, np.array([1., 1.]), np.array([-1, 1]), np.array([0, 1]))
        self.assertTrue(np.allclose(q, [0., 2/3 - 1/2, 1 - 1/2]))
        q = prefix_expected_iou(3, np.array([0., 0.]), np.array([-1, 1]), np.array([0, 1]))
        self.assertTrue(np.allclose(q, [0., 2/3 - 1, .5 - 1]))


if __name__ == "__main__":
    unittest.main()

