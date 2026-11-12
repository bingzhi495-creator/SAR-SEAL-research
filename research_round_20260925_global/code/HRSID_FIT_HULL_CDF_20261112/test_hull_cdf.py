"""Behavioral tests for subpixel polygon projection oracles."""

import unittest

from shapely.affinity import translate
from shapely.geometry import MultiPolygon, Polygon, box

from hull_cdf import projected_choices


SHIFTS = ("w1_shift", "mean_shift", "median_shift", "q3_mean_shift", "q3_median_shift")


class ProjectedChoicesTests(unittest.TestCase):
    def test_pure_rigid_translation_agrees_on_all_choices(self):
        pred = box(0, 0, 4, 2)
        gt = translate(pred, xoff=0.6, yoff=0.8)

        result = projected_choices(gt, pred, (3, 4))

        for name in SHIFTS:
            self.assertEqual(result[name], 1.0)
        self.assertAlmostEqual(result["w1_costs"][result["menu"].index(1.0)], 0, places=12)

    def test_remote_low_mass_tail_changes_mean_but_not_w1(self):
        pred = box(0, 0, 1, 10)
        gt = MultiPolygon([box(0, 0, 1, 9), box(9, 0, 10, 1)])

        result = projected_choices(gt, pred, (1, 0))

        self.assertAlmostEqual(result["mean_delta"], .9, places=12)
        self.assertEqual(result["mean_shift"], 1.0)
        self.assertEqual(result["w1_shift"], 0.0)
        self.assertEqual(result["median_shift"], 0.0)
        self.assertEqual(result["q3_mean_shift"], 0.0)
        self.assertEqual(result["q3_median_shift"], 0.0)

    def test_half_pixel_translation_and_fine_grid_agree(self):
        pred = box(100, 100, 101, 104)
        gt = translate(pred, xoff=-.5)

        for step in (.125, .0625):
            result = projected_choices(gt, pred, (1, 0), step=step)

            for name in SHIFTS:
                self.assertEqual(result[name], -.5)
            for shift, cost in zip(result["menu"], result["w1_costs"], strict=True):
                self.assertAlmostEqual(cost, abs(shift + .5), places=12)

    def test_zero_translation_and_nearest_shift_ties_favor_no_edit(self):
        pred = box(0, 0, 1, 1)
        for gt in (pred, translate(pred, xoff=.25), translate(pred, xoff=-.25)):
            result = projected_choices(gt, pred, (1, 0))

            for name in SHIFTS:
                self.assertEqual(result[name], 0.0)

    def test_invalid_area_or_axis_is_rejected(self):
        pred = box(0, 0, 2, 3)
        with self.assertRaises(ValueError):
            projected_choices(Polygon(), pred, (1, 0))
        with self.assertRaises(ValueError):
            projected_choices(pred, pred, (0, 0))


if __name__ == "__main__":
    unittest.main()
