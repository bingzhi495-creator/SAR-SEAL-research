import unittest
from fractions import Fraction

from fixed_cell_score_v2 import hat_value, optimize_cell


class ExactBreakpointsTest(unittest.TestCase):
    def test_breakpoints_keep_zero_background_and_flip_ties_together(self):
        result = optimize_cell(
            [Fraction(0), Fraction(-2), Fraction(-1), Fraction(-1)],
            [Fraction(1)] * 4,
            [True, True, True, False],
        )
        self.assertEqual((result.delta_i, result.delta_u, result.flips), (2, 1, 3))
        self.assertTrue(1 < result.coefficient <= 2)


    def test_hats_share_open_support_and_peak_at_their_own_knots(self):
        for peak in (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)):
            self.assertEqual(hat_value(Fraction(0), peak), 0)
            self.assertEqual(hat_value(Fraction(1), peak), 0)
            self.assertEqual(hat_value(peak, peak), 1)
            self.assertGreater(hat_value(Fraction(1, 10), peak), 0)


    def test_cell_with_no_accessible_breakpoint_keeps_noop(self):
        result = optimize_cell(
            [Fraction(8), Fraction(-8)],
            [Fraction(1), Fraction(1)],
            [False, True],
        )
        self.assertEqual((result.delta_i, result.delta_u, result.flips, result.coefficient), (0, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
