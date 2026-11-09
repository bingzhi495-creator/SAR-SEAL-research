import unittest

import numpy as np

from score_box_shift import make_row, strict


class FixedBoxShiftTests(unittest.TestCase):
    def test_signed_shift_geometry_and_box_axes(self):
        pred = np.zeros((8, 8), dtype=bool)
        gt = pred.copy()
        pred[1:4, 1:4] = True
        gt[2:5, 0:3] = True
        row = make_row(pred, gt, [.25, .5, .3, .3], protected=False)
        self.assertEqual(row["target"], (1, -1))
        np.testing.assert_array_equal(row["box_gap"], [1.5, -.5])
        np.testing.assert_array_equal(row["mask_gap"], [0., 0.])
        self.assertNotIn((0, -2), row["table"])
        self.assertEqual(row["table"][(1, -1)], (9, 0, 0, 10))
        self.assertTrue(strict(row["table"][(1, -1)]))

    def test_zero_target_does_not_disable_strict_scoring(self):
        pred = np.zeros((10, 10), dtype=bool)
        pred[2:6, 2:6] = True
        row = make_row(pred, pred, [.4, .4, .4, .4], protected=True)
        self.assertEqual(row["target"], (0, 0))
        self.assertFalse(strict(row["table"][(1, 0)]))


if __name__ == "__main__":
    unittest.main()
