import unittest

import numpy as np

from translation_probe import measure


class TranslationProbeTest(unittest.TestCase):
    def test_known_up_shift_preserves_area_and_recovers_mask(self):
        truth = np.zeros((8, 8), dtype=bool)
        truth[2:6, 2:6] = True
        pred = np.zeros_like(truth)
        pred[3:7, 2:6] = True

        result = measure(pred, truth)

        self.assertEqual(result["base"], dict(tp=12, fp=4, fn=4, edits=0, dy=0, dx=0))
        self.assertEqual(result["fixed_up"], dict(tp=16, fp=0, fn=0, edits=8, dy=-1, dx=0))
        self.assertEqual(result["oracle1"], result["fixed_up"])
        self.assertEqual(result["oracle2"], result["fixed_up"])
        self.assertEqual((result["dy_pixels"], result["dx_pixels"]), (1.0, 0.0))


    def test_canvas_edge_rejects_cropping_or_wraparound(self):
        pred = np.zeros((7, 7), dtype=bool)
        pred[0:2, 3] = True
        truth = np.zeros_like(pred)
        truth[0, 3] = True

        result = measure(pred, truth)

        self.assertIsNone(result["fixed_up"])
        self.assertEqual(result["oracle2"], result["base"])
        self.assertEqual(result["fixed_down"], dict(tp=0, fp=2, fn=1, edits=2, dy=1, dx=0))

    def test_radius_two_can_recover_beyond_the_radius_one_search(self):
        pred = np.zeros((8, 8), dtype=bool)
        pred[3:5, 2:5] = True
        truth = np.zeros_like(pred)
        truth[3:5, 4:7] = True

        result = measure(pred, truth)

        self.assertEqual(result["oracle1"], dict(tp=4, fp=2, fn=2, edits=4, dy=0, dx=1))
        self.assertEqual(result["oracle2"], dict(tp=6, fp=0, fn=0, edits=8, dy=0, dx=2))

    def test_equal_iou_and_edits_prefer_smaller_displacement_then_tuple(self):
        pred = np.zeros((7, 7), dtype=bool)
        pred[3, 3] = True
        truth = np.zeros_like(pred)
        truth[3, 2] = truth[3, 4] = truth[4, 4] = True

        result = measure(pred, truth)

        self.assertEqual(result["oracle1"], dict(tp=1, fp=0, fn=2, edits=2, dy=0, dx=-1))

    def test_ninety_percent_area_upper_bound_is_inclusive(self):
        truth = np.ones((2, 5), dtype=bool)
        pred = truth.copy()
        pred[0, 0] = False
        self.assertFalse(measure(pred, truth)["area_impossible"])

        pred[0, 1] = False
        self.assertTrue(measure(pred, truth)["area_impossible"])

    def test_elongation_is_finite_and_rotation_invariant(self):
        point = np.ones((1, 1), dtype=bool)
        self.assertAlmostEqual(measure(point, point)["elongation"], 1.0)

        horizontal = np.ones((1, 5), dtype=bool)
        vertical = np.rot90(horizontal)
        self.assertAlmostEqual(measure(horizontal, horizontal)["elongation"], 5.0)
        self.assertAlmostEqual(measure(vertical, vertical)["elongation"], 5.0)

    def test_invalid_or_empty_masks_are_rejected(self):
        good = np.ones((2, 2), dtype=bool)
        with self.assertRaisesRegex(ValueError, "Boolean masks"):
            measure(good.astype(np.uint8), good)
        with self.assertRaisesRegex(ValueError, "Boolean masks"):
            measure(good, good[:, :1])
        with self.assertRaisesRegex(ValueError, "nonempty foreground"):
            measure(good, np.zeros_like(good))


if __name__ == "__main__":
    unittest.main()
