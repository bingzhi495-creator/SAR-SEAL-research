"""Synthetic tests for the source-max guard."""
import unittest

from strict_guard import calibrated_cutoff, keep_edit


class GuardTests(unittest.TestCase):
    def test_small_calibration_abstains(self):
        self.assertIsNone(calibrated_cutoff([.02] * 8))
        self.assertFalse(keep_edit(.99, None))

    def test_exact_order_clip_and_strict_tie(self):
        self.assertEqual(calibrated_cutoff([-.1] * 9), 0.)
        self.assertEqual(calibrated_cutoff([.01] * 8 + [.09]), .09)
        self.assertFalse(keep_edit(.09, .09))
        self.assertTrue(keep_edit(.10, .09))


if __name__ == "__main__":
    unittest.main()
