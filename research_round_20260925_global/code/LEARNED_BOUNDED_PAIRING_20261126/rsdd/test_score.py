import unittest
from score import assess, close_enough


class ScoreTests(unittest.TestCase):
    def test_gate_uses_effect_direction_and_coverage(self):
        self.assertEqual(assess(.30, (.01, .12), 20, 8)["decision"], "positive")
        self.assertEqual(assess(-.30, (-.12, -.01), 20, 8)["decision"], "negative")
        self.assertEqual(assess(.40, (-.01, .12), 20, 8)["decision"], "descriptive")
        self.assertEqual(assess(.40, (.02, .12), 19, 8)["decision"], "descriptive")
        self.assertEqual(assess(.40, (.02, .12), 20, 7)["decision"], "descriptive")

    def test_practical_bands_include_whole_paired_interval(self):
        self.assertTrue(assess(.15, (-.15, .15), 21, 9)["practical_within_bands"])
        self.assertFalse(assess(.15, (-.151, .10), 21, 9)["practical_within_bands"])

    def test_original_coco_comparison_rejects_different_value(self):
        close_enough({"AP": .25, "APl": -1.0}, {"AP": .25, "APl": None})
        with self.assertRaisesRegex(ValueError, "Archived raster score parity"):
            close_enough({"AP": .26}, {"AP": .25})


if __name__ == "__main__":
    unittest.main()
