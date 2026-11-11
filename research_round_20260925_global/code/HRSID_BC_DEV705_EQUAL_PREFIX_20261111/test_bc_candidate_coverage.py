import unittest

from bc_candidate_coverage import label_coverage


class BCCoverageTest(unittest.TestCase):
    def test_paired_size_counts_are_derived_from_the_same_transition_rows(self):
        transition = dict(neither=1, center_only=5, native_only=2, both=3)
        original = dict(
            arms=dict(native=dict(predictions=12), center=dict(predictions=12)),
            any_candidate_transitions={".90": {"small": transition}},
        )
        result = label_coverage(original)
        self.assertEqual(result["arms"]["stepwise_center"]["predictions"], 12)
        self.assertEqual(result["any_candidate_transitions"][".90"]["small"],
                         dict(neither=1, one_pass_only=5, stepwise_center_only=2, both=3))
        self.assertEqual(result["any_candidate_by_size"][".90"]["small"],
                         dict(stepwise_center=5, one_pass=8, one_pass_minus_stepwise_center=3))
        self.assertIn("native", original["arms"])


if __name__ == "__main__":
    unittest.main()
