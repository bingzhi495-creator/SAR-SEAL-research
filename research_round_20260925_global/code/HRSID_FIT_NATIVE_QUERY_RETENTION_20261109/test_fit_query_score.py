import unittest

import numpy as np
from fit_query_score import summarize
from query_retention import analyze_image


class GateTests(unittest.TestCase):
    def test_strict_recovery_and_gate(self):
        # 12 groups, each has one missing strict match. One empty native
        # candidate can be replaced by the next nonempty ship candidate.
        case = analyze_image(np.array([[0.], [.95], [.70]]), np.array([.9, .8, .7]),
                             [0], np.array([True, False, False]))
        result = summarize([case] * 12, main=True)
        self.assertEqual(result["eligible_gt"], 12)
        self.assertEqual(result["policies"]["all_queries"]["strict_gain_groups"], 12)
        self.assertEqual(result["policies"]["nonempty_same_count"]["strict_paired_interval"], [1., 1.])
        self.assertEqual(result["decision"], "ADVANCE_EQUAL_COUNT_FILTER_TO_PROSPECTIVE_CONTROL")
        held = summarize([case] * 9, main=True)
        self.assertEqual(held["decision"], "STOP_SELECTION_SHORTFALL_HYPOTHESIS_FOR_THIS_COHORT")


if __name__ == "__main__":
    unittest.main()
