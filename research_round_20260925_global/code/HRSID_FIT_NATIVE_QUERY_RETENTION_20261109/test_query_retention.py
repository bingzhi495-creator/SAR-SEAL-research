import json
import unittest

import numpy as np

from query_retention import analyze_image


class QueryRetentionTests(unittest.TestCase):
    def test_same_count_controls_and_stable_score_ties(self):
        overlap = np.array([[.91, 0], [0, 0], [0, .90], [.80, .80]])
        scores = [.50, .95, .50, .40]
        empty = [False, True, False, False]
        result = analyze_image(overlap, scores, [0, 3], empty)
        self.assertEqual(result["eligible_gt"], 2)
        self.assertEqual(result["native_near_strict"], [False, True])
        policies = result["policies"]
        self.assertEqual(
            {name: row["candidates"] for name, row in policies.items()},
            dict(native=2, ship_same_count=2, nonempty_same_count=2, all_queries=4),
        )
        self.assertEqual(
            {name: row["empty_candidates"] for name, row in policies.items()},
            dict(native=0, ship_same_count=1, nonempty_same_count=0, all_queries=1),
        )
        self.assertEqual(policies["native"]["strict_reach"], [True, False])
        self.assertEqual(policies["ship_same_count"]["strict_reach"], [True, False])
        self.assertEqual(policies["nonempty_same_count"]["strict_reach"], [True, True])
        self.assertEqual(policies["all_queries"]["strict_reach"], [True, True])
        self.assertEqual(json.loads(json.dumps(result)), result)


    def test_matching_keeps_maximum_and_rank_greedy_distinct(self):
        overlap = [[.92, .91], [.90, 0.0]]
        result = analyze_image(overlap, [.10, .90], [0, 1], [False, False])
        policies = result["policies"]
        self.assertEqual(
            policies["native"]["thresholds"]["0.90"],
            dict(independent=2, maximum_one_to_one=2, native_greedy=1),
        )
        self.assertEqual(
            policies["all_queries"]["thresholds"]["0.90"],
            dict(independent=2, maximum_one_to_one=2, native_greedy=2),
        )

    def test_empty_population_and_refill_keep_exact_counts(self):
        scores = [.20, .90, .80]
        empty = [False, True, True]
        result = analyze_image([[.90], [0], [0]], scores, [], empty)
        self.assertEqual(result["policies"]["native"]["strict_reach"], [False])
        self.assertEqual(result["policies"]["all_queries"]["strict_reach"], [True])
        result = analyze_image(np.zeros((3, 0)), scores, [1, 2], empty)
        row = result["policies"]["nonempty_same_count"]
        self.assertEqual((row["candidates"], row["empty_candidates"]), (2, 1))
        self.assertEqual(row["strict_reach"], [])
        self.assertEqual(
            row["thresholds"]["0.75"],
            dict(independent=0, maximum_one_to_one=0, native_greedy=0),
        )

    def test_rejects_duplicate_native_ids_and_inconsistent_empty_rows(self):
        with self.assertRaisesRegex(ValueError, "unique native query IDs"):
            analyze_image([[.5], [0]], [.5, .2], [0, 0], [False, True])
        with self.assertRaisesRegex(ValueError, "zero IoU"):
            analyze_image([[.5], [0]], [.5, .2], [0], [True, True])

if __name__ == "__main__":
    unittest.main()
