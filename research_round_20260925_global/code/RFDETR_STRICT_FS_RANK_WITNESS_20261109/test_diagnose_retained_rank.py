from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_retained_rank import compare_first_ranks, witness


class RankWitnessTests(unittest.TestCase):
    def test_same_candidate_can_supply_two_independent_witnesses(self):
        matrix = np.array([[0.90, 0.91], [0.89, 0.80], [0.95, 0.20]])
        result = witness(matrix)
        self.assertEqual(result["first_rank"].tolist(), [1, 1])
        self.assertEqual(result["best_iou"].tolist(), [0.95, 0.91])
        self.assertEqual(result["injective"], {"1": 1, "10": 2, "100": 2})


    def test_pair_separates_new_candidates_from_shared_rank_crossings(self):
        f = np.array([0, 12, 7, 0, 5, 9, 20])
        s = np.array([8, 0, 10, 0, 14, 7, 18])
        result = compare_first_ranks(f, s)
        self.assertEqual(result["availability"], {"both": 4, "S_only": 1, "F_only": 1, "neither": 1})
        self.assertEqual(result["both_rank_direction"], {"S_earlier": 2, "S_later": 2, "same": 0})
        self.assertEqual(result["cutoff_decomposition"]["10"],
                         {"S_only_inside": 1, "F_only_inside": 0, "both_entered": 0, "both_left": 1, "net_S_minus_F": 0})

if __name__ == "__main__":
    unittest.main()

