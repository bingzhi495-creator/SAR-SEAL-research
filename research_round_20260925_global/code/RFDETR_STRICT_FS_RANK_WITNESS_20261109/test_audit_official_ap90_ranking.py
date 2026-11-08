from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_official_ap90_ranking import global_ranking


class RankingTest(unittest.TestCase):
    def test_global_rank_respects_original_order_when_scores_are_tied(self):
        rows = [
            {"dtScores": [0.95, 0.8, 0.7], "dtMatches": np.array([[0, 1, 0]]), "dtIgnore": np.array([[False, False, True]])},
            {"dtScores": [0.8, 0.6], "dtMatches": np.array([[0, 2]]), "dtIgnore": np.array([[False, False]])},
        ]
        self.assertEqual(global_ranking(rows, 0)[0].tolist(), [0, 1, 0, 2])


if __name__ == "__main__":
    unittest.main()
