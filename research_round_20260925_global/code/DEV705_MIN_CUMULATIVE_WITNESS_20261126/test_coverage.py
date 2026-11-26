import sys
from pathlib import Path
from itertools import permutations
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coverage import evaluate, minimum


class WitnessTests(unittest.TestCase):
    def test_infeasible_cap_can_have_smaller_total(self):
        coords = np.array([[19224, 16025], [18413, 15219],
                           [12811, 8014], [13618, 12821]])
        self.assertEqual(
            evaluate(coords[[2, 1]].ravel(), [12811, 15219, 19224, 16025],
                     [19224, 15219, 12811, 8014]),
            (False, 97, 95, 90),
        )

    def test_capped_total_is_not_unrestricted_total(self):
        coords = np.array([[13618, 8821], [11210, 8017],
                           [13614, 13619], [16017, 19223]])
        self.assertEqual(
            evaluate(coords[[0, 2, 3]].ravel(), [13618, 13619, 16017, 8017, 13614, 8821],
                     [13618, 8017, 13614, 13619, 16017, 19223]),
            (False, 122, 111, 116),
        )

    def test_any_matching_can_satisfy_cap(self):
        # The minimum diagonal totals 10, yet its longest edge exceeds 9.
        # The off diagonal totals 18 and both its edges meet the cap.
        cost = np.array([[0, 9], [9, 10]], dtype=np.int64)
        self.assertEqual(minimum(cost), 10)
        self.assertEqual(minimum(cost > 9), 0)

    def test_against_enumeration(self):
        rng = np.random.Generator(np.random.PCG64(5217))
        for n in range(1, 9):
            for _ in range(6):
                costs = rng.integers(60, size=(n, n))
                limit = rng.integers(60)
                totals = []
                feasible = False
                for cols in permutations(range(n)):
                    edges = [costs[j, col] for j, col in enumerate(cols)]
                    totals.append(sum(edges))
                    feasible |= max(edges) <= limit
                self.assertEqual(minimum(costs), min(totals))
                self.assertEqual(minimum(costs > limit) == 0, feasible)


if __name__ == "__main__":
    unittest.main()
