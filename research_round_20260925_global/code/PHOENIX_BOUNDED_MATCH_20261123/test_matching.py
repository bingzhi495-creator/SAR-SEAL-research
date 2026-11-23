"""Synthetic validation of the bounded cardinality matcher."""
from itertools import combinations, permutations
import unittest

import numpy as np
from scipy.optimize import linear_sum_assignment

from matching import bounded_endpoints


def posterior(value):
    return int(float(value) * (1 << 149))


def pairs(dist, chosen_adds, chosen_drops, cap):
    for ordering in permutations(chosen_drops):
        lengths = tuple(int(dist[a, b]) for a, b in zip(chosen_adds, ordering))
        if max(lengths) <= cap:
            yield sum(lengths)


def check_case(case, coords, score, k, width):
    ay, ax = np.divmod(coords[:, 0], width)
    dy, dx = np.divmod(coords[:, 1], width)
    dist = (ay[:, None] - dy) ** 2 + (ax[:, None] - dx) ** 2
    _, permutation = linear_sum_assignment(dist)
    stable_add = tuple(np.argsort(-score[:, 0], kind="stable"))
    stable_drop = tuple(np.argsort(score[:, 1], kind="stable"))
    pa = sorted(range(len(coords)), key=lambda i: -(posterior(score[i, 0]) - posterior(score[permutation[i], 1])))[:k]
    pd = permutation[pa]
    cap = max(int(dist[i, j]) for i, j in zip(pa, pd))
    ia, idrop = np.array(stable_add[:k]), np.array(stable_drop[:k])
    options = []
    for adds in combinations(range(len(coords)), k):
        for drops in combinations(range(len(coords)), k):
            lengths = tuple(pairs(dist, adds, drops, cap))
            if lengths:
                gain = sum(posterior(score[i, 0]) for i in adds) - sum(posterior(score[j, 1]) for j in drops)
                key = (gain, tuple(i in adds for i in stable_add), tuple(i in drops for i in stable_drop), -min(lengths))
                options.append((key, set(adds), set(drops)))
    expected, adds, drops = max(options)
    aa, dd, counts = bounded_endpoints(coords, score, k, width, np.array(pa), pd, ia, idrop)
    case.assertEqual(set(aa), adds)
    case.assertEqual(set(dd), drops)
    case.assertEqual(sum(int(dist[a, d]) for a, d in zip(aa, dd)), -expected[3])
    case.assertTrue(all(int(dist[a, d]) <= cap for a, d in zip(aa, dd)))
    case.assertEqual(counts["independent_feasible"], int(any(pairs(dist, tuple(ia), tuple(idrop), cap))))
    return counts


class BoundedMatchTests(unittest.TestCase):
    def test_junction_allocates_an_alternative_to_fixed_pair(self):
        size = 12
        coords = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        tiny = np.float32(2.0 ** -149)
        score = np.array([[tiny, 0.0], [0.0, tiny], [0.5, 0.25], [0.5, 0.25], [0.5, 1.0]], dtype=np.float32)
        check_case(self, coords, score, 2, size)

    def test_matches_exhaustive_search(self):
        rng = np.random.default_rng(13423)
        tested = [0, 0]
        for _ in range(180):
            n = int(rng.integers(2, 7))
            k = int(rng.integers(1, min(4, n) + 1))
            coords = rng.choice(64, (n, 2), replace=False)
            if rng.integers(2):
                score = rng.choice(np.array([0, 2.0 ** -149, 2.0 ** -148, 0.2, 0.5, 1], dtype=np.float32), (n, 2))
            else:
                score = rng.random((n, 2), dtype=np.float32)
            counts = check_case(self, coords, score, k, 8)
            tested[counts["independent_feasible"]] += 1
        self.assertGreater(min(tested), 15)


if __name__ == "__main__":
    unittest.main()
