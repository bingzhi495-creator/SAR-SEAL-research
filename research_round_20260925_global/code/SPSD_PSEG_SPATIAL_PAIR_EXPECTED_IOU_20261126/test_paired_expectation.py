from itertools import product
import unittest

import numpy as np

from paired_expectation import prepare


def brute(probabilities, pairs, kappa, outside, selected):
    area = outside + int(selected.sum())
    answer = np.zeros(2)
    paired_sites = set(pairs.ravel())
    for ref in range(2):
        p = probabilities[:, ref]
        for truth_tuple in product((False, True), repeat=len(p)):
            truth = np.asarray(truth_tuple, dtype=bool)
            weight = 1.0
            for i, j in pairs:
                lo, hi = max(0.0, p[i] + p[j] - 1), min(p[i], p[j])
                independent = p[i] * p[j]
                target = hi if kappa[ref] >= 0 else lo
                joint = independent + abs(kappa[ref]) * (target - independent)
                table = np.array([[1 - p[i] - p[j] + joint, p[j] - joint],
                                  [p[i] - joint, joint]])
                weight *= table[int(truth[i]), int(truth[j])]
            for i in set(range(len(p))) - paired_sites:
                weight *= p[i] if truth[i] else 1 - p[i]
            answer[ref] += weight * (outside + (truth & selected).sum()) / (area + (truth & ~selected).sum())
    return answer


class PairedExpectationTest(unittest.TestCase):
    def test_exact_convolution_reproduces_brute_force_all_endpoint_selections(self):
        p = np.array([[.19, .7], [.72, .19], [.28, .33], [1, .64], [0, .5], [.63, 1.]])
        pairs = np.array([[0, 3], [1, 2]])
        kappa = np.array([.91, -.88])
        outside = 3
        evaluator = prepare(p, pairs, kappa, outside)
        for selected in product((False, True), repeat=len(p)):
            chosen = np.array(selected, dtype=bool)
            area = outside + int(chosen.sum())
            np.testing.assert_allclose(evaluator.utility(chosen, area), brute(p, pairs, kappa, outside, chosen),
                                       rtol=0, atol=1e-14)


    def test_zero_dependence_equals_ungrouped_and_batch_bounds_hold(self):
        p = np.array([[.45, .78], [.87, .1], [.38, .77], [.63, .89]])
        pairs = np.array([[0, 2], [1, 3]])
        outside = 5
        chosen = np.array(list(product((False, True), repeat=len(p))))
        areas = outside + chosen.sum(axis=1)
        neutral = prepare(p, pairs, np.zeros(2), outside)
        ungrouped = prepare(p, [], np.zeros(2), outside)
        for beta in ([1., -1.], [-.43, .83]):
            with self.subTest(beta=beta):
                dependent = prepare(p, pairs, beta, outside)
                bounds = dependent.bound(chosen, areas)
                for row, area, limit in zip(chosen, areas, bounds, strict=True):
                    expected = neutral.utility(row, area)
                    np.testing.assert_allclose(expected, ungrouped.utility(row, area), rtol=0, atol=1e-14)
                    np.testing.assert_array_equal(neutral.bound(row, area), np.zeros(2))
                    np.testing.assert_allclose(limit, dependent.bound(row, area), rtol=0, atol=1e-14)
                    self.assertTrue(np.all(np.abs(dependent.utility(row, area) - expected) <= limit + 1e-14))
        same_count = areas == outside + 2
        np.testing.assert_allclose(dependent.bound(chosen[same_count], outside + 2), bounds[same_count])

    def test_dependence_reverses_equal_area_winner_with_unchanged_marginals(self):
        p = np.repeat(np.array([[.5], [.5], [.49], [0.]]), 2, axis=1)
        independent = prepare(p, [[0, 1]], [0., 0.], 1)
        dependent = prepare(p, [[0, 1]], [1., 1.], 1)
        first = np.array([True, False, False, False])
        third = np.array([False, False, True, False])
        np.testing.assert_allclose([independent.utility(first, 2), independent.utility(third, 2)],
                                   [[.533125] * 2, [.5277083333333333] * 2], rtol=0, atol=1e-14)
        np.testing.assert_allclose([dependent.utility(first, 2), dependent.utility(third, 2)],
                                   [[.5016666666666667] * 2, [.55875] * 2], rtol=0, atol=1e-14)

    def test_rejects_ambiguous_support_and_wrong_area(self):
        p = np.full((4, 2), .5)
        with self.assertRaises(ValueError):
            prepare(p, [[0, 1], [1, 2]], [.5, .5], 5)
        evaluator = prepare(p, [[0, 1]], [.5, .5], 5)
        with self.assertRaises(ValueError):
            evaluator.utility(np.array([True, False, True, False]), 6)
        with self.assertRaises(ValueError):
            evaluator.bound(np.array([1, 0, 1, 0]), 7)


if __name__ == "__main__":
    unittest.main()

