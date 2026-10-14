"""NumPy/stdlib SYNTHETIC_ONLY behavior tests; no science files or training."""
import importlib.util
import itertools
from pathlib import Path
import unittest
import numpy as np

SOURCE = Path(__file__).with_name('N16_SCOUT_ACTION_LABELS_AND_SELECTION_UNBOUND_20261014_01.py')
SPEC = importlib.util.spec_from_file_location('_N16_synthetic_helpers', SOURCE)
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)


class HelperTests(unittest.TestCase):
    def test_utility_truth_table_distinguishes_real_action_from_native_error(self):
        P = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.bool_)
        G = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.bool_)
        A = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.bool_)
        saved = [x.copy() for x in (P, G, A)]
        labels = HELPER.utility_labels(P, G, A)
        self.assertEqual(HELPER.UTILITY_NAMES, ('help', 'harm', 'noop'))
        np.testing.assert_array_equal(labels, [2, 1, 2, 0, 0, 2, 1, 2])
        self.assertEqual(labels.dtype, np.int64)
        for value, before in zip((P, G, A), saved):
            np.testing.assert_array_equal(value, before)

    def test_error_labels_are_FP_FN_correct_independent_of_PR_action(self):
        P = np.array([False, False, True, True])
        G = np.array([False, True, False, True])
        labels = HELPER.error_labels(P, G)
        self.assertEqual(HELPER.ERROR_NAMES, ('FP', 'FN', 'correct'))
        np.testing.assert_array_equal(labels, [2, 1, 0, 2])
        # Both native errors may remain utility noop when PR does not flip.
        np.testing.assert_array_equal(HELPER.utility_labels(P, G, P), [2, 2, 2, 2])
        self.assertEqual(labels.dtype, np.int64)

    def test_scores_use_softmax_help_minus_harm_and_native_error_probability(self):
        logits = np.log(np.array([[2.0, 1.0, 1.0], [1.0, 3.0, 4.0]]))
        np.testing.assert_allclose(HELPER.utility_score(logits), [0.25, -0.25], rtol=0, atol=1e-15)
        np.testing.assert_allclose(HELPER.error_score(logits), [0.5, 0.0], rtol=0, atol=1e-15)
        high = np.array([[1000.0, 1000.0, 1000.0], [1000.0, -1000.0, 0.0]])
        np.testing.assert_allclose(HELPER.utility_score(high), [0.0, 1.0], rtol=0, atol=1e-15)
        np.testing.assert_allclose(HELPER.error_score(high), [1.0 / 3.0, 1.0], rtol=0, atol=1e-15)

    def test_atmost_and_exact_use_only_real_C_with_locked_stable_ties(self):
        L = np.array([[0.3, -0.1, 0.1], [0.9, -0.2, 0.1]], dtype=np.float32)
        C = np.array([[1, 1, 1], [0, 1, 1]], dtype=np.bool_)
        score = np.array([[0.5, 0.5, 0.5], [1000.0, 0.0, -0.5]])
        main = HELPER.choose_candidates(L, C, score, 2)
        np.testing.assert_array_equal(np.flatnonzero(main), [1, 2])
        # K4 main keeps only three positive physical candidates; exact fills
        # with the actual zero-score point4, then may use actual negative point5.
        np.testing.assert_array_equal(np.flatnonzero(HELPER.choose_candidates(L, C, score, 4)), [0, 1, 2])
        np.testing.assert_array_equal(np.flatnonzero(HELPER.choose_candidates(L, C, score, 4, mode='exact')), [0, 1, 2, 4])
        np.testing.assert_array_equal(HELPER.choose_candidates(L, C, score, 5, mode='exact'), C)
        self.assertFalse(np.any(main & ~C))
        np.testing.assert_array_equal(HELPER.choose_candidates(L, C, score, 0), np.zeros_like(C))

    def test_selected_PR_changes_only_selected_real_C_points_bitwise(self):
        L = np.array([-0.0, 0.1, -0.2, 0.3], dtype=np.float32)
        delta = np.array([100.0, -1.0, 1.0, -100.0], dtype=np.float32)
        C = np.array([0, 1, 1, 0], dtype=np.bool_)
        selected = np.array([0, 1, 0, 0], dtype=np.bool_)
        output = HELPER.apply_selected_residual(L, delta, C, selected)
        self.assertEqual(output.dtype, L.dtype)
        np.testing.assert_array_equal(output[~selected].view(np.uint32), L[~selected].view(np.uint32))
        self.assertFalse(bool((output > 0)[1]))
        np.testing.assert_array_equal(L.view(np.uint32), np.array([-0.0, 0.1, -0.2, 0.3], dtype=np.float32).view(np.uint32))
        with self.assertRaises(ValueError):
            HELPER.apply_selected_residual(L, delta, C, np.array([1, 0, 0, 0], dtype=np.bool_))

    def test_dense_error_control_ranks_only_actual_PR_flip_opportunity(self):
        delta = np.array([False, True, True], dtype=np.bool_)
        logits = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-np.log(3.0), -np.log(3.0), np.log(2.0)]])
        # No-flip point0 has score0 despite p_error=2/3. Actual flips1/2
        # carry 2*p_error-1; this is a ranking score, not a new correction.
        actual = HELPER.dense_error_flip_score(delta, logits)
        np.testing.assert_allclose(actual, [0.0, 1.0 / 3.0, -0.5], rtol=0, atol=1e-15)
        self.assertEqual(actual.dtype, np.float64)
        np.testing.assert_array_equal(delta, [False, True, True])
        selected = HELPER.choose_candidates(np.zeros(3), np.ones(3, dtype=np.bool_), actual, 3)
        np.testing.assert_array_equal(selected, [False, True, False])
        with self.assertRaises(TypeError):
            HELPER.dense_error_flip_score(delta.astype(np.float32), logits)

    def test_MagNet_binary_hand_calculation_and_stable_extremes(self):
        log3 = np.log(3.0)
        L = np.array([0.0, log3, -log3])
        u = np.array([log3, 0.0, 2.0 * log3])
        score = HELPER.magnet_score(L, u)
        np.testing.assert_allclose(score, [0.5, 0.25, 0.25], rtol=0, atol=2e-16)
        # Literal middle-range formula is only an independent algebra check,
        # not the numerically binding implementation at saturated logits.
        p = 1.0 / (1.0 + np.exp(-L))
        q = 1.0 / (1.0 + np.exp(-(L + u)))
        literal = (1.0 - np.abs(2.0 * p - 1.0)) * np.abs(2.0 * q - 1.0)
        np.testing.assert_allclose(score, literal, rtol=0, atol=2e-16)
        extreme_L = np.array([-1000.0, -20.0, 0.0, 20.0, 40.0, 1000.0])
        extreme_u = np.zeros_like(extreme_L)
        with np.errstate(all='raise'):
            stable = HELPER.magnet_score(extreme_L, extreme_u)
        self.assertTrue(np.all(np.isfinite(stable)))
        self.assertTrue(np.all((stable >= 0) & (stable <= 1)))
        self.assertGreater(stable[-2], 0.0)  # Preserve mathematical uncertainty atL40.
        self.assertEqual(stable[2], 0.0)

    def test_rejects_implicit_or_masked_labels_and_invalid_physical_budget(self):
        P = np.array([True, False])
        G = np.array([False, True])
        masked = np.ma.array(G, mask=[True, False])
        with self.assertRaises(TypeError):
            HELPER.utility_labels(P, masked, P)
        with self.assertRaises(TypeError):
            HELPER.error_labels(P.astype(np.uint8), G)
        with self.assertRaises(ValueError):
            HELPER.utility_labels(P, G[:1], P)
        L = np.array([0.1, 0.2])
        C = np.array([True, False])
        score = np.array([1.0, 1000.0])
        for bad in (True, 1.0, np.array(1)):
            with self.assertRaises(TypeError):
                HELPER.choose_candidates(L, C, score, bad)
        for bad in (-1, 2):
            with self.assertRaises(ValueError):
                HELPER.choose_candidates(L, C, score, bad, mode='exact')
        with self.assertRaises(ValueError):
            HELPER.choose_candidates(L, C, np.array([1.0, np.nan]), 1)
        with self.assertRaises(TypeError):
            HELPER.choose_candidates(L, C.astype(np.uint8), score, 1)

    def test_empty_K0_and_exact_numeric_ties_have_no_GT_or_rank_tolerance(self):
        empty_bool, empty_float = np.zeros(0, dtype=np.bool_), np.zeros(0, dtype=np.float32)
        self.assertEqual(HELPER.utility_labels(empty_bool, empty_bool, empty_bool).shape, (0,))
        self.assertEqual(HELPER.error_labels(empty_bool, empty_bool).shape, (0,))
        self.assertEqual(HELPER.utility_score(np.zeros((0, 3), dtype=np.float32)).shape, (0,))
        for mode in ('atmost', 'exact'):
            self.assertEqual(HELPER.choose_candidates(empty_float, empty_bool, empty_float, 0, mode=mode).shape, (0,))
        L = np.array([0.01, 100.0, 0.0, -0.0])
        C = np.ones(4, dtype=np.bool_)
        score = np.array([0.5, np.nextafter(0.5, np.inf), 0.0, -0.0])
        # One representable score increment outranks native uncertainty; no epsilon tie.
        np.testing.assert_array_equal(np.flatnonzero(HELPER.choose_candidates(L, C, score, 1)), [1])
        np.testing.assert_array_equal(np.flatnonzero(HELPER.choose_candidates(L, C, score, 4)), [0, 1])
        tiny = np.array([0.0, np.nextafter(0.0, 1.0), -1.0, 0.0])
        np.testing.assert_array_equal(np.flatnonzero(HELPER.choose_candidates(L, C, tiny, 1)), [1])
        with self.assertRaises(TypeError):
            HELPER.choose_candidates(L, C, score, 1, G=np.ones(4, dtype=np.bool_))

    def test_all_four_point_physical_pools_match_independent_Python_order(self):
        L = np.array([-0.2, 0.1, -0.1, 0.0])
        for pool in range(16):
            C = np.array([bool(pool & (1 << i)) for i in range(4)])
            for values in itertools.product((-1.0, 0.0, 1.0), repeat=4):
                score = np.array(values)
                for mode in ('atmost', 'exact'):
                    physical = [i for i in range(4) if C[i] and (mode == 'exact' or score[i] > 0)]
                    ranking = sorted(physical, key=lambda i: (-float(score[i]), abs(float(L[i])), i))
                    for K in range(int(C.sum()) + 1):
                        expected = np.zeros(4, dtype=np.bool_)
                        expected[ranking[:K]] = True
                        np.testing.assert_array_equal(HELPER.choose_candidates(L, C, score, K, mode=mode), expected)


if __name__ == '__main__':
    unittest.main()
