# encoding: utf-8
"""SYNTHETIC_ONLY tracer: shared-union F lookup for one native query.

The core under test must not load real caches, weights, images, or models.
Root owns test execution and will add K0 and missing-F rejection cases later.
"""
import unittest

import numpy as np

import N23_FOURHEAD_DEV_CORE_UNBOUND_20261017_01 as core


class FourHeadDevQueryPlanContractTests(unittest.TestCase):
    def test_Omega_coordinates_lookup_their_actual_nonidentity_union_positions(self):
        native_L = np.array([[-1.0, 3.0, 2.0], [-4.0, 5.0, -2.0]], dtype=np.float32)
        union_flat_indices = np.array([0, 1, 2, 4, 5], dtype=np.int64)

        plan = core.query_plan(np, native_L, union_flat_indices, expected_K=3)

        np.testing.assert_array_equal(plan['Omega_flat_indices'], np.array([0, 2, 5], dtype=np.int64))
        np.testing.assert_array_equal(plan['union_positions'], np.array([0, 2, 4], dtype=np.int64))
        self.assertEqual(plan['Omega_flat_indices'].dtype, np.dtype('int64'))
        self.assertEqual(plan['union_positions'].dtype, np.dtype('int64'))
        self.assertEqual(plan['K'], 3)
        # Coordinate 5 must address union row 4, rather than F row 5 or Omega row 2.
        shared_F_rows = np.array([100, 101, 102, 104, 105], dtype=np.float32)
        np.testing.assert_array_equal(shared_F_rows[plan['union_positions']], [100, 102, 105])

    def test_only_Omega_changes_and_zero_is_background(self):
        native = np.array([[-1., 3., 2.], [-4., 5., -2.]], dtype=np.float32)
        indices = np.array([0, 2, 5], dtype=np.int64)
        logits = np.array([1., 0., 2.], dtype=np.float32)
        result = core.apply_query_logits(np, native, indices, logits)
        np.testing.assert_array_equal(result, [[True, True, False], [False, True, True]])
        np.testing.assert_array_equal(native, [[-1., 3., 2.], [-4., 5., -2.]])

    def test_missing_feature_and_changed_K_are_rejected(self):
        native = np.array([[-1., 3., 2.], [-4., 5., -2.]], dtype=np.float32)
        with self.assertRaises(ValueError):
            core.query_plan(np, native, np.array([0, 2], dtype=np.int64), expected_K=3)
        with self.assertRaises(ValueError):
            core.query_plan(np, native, np.array([0, 2, 5], dtype=np.int64), expected_K=2)

    def test_K0_preserves_native_and_non_Omega_writes_are_rejected(self):
        native = np.array([[-3., 4.]], dtype=np.float32)
        plan = core.query_plan(np, native, np.array([1], dtype=np.int64), expected_K=0)
        self.assertEqual(plan['K'], 0)
        np.testing.assert_array_equal(core.apply_query_logits(np, native, plan['Omega_flat_indices'],
                                                            np.empty(0, dtype=np.float32)), [[False, True]])
        with self.assertRaises(ValueError):
            core.apply_query_logits(np, native, np.array([1], dtype=np.int64), np.array([-1.], dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
