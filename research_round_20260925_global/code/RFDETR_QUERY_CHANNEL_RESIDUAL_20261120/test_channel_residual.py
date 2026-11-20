"""Synthetic channel vectors only, with no model or dataset input."""
from hashlib import sha256
import unittest

import numpy as np

from channel_residual import point_feature_controls, rademacher_matrix


def inputs(k=2):
    old = np.arange(k*2*14, dtype=np.float32).reshape(k, 2, 14) / 57
    spatial = np.zeros((k, 2, 256), dtype=np.float64)
    for i, row in enumerate(spatial.reshape(-1, 256)):
        row[i] = 1.
    query = np.ones(256, dtype=np.float64)
    coords = np.arange(k*2, dtype=np.int64).reshape(k, 2)
    layers = np.zeros((k, 2), dtype=np.int64)
    return old, spatial, query, coords, layers


class ResidualTests(unittest.TestCase):
    def test_permutation_preserves_candidate_sign_layer_and_immutable_pixel_order(self):
        old, spatial, query, coords, _ = inputs(3)
        layers = np.array([[0,0],[0,1],[1,1]], dtype=np.int64)
        result = point_feature_controls(old, spatial, query, coords, layers, candidate_id=(1,7,99))
        source = result['permutation_source_flat_indices']
        np.testing.assert_array_equal(source, [[2,1],[0,5],[4,3]])
        self.assertEqual(sorted(source.reshape(-1).tolist()), list(range(6)))
        self.assertEqual(result['metadata']['singleton_points_kept'], 2)
        for receiver, donor in enumerate(source.reshape(-1)):
            self.assertEqual(receiver % 2, donor % 2)
            self.assertEqual(layers.reshape(-1)[receiver], layers.reshape(-1)[donor])
        np.testing.assert_array_equal(result['residuals']['query_conditioned_permuted'],
            result['residuals']['query_conditioned_centered'].reshape(-1,16)[source.reshape(-1)].reshape(3,2,16))
        order = np.array([2,0,1])
        reordered = point_feature_controls(old[order], spatial[order], query, coords[order], layers[order],
                                           candidate_id=(1,7,99))
        np.testing.assert_array_equal(reordered['residuals']['query_conditioned_permuted'],
                                      result['residuals']['query_conditioned_permuted'][order])
        repeated = point_feature_controls(old, spatial, query, coords, layers, candidate_id=(1,7,99))
        np.testing.assert_array_equal(source, repeated['permutation_source_flat_indices'])
        np.testing.assert_array_equal(result['features']['query_conditioned_permuted'][...,:14], old)
        np.testing.assert_array_equal(result['coordinates'], coords)

    def test_constant_vectors_zero_query_and_float64_centering(self):
        old, spatial, query, coords, layers = inputs()
        constant = point_feature_controls(old, np.full_like(spatial,7.), np.full_like(query,3.),
                                          coords, layers, candidate_id=(1,7,99))
        for residual in constant['residuals'].values():
            np.testing.assert_array_equal(residual, np.zeros((2,2,16),dtype=np.float32))
        zero_q = point_feature_controls(old, spatial, np.zeros_like(query), coords, layers, candidate_id=(1,7,99))
        self.assertTrue(np.any(zero_q['residuals']['ordinary_centered'] != 0))
        self.assertFalse(np.any(zero_q['residuals']['query_conditioned_centered']))
        # A constant F with a nonuniform Q can retain a query-only channel signal.
        varied_q = np.arange(256,dtype=np.float64)
        query_only = point_feature_controls(old, np.full_like(spatial,7.), varied_q, coords, layers,
                                            candidate_id=(1,7,99))
        self.assertFalse(np.any(query_only['residuals']['ordinary_centered']))
        self.assertTrue(np.any(query_only['residuals']['query_conditioned_centered']))
        # Casting first would destroy this within-channel variation.
        offset = np.broadcast_to(np.arange(256,dtype=np.float64)+2.**40, (2,2,256)).copy()
        precise = point_feature_controls(old, offset, query, coords, layers, candidate_id=(1,7,99))
        expected = ((np.arange(256,dtype=np.float64)-127.5) @ rademacher_matrix().astype(np.float64)/16).astype(np.float32)
        np.testing.assert_array_equal(precise['residuals']['ordinary_centered'], np.broadcast_to(expected,(2,2,16)))
        empty = point_feature_controls(*inputs(0), candidate_id=(1,7,99))
        self.assertEqual(empty['features']['zero'].shape, (0,2,30))
        self.assertEqual(empty['metadata']['singleton_points_kept'], 0)

    def test_duplicate_odd_malformed_nonfinite_and_float32_overflow_rejected(self):
        original = inputs()
        def replaced(index, value):
            arguments = list(original)
            arguments[index] = value
            return arguments
        duplicated = original[3].copy()
        duplicated[1,1] = duplicated[0,0]
        bad_query = original[2].copy()
        bad_query[0] = np.nan
        overflowing = original[1].copy()
        overflowing[0,0,0] = 1e300
        invalid = [replaced(3,duplicated), replaced(3,np.arange(3)),
                   replaced(1,np.zeros((3,256),dtype=np.float64)),
                   replaced(0,np.zeros((2,2,13),dtype=np.float32)),
                   replaced(0,original[0].astype(np.float64)),
                   replaced(2,bad_query), replaced(2,np.ones((256,1))),
                   replaced(1,original[1].astype(np.float16)), replaced(1,overflowing),
                   replaced(4,np.zeros((2,2),dtype=float)), replaced(4,np.array([[0,-1],[0,0]])),
                   replaced(3,original[3].astype(float)), inputs(33)]
        for i, arguments in enumerate(invalid):
            with self.subTest(case=i), self.assertRaises(ValueError):
                point_feature_controls(*arguments, candidate_id=(1,7,99))
        for candidate in ([1,7,99], (1,True,99), (1,0,99)):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                point_feature_controls(*original, candidate_id=candidate)

    def test_fixed_projection_and_equal_dot_different_channel_residual(self):
        old, spatial, query, coords, layers = inputs()
        original = [v.copy() for v in (old, spatial, query, coords, layers)]
        random_before = np.random.get_state()
        matrix = rademacher_matrix()
        self.assertEqual(matrix.shape, (256, 16))
        self.assertEqual(matrix.dtype, np.int8)
        self.assertEqual(set(np.unique(matrix)), {-1, 1})
        self.assertEqual(sha256(matrix.tobytes()).hexdigest(),
                         '96db66d62d88541d31254f408a18330c0b03f423364b64d2f50cdf19ce466448')
        result = point_feature_controls(old, spatial, query, coords, layers, candidate_id=(1, 7, 99))
        self.assertEqual(set(result['features']),
                         {'zero', 'ordinary_centered', 'query_conditioned_centered', 'query_conditioned_permuted'})
        self.assertEqual(float(spatial[0, 0] @ query), float(spatial[0, 1] @ query))
        residual = result['residuals']['query_conditioned_centered']
        self.assertFalse(np.array_equal(residual[0, 0], residual[0, 1]))
        expected = ((spatial-spatial.mean(axis=-1, keepdims=True)) @ matrix.astype(np.float64) / 16).astype(np.float32)
        np.testing.assert_array_equal(residual, expected)
        np.testing.assert_array_equal(residual, result['residuals']['ordinary_centered'])
        for value in result['features'].values():
            self.assertEqual(value.shape, (2, 2, 30))
            self.assertEqual(value.dtype, np.float32)
            np.testing.assert_array_equal(value[..., :14], old)
            self.assertTrue(np.isfinite(value).all())
        np.testing.assert_array_equal(result['coordinates'], coords)
        for before, after in zip(original, (old, spatial, query, coords, layers)):
            np.testing.assert_array_equal(before, after)
        random_after = np.random.get_state()
        self.assertEqual(random_before[0], random_after[0])
        np.testing.assert_array_equal(random_before[1], random_after[1])
        self.assertEqual(random_before[2:], random_after[2:])


if __name__ == '__main__':
    unittest.main()
