from unittest import TestCase

import numpy as np

from pair_iou import brier, fit_kappa, graphs


class SpatialPairsTest(TestCase):
    def test_graphs_pair_adjacent_vertices_and_rewire_with_same_sign_counts(self):
        coords = np.array([[[0, 0], [0, 1]], [[5, 1], [5, 2]], [[8, 5], [8, 6]],
                           [[11, 6], [23, 3]]], dtype=np.int32)
        edges = graphs(coords, "scene:1")
        local, control = edges["local"], edges["nonspatial"]
        np.testing.assert_equal(local, [[0, 1], [2, 3], [4, 5]])
        self.assertTrue(np.all(np.abs(coords.reshape(-1, 2)[local[:, 0]] -
                                      coords.reshape(-1, 2)[local[:, 1]]).sum(axis=1) == 1))
        self.assertEqual(sorted(local.ravel()), sorted(control.ravel()))
        self.assertEqual({tuple(row % 2) for row in control}, {(0, 1)})
        self.assertFalse(set(map(tuple, local)) & set(map(tuple, control)))

    def test_proper_brier_fit_allows_both_correlation_signs(self):
        q = np.full((4, 2), 0.5)
        w = np.ones(4)
        for y, expected in (([[0, 0], [0, 0], [1, 1], [1, 1]], 1),
                            ([[0, 1], [0, 1], [1, 0], [1, 0]], -1)):
            with self.subTest(expected=expected):
                y = np.asarray(y)
                kappa = fit_kappa(q, y, w)
                self.assertAlmostEqual(kappa, expected)
                self.assertLess(brier(q, y, kappa).mean(), brier(q, y, 0).mean())
