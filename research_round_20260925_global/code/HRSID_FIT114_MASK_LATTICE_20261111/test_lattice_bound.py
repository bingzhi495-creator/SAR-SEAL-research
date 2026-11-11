import unittest
from itertools import product

import numpy as np

import lattice_bound
from lattice_bound import axis_cell_ids


class LatticeBoundTests(unittest.TestCase):
    def test_half_pixel_coordinates_partition_clamped_runs_and_knots(self):
        self.assertEqual(axis_cell_ids(3, 9).tolist(), [-1, 0, 0, 0, 1, 1, 1, 2, 2])
        np.testing.assert_array_equal(axis_cell_ids(3, 3), [0, 1, 2])
        for nodes in (144, 288):
            ids = axis_cell_ids(nodes, 800)
            expected = np.floor((np.arange(800) + 0.5) * nodes / 800 - 0.5)
            np.testing.assert_array_equal(ids, expected)

    def test_weighted_one_transition_cost_matches_brute_force(self):
        truth = np.array(list(product((False, True), repeat=6)))
        axis = np.arange(truth.shape[1])
        for fn, fp in ((10, 9), (4, 3)):
            expected = []
            for row in truth:
                candidates = [axis < cut for cut in range(len(row) + 1)]
                candidates += [~candidate for candidate in candidates]
                expected.append(min(fn * np.count_nonzero(row & ~pred)
                                    + fp * np.count_nonzero(~row & pred)
                                    for pred in candidates))
            got = lattice_bound.one_transition_cost(truth, -1, fn, fp)
            np.testing.assert_array_equal(got, expected)

    def test_interior_one_pixel_island_has_a_certified_error(self):
        truth = np.zeros((12, 12), dtype=bool)
        truth[5, 5] = True
        self.assertEqual(lattice_bound.cell_cost_bounds(truth, 2), {"75": 4, "90": 10})

    def test_cell_bound_does_not_exceed_independent_bilinear_errors(self):
        random = np.random.default_rng(523)
        nodes, pixels = 4, 18
        location = np.clip((np.arange(pixels) + 0.5) * nodes / pixels - 0.5, 0, nodes - 1)
        for _ in range(8):
            truth = random.random((pixels, pixels)) < 0.4
            bound = lattice_bound.cell_cost_bounds(truth, nodes)
            for _ in range(8):
                coarse = random.standard_normal((nodes, nodes))
                horizontal = [np.interp(location, np.arange(nodes), row) for row in coarse]
                pred = np.stack([np.interp(location, np.arange(nodes), column)
                                 for column in np.array(horizontal).T], axis=1) > 0
                fn = np.count_nonzero(truth & ~pred)
                fp = np.count_nonzero(~truth & pred)
                self.assertLessEqual(bound["90"], 10 * fn + 9 * fp)
                self.assertLessEqual(bound["75"], 4 * fn + 3 * fp)
            self.assertEqual(lattice_bound.cell_cost_bounds(pred, nodes), {"75": 0, "90": 0})

    def test_threshold_certificate_is_strict_and_paired(self):
        rows = [
            ("a", 4, {144: {"75": 4, "90": 10}, 288: {"75": 0, "90": 0}}),
            ("a", 9, {144: {"75": 5, "90": 9}, 288: {"75": 4, "90": 8}}),
            ("b", 1024, {144: {"75": 0, "90": 0}, 288: {"75": 0, "90": 0}}),
        ]
        result = lattice_bound.summarize(rows)
        small = result["small"]
        self.assertEqual((small["instances"], small["source_groups"]), (2, 1))
        self.assertEqual(small["144"]["75"]["certified_below_threshold"], 0)
        self.assertEqual(small["144"]["90"]["certified_below_threshold"], 1)
        self.assertEqual(small["paired_288_vs_144"]["90"]["certified_144_only"], 1)
        self.assertEqual(result["medium"]["instances"], 1)
        self.assertEqual(result["large"]["instances"], 0)


if __name__ == "__main__":
    unittest.main()
