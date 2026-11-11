import unittest

import numpy as np

import lattice_witness
from lattice_witness import resize_bilinear


class LatticeWitnessTests(unittest.TestCase):
    def test_resize_uses_half_pixel_coordinates_and_edge_clamp(self):
        field = np.array([[0.0, 1.0], [2.0, 3.0]])
        axes = np.array([0.0, 0.25, 0.75, 1.0])
        np.testing.assert_array_equal(resize_bilinear(field, (4, 4)), 2 * axes[:, None] + axes)
        np.testing.assert_array_equal(resize_bilinear(field, (1, 1)), [[1.5]])

    def test_signed_edt_counts_outside_image_as_background(self):
        mask = np.ones((3, 3), dtype=bool)
        expected = np.ones((3, 3))
        expected[1, 1] = 2
        np.testing.assert_array_equal(lattice_witness.signed_distance(mask), expected)

    def test_threshold_ambiguity_counts_both_errors_conservatively(self):
        truth = np.array([[True, True, False, False]])
        field = np.array([[0.1, 0.0, 0.0, -0.1]])
        result = lattice_witness.threshold_witnesses(truth, field)[0.0]
        self.assertEqual(result, (1, 3, 2))

    def test_constructed_half_plane_is_perfect_at_zero_threshold(self):
        truth = np.zeros((12, 12), dtype=bool)
        truth[:, :6] = True
        coarse = resize_bilinear(lattice_witness.signed_distance(truth), (2, 2))
        native = resize_bilinear(coarse, truth.shape)
        self.assertEqual(lattice_witness.threshold_witnesses(truth, native)[0.0], (72, 72, 0))

    def test_summary_distinguishes_constructed_impossible_and_unknown(self):
        weights = {"75": (4, 3), "90": (10, 9)}
        poor = dict.fromkeys(lattice_witness.THRESHOLDS, (8, 10, 0))
        improved = poor.copy()
        improved[0.5] = (9, 10, 0)
        plain = {"75": 0, "90": 0}
        impossible90 = {"75": 0, "90": 11}
        rows = [
            ("a", 10, {144: impossible90, 288: plain}, {144: poor, 288: improved}),
            ("b", 10, {144: plain, 288: plain}, {144: poor, 288: poor}),
        ]
        got = lattice_witness.summarize(rows, (144, 288), weights)["small"]
        self.assertEqual(got["144"]["90"]["constructive_witnesses"], 0)
        self.assertEqual(got["144"]["90"]["analytic_certified_below"], 1)
        self.assertEqual(got["144"]["90"]["unknown"], 1)
        self.assertEqual(got["288"]["90"]["constructive_witnesses"], 1)
        self.assertEqual(got["288"]["90"]["zero_threshold_witnesses"], 0)
        self.assertEqual(got["paired_288_vs_144"]["90"]["144_impossible_288_witness"], 1)


if __name__ == "__main__":
    unittest.main()
