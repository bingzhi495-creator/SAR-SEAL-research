from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_grid_cancellation as c


class GridTests(unittest.TestCase):
    def test_native_half_pixel_and_clamp(self):
        p = np.array([0, 5, 10, 15])
        ids, w = c.interpolation(p, (4, 4), 2)
        np.testing.assert_allclose(c.gather(np.arange(4), ids, w), [0, .75, 2.25, 3])
        np.testing.assert_allclose(w.sum(axis=1), 1.)

    def test_sparse_operator_and_adjoint(self):
        p = np.array([0, 2, 3, 5, 8, 9, 14])
        ids, w = c.interpolation(p, (3, 5), 4)
        nodes = np.arange(16.) / 8
        g = np.array([1, -2, .5, 1, -.5, 2, -3.])
        self.assertAlmostEqual(g @ c.gather(nodes, ids, w), nodes @ c.scatter(g, ids, w, 4))

    def test_opposed_gradients_cancel_only_when_nodes_shared(self):
        p = np.array([0, 3])
        g = np.array([.75, -.75])
        for size, expected in ((1, 1.), (2, 0.)):
            ids, w = c.interpolation(p, (2, 2), size)
            measured = c.cancellation(g, ids, w, size)
            self.assertEqual(measured, expected)

    def test_local_sign_reversal_is_only_native_scanline_witness(self):
        p = np.array([3, 4, 5])
        gt = np.array([False, True, False])
        self.assertEqual(c.nonmonotone_scanlines(p, gt, (3, 3), 1), (1, 0))
        self.assertEqual(c.nonmonotone_scanlines(p, gt, (3, 3), 3), (0, 0))

    def test_correct_labels_recover_frozen_target(self):
        z = np.array([-1., -.1, .1, 1.])
        action = np.array([1, 1, -1, -1])
        flip = np.array([True, False, True, False])
        np.testing.assert_array_equal(c.targets(z, action, flip), [True, False, False, True])


    def test_same_errors_and_directional_gradient_readout(self):
        p = np.array([0, 1, 2, 3])
        z = np.array([-.1, .1, -1., 1.])
        sign = np.array([1, -1, 1, -1])
        errors = np.array([True, True, False, False])
        coarse, _ = c.instance_report(p, z, sign, errors, (2, 2), 1)
        fine, _ = c.instance_report(p, z, sign, errors, (2, 2), 2)
        self.assertEqual(coarse["error_BCE_cancellation"], 1.)
        self.assertEqual(fine["error_BCE_cancellation"], 0.)
        for key in ("full_BCE_nonhelpful_on_add_errors", "full_BCE_nonhelpful_on_remove_errors"):
            self.assertEqual(coarse[key], 1.)
            self.assertEqual(fine[key], 0.)
        self.assertEqual(fine["full_BCE_adverse_on_correct"], 0.)


if __name__ == "__main__":
    unittest.main()
