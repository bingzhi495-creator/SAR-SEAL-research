import unittest

import numpy as np

from bilinear_area import pixel_cell_average


def numerical_average(grid, native_hw, ordinal, samples=768):
    """Independent composite midpoint quadrature over native cell area."""
    height, width = native_hw
    row, col = divmod(ordinal, width)
    offset = (np.arange(samples, dtype=np.float64) + .5) / samples
    # Continuous extension of the stock native-center projection.
    sy = np.clip((row + offset) * grid.shape[0] / height - .5, 0., grid.shape[0] - 1)
    sx = np.clip((col + offset) * grid.shape[1] / width - .5, 0., grid.shape[1] - 1)
    ya, xa = sy.astype(int), sx.astype(int)
    yb = np.minimum(ya + 1, grid.shape[0] - 1)
    xb = np.minimum(xa + 1, grid.shape[1] - 1)
    wy = (sy - ya)[:, None]
    wx = (sx - xa)[None, :]
    low = (1. - wx) * grid[ya[:, None], xa] + wx * grid[ya[:, None], xb]
    high = (1. - wx) * grid[yb[:, None], xa] + wx * grid[yb[:, None], xb]
    return ((1. - wy) * low + wy * high).mean()


class TestBilinearArea(unittest.TestCase):
    def test_constants_across_rectangular_and_degenerate_axes(self):
        for source, native in (((3, 5), (8, 2)), ((1, 7), (6, 1)), ((2, 2), (2, 2))):
            field = np.full(source, .375, dtype=np.float32)
            indices = np.array([[0, native[1]-1], [native[0]*native[1]-1, 0]])
            actual = pixel_cell_average(field, native, indices)
            self.assertEqual(actual.dtype, np.float32)
            np.testing.assert_array_equal(actual, np.full(indices.shape, .375))

    def test_source_equals_native_edges_and_center(self):
        ramp = np.arange(4, dtype=np.float32).reshape(1, 4)
        np.testing.assert_array_equal(pixel_cell_average(ramp, (1, 4), [0, 1, 2, 3]),
                                      np.array([.125, 1., 2., 2.875], dtype=np.float32))
        impulse = np.zeros((3, 3), dtype=np.float32)
        impulse[1, 1] = 1
        expected = np.outer([.125, .75, .125], [.125, .75, .125]).ravel()
        np.testing.assert_array_equal(pixel_cell_average(impulse, (3, 3), np.arange(9)), expected)
        edge = np.array([[1., 0.]], dtype=np.float32)
        np.testing.assert_array_equal(pixel_cell_average(edge, (1, 4), [0, 1, 2, 3]),
                                      np.array([1., .75, .25, 0.], dtype=np.float32))

    def test_each_axis_affine_on_interior_cells(self):
        yy, xx = np.indices((5, 7), dtype=np.float32)
        field = 2. * yy + 3. * xx
        native = (8, 12)
        y, x = (2, 6), (2, 9)
        indices = [a * native[1] + b for a, b in zip(y, x)]
        expected = [2. * ((a+.5) * 5 / 8 - .5) + 3. * ((b+.5) * 7 / 12 - .5) for a, b in zip(y, x)]
        np.testing.assert_array_equal(pixel_cell_average(field, native, indices),
                                      np.array(expected, dtype=np.float32))

    def test_all_native_cells_conserve_average(self):
        field = (np.arange(35, dtype=np.float32).reshape(5, 7) % 11) / 16
        for native in ((1, 1), (2, 9), (8, 3), (7, 4), (5, 7)):
            values = pixel_cell_average(field, native, np.arange(native[0]*native[1]))
            self.assertAlmostEqual(float(np.mean(values, dtype=np.float64)),
                                   float(np.mean(field, dtype=np.float64)), delta=2e-8)

    def test_against_independent_numerical_quadrature(self):
        rng = np.random.default_rng(1729)
        for source, native in (((3, 7), (9, 2)), ((13, 2), (3, 11)), ((2, 6), (7, 1)),
                               ((1, 9), (1, 6)), ((9, 4), (8, 13)), ((4, 5), (4, 5))):
            field = rng.random(source, dtype=np.float32)
            count = native[0] * native[1]
            indices = np.array([count-1, 0, count//2, count//3, 0], dtype=np.int64)
            actual = pixel_cell_average(field, native, indices)
            expected = [numerical_average(field, native, int(x)) for x in indices]
            np.testing.assert_allclose(actual, expected, rtol=0., atol=3e-6)

    def test_boundaries_and_input_validation(self):
        field = np.ones((2, 3), dtype=np.float32)
        self.assertEqual(pixel_cell_average(field, (2, 3), np.empty((2, 0), dtype=np.int32)).shape,
                         (2, 0))
        self.assertEqual(pixel_cell_average(field, (1, 1), np.int64(0)).shape, ())
        for endpoints in ([-1], [6], [float(1)], np.array([2**63], dtype=np.uint64)):
            with self.assertRaises(ValueError):
                pixel_cell_average(field, (2, 3), endpoints)
        for grid in (field.astype(np.float64), np.empty((0, 2), dtype=np.float32),
                     np.full((2, 3), np.nan, dtype=np.float32)):
            with self.assertRaises(ValueError):
                pixel_cell_average(grid, (2, 3), [0])
        for native in ((0, 2), (2., 3), (True, 2), (2,)):
            with self.assertRaises(ValueError):
                pixel_cell_average(field, native, [0])


if __name__ == "__main__":
    unittest.main()

