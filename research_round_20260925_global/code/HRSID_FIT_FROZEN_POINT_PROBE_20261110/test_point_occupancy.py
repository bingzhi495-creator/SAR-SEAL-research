import unittest
from unittest.mock import patch

import numpy as np

from point_occupancy import bitmap_cell_occupancy, cell_occupancy, polygon_cell_occupancy
from vector_label_geometry import UnsupportedPolygon


class PointOccupancyTest(unittest.TestCase):
    def test_fractional_bitmap_and_polygon_targets_preserve_integer_bank_order(self):
        bitmap = np.asarray([[True, False, False], [True, True, False]])
        polygon = [[.25, .5, 1.75, .5, 1.75, 1.5, .25, 1.5]]
        steps = [dict(kind="resize", before=[2, 3], after=[1, 2])]
        cells = np.asarray([[1, 0], [0, 0], [1, 0]], dtype=np.int64)
        targets = cell_occupancy(bitmap, polygon, steps, cells)
        self.assertEqual(set(targets), {"bitmap", "polygon"})
        self.assertEqual(targets["bitmap"].dtype, np.float64)
        self.assertEqual(targets["polygon"].dtype, np.float64)
        np.testing.assert_allclose(targets["bitmap"], [1 / 6, 5 / 6, 1 / 6], rtol=0, atol=1e-15)
        np.testing.assert_allclose(targets["polygon"], [1 / 12, 5 / 12, 1 / 12], rtol=0, atol=1e-15)

    def test_resize_crop_resize_and_flip_match_on_aligned_square_boundaries(self):
        bitmap = np.zeros((4, 6), dtype=np.bool_)
        bitmap[1:2, :3] = True
        polygon = [[0., 1., 3., 1., 3., 2., 0., 2.]]
        steps = [
            dict(kind="resize", before=[4, 6], after=[8, 12]),
            dict(kind="crop", before=[8, 12], after=[4, 5], top=1, left=2),
            dict(kind="resize", before=[4, 5], after=[8, 10]),
            dict(kind="hflip", before=[8, 10], after=[8, 10]),
        ]
        cells = [[0, 3], [1, 3], [2, 2], [9, 5], [9, 6]]
        targets = cell_occupancy(bitmap, polygon, steps, cells)
        for target in targets.values():
            np.testing.assert_array_equal(target, [0., 0., 1., 1., 0.])

    def test_overlapping_polygon_parts_are_unioned_before_cell_intersection(self):
        polygon = [
            [-.5, 0., 1.25, 0., 1.25, 1., -.5, 1.],
            [.75, 0., 2.5, 0., 2.5, 1., .75, 1.],
        ]
        cells = np.asarray([[2, 0], [0, 0], [1, 0]], dtype=np.uint16)
        np.testing.assert_array_equal(
            polygon_cell_occupancy(polygon, (1, 3), [], cells), [.5, 1., 1.],
        )

    def test_crop_can_remove_both_valid_representations(self):
        bitmap = np.asarray([[True, False, False]])
        polygon = [[0., 0., 1., 0., 1., 1., 0., 1.]]
        steps = [dict(kind="crop", before=[1, 3], after=[1, 2], top=0, left=1)]
        targets = cell_occupancy(bitmap, polygon, steps, [[0, 0], [1, 0]])
        for target in targets.values():
            np.testing.assert_array_equal(target, [0., 0.])

    def test_local_and_prefix_integrals_agree_for_every_cell_with_anisotropic_flip(self):
        row, col = np.indices((9, 13))
        bitmap = (5 * row + 7 * col + row * col) % 11 < 4
        steps = [
            dict(kind="hflip", before=[9, 13], after=[9, 13]),
            dict(kind="resize", before=[9, 13], after=[7, 19]),
            dict(kind="crop", before=[7, 19], after=[5, 12], top=1, left=4),
            dict(kind="resize", before=[5, 12], after=[6, 3]),
        ]
        rows, cols = np.indices((6, 3))
        cells = np.column_stack((cols.ravel(), rows.ravel()))[::-1]
        local = bitmap_cell_occupancy(bitmap, steps, cells)
        with patch("point_occupancy.MAX_LOCAL_INTERSECTION_ELEMENTS", 0):
            prefix = bitmap_cell_occupancy(bitmap, steps, cells)
        self.assertGreaterEqual(np.count_nonzero((local > 0.) & (local < 1.)), len(cells) // 2)
        np.testing.assert_array_equal(local, prefix)

    def test_rejects_noninteger_and_out_of_grid_banks_and_accepts_empty_integer_bank(self):
        bitmap = np.asarray([[True, False]])
        polygon = [[0., 0., 1., 0., 1., 1., 0., 1.]]
        for bad in ([[1., 0.]], [[True, False]], [[-1, 0]], [[2, 0]], [[0, 1]], [0, 0]):
            with self.subTest(cells=bad), self.assertRaises(ValueError):
                cell_occupancy(bitmap, polygon, [], bad)
        targets = cell_occupancy(bitmap, polygon, [], np.empty((0, 2), dtype=np.int32))
        for target in targets.values():
            self.assertEqual(target.shape, (0,))
            self.assertEqual(target.dtype, np.float64)

    def test_nonpolygon_is_ineligible_without_preventing_bitmap_only_measurement(self):
        bitmap = np.asarray([[True, False]])
        rle = {"size": [1, 2], "counts": "x"}
        with self.assertRaises(UnsupportedPolygon):
            cell_occupancy(bitmap, rle, [], [[0, 0]])
        np.testing.assert_array_equal(bitmap_cell_occupancy(bitmap, [], [[0, 0]]), [1.])


if __name__ == "__main__":
    unittest.main()
