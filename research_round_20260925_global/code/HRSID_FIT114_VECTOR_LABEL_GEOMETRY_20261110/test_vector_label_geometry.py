import unittest

import numpy as np

from vector_label_geometry import UnsupportedPolygon, binary_iou, final_polygons, polygon_raster, vector_footprint_metrics


class VectorLabelGeometryTest(unittest.TestCase):
    def test_resize_crop_resize_flip_uses_pixel_edge_coordinates(self):
        source = [[0., 1., 3., 1., 3., 2., 0., 2.]]
        steps = [
            dict(kind="resize", before=[4, 6], after=[8, 12]),
            dict(kind="crop", before=[8, 12], after=[4, 5], top=1, left=2),
            dict(kind="resize", before=[4, 5], after=[8, 10]),
            dict(kind="hflip", before=[8, 10], after=[8, 10]),
        ]
        parts, hw = final_polygons(source, (4, 6), steps)
        self.assertEqual(hw, (8, 10))
        self.assertEqual(len(parts), 1)
        self.assertEqual((float(parts[0][:, 0].min()), float(parts[0][:, 0].max())), (2., 10.))
        expected = np.zeros(hw, dtype=np.bool_)
        expected[2:6, 2:10] = True
        np.testing.assert_array_equal(polygon_raster(parts, hw), expected)



    def test_crop_preserves_disconnected_polygon_pieces(self):
        source = [[0., 0., 6., 0., 6., 6., 4., 6., 4., 2., 2., 2., 2., 6., 0., 6.]]
        steps = [dict(kind="crop", before=[6, 6], after=[3, 6], top=3, left=0)]
        parts, hw = final_polygons(source, (6, 6), steps)
        self.assertEqual(len(parts), 2)
        expected = np.zeros(hw, dtype=np.bool_)
        expected[:, :2] = expected[:, 4:] = True
        np.testing.assert_array_equal(polygon_raster(parts, hw), expected)

    def test_continuous_polygon_area_is_not_a_raster_self_comparison(self):
        source = [[.25, 0., 1.25, 0., 1.25, 1., .25, 1.]]
        parts, hw = final_polygons(source, (1, 2), [])
        left = np.asarray([[True, False]])
        right = np.asarray([[False, True]])
        values = vector_footprint_metrics(parts, hw, {"left": left, "right": right})
        self.assertAlmostEqual(values["visible_polygon_area"], 1.)
        self.assertAlmostEqual(values["by_arm"]["left"]["iou"], .75 / 1.25)
        self.assertAlmostEqual(values["by_arm"]["right"]["iou"], .25 / 1.75)
        self.assertEqual(binary_iou(left, right), 0.)

    def test_nonpolygon_and_self_intersecting_labels_are_explicitly_ineligible(self):
        for source in ({"size": [4, 4], "counts": "1"}, [[0., 0., 2., 2., 2., 0., 0., 2.]]):
            with self.assertRaises(UnsupportedPolygon):
                final_polygons(source, (4, 4), [])


if __name__ == "__main__":
    unittest.main()
