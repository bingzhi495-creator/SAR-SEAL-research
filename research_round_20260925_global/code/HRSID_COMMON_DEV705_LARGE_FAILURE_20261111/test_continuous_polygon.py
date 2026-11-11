import unittest

import numpy as np

from continuous_polygon import measure_pair, paired_cluster
from vector_label_geometry import polygon_raster, polygon_rings
from vector_population_math import run_footprint_metrics


class ContinuousPolygonTest(unittest.TestCase):
    def test_fractional_polygon_and_raster_agree_on_same_fixed_pair(self):
        annotation = [[.2, .2, 2.2, .2, 2.2, 1.2, .2, 1.2]]
        rings = polygon_rings(annotation)
        raster = polygon_raster(rings, (4, 4))
        native = np.zeros((4, 4), dtype=bool)
        native[0, 0] = True
        center = raster.copy()
        result = measure_pair(raster, dict(native=native, center=center), rings, run_footprint_metrics)
        self.assertAlmostEqual(result['native']['raster_iou'], .5)
        self.assertEqual(result['center']['raster_iou'], 1.)
        self.assertAlmostEqual(result['native']['polygon_iou'], .64 / 2.36)
        self.assertAlmostEqual(result['center']['polygon_iou'], 1.44 / 2.56)
        self.assertAlmostEqual(result['self_polygon_iou'], 1.44 / 2.56)
        self.assertAlmostEqual(result['center']['polygon_fp_rate'], .28)
        self.assertAlmostEqual(result['center']['polygon_fn_rate'], .28)
        self.assertEqual(result['center']['raster_margin90'], 2)
        self.assertAlmostEqual(result['margin_attribution_error'], 0.)


    def test_object_and_group_means_and_paired_representation_contrast(self):
        values = np.array([[.10, .09], [.10, .08], [-.10, -.04]])
        result = paired_cluster(values, ['g1', 'g1', 'g2'])
        self.assertAlmostEqual(result['raster']['object_mean'], 1/30)
        self.assertAlmostEqual(result['raster']['group_equal_mean'], 0.)
        self.assertAlmostEqual(result['polygon']['group_equal_mean'], .0225)
        self.assertAlmostEqual(result['polygon_minus_raster']['object_mean'], .01)
        self.assertEqual(result['polygon_minus_raster']['group_signs'], [1, 0, 1])


if __name__ == '__main__':
    unittest.main()
