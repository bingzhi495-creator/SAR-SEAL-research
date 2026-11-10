import unittest
import numpy as np
from polygon_support import continuous_polygon, polygon_scores, support_cell


class PolygonAndSupportTests(unittest.TestCase):
    def test_integer_trace_moves_a_fractional_polygon_in_edge_coordinates(self):
        segmentation = [[1.25,.25,2.25,.25,2.25,1.25,1.25,1.25]]
        steps = [
            ("pad",(3,5),(5,7),(1,2)),
            ("crop",(5,7),(3,4),(2,2,6,5)),
            ("flip",(3,4),(3,4),()),
            ("resize",(3,4),(6,2),(.5,2.)),
        ]
        ref, pre_hw, out_hw = continuous_polygon(segmentation, (3,5), steps)
        self.assertEqual(pre_hw, (3,4))
        self.assertEqual(out_hw, (6,2))
        self.assertEqual(ref.bounds, (1.375,.5,1.875,2.5))
        active = np.zeros((6,2),dtype=bool)
        active[1,1] = True
        out = polygon_scores(ref,pre_hw,{"active":active,"empty":np.zeros_like(active)})
        self.assertAlmostEqual(out["active"]["iou"],1/3)
        self.assertAlmostEqual(out["active"]["signed_y_px"],0.)
        self.assertAlmostEqual(out["active"]["signed_x_px"],-.25)
        self.assertEqual(out["empty"]["iou"],0.)
        self.assertIsNone(out["empty"]["signed_y_px"])

    def test_single_pixel_rescue_keeps_row_major_ties_and_fractional_mass(self):
        source = np.ones((3,3),dtype=bool)
        source[1,:] = False
        kept, data = support_cell(source,(2,2))
        self.assertEqual(np.argwhere(kept).tolist(),[[0,0]])
        self.assertAlmostEqual(data["highest_cell_coverage"],2/3)
        self.assertAlmostEqual(data["area"],8/3)

