import unittest
import numpy as np
import shapely

from full_continuous_ap import boxes_from_counts, overlap_area, plausible


class ContinuousGeometry(unittest.TestCase):
    def test_column_wrap_retains_exact_disjoint_foreground(self):
        # Foreground starts at (x=0,y=2), crosses two column boundaries,
        # and ends with (x=2,y=0) on a height-3 canvas.
        bounds = boxes_from_counts([2,5,5], 3, 4)
        geometry = shapely.box(.5,.25,2.5,2.75)
        self.assertAlmostEqual(float(shapely.area(shapely.union_all(shapely.box(*bounds.T)))), 5.)
        self.assertAlmostEqual(overlap_area(bounds, geometry), .375 + 2.5 + .375)

    def test_pruning_keeps_equal_threshold_and_rejects_impossible(self):
        dt=np.array([[0.,0.,2.,2.],[2.,0.,4.,2.],[0.,0.,2.,2.]])
        areas=np.array([4.,4.,12.])
        self.assertEqual(plausible(dt,areas,np.array([0.,0.,2.,2.]),8.).tolist(),[True,False,False])


if __name__ == "__main__":
    unittest.main()
