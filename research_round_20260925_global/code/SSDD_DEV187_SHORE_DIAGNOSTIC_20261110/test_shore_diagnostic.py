from pathlib import Path
import sys
import unittest

import numpy as np
from pycocotools import mask

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shore_diagnostic import bbox_independent_fp, zones


class ShoreZonesTest(unittest.TestCase):
    def test_band_uses_original_grid_and_real_sea_land_adjacency(self):
        land = np.zeros((4, 12), dtype=bool)
        land[:, :6] = True
        result = zones(land)
        self.assertTrue(np.all(result[:, :2] == 0))
        self.assertTrue(np.all(result[:, 2:10] == 1))
        self.assertTrue(np.all(result[:, 10:] == 2))
        self.assertTrue(np.all(zones(np.zeros((2, 2), dtype=bool)) == 2))
        self.assertTrue(np.all(zones(np.ones((2, 2), dtype=bool)) == 0))

    def test_unmatched_mask_is_partitioned_once_excluding_matches_and_ignored_rows(self):
        grid = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.uint8)
        rle = mask.encode(np.asfortranarray(np.ones_like(grid)))
        predictions = [dict(image_id=7, segmentation=rle)] * 3
        records = [dict(image_id=7, dtIds=[1, 2, 3], dtMatches=[[4, 0, 0]], dtIgnore=[[False, True, False]])]
        result = bbox_independent_fp(records, predictions, {7: ("land_containing", grid)})
        self.assertEqual(result["false_positive_masks"], 1)
        self.assertEqual(result["pixel_mass_by_zone"], dict(land_interior=2, shore_4px=2, water_interior=2))
        self.assertEqual(result["false_positive_pixel_mass_over_all_pixels"], 1.)


if __name__ == "__main__":
    unittest.main()
