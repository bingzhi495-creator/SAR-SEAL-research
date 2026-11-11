import unittest

import numpy as np
import torch

from tiled_band import band_window, registered_values


class TiledBandTest(unittest.TestCase):
    def test_edge_tile_contains_exact_canvas_limited_square_band(self):
        pred = np.zeros((800, 800), dtype=bool)
        pred[790:800, 794:800] = True
        crop, yx = band_window(pred)
        self.assertEqual(crop, (400, 400, 800, 800))
        self.assertIn((799, 799), {tuple(p) for p in yx})
        self.assertIn((788, 792), {tuple(p) for p in yx})
        self.assertNotIn((794, 796), {tuple(p) for p in yx})
        self.assertTrue(np.all(yx >= 0) and np.all(yx < 800))
        pred[400, 200:640] = True
        self.assertIsNone(band_window(pred))

    def test_half_pixel_render_and_tile_origin(self):
        field = torch.tensor([[0., 4.], [0., 4.]])
        pts = np.array([[60, 40], [60, 41], [60, 42], [60, 43]])
        self.assertEqual(registered_values(field, (40, 60, 44, 64), pts).tolist(), [0., 1., 3., 4.])


if __name__ == "__main__":
    unittest.main()
