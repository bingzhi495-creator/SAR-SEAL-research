import unittest
import numpy as np
from score_distance_fusion import signed_distance, choose_by_score


class FixedDistanceSelection(unittest.TestCase):
    def test_positive_foreground_and_false_border(self):
        a = np.ones((3, 5), dtype=bool)
        values = signed_distance(a)
        self.assertEqual(values[0, 0], 1)
        self.assertEqual(values[1, 2], 2)
        b = np.zeros((3, 5), dtype=bool)
        b[0, 0] = True
        values = signed_distance(b)
        self.assertEqual(values[0, 0], 1)
        self.assertEqual(values[1, 1], -np.sqrt(2))
        self.assertEqual(values[2, 4], -3)

    def test_native_row_major_ties_and_equal_count(self):
        a = np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)
        support = np.array([[1, 0, 1], [1, 1, 1]], dtype=bool)
        q = np.array([[0, 9, 0], [2, -1, 2]], dtype=float)
        result = choose_by_score(a, support, q, 1, 2)
        self.assertEqual(np.flatnonzero(result ^ a).tolist(), [0, 3, 4])

    def test_rectangular_crop_and_donor_order_preserve_native_band_ranking(self):
        from scipy.ndimage import binary_dilation, binary_erosion
        from score_distance_fusion import signed_distance_sum, tight_bounds

        rng = np.random.default_rng(5202)
        structure = np.ones((3, 3), dtype=bool)
        for _ in range(80):
            h, w = (int(x) for x in rng.integers(8, 25, size=2))
            y = int(rng.integers(0, h))
            x = int(rng.integers(0, w))
            masks = []
            for index in range(3):
                mask = np.zeros((h, w), dtype=bool)
                dy, dx = (int(v) for v in rng.integers(-2, 3, size=2))
                cy, cx = min(h-1, max(0, y+dy)), min(w-1, max(0, x+dx))
                mask[max(0, cy-2):min(h, cy+3), max(0, cx-2):min(w, cx+3)] = True
                if index:
                    mask[cy, cx] = False
                masks.append(mask)
            anchor, left, right = masks
            band = (binary_dilation(anchor, structure, iterations=2, border_value=0)
                    & ~binary_erosion(anchor, structure, iterations=2, border_value=0))
            crop = tight_bounds(masks)
            fields = tuple(signed_distance(mask) for mask in masks)
            full = signed_distance_sum(fields)
            self.assertTrue(np.array_equal(full, signed_distance_sum((fields[2], fields[0], fields[1]))))
            cropped = signed_distance_sum(tuple(signed_distance(mask[crop]) for mask in masks))
            self.assertTrue(np.array_equal(cropped[band[crop]], full[crop][band[crop]]))
            both = band & (left != anchor) & (right != anchor)
            any_donor = band & ((left != anchor) | (right != anchor))
            adds = min(2, int((both & ~anchor).sum()))
            drops = min(2, int((both & anchor).sum()))
            self.assertTrue(np.array_equal(choose_by_score(anchor, band, full, adds, drops),
                                           choose_by_score(anchor, any_donor, full, adds, drops)))
