import unittest

import numpy as np

from reference_resize import center_reference, inverse_scales


class ReferenceResizeTests(unittest.TestCase):
    def test_resize_uses_output_pixel_centers_and_exact_ties(self):
        raw = np.array([[[False, True, True, False], [True, False, False, True]]])
        result = center_reference(raw, [dict(kind="resize", before=(2, 4), after=(1, 2))])
        np.testing.assert_array_equal(result, [[[False, True]]])

    def test_resize_crop_resize_flip_composes_before_rasterization(self):
        raw = np.zeros((2, 5, 7), dtype=bool)
        raw[0, 2, [5, 2]] = True
        raw[0, 4, [4, 1]] = True
        raw[1, 1, 5] = True
        raw[1, 4, 1] = True
        operations = [
            dict(kind="resize", before=(5, 7), after=(4, 5)),
            dict(kind="crop", before=(4, 5), after=(3, 3), top=1, left=1),
            dict(kind="resize", before=(3, 3), after=(2, 4)),
            dict(kind="hflip", before=(2, 4), after=(2, 4)),
        ]
        expected = np.array([
            [[True, False, True, False], [False, True, False, True]],
            [[False, False, False, False], [False, False, False, True]],
        ])
        np.testing.assert_array_equal(center_reference(raw, operations), expected)
        self.assertEqual(inverse_scales(operations), (15 / 8, -21 / 20))
