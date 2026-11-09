import unittest

import numpy as np
from pycocotools import mask
import torch
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as vision

from vertical_phase_resize import resize_masks_vertical


class VerticalPhaseResizeTest(unittest.TestCase):
    def test_native_matches_stock_mask_default_at_admitted_canvas(self):
        source = torch.zeros((2, 800, 800), dtype=torch.bool)
        source[0, 200:214, 95:108] = True
        source[1, 388:397, 533:584] = True
        source[1, 390:395, 529:588] = True
        stock = vision.resize(tv_tensors.Mask(source), [576, 576]).as_subclass(torch.Tensor)

        result = resize_masks_vertical(source, (576, 576), "native")

        self.assertTrue(torch.equal(result, stock))
        self.assertIs(type(result), torch.Tensor)
        self.assertIs(result.dtype, torch.bool)


    def test_each_mode_changes_only_y_on_rectangular_coordinate_basis(self):
        h, w = 8, 7
        out_h, out_w = 5, 4
        source = torch.eye(h * w, dtype=torch.bool).reshape(h * w, h, w)
        y = torch.arange(out_h)
        x = torch.arange(out_w) * w // out_w
        expected_y = {
            "native": y * h // out_h,
            "center_y": (2 * y + 1) * h // (2 * out_h),
            "reflected_y": h - 1 - (out_h - 1 - y) * h // out_h,
        }
        for mode, rows in expected_y.items():
            with self.subTest(mode=mode):
                expected = source[:, rows][:, :, x]
                self.assertTrue(torch.equal(resize_masks_vertical(source, [out_h, out_w], mode), expected))

    def test_stock_native_parity_on_coco_rasterized_rows_and_noncontiguous_masks(self):
        polygons = [
            [1.7, 2.1, 8.6, 3.6, 10.2, 5.5, 6.0, 7.3, 1.2, 4.7],
            [8.4, 9.2, 13.6, 8.6, 20.2, 10.4, 19.8, 11.7, 12.1, 10.3],
        ]
        pixels = np.moveaxis(mask.decode(mask.frPyObjects(polygons, 13, 23)), -1, 0)
        source = torch.from_numpy(pixels).bool()
        for oriented in (source, source.transpose(-2, -1)):
            for size in ((3, 12), (26, 11), (33, 19), (13, 23)):
                with self.subTest(shape=oriented.shape, size=size):
                    expected = vision.resize(tv_tensors.Mask(oriented), list(size))
                    observed = resize_masks_vertical(tv_tensors.Mask(oriented), size, "native")
                    self.assertIs(type(observed), tv_tensors.Mask)
                    self.assertTrue(torch.equal(observed, expected))

    def test_all_modes_use_stock_x_when_height_does_not_change(self):
        generator = torch.Generator().manual_seed(5219)
        source = torch.rand((2, 61, 800), generator=generator) > .81
        expected = vision.resize(tv_tensors.Mask(source), [61, 576])
        for mode in ("native", "center_y", "reflected_y"):
            with self.subTest(mode=mode):
                self.assertTrue(torch.equal(resize_masks_vertical(source, (61, 576), mode), expected))

    def test_reflected_mode_matches_actual_legacy_resize_on_flipped_mask(self):
        source = torch.rand((3, 800, 67), generator=torch.Generator().manual_seed(5213)) > .87
        stock_reflected = vision.resize(tv_tensors.Mask(source.flip((-2,))), [576, 39]).flip((-2,))
        self.assertTrue(torch.equal(resize_masks_vertical(source, (576, 39), "reflected_y"), stock_reflected))

    def test_noop_alias_empty_shape_and_wrapped_dtype_match_stock(self):
        source = tv_tensors.Mask(torch.ones((2, 4, 5), dtype=torch.bool))
        empty = tv_tensors.Mask(torch.empty((0, 4, 5), dtype=torch.bool))
        stock_identity = vision.resize(source, [4, 5])
        stock_empty = vision.resize(empty, [2, 7])
        for mode in ("native", "center_y", "reflected_y"):
            with self.subTest(mode=mode):
                unchanged = resize_masks_vertical(source, (4, 5), mode)
                out = resize_masks_vertical(empty, (2, 7), mode)
                self.assertIs(type(unchanged), type(stock_identity))
                self.assertEqual(unchanged.data_ptr(), stock_identity.data_ptr())
                self.assertIsNot(unchanged, source)
                self.assertIs(type(out), type(stock_empty))
                self.assertEqual(out.shape, stock_empty.shape)
                self.assertIs(out.dtype, stock_empty.dtype)
        plain = source.as_subclass(torch.Tensor)
        result = resize_masks_vertical(plain, (4, 5), "native")
        self.assertIs(type(result), torch.Tensor)
        self.assertEqual(result.data_ptr(), plain.data_ptr())

    def test_rejects_invalid_mode_mask_or_output_canvas(self):
        source = torch.zeros((2, 4, 5), dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "center_y or reflected_y"):
            resize_masks_vertical(source, (3, 6), "center")
        for invalid in (source.float(), source[0], source[:, :0]):
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype):
                with self.assertRaisesRegex(ValueError, "CPU Boolean instance masks"):
                    resize_masks_vertical(invalid, (3, 6), "native")
        for size in ((0, 6), (True, 6), (3,), (3.0, 6)):
            with self.subTest(size=size):
                with self.assertRaisesRegex(ValueError, "integer output height and width"):
                    resize_masks_vertical(source, size, "native")


if __name__ == "__main__":
    unittest.main()
