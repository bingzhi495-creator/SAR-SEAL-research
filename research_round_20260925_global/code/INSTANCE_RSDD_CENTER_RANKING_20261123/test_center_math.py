import unittest

import numpy as np
import torch
from torch.nn import functional as F

from center_math import border_and_zero, eligible


class CenterTest(unittest.TestCase):
    def test_half_open_and_adjacent_fp32_edges(self):
        indices = np.array([0, 1, 5, 6, 2, 7, 10, 11])
        edges = np.array([0.5, 0.5, 2.5, 2.5], dtype=np.float32)
        np.testing.assert_array_equal(eligible(indices, 5, edges),
                                      [True] * 4 + [False] * 4)
        edges[2:] = np.nextafter(edges[2:], np.float32(np.inf))
        self.assertTrue(eligible(indices, 5, edges).all())
        edges[0] = np.nextafter(edges[0], np.float32(np.inf))
        self.assertFalse(eligible([0], 5, edges)[0])
        self.assertEqual(eligible([], 5, edges).dtype, np.bool_)

    def test_padding_weights_inside_continuous_box(self):
        width = 514
        box = torch.tensor([[0.5, 0.5, 512.5, 512.5]])
        logits = torch.zeros((1, 224, 224))
        sides = ([0, 100 * width, 100 * width + 1], [100 * width + 2, 100 * width + 511])
        self.assertTrue(all(eligible(side, width, box[0]).all() for side in sides))
        zeros, border, edge = border_and_zero(logits, box, sides, (width, width))
        np.testing.assert_array_equal(zeros[0], [0.125, 0.25, 0.46875])
        np.testing.assert_array_equal(zeros[1], [0.5, 0.46875])
        np.testing.assert_array_equal(edge[0], [True, True, True])
        np.testing.assert_array_equal(edge[1], [False, True])
        for row in border:
            np.testing.assert_array_equal(row, np.full(len(row), 0.5, np.float32))

    def test_sigmoid_precedes_bilinear(self):
        width = 514
        box = torch.tensor([[0.5, 0.5, 512.5, 512.5]])
        logits = torch.zeros((1, 224, 224))
        logits[:, :, 111] = -8.0
        logits[:, :, 112] = 2.0
        zeros, border, edge = border_and_zero(logits, box, ([], [256 * width + 256]),
                                              (width, width))
        expected = torch.tensor([-8.0, 2.0]).sigmoid().mean().item()
        self.assertEqual(zeros[1].item(), expected)
        self.assertEqual(border[1].item(), expected)
        self.assertFalse(edge[1].item())
        self.assertEqual(zeros[0].size, 0)

    def test_readback_vs_full_native_zero_padded_paste(self):
        height, width = 59, 97
        box = torch.tensor([[7.03125, 1.5, 88.125, 55.625]])
        logits = torch.randn((1, 224, 224), generator=torch.Generator().manual_seed(14))
        y = (torch.arange(height) + 0.5 - box[0, 1]) / (box[0, 3] - box[0, 1])
        x = (torch.arange(width) + 0.5 - box[0, 0]) / (box[0, 2] - box[0, 0])
        yy, xx = torch.meshgrid(2*y-1, 2*x-1, indexing="ij")
        whole = F.grid_sample(logits[None].sigmoid(), torch.stack((xx, yy), dim=-1)[None],
                              mode="bilinear", padding_mode="zeros", align_corners=False)
        native = np.arange(height * width)
        native = native[eligible(native, width, box[0])]
        sides = (native[::2], native[1::2][::-1])
        zeros, border, edge = border_and_zero(logits, box, sides, (height, width))
        expected = whole.numpy().ravel().view(np.uint32)
        for i in range(2):
            np.testing.assert_array_equal(zeros[i].view(np.uint32), expected[sides[i]])
            np.testing.assert_array_equal(zeros[i][~edge[i]].view(np.uint32),
                                          border[i][~edge[i]].view(np.uint32))
            self.assertGreater((~edge[i]).sum(), 10)

    def test_empty_sides_and_invalid_inputs(self):
        logits = torch.zeros((1, 224, 224))
        box = torch.tensor([[0.0, 0.0, 4.0, 3.0]])
        zeros, border, edge = border_and_zero(logits, box, ([], []), (3, 4))
        self.assertEqual([row.dtype for row in (*zeros, *border, *edge)],
                         [np.float32] * 4 + [np.bool_] * 2)
        self.assertTrue(all(row.shape == (0,) for row in (*zeros, *border, *edge)))
        for sides in (([-1], []), ([], [12]), ([1.1], [])):
            with self.assertRaises(ValueError):
                border_and_zero(logits, box, sides, (3, 4))


if __name__ == "__main__":
    unittest.main()
