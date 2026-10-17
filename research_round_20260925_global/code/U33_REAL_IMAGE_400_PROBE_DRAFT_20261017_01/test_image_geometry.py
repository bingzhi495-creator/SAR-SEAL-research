"""Pending one real-Torch 4/8 geometry contract, not an executed result.

Import the missing factory before Torch so Root can obtain a clean stdlib-only
MissingModule RED. This analytic fixture does not certify size400 or real images.
"""

import image_geometry as factory

import unittest

import torch


class ImageGeometryContract(unittest.TestCase):
    def test_non_square_integer_content_padding_and_original_residual_alignment(self):
        geometry = factory.make_geometry(torch, ori_shape=(3, 5), content_shape=(5, 8),
                                         canvas_size=8, branch_size=4)
        image = torch.tensor([51, 102, 204], dtype=torch.uint8).view(1, 3, 1, 1)
        image = image.expand(1, 3, 3, 5).clone()
        prepared = geometry.prepare_rgb(image)
        # The first five canvas rows contain the constant image, then three
        # black rows. Half-pixel 8->4 sampling gives [1,1,1/2,0]. Width is full.
        channel_values = torch.tensor([.2, .4, .8], dtype=torch.float32).view(1, 3, 1, 1)
        row_support = torch.tensor([1., 1., .5, 0.], dtype=torch.float32).view(1, 1, 4, 1)
        expected_prepared = (channel_values*row_support).expand(1, 3, 4, 4)
        self.assertEqual(tuple(prepared.shape), (1, 3, 4, 4))
        self.assertEqual(prepared.dtype, torch.float32)
        torch.testing.assert_close(prepared, expected_prepared, rtol=0, atol=1e-6)

        delta = torch.tensor([0., 1., 2., 3.], dtype=torch.float32).view(1, 1, 4, 1)
        delta = delta.expand(1, 1, 4, 4).clone()
        rendered = geometry.render_delta(delta)
        # 4->8 gives [0,1/4,3/4,5/4,7/4,9/4,11/4,3]. Crop the first five
        # rows, then 5->3 samples at [1/3,2,11/3], giving [1/12,3/4,19/12].
        # These expected values are analytic constants, not another interpolate.
        expected_rendered = torch.tensor([1/12, 3/4, 19/12], dtype=torch.float32)
        expected_rendered = expected_rendered.view(1, 1, 3, 1).expand(1, 1, 3, 5)
        self.assertEqual(tuple(rendered.shape), (1, 1, 3, 5))
        torch.testing.assert_close(rendered, expected_rendered, rtol=0, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
