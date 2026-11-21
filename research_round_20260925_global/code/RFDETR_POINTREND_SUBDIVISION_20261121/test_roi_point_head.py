import unittest

import torch
from torch.nn import functional as F

from pointrend_subdivision import subdivide_binary_mask
from roi_point_head import (
    BinaryPointHead, binary_point_training_loss, coarse_roi_logits,
    fine_features_at_roi_points, paste_probabilities_at_pixels, sample_training_points,
)


class TestROIAdapter(unittest.TestCase):
    def test_fractional_boxes_sample_one_shared_ramp_with_half_pixel_centers(self):
        # A 4x6 feature covers an 8x12 image. Feature [j,i] is at (2i+1,2j+1).
        y, x = torch.meshgrid(torch.arange(4), torch.arange(6), indexing="ij")
        image_features = (10 * y + x).float()[None, None]
        boxes = torch.tensor([[1., 1., 9., 5.], [2., 2., 8., 6.]])
        roi_xy = torch.tensor([[[0., 0.], [.5, .5]], [[.5, .25], [1., 1.]]])
        actual = fine_features_at_roi_points(image_features, boxes, roi_xy, (8, 12))
        torch.testing.assert_close(actual[:, 0], torch.tensor([[0., 12.], [12., 28.5]]))

    def test_coarse_uses_aligned_query_planes_and_keeps_signed_logits(self):
        ramp = torch.arange(49).view(7, 7).float() - 24
        planes = torch.stack([ramp, -ramp])[:, None]
        boxes = torch.tensor([[0., 0., 14., 14.], [0., 0., 14., 14.]])
        actual = coarse_roi_logits(planes, boxes, (14, 14))
        torch.testing.assert_close(actual, planes, atol=4e-6, rtol=1e-6)

    def test_paste_interpolates_probabilities_with_zero_border_at_pixel_centers(self):
        logits = torch.tensor([[[[4., -2.], [4., -2.]]]])
        box = torch.tensor([[.5, 0., 2.5, 2.]])
        pixels = torch.tensor([[[0, 0], [1, 0], [3, 0]]])
        actual = paste_probabilities_at_pixels(logits, box, pixels, (2, 4))
        p, q = torch.sigmoid(torch.tensor([4., -2.]))
        torch.testing.assert_close(actual[0], torch.stack([p / 2, (p + q) / 2, p * 0]))
        with self.assertRaises(ValueError):
            paste_probabilities_at_pixels(logits, torch.tensor([[0., 0., 0., 2.]]), pixels, (2, 4))

    def test_author_sampling_returns_147_uncertain_then_49_fresh_uniform_points(self):
        coarse = torch.arange(49).view(1, 1, 7, 7).float() - 19.7
        coarse = torch.cat([coarse, coarse + 7.2])
        actual = sample_training_points(coarse, torch.Generator().manual_seed(11))
        rng = torch.Generator().manual_seed(11)
        pool = torch.rand(2, 588, 2, generator=rng)
        raw = F.grid_sample(coarse, (2 * pool - 1)[:, :, None], align_corners=False)[:, 0, :, 0]
        expected = pool.gather(1, raw.abs().argsort()[:, :147, None].expand(-1, -1, 2))
        torch.testing.assert_close(actual[:, :147], expected, atol=0, rtol=0)
        torch.testing.assert_close(actual[:, 147:], torch.rand(2, 49, 2, generator=rng), atol=0, rtol=0)

    def test_single_binary_step_trains_only_head_against_soft_sampled_bitmask(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(513)
            head = BinaryPointHead()
        self.assertEqual(sum(p.numel() for p in head.parameters()), 198402)
        feature = torch.linspace(0, 1, 256 * 4 * 4).view(1, 256, 4, 4).requires_grad_()
        boxes = torch.tensor([[0., 0., 8., 8.]])
        coarse = torch.linspace(-2, 2, 49).view(1, 1, 7, 7).requires_grad_()
        truth = torch.arange(8).repeat(8, 1).ge(4).unsqueeze(0)
        loss = binary_point_training_loss(head, feature, coarse, boxes, truth, (8, 8),
                                          torch.Generator().manual_seed(23))
        coords = sample_training_points(coarse, torch.Generator().manual_seed(23))
        native_grid = (2 * coords - 1).unsqueeze(-2)
        sample = lambda plane: F.grid_sample(plane, native_grid, align_corners=False).squeeze(-1)
        target = sample(truth[:, None].float())
        self.assertTrue(((target > 0) & (target < 1)).any())
        expected = F.binary_cross_entropy_with_logits(head(sample(feature), sample(coarse)), target)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNone(feature.grad)
        self.assertIsNone(coarse.grad)
        self.assertTrue(all(p.grad is not None and p.grad.norm() > 0 for p in head.parameters()))

    def test_real_head_callback_renders_two_distinct_boxes_and_pastes_native_pixels(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(13)
            head = BinaryPointHead().eval()
        with torch.no_grad():
            # Hold all other weights at zero and route one channel from the RF
            # feature through every hidden layer. This yields known constant
            # interiors on the left and right halves of the synthetic image.
            for parameter in head.parameters():
                parameter.zero_()
            for layer in (head.fc1, head.fc2, head.fc3, head.predictor):
                layer.weight[0, 0, 0] = 1
        basis = torch.zeros(1, 256, 8, 16)
        basis[0, 0, :, :8] = 1
        basis[0, 0, :, 8:] = 3
        boxes = torch.tensor([[1., 1., 7., 7.], [9., 1., 15., 7.]])
        size = (8, 16)
        coarse = coarse_roi_logits(torch.zeros(2, 1, 8, 16), boxes, size)
        result = subdivide_binary_mask(
            coarse, lambda xy, c: head(fine_features_at_roi_points(basis, boxes, xy, size), c),
            initial_size=(7, 7), native_size=(28, 28), points_per_step=49,
        )
        self.assertEqual(result.point_evaluations, 2 * 3 * 49)
        points = torch.tensor([[[3, 3], [13, 3]], [[3, 3], [13, 3]]])
        pasted = paste_probabilities_at_pixels(result.logits, boxes, points, size)
        torch.testing.assert_close(pasted, torch.diag(torch.tensor([1., 3.]).sigmoid()))


if __name__ == "__main__":
    unittest.main()
