import unittest

import torch
from rfdetr.models.heads.segmentation import SegmentationHead, point_sample

from cached_final_query import capture_final_query, matched_dense_logits, projection_parameters, sample_integer_cells


class CachedFinalQueryTest(unittest.TestCase):
    def test_reconstructs_the_actual_final_native_masks_for_the_same_query_indices(self):
        with torch.random.fork_rng():
            torch.manual_seed(253_001)
            head = SegmentationHead(in_dim=7, num_blocks=3, bottleneck_ratio=2, downsample_ratio=2).eval()
            spatial = torch.randn(1, 7, 4, 7)
            queries = [torch.randn(1, 11, 7) for _ in range(3)]
            with torch.no_grad():
                head.bias.fill_(.31)
                with capture_final_query(head) as capture:
                    native, = head(spatial, queries, (12, 18), final_layer_only=True)
                cache = capture.seal(native)
                selected = matched_dense_logits(head.query_features_proj, cache, [8, 2, 8])
            torch.testing.assert_close(selected, native[0, [8, 2, 8]], rtol=0, atol=0)


    def test_point_sample_uses_the_same_integer_cell_centers_as_the_native_sampler(self):
        logits = torch.arange(24, dtype=torch.float32).view(2, 3, 4) / 7
        cells = torch.tensor([[[0, 0], [8, 6], [2, 3]], [[8, 6], [1, 2], [0, 0]]])
        unit = (cells.to(torch.float64) + .5) / torch.tensor([9, 7])
        expected = torch.cat([
            point_sample(logits[i:i+1, None], unit[i:i+1].float(), align_corners=False)[:, 0]
            for i in range(2)
        ])
        actual = sample_integer_cells(logits, cells, (7, 9))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        common = sample_integer_cells(logits, cells[0], (7, 9))
        expanded = sample_integer_cells(logits, cells[0].expand(2, -1, -1), (7, 9))
        torch.testing.assert_close(common, expanded, rtol=0, atol=0)


    def test_cached_update_matches_native_gradient_and_updates_only_original_query_linear(self):
        with torch.random.fork_rng():
            torch.manual_seed(253_002)
            head = SegmentationHead(in_dim=8, num_blocks=2, downsample_ratio=2).eval()
            spatial = torch.randn(1, 8, 3, 5)
            queries = [torch.randn(1, 12, 8) for _ in range(2)]
            params = projection_parameters(head.query_features_proj)
            self.assertIs(params[0], head.query_features_proj.weight)
            self.assertIs(params[1], head.query_features_proj.bias)
            head.requires_grad_(False)
            for param in params:
                param.requires_grad_(True)
            before = {name: tensor.detach().clone() for name, tensor in head.state_dict().items()}
            with torch.inference_mode():
                with capture_final_query(head) as capture:
                    native, = head(spatial, queries, (10, 14), final_layer_only=True)
                cache = capture.seal(native)
            self.assertTrue(all(not tensor.requires_grad and not torch.is_inference(tensor)
                                for tensor in (cache.spatial, cache.queries, cache.bias)))

            ids = [7, 2]
            cells = torch.tensor([[[13, 9], [0, 0], [3, 3]], [[3, 6], [0, 9], [9, 4]]])
            targets = torch.tensor([[.2, 0., 1.], [.3, 1., .5]])
            unit = ((cells.double() + .5) / torch.tensor([14, 10])).float()
            native_grad, = head(spatial, queries, (10, 14), final_layer_only=True)
            native_points = point_sample(native_grad[0, ids, None], unit, align_corners=False)[:, 0]
            native_loss = torch.nn.functional.binary_cross_entropy_with_logits(native_points, targets)
            expected_grad = torch.autograd.grad(native_loss, params)
            cached = matched_dense_logits(head.query_features_proj, cache, ids)
            actual_points = sample_integer_cells(cached, cells, (10, 14))
            torch.testing.assert_close(actual_points, native_points, rtol=0, atol=0)
            torch.nn.functional.binary_cross_entropy_with_logits(actual_points, targets).backward()
            for param, expected in zip(params, expected_grad, strict=True):
                self.assertGreater(float(expected.abs().sum()), 0.)
                torch.testing.assert_close(param.grad, expected, rtol=0, atol=0)
            with torch.no_grad():
                for param in params:
                    param.add_(param.grad, alpha=-.01)
                updated_native, = head(spatial, queries, (10, 14), final_layer_only=True)
                updated_cached = matched_dense_logits(head.query_features_proj, cache, ids)
            torch.testing.assert_close(updated_cached, updated_native[0, ids], rtol=0, atol=0)
            self.assertEqual(
                {name for name, tensor in head.state_dict().items() if not torch.equal(tensor, before[name])},
                {"query_features_proj.weight", "query_features_proj.bias"},
            )
            self.assertTrue(all(param.grad is None for param in head.parameters() if not param.requires_grad))

    def test_rejects_auxiliary_capture_and_does_not_leave_hooks_on_the_native_head(self):
        with torch.random.fork_rng(), torch.no_grad():
            torch.manual_seed(253_003)
            head = SegmentationHead(in_dim=5, num_blocks=2).eval()
            spatial = torch.randn(1, 5, 3, 3)
            queries = [torch.randn(1, 7, 5) for _ in range(2)]
            with capture_final_query(head) as repeated:
                native = head(spatial, queries, (12, 12), final_layer_only=False)
            with self.assertRaisesRegex(ValueError, "one closed final-only"):
                repeated.seal(native[-1])
            with capture_final_query(head) as final:
                native, = head(spatial, queries, (12, 12), final_layer_only=True)
            cache = final.seal(native)
            projection = head.query_features_proj
            for invalid in ([7], [-1], [1.0], [[0]], [False]):
                with self.subTest(ids=invalid), self.assertRaises(ValueError):
                    matched_dense_logits(projection, cache, invalid)
            with self.assertRaisesRegex(ValueError, "outside the final image"):
                sample_integer_cells(native[0, :1], [[12, 0]], (12, 12))


if __name__ == "__main__":
    unittest.main()
