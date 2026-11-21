import unittest

import torch

from pointrend_subdivision import subdivide_binary_mask


class BinarySubdivisionTests(unittest.TestCase):
    def test_non_dyadic_regular_centers_use_author_affine_grid_rounding(self):
        coarse = torch.ones(1, 1, 7, 7)
        coords = []

        def predict(xy, samples):
            coords.append(xy)
            return samples

        subdivide_binary_mask(
            coarse, predict, initial_size=(7, 7), native_size=(7, 7), points_per_step=1
        )
        transform = torch.tensor([[[.5, 0., .5], [0., .5, .5]]])
        expected = torch.nn.functional.affine_grid(transform, (1, 1, 7, 7), align_corners=False)
        self.assertTrue(torch.equal(coords[0], expected.reshape(1, 49, 2)))

    def test_rectangular_native_grid_changes_only_selected_pixel_for_each_mask(self):
        first = torch.tensor([[[[0.125, 6., 14.], [3., 8., 21.]]]], dtype=torch.float64)
        coarse = torch.cat((first, first.flip((-2, -1))))
        calls = []

        def predict(xy, samples):
            calls.append(xy)
            if len(calls) == 1:
                return coarse.flatten(2)
            return torch.full_like(samples, -99)

        out = subdivide_binary_mask(
            coarse, predict, initial_size=(2, 3), native_size=(3, 5), points_per_step=1
        )
        centers = torch.tensor([[[0.1, 1 / 6]], [[0.9, 5 / 6]]], dtype=torch.float64)
        torch.testing.assert_close(calls[-1], centers, rtol=0, atol=2e-16)
        expected = torch.nn.functional.interpolate(
            coarse, size=(3, 5), mode="bilinear", align_corners=False
        )
        expected[0, 0, 0, 0] = expected[1, 0, 2, 4] = -99
        self.assertTrue(torch.equal(out.logits, expected))

    def test_empty_query_batch_has_native_shape_and_no_point_evaluations(self):
        def fail_if_called(xy, samples):
            self.fail("a batch with no queries invoked the head")

        out = subdivide_binary_mask(
            torch.empty(0, 1, 2, 3), fail_if_called, initial_size=(1, 2),
            native_size=(5, 7), points_per_step=4
        )
        self.assertEqual(tuple(out.logits.shape), (0, 1, 5, 7))
        self.assertEqual((out.stages, out.point_evaluations), ((), 0))

    def test_dense_stage_promotion_and_realized_counts_match_callback_work(self):
        coarse = torch.ones((2, 1, 2, 2))
        actual = []

        def predict(xy, samples):
            actual.append(xy.shape[0] * xy.shape[1])
            return samples

        out = subdivide_binary_mask(
            coarse, predict, initial_size=(2, 2), native_size=(16, 16), points_per_step=16
        )
        self.assertEqual([stage.size for stage in out.stages], [(4, 4), (8, 8), (16, 16)])
        self.assertEqual([stage.points_per_mask for stage in out.stages], [16, 16, 16])
        self.assertEqual(actual, [32, 32, 32])
        self.assertEqual([stage.point_evaluations for stage in out.stages], actual)
        self.assertEqual(out.point_evaluations, 96)

    def test_feedback_drives_selection_but_callback_always_samples_original_coarse(self):
        coarse = torch.tensor([[[[-4., 2.]]]])
        calls = []

        def predict(xy, samples):
            calls.append((xy.clone(), samples.clone()))
            if len(calls) == 1:
                return samples
            return torch.full_like(samples, -3. if len(calls) == 2 else 9.)

        out = subdivide_binary_mask(
            coarse, predict, initial_size=(1, 2), native_size=(1, 8), points_per_step=1
        )
        expected_centers = [[[.25, .5], [.75, .5]], [[.625, .5]], [[.8125, .5]]]
        expected_samples = [[-4., 2.], [.5], [1.75]]
        for (xy, samples), centers, values in zip(calls, expected_centers, expected_samples):
            self.assertTrue(torch.equal(xy[0], torch.tensor(centers)))
            self.assertTrue(torch.equal(samples[0, 0], torch.tensor(values)))
        self.assertEqual(len(calls), 3)
        expected = [-4., -3.625, -2.875, -2.625, -2.875, -1.75, 9., 2.]
        self.assertTrue(torch.equal(out.logits[0, 0, 0], torch.tensor(expected)))
        self.assertTrue(torch.equal(coarse.flatten(), torch.tensor([-4., 2.])))


if __name__ == "__main__":
    unittest.main()
