"""Synthetic CPU checks for cached strata and exact complement rank selection."""
import unittest

import torch

from point_allocation import _select_complement, prepare_strata, sample_target_points_prepared


def make_generator(seed=5201):
    return torch.Generator(device="cpu").manual_seed(seed)


class PointAllocationTest(unittest.TestCase):
    def test_complement_rank_select_for_every_small_boundary_set(self):
        for total in range(1, 9):
            for bits in range(1 << total):
                members = torch.tensor([p for p in range(total) if bits & (1 << p)], dtype=torch.int64)
                expected = [p for p in range(total) if not bits & (1 << p)]
                gaps = members - torch.arange(len(members))
                ranks = torch.arange(len(expected))
                self.assertEqual(_select_complement(gaps, ranks).tolist(), expected, (total, bits))


    def test_prepared_sampling_matches_direct_complement_and_p66_weights(self):
        masks = torch.zeros((1, 32, 32), dtype=torch.bool)
        masks[0, 10, 19] = True
        coords, weights, indices = sample_target_points_prepared(
            masks, torch.tensor([0]), 40, arm="D", generator=make_generator(), strata=prepare_strata(masks),
        )
        members = [y * 32 + x for y in range(8, 13) for x in range(17, 22)]
        member_set = set(members)
        complement = [index for index in range(1024) if index not in member_set]
        direct = make_generator()
        expected = torch.cat((torch.tensor(members)[torch.randint(25, (2,), generator=direct)],
                              torch.tensor(complement)[torch.randint(999, (38,), generator=direct)]))
        self.assertTrue(torch.equal(indices[0], expected))
        torch.testing.assert_close(coords[0, :, 0], (expected % 32 + 0.5) / 32, rtol=0, atol=0)
        torch.testing.assert_close(coords[0, :, 1], (expected // 32 + 0.5) / 32, rtol=0, atol=0)
        values = torch.tensor([5.0] * 2 + [2.0] * 38)
        torch.testing.assert_close((weights * values).mean(), torch.tensor((25 * 5 + 999 * 2) / 1024))

    def test_grouped_independent_draws_return_to_original_interleaved_match_order(self):
        masks = torch.zeros((2, 32, 32), dtype=torch.bool)
        masks[0, 10, 19] = True
        masks[1, 8:15, 10:18] = True
        cache = prepare_strata(masks)
        grouped = sample_target_points_prepared(masks, torch.tensor([1, 1, 0, 0]), 40, arm="D",
                                                generator=make_generator(), strata=cache)
        interleaved = sample_target_points_prepared(masks, torch.tensor([1, 0, 1, 0]), 40, arm="D",
                                                    generator=make_generator(), strata=cache)
        restore = torch.tensor([0, 2, 1, 3])
        for original, output in zip(grouped, interleaved):
            torch.testing.assert_close(output, original[restore], rtol=0, atol=0)
        self.assertFalse(torch.equal(interleaved[2][0], interleaved[2][2]))
        self.assertFalse(torch.equal(interleaved[2][1], interleaved[2][3]))

    def test_degenerate_d_and_uniform_b_use_same_seeded_draws(self):
        for filled in (False, True):
            masks = torch.full((1, 1, 2), filled, dtype=torch.bool)
            targets = torch.tensor([0])
            result_b = sample_target_points_prepared(masks, targets, 8, arm="B",
                                                     generator=make_generator(), strata=None)
            result_d = sample_target_points_prepared(masks, targets, 8, arm="D",
                                                     generator=make_generator(), strata=prepare_strata(masks))
            for b, d in zip(result_b, result_d):
                torch.testing.assert_close(b, d, rtol=0, atol=0)
            self.assertLess(result_d[2].unique().numel(), 8)

    def test_cache_guards_population_and_rejects_ordinary_in_place_mutation(self):
        masks = torch.zeros((1, 5, 5), dtype=torch.bool)
        cache = prepare_strata(masks)
        targets = torch.tensor([0])
        with self.assertRaisesRegex(ValueError, "exact immutable"):
            sample_target_points_prepared(masks.clone(), targets, 4, arm="D",
                                          generator=make_generator(), strata=cache)
        masks[0, 1, 1] = True
        with self.assertRaisesRegex(ValueError, "exact immutable"):
            sample_target_points_prepared(masks, targets, 4, arm="D",
                                          generator=make_generator(), strata=cache)
        with self.assertRaisesRegex(ValueError, "outside"):
            sample_target_points_prepared(masks, torch.tensor([1]), 4, arm="B",
                                          generator=make_generator(), strata=None)

    def test_empty_and_inference_target_tensors_are_supported(self):
        with torch.inference_mode():
            masks = torch.zeros((0, 5, 5), dtype=torch.bool)
            rng = make_generator()
            original_state = rng.get_state()
            result = sample_target_points_prepared(masks, torch.empty(0, dtype=torch.int64), 4,
                                                    arm="D", generator=rng, strata=prepare_strata(masks))
            self.assertEqual([part.shape for part in result], [(0, 4, 2), (0, 4), (0, 4)])
            self.assertTrue(torch.equal(rng.get_state(), original_state))


if __name__ == "__main__":
    unittest.main()
