import struct
import unittest
from dataclasses import replace

from fixed_candidate_adapter import Candidate, BackendResult, refine_candidates


def candidates():
    return tuple(
        Candidate(q, q % 2, -0.0 if q == 0 else (q + 0.12345678901234568) / 101,
                  ((1, 0, 0), (0, q % 2, 0)))
        for q in range(99, -1, -1)
    )


class AdapterTests(unittest.TestCase):
    def test_one_indexed_batch_preserves_all_identities_and_scores(self):
        original = candidates()
        calls = []

        def backend(*, query_ids, masks, image_size, iters):
            calls.append((query_ids, masks, image_size, iters))
            output = tuple(((0, 1, 0), (q % 2, 0, 0)) for q in query_ids)
            return BackendResult(query_ids, output)

        result = refine_candidates(original, (2, 3), backend)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], tuple(c.query_id for c in original))
        self.assertEqual(calls[0][2:], ((2, 3), 5))
        self.assertEqual(len(result.candidates), 100)
        for before, after in zip(original, result.candidates):
            self.assertEqual((after.query_id, after.category_id),
                             (before.query_id, before.category_id))
            self.assertEqual(struct.pack('>d', after.score), struct.pack('>d', before.score))
            self.assertEqual(after.mask, ((0, 1, 0), (before.query_id % 2, 0, 0)))
        self.assertEqual(result.empty_policy, 'reject')
        self.assertFalse(result.is_adaptation)
        self.assertEqual(result.backend_calls, 1)
        self.assertEqual(result.configured_iterations, 5)

    def test_reordered_backend_identities_are_rejected(self):
        def backend(*, query_ids, masks, **kwargs):
            return BackendResult(tuple(reversed(query_ids)), masks)

        with self.assertRaises(ValueError):
            refine_candidates(candidates(), (2, 3), backend)

    def test_wrong_input_geometry_is_rejected_before_backend(self):
        original = list(candidates())
        original[0] = replace(original[0], mask=((1, 0), (0, 0)))
        calls = []

        def backend(*, query_ids, masks, **kwargs):
            calls.append(True)
            return BackendResult(query_ids, masks)

        with self.assertRaises(ValueError):
            refine_candidates(original, (2, 3), backend)
        self.assertEqual(calls, [])

    def test_non_binary_input_is_not_silently_converted(self):
        for value in (255, 0.5):
            with self.subTest(value=value):
                original = list(candidates())
                original[0] = replace(original[0], mask=((value, 0, 0), (0, 0, 0)))
                calls = []

                def backend(*, query_ids, masks, **kwargs):
                    calls.append(True)
                    return BackendResult(query_ids, masks)

                with self.assertRaises(ValueError):
                    refine_candidates(original, (2, 3), backend)
                self.assertEqual(calls, [])

    def test_candidate_population_must_be_exactly_one_hundred(self):
        original = candidates()
        for batch in (original[:-1], original + (original[0],)):
            calls = []

            def backend(*, query_ids, masks, **kwargs):
                calls.append(True)
                return BackendResult(query_ids, masks)

            with self.assertRaises(ValueError):
                refine_candidates(batch, (2, 3), backend)
            self.assertEqual(calls, [])

    def test_default_empty_mask_policy_rejects_without_backend(self):
        original = list(candidates())
        original[17] = replace(original[17], mask=((0, 0, 0), (0, 0, 0)))
        calls = []

        def backend(*, query_ids, masks, **kwargs):
            calls.append(True)
            return BackendResult(query_ids, masks)

        with self.assertRaises(ValueError):
            refine_candidates(original, (2, 3), backend)
        self.assertEqual(calls, [])

    def test_backend_cannot_silently_drop_a_mask(self):
        def backend(*, query_ids, masks, **kwargs):
            return BackendResult(query_ids, masks[:-1])

        with self.assertRaises(ValueError):
            refine_candidates(candidates(), (2, 3), backend)

    def test_backend_output_must_obey_original_geometry_and_binary_contract(self):
        for bad in (((1, 0), (0, 0)), ((255, 0, 0), (0, 0, 0))):
            with self.subTest(bad=bad):
                def backend(*, query_ids, masks, **kwargs):
                    return BackendResult(query_ids, (bad,) + masks[1:])

                with self.assertRaises(ValueError):
                    refine_candidates(candidates(), (2, 3), backend)

    def test_explicit_empty_passthrough_refills_original_indices(self):
        original = list(candidates())
        empty_indices = {0, 17, 99}
        empty = ((0, 0, 0), (0, 0, 0))
        for index in empty_indices:
            original[index] = replace(original[index], mask=empty)
        expected_ids = tuple(c.query_id for i, c in enumerate(original) if i not in empty_indices)
        calls = []

        def backend(*, query_ids, masks, image_size, iters):
            calls.append((query_ids, masks, image_size, iters))
            return BackendResult(query_ids, tuple(((0, 1, 0), (q % 2, 0, 0)) for q in query_ids))

        result = refine_candidates(original, (2, 3), backend, empty_policy='passthrough')
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], expected_ids)
        self.assertEqual(calls[0][2:], ((2, 3), 5))
        self.assertTrue(result.is_adaptation)
        self.assertEqual(result.empty_policy, 'passthrough')
        self.assertEqual(len(result.candidates), 100)
        for i, (before, after) in enumerate(zip(original, result.candidates)):
            self.assertEqual((after.query_id, after.category_id), (before.query_id, before.category_id))
            self.assertEqual(struct.pack('>d', after.score), struct.pack('>d', before.score))
            expected = empty if i in empty_indices else ((0, 1, 0), (before.query_id % 2, 0, 0))
            self.assertEqual(after.mask, expected)

    def test_duplicate_query_identity_is_rejected_before_backend(self):
        original = list(candidates())
        original[0] = replace(original[0], query_id=original[1].query_id)
        calls = []

        def backend(*, query_ids, masks, **kwargs):
            calls.append(True)
            return BackendResult(query_ids, masks)

        with self.assertRaises(ValueError):
            refine_candidates(original, (2, 3), backend)
        self.assertEqual(calls, [])

    def test_all_empty_passthrough_keeps_population_without_backend(self):
        original = tuple(replace(c, mask=((0, 0, 0), (0, 0, 0))) for c in candidates())

        def backend(**kwargs):
            self.fail('All-empty passthrough must not call backend')

        result = refine_candidates(original, (2, 3), backend, empty_policy='passthrough')
        self.assertEqual(result.candidates, original)
        self.assertEqual(result.backend_calls, 0)
        self.assertTrue(result.is_adaptation)

    def test_backend_failure_propagates_without_retry_or_fallback(self):
        calls = []

        def backend(**kwargs):
            calls.append(True)
            raise RuntimeError('fixture backend failed')

        with self.assertRaisesRegex(RuntimeError, 'fixture backend failed'):
            refine_candidates(candidates(), (2, 3), backend)
        self.assertEqual(calls, [True])

    def test_unknown_empty_policy_is_not_accepted(self):
        def backend(**kwargs):
            self.fail('Invalid policy must fail before backend')

        with self.assertRaises(ValueError):
            refine_candidates(candidates(), (2, 3), backend, empty_policy='automatic')


if __name__ == '__main__':
    unittest.main()
