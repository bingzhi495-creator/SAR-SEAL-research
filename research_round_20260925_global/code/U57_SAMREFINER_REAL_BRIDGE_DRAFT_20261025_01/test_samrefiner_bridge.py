import os
import hashlib
import importlib.util
from pathlib import Path
import struct
import sys
import sysconfig
import unittest

_fixture_tmp = Path(__file__).parent / 'fixture_tmp'
_fixture_tmp.mkdir(exist_ok=True)
os.environ['TEMP'] = str(_fixture_tmp)
os.environ['TMP'] = str(_fixture_tmp)
sys.path.append(sysconfig.get_path('purelib'))

import numpy as np

from samrefiner_bridge import make_samrefiner_backend


class PoisonAuxiliary:
    def __array__(self, *args, **kwargs):
        raise AssertionError('SAM quality/logit auxiliaries must not be converted')


class BridgeTests(unittest.TestCase):
    def test_one_author_call_converts_only_masks_and_echoes_declared_ids(self):
        path = 'synthetic_path_not_opened.png'
        sam = object()
        ids = (17, 2, 91)
        masks = np.zeros((3, 2, 3), dtype=np.bool_)
        masks[:, 0, 0] = True
        calls = []

        def author(image_path, coarse_masks, loaded_sam, **kwargs):
            calls.append((image_path, coarse_masks, loaded_sam, kwargs))
            self.assertEqual(coarse_masks.dtype, np.uint8)
            self.assertEqual(coarse_masks.shape, (3, 2, 3))
            self.assertTrue(coarse_masks.flags.c_contiguous)
            np.testing.assert_array_equal(coarse_masks, masks.astype(np.uint8))
            output = np.zeros_like(coarse_masks)
            output[:, 1, 2] = 1
            return output, PoisonAuxiliary(), PoisonAuxiliary()

        backend = make_samrefiner_backend(path, sam, author)
        result = backend(query_ids=ids, masks=masks, image_size=(2, 3), iters=5)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], path)
        self.assertIs(calls[0][2], sam)
        self.assertEqual(calls[0][3], dict(
            resize_transform=None, use_point=True, use_box=True, use_mask=True,
            add_neg=True, iters=5, margin=0.0, gamma=4.0, strength=30,
            use_samhq=False, ddp=False, is_train=False))
        self.assertEqual(result.query_ids, ids)
        self.assertEqual(result.masks, tuple(((0, 0, 0), (0, 0, 1)) for _ in ids))
        self.assertNotIn('torch', sys.modules)

    def test_invalid_input_array_never_reaches_author(self):
        cases = [np.ones((3, 2, 2), dtype=np.uint8),
                 np.ones((2, 2, 3), dtype=np.uint8),
                 np.ones((2, 3), dtype=np.uint8),
                 np.full((3, 2, 3), 255, dtype=np.uint8),
                 np.full((3, 2, 3), 0.5, dtype=np.float32)]
        for masks in cases:
            with self.subTest(shape=masks.shape, dtype=masks.dtype):
                calls = []

                def author(path, batch, sam, **kwargs):
                    calls.append(True)
                    return batch, None, None

                backend = make_samrefiner_backend('not_opened.png', object(), author)
                with self.assertRaises(ValueError):
                    backend(query_ids=(17, 2, 91), masks=masks, image_size=(2, 3), iters=5)
                self.assertEqual(calls, [])

    def test_invalid_declarations_are_rejected_before_author(self):
        cases = [((17, 2, 91), (2, 3), 1, (3, 2, 3)),
                 ((17, 17, 91), (2, 3), 5, (3, 2, 3)),
                 ((17, 2, 91), (2, 3, 1), 5, (3, 2, 3, 1)),
                 ((17, 2, 91), (0, 3), 5, (3, 0, 3)),
                 ((), (2, 3), 5, (0, 2, 3)),
                 (tuple(range(101)), (2, 3), 5, (101, 2, 3))]
        for ids, size, iters, shape in cases:
            with self.subTest(count=len(ids), size=size, iters=iters):
                calls = []

                def author(path, batch, sam, **kwargs):
                    calls.append(True)
                    return batch, None, None

                backend = make_samrefiner_backend('not_opened.png', object(), author)
                with self.assertRaises(ValueError):
                    backend(query_ids=ids, masks=np.ones(shape, dtype=np.uint8),
                            image_size=size, iters=iters)
                self.assertEqual(calls, [])

    def test_author_return_contract_is_checked_without_thresholding(self):
        good = np.zeros((3, 2, 3), dtype=np.uint8)
        cases = [good, [good, None, None], (good, None),
                 (good.tolist(), None, None),
                 (good[:2], None, None),
                 (np.zeros((3, 2, 2), dtype=np.uint8), None, None),
                 (good.astype(np.float32), None, None),
                 (good.astype(np.bool_), None, None),
                 (np.full((3, 2, 3), 255, dtype=np.uint8), None, None)]
        for output in cases:
            with self.subTest(container=type(output).__name__):
                calls = []

                def author(path, batch, sam, **kwargs):
                    calls.append(True)
                    return output

                backend = make_samrefiner_backend('not_opened.png', object(), author)
                with self.assertRaises(ValueError):
                    backend(query_ids=(17, 2, 91), masks=np.ones_like(good),
                            image_size=(2, 3), iters=5)
                self.assertEqual(calls, [True])

    def test_input_copy_is_contiguous_and_cannot_modify_caller_array(self):
        source = np.ones((3, 2, 3), dtype=np.uint8)[:, :, ::-1]
        source.setflags(write=False)
        before = source.copy()

        def author(path, batch, sam, **kwargs):
            self.assertTrue(batch.flags.c_contiguous)
            self.assertFalse(np.shares_memory(batch, source))
            batch[:] = 0
            return batch, None, None

        backend = make_samrefiner_backend('not_opened.png', object(), author)
        result = backend(query_ids=(17, 2, 91), masks=source, image_size=(2, 3), iters=5)
        np.testing.assert_array_equal(source, before)
        self.assertEqual(result.masks, tuple(((0, 0, 0), (0, 0, 0)) for _ in range(3)))

    def test_author_failure_propagates_without_retry_or_passthrough(self):
        calls = []

        def author(*args, **kwargs):
            calls.append(True)
            raise RuntimeError('fixture author failure')

        backend = make_samrefiner_backend('not_opened.png', object(), author)
        with self.assertRaisesRegex(RuntimeError, 'fixture author failure'):
            backend(query_ids=(17,), masks=np.ones((1, 2, 3), dtype=np.uint8),
                    image_size=(2, 3), iters=5)
        self.assertEqual(calls, [True])

    def test_integrates_with_readonly_frozen_adapter_and_preserves_s0(self):
        source = Path(__file__).parent.parent / (
            'U57_SAMREFINER_FIXED_CANDIDATE_ADAPTER_DRAFT_20261025_01'
        ) / 'fixed_candidate_adapter.py'
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                         'eb0609708bd189ebfda381c9ef5700c2bdba53fbeba0837febeff24fdbb9cac2')
        spec = importlib.util.spec_from_file_location('frozen_candidate_bridge_fixture', str(source))
        adapter = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = adapter
        spec.loader.exec_module(adapter)
        original = []
        empty_indices = {17, 99}
        for i, query_id in enumerate(range(99, -1, -1)):
            bits = [int(bool((query_id + 1) & (1 << bit))) for bit in range(9)]
            mask = tuple(tuple(bits[row * 3:row * 3 + 3]) for row in range(3))
            if i in empty_indices:
                mask = ((0, 0, 0),) * 3
            original.append(adapter.Candidate(query_id, query_id % 2,
                            -0.0 if query_id == 0 else query_id / 101.0, mask))
        calls = []

        def author(path, batch, sam, **kwargs):
            calls.append(batch.shape)
            return np.flip(batch, axis=-1).copy(), PoisonAuxiliary(), PoisonAuxiliary()

        backend = make_samrefiner_backend('not_opened.png', object(), author)
        result = adapter.refine_candidates(original, (3, 3), backend, empty_policy='passthrough')
        self.assertEqual(calls, [(98, 3, 3)])
        self.assertTrue(result.is_adaptation)
        self.assertEqual(len(result.candidates), 100)
        for i, (before, after) in enumerate(zip(original, result.candidates)):
            self.assertEqual((after.query_id, after.category_id), (before.query_id, before.category_id))
            self.assertEqual(struct.pack('>d', after.score), struct.pack('>d', before.score))
            expected = before.mask if i in empty_indices else tuple(tuple(reversed(row)) for row in before.mask)
            self.assertEqual(after.mask, expected)
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main()
