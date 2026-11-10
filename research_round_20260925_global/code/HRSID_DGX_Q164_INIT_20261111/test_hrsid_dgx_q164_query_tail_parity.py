"""Synthetic source and bitwise query-tail invariance checks."""
from hashlib import sha256
from unittest import TestCase

import hrsid_dgx_q164_query_tail_parity as subject


def digest(state):
    return sha256(b"".join(name.encode() + tensor.detach().contiguous().numpy().tobytes()
                           for name, tensor in sorted(state.items()))).hexdigest()


class TailTests(TestCase):
    def test_registered_tail_copy_preserves_prefix_and_every_other_tensor(self):
        import torch
        model = torch.nn.Module()
        model.query_feat = torch.nn.Embedding(13 * 164, 2)
        model.register_parameter("untouched", torch.nn.Parameter(torch.tensor([2.])))
        original = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        prefix = original[subject.QUERY_KEY].reshape(13, 164, 2)[:, :100].contiguous()
        tail = torch.arange(13 * 64 * 2, dtype=torch.float32).reshape(13, 64, 2)
        shadow = {name: tensor.clone() for name, tensor in original.items()}
        shadow[subject.QUERY_KEY].reshape(13, 164, 2)[:, 100:].copy_(tail)
        expected = digest(shadow)
        witness = subject.copy_query_tail(model, tail, torch, digest, digest(original), expected,
                                          digest({subject.QUERY_KEY: prefix}), tensor_count=2)
        self.assertTrue(witness["tail_copy_performed"])
        self.assertEqual(witness["canonical_q164_sha256"], expected)
        self.assertEqual(witness["unchanged_other_tensors"], 1)
        self.assertEqual(witness["query_prefix_before_sha256"], witness["query_prefix_after_sha256"])
        self.assertEqual(witness["changed_tensor_names"], [subject.QUERY_KEY])
        self.assertEqual(digest(model.state_dict()), expected)



    def test_wrong_shadow_never_mutates_registered_weight(self):
        import torch

        model = torch.nn.Module()
        model.query_feat = torch.nn.Embedding(13 * 164, 2)
        model.register_parameter("untouched", torch.nn.Parameter(torch.tensor([2.])))
        original = model.state_dict()
        before = digest(original)
        query = original[subject.QUERY_KEY].reshape(13, 164, 2)
        prefix_sha = digest({subject.QUERY_KEY: query[:, :100].contiguous()})
        tail = torch.ones((13, 64, 2), dtype=torch.float32)
        with self.assertRaises(subject.TailMismatch) as mismatch:
            subject.copy_query_tail(model, tail, torch, digest, before, "a" * 64, prefix_sha, tensor_count=2)
        self.assertFalse(mismatch.exception.audit["tail_copy_performed"])
        self.assertEqual(digest(model.state_dict()), before)

    def test_raw_little_endian_binary_checks_typed_tail_and_exact_shape(self):
        from pathlib import Path
        from types import SimpleNamespace

        import numpy as np
        import torch

        array = np.arange(np.prod(subject.TAIL_SHAPE), dtype="<f4").reshape(subject.TAIL_SHAPE)
        raw = array.tobytes(order="C")
        pin = sha256(raw).hexdigest()
        typed = digest({subject.QUERY_KEY: torch.from_numpy(array)})
        loader = SimpleNamespace(pinned=lambda path, sha, limit: raw if sha == pin and limit == len(raw) else b"")
        result = subject.load_tail(loader, (Path.cwd() / "synthetic") / subject.TAIL_NAME, pin, typed, digest, torch)
        self.assertEqual(tuple(result.shape), subject.TAIL_SHAPE)
        self.assertTrue(torch.equal(result, torch.from_numpy(array)))
        with self.assertRaisesRegex(ValueError, "canonical Windows query-tail tensor"):
            subject.load_tail(loader, (Path.cwd() / "synthetic") / subject.TAIL_NAME, pin, "a" * 64, digest, torch)

