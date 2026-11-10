"""Synthetic invariance checks for the narrow P291 position substitution."""
from unittest import TestCase

import hrsid_dgx_q164_position_parity as subject


class LocationTests(TestCase):
    def test_expansion_tail_is_only_localized_after_all_other_invariants(self):
        ref = "refpoint_embed.weight"
        query = "query_feat.weight"
        stock = {"position": "canonical", "other": "same", ref: "prefix", query: "prefix"}
        final = dict(stock, **{ref: "plus zeros", query: "plus random"})
        locations = subject.localize_expansion(stock, final, (ref, query), True, True, False)
        self.assertEqual(locations, ["query_feat.weight[13 groups, rows 100:164]"])
        self.assertEqual(subject.localize_expansion(stock, final, (ref, query), True, True, True), [])
        changed = dict(final, other="wrong")
        self.assertEqual(subject.localize_expansion(stock, changed, (ref, query), True, True, False), ["other"])
        self.assertEqual(subject.localize_expansion(stock, final, (ref, query), False, True, False),
                         ["original per-group query prefixes"])



    def test_registered_position_copy_preserves_other_tensors_and_wrong_raw_never_loads_reference(self):
        from hashlib import sha256
        from types import SimpleNamespace
        from unittest.mock import Mock

        import torch

        def tensor_digest(state):
            digest = sha256()
            for key, tensor in sorted(state.items()):
                digest.update(key.encode())
                digest.update(tensor.detach().contiguous().numpy().tobytes())
            return digest.hexdigest()

        model = torch.nn.Module()
        model.register_parameter("position", torch.nn.Parameter(torch.tensor([1., 2.])))
        model.register_parameter("keep", torch.nn.Parameter(torch.tensor([3., 4.])))
        reference = torch.tensor([8., 9.])
        before = model.state_dict()
        p290 = SimpleNamespace(POSITION="position", P242_RAW_Q100=tensor_digest(before),
                               P242_RAW_POSITION=tensor_digest({"position": before["position"]}),
                               Q100=tensor_digest(dict(before, position=reference)),
                               WINDOWS_POSITION=tensor_digest({"position": reference}))
        load_ref = Mock(return_value=reference)
        audit, after = subject.canonicalize(model, load_ref, torch, tensor_digest, p290, tensor_count=2)
        self.assertEqual(audit["changed_q100_tensor_names"], ["position"])
        self.assertEqual(audit["unchanged_non_position_tensors"], 1)
        self.assertEqual(audit["non_position_before_sha256"], audit["non_position_after_sha256"])
        self.assertTrue(torch.equal(model.position, reference))
        self.assertTrue(torch.equal(model.keep, torch.tensor([3., 4.])))
        self.assertEqual(after["position"], p290.WINDOWS_POSITION)
        load_ref.reset_mock()
        with self.assertRaises(subject.NumericMismatch) as mismatch:
            subject.canonicalize(model, load_ref, torch, tensor_digest, p290, tensor_count=2)
        self.assertFalse(mismatch.exception.details["reference_tensor_substitution"])
        load_ref.assert_not_called()

    def test_wrong_shadow_does_not_change_registered_tensor(self):
        from hashlib import sha256
        from types import SimpleNamespace

        import torch

        def digest(state):
            return sha256(b"".join(key.encode() + value.detach().numpy().tobytes()
                                   for key, value in sorted(state.items()))).hexdigest()

        model = torch.nn.Module()
        model.register_parameter("position", torch.nn.Parameter(torch.tensor([1.])))
        model.register_parameter("keep", torch.nn.Parameter(torch.tensor([2.])))
        before = model.state_dict()
        reference = torch.tensor([3.])
        p290 = SimpleNamespace(POSITION="position", P242_RAW_Q100=digest(before),
                               P242_RAW_POSITION=digest({"position": before["position"]}),
                               Q100="a" * 64, WINDOWS_POSITION=digest({"position": reference}))
        with self.assertRaises(subject.NumericMismatch) as mismatch:
            subject.canonicalize(model, lambda: reference, torch, digest, p290, tensor_count=2)
        self.assertFalse(mismatch.exception.details["reference_tensor_substitution"])
        self.assertTrue(torch.equal(model.position, torch.tensor([1.])))
        self.assertEqual(digest(model.state_dict()), p290.P242_RAW_Q100)

