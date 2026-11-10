"""Synthetic native-state checks for the DGX canonical Q100 initializer."""

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import ssdd_phase_init_dgx_canonical as adapter
from ssdd_phase_init import STATE, STATE_SHA, load_api


digest_state = load_api(STATE, STATE_SHA).tensor_dict_sha256


class OriginalInitializer:
    def __init__(self, module, expected):
        self.module = module
        self.expected = expected
        self.guard_calls = 0

    def describe_q100(self, module, digest):
        values = module.model.state_dict()
        return dict(online_tensor_sha256=digest(values), state_tensors=len(values))

    def initialize_q100(self, fit_layout, output, coco_path):
        witness = self.describe_q100(self.module, digest_state)
        self.guard_calls += 1
        if witness["online_tensor_sha256"] != self.expected:
            raise ValueError("Original complete Q100 digest required")
        return fit_layout, output, self.module, witness


class CanonicalInitializationTest(unittest.TestCase):
    def start_patch(self, patcher):
        self.addCleanup(patcher.stop)
        patcher.start()

    def setUp(self):
        root = TemporaryDirectory()
        self.addCleanup(root.cleanup)
        self.file = Path(root.name).resolve() / "position.pt"
        self.key = "position"
        self.native = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)
        self.reference = self.native + 0.25
        model = torch.nn.Module()
        model.register_parameter(self.key, torch.nn.Parameter(self.native.clone()))
        model.register_parameter("other", torch.nn.Parameter(torch.tensor([1.0, 2.0])))
        self.module = SimpleNamespace(model=model)
        native_state = model.state_dict()
        reference_state = dict(native_state, **{self.key: self.reference})
        self.reference_digest = digest_state(reference_state)
        self.original = OriginalInitializer(self.module, self.reference_digest)
        self.start_patch(patch.multiple(
            adapter, POSITION_KEY=self.key, POSITION_SHAPE=(1, 3, 2),
            NATIVE_DGX_SHA256=digest_state(native_state),
            NATIVE_DGX_POSITION_SHA256=digest_state({self.key: self.native}),
            CANONICAL_POSITION_SHA256=digest_state({self.key: self.reference}),
            CANONICAL_Q100_SHA256=self.reference_digest,
        ))
        self.start_patch(patch.object(adapter, "_load_original", return_value=self.original))

    def save_position(self, value):
        torch.save(value, self.file)
        size = self.file.stat().st_size
        self.start_patch(patch.object(adapter, "POSITION_FILE_BYTES", size))
        return sha256(self.file.read_bytes()).hexdigest()

    def initialize(self, expected_sha):
        initialize = adapter.make_canonical_initializer(self.file, expected_sha)
        return initialize("fit", "output", "coco")

    def test_exact_position_reconstructs_original_guard_without_replacing_parameter(self):
        digest = self.save_position({self.key: self.reference})
        parameter = self.module.model.get_parameter(self.key)
        other = self.module.model.get_parameter("other").detach().clone()
        model, train, module, report = self.initialize(digest)

        self.assertEqual((model, train), ("fit", "output"))
        self.assertIs(module, self.module)
        self.assertEqual(report["online_tensor_sha256"], self.reference_digest)
        self.assertEqual(self.original.guard_calls, 1)
        self.assertIs(parameter, self.module.model.get_parameter(self.key))
        self.assertTrue(torch.equal(parameter, self.reference))
        self.assertTrue(torch.equal(self.module.model.get_parameter("other"), other))


    def assert_native_position_unchanged(self):
        self.assertTrue(torch.equal(self.module.model.get_parameter(self.key), self.native))
        self.assertEqual(self.original.guard_calls, 0)

    def test_changed_other_tensor_rejected_before_artifact_read(self):
        with torch.no_grad():
            self.module.model.get_parameter("other").add_(1.0)
        with self.assertRaisesRegex(ValueError, "Exact original DGX Q100"):
            self.initialize("1" * 64)
        self.assert_native_position_unchanged()

    def test_native_position_digest_independently_required(self):
        with patch.object(adapter, "NATIVE_DGX_POSITION_SHA256", "1" * 64):
            with self.assertRaisesRegex(ValueError, "Exact native DGX Q100 position"):
                self.initialize("1" * 64)
        self.assert_native_position_unchanged()

    def test_position_file_digest_mismatch_rejected(self):
        self.save_position({self.key: self.reference})
        with self.assertRaisesRegex(ValueError, "position file SHA-256"):
            self.initialize("1" * 64)
        self.assert_native_position_unchanged()

    def test_position_file_with_additional_tensor_rejected(self):
        digest = self.save_position({self.key: self.reference, "other": torch.tensor([2.0])})
        with self.assertRaisesRegex(ValueError, "Exactly one registered"):
            self.initialize(digest)
        self.assert_native_position_unchanged()

    def test_nonfinite_position_rejected_even_with_bound_file_digest(self):
        nonfinite = self.reference.clone()
        nonfinite[0, 0, 0] = float("nan")
        digest = self.save_position({self.key: nonfinite})
        with self.assertRaisesRegex(ValueError, "Exact finite canonical"):
            self.initialize(digest)
        self.assert_native_position_unchanged()

    def test_complete_reference_state_checked_before_parameter_update(self):
        digest = self.save_position({self.key: self.reference})
        with patch.object(adapter, "CANONICAL_Q100_SHA256", "1" * 64):
            with self.assertRaisesRegex(ValueError, "Exact canonical Q100 reconstruction"):
                self.initialize(digest)
        self.assert_native_position_unchanged()

if __name__ == "__main__":
    unittest.main()
