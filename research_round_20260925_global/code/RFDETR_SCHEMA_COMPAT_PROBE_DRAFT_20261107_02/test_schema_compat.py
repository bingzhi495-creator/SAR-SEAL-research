"""Synthetic standard-library tests. No Torch, weights, images or model run."""
import math
import unittest
from collections import OrderedDict
from types import SimpleNamespace


class TensorStub:
    def __init__(self, shape=(0, 0), dtype="torch.bool", device="cpu"):
        self.shape = shape
        self.dtype = dtype
        self.device = SimpleNamespace(type=device)

    def numel(self):
        return math.prod(self.shape)

    def clone(self):
        return TensorStub(self.shape, self.dtype, self.device.type)


class ParameterStub(TensorStub):
    pass


TORCH_STUB = SimpleNamespace(
    bool="torch.bool", nn=SimpleNamespace(Parameter=ParameterStub),
    is_tensor=lambda value: isinstance(value, TensorStub),
)


class ModelStub:
    def __init__(self):
        self.use_grouppose_keypoints = False
        self.num_keypoints_per_class = []
        self.keypoint_embed = None
        self.buffer = TensorStub()
        self.expected = OrderedDict([
            ("class_embed.weight", TensorStub((91, 8), "torch.float32")),
            ("_kp_active_mask", self.buffer),
        ])

    def state_dict(self):
        return self.expected

    def get_buffer(self, name):
        if name != "_kp_active_mask":
            raise AttributeError(name)
        return self.buffer


class SchemaCompatTests(unittest.TestCase):
    def setUp(self):
        self.model = ModelStub()
        self.original = OrderedDict([
            ("class_embed.weight", TensorStub((91, 8), "torch.float32")),
        ])
        self.original._metadata = OrderedDict([("", {"version": 1})])

    def complete(self):
        from schema_compat import complete_legacy_schema_state
        return complete_legacy_schema_state(TORCH_STUB, self.model, self.original)

    def test_adds_only_known_empty_buffer_preserving_source_and_metadata(self):
        old_items = list(self.original.items())
        completed, metadata = self.complete()
        self.assertIsNot(completed, self.original)
        self.assertEqual(list(self.original.items()), old_items)
        for name, tensor in self.original.items():
            self.assertIs(completed[name], tensor)
        self.assertIs(completed._metadata, self.original._metadata)
        self.assertEqual(set(completed) - set(self.original), {"_kp_active_mask"})
        added = completed["_kp_active_mask"]
        self.assertIsNot(added, self.model.buffer)
        self.assertEqual((added.shape, added.dtype, added.numel()), ((0, 0), "torch.bool", 0))
        self.assertEqual(metadata["added_keys"], ["_kp_active_mask"])
        self.assertFalse(metadata["strict_original_state_load"])
        self.assertFalse(metadata["strict_completed_state_load"])
        self.assertFalse(metadata["qualified"])

    def test_rejects_every_other_missing_or_unexpected_key_set(self):
        cases = {
            "another_missing": OrderedDict(),
            "unexpected": OrderedDict(self.original, unexpected=TensorStub()),
            "already_present": OrderedDict(self.original, _kp_active_mask=TensorStub()),
            "nested_key": OrderedDict(self.original, **{"head._kp_active_mask": TensorStub()}),
        }
        for case, state in cases.items():
            with self.subTest(case=case):
                self.original = state
                with self.assertRaises(ValueError):
                    self.complete()

    def test_requires_explicit_disabled_keypoints_and_empty_schema(self):
        cases = [
            ("use_grouppose_keypoints", True),
            ("use_grouppose_keypoints", 0),
            ("num_keypoints_per_class", [0]),
            ("num_keypoints_per_class", None),
            ("keypoint_embed", object()),
        ]
        for name, value in cases:
            with self.subTest(field=name, value=value):
                self.model = ModelStub()
                setattr(self.model, name, value)
                with self.assertRaises(ValueError):
                    self.complete()

    def test_rejects_nonempty_wrong_shape_dtype_device_or_parameter_buffer(self):
        cases = [
            TensorStub((1, 1)),
            TensorStub((1, 0)),
            TensorStub(dtype="torch.float32"),
            TensorStub(device="cuda"),
            ParameterStub(),
            object(),
        ]
        for buffer in cases:
            with self.subTest(buffer=buffer):
                self.model = ModelStub()
                self.model.buffer = buffer
                self.model.expected["_kp_active_mask"] = buffer
                with self.assertRaises(ValueError):
                    self.complete()

    def test_rejects_other_tensor_schema_differences_without_mutating_source(self):
        cases = [
            TensorStub((90, 8), "torch.float32"),
            TensorStub((91, 8), "torch.float64"),
            TensorStub((91, 8), "torch.float32", "cuda"),
            object(),
        ]
        for tensor in cases:
            with self.subTest(tensor=tensor):
                self.original["class_embed.weight"] = tensor
                old_items = list(self.original.items())
                old_metadata = self.original._metadata
                with self.assertRaises(ValueError):
                    self.complete()
                self.assertEqual(list(self.original.items()), old_items)
                self.assertIs(self.original._metadata, old_metadata)


if __name__ == "__main__":
    unittest.main()
