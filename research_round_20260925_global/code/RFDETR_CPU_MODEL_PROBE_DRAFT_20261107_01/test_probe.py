"""Synthetic control tests only: no real Torch, models, checkpoints or images."""
import importlib
import contextlib
import json
import pickle
import sys
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch


class TensorStub:
    """Metadata/token only: no tensor arithmetic or numerical result is simulated."""
    def __init__(self, shape=(1,), token=1, dtype="float32", finite=True):
        self.shape, self.token, self.dtype, self.finite = shape, token, dtype, finite
        self.device = SimpleNamespace(type="cpu")

    def numel(self):
        n = 1
        for size in self.shape:
            n *= size
        return n

    def new_tensor(self, values):
        return TensorStub((len(values),), token=tuple(values))

    def view(self, *shape):
        return TensorStub(shape, token=self.token)

    def __sub__(self, other):
        return TensorStub(self.shape)

    def __truediv__(self, other):
        return TensorStub(self.shape)


class BoolStub:
    def __init__(self, value):
        self.value = value

    def all(self):
        return self

    def item(self):
        return self.value


@dataclass
class DefaultsStub:
    force_no_pretrain: bool = False


class ModelStub:
    def __init__(self, runtime):
        self.runtime, self.eval_called = runtime, False

    def state_dict(self):
        return self.runtime.expected

    def load_state_dict(self, state, strict):
        return self.runtime.strict_load(state, strict)

    def to(self, **kwargs):
        assert kwargs == {"device": "cpu", "dtype": "float32"}
        return self

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, image):
        assert self.eval_called and self.runtime.threads == 2
        assert image.shape == (1, 3, 312, 312)
        self.runtime.forward_calls += 1
        return self.runtime.outputs


class RuntimeStub:
    def __init__(self, missing_key=False):
        self.strict_calls, self.forward_calls, self.load_calls = [], 0, []
        self.threads, self.rng = 7, 101
        self.expected = {"weight": TensorStub(), "bias": TensorStub(token=2)}
        self.saved = dict(self.expected)
        self.outputs = {"pred_logits": TensorStub((1, 100, 91)),
                        "pred_boxes": TensorStub((1, 100, 4)),
                        "pred_masks": TensorStub((1, 100, 78, 78))}
        if missing_key:
            self.saved.pop("bias")
        self.model = ModelStub(self)
        self.torch = SimpleNamespace(
            load=self.load, get_num_threads=lambda: self.threads,
            set_num_threads=lambda n: setattr(self, "threads", n),
            get_rng_state=lambda: self.rng, set_rng_state=lambda s: setattr(self, "rng", s),
            random=SimpleNamespace(default_generator=SimpleNamespace(manual_seed=lambda s: setattr(self, "rng", s))),
            is_tensor=lambda x: isinstance(x, TensorStub), isfinite=lambda x: BoolStub(x.finite),
            equal=lambda a, b: a.shape == b.shape and a.dtype == b.dtype and a.token == b.token,
            float32="float32", full=lambda shape, value, **kw: TensorStub(shape, token=value),
            inference_mode=contextlib.nullcontext,
        )

    def load(self, path, **kwargs):
        self.load_calls.append(kwargs)
        return {"model": self.saved}

    def strict_load(self, state, strict):
        self.strict_calls.append(strict)
        if set(state) != set(self.expected):
            raise RuntimeError('Missing key(s) in state_dict: "bias" [synthetic]')
        self.expected = dict(state)
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    def build(self, cfg, defaults):
        self.cfg, self.defaults = cfg, defaults
        return self.model

    def import_module(self, name):
        modules = {
            "torch": self.torch,
            "rfdetr.config": SimpleNamespace(RFDETRSegNanoConfig=lambda **kw: SimpleNamespace(**kw)),
            "rfdetr.models": SimpleNamespace(MODEL_DEFAULTS=DefaultsStub(), build_model_from_config=self.build),
        }
        return modules[name]


class SafeLoadTests(unittest.TestCase):
    def test_bytes_path_is_rejected_without_runtime_import(self):
        probe = importlib.import_module("probe")
        with patch.object(probe.importlib, "import_module") as runtime_import:
            result = probe.run_probe(b"synthetic-not-a-checkpoint.pt")
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["qualified"])
        runtime_import.assert_not_called()
        json.dumps(result, allow_nan=False)

    def test_rejects_unsafe_checkpoint_without_pickle_fallback(self):
        probe = importlib.import_module("probe")
        calls = []

        def load(path, **kwargs):
            calls.append(kwargs)
            raise pickle.UnpicklingError("synthetic unsafe global")

        fake_torch = SimpleNamespace(load=load, serialization=SimpleNamespace())
        with patch.object(probe.importlib, "import_module", return_value=fake_torch):
            result = probe.run_probe("synthetic-not-a-checkpoint.pt")
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["qualified"])
        self.assertEqual(len(calls), 1)
        self.assertTrue(all(c["weights_only"] is True for c in calls))
        self.assertTrue(all(c["map_location"] == "cpu" for c in calls))

    def test_namespace_retry_is_bounded_and_never_uses_pickle(self):
        probe = importlib.import_module("probe")
        loads, allowed = [], []

        def load(path, **kwargs):
            loads.append(kwargs)
            raise pickle.UnpicklingError("synthetic unsupported global")

        def safe_globals(classes):
            allowed.extend(classes)
            return contextlib.nullcontext()

        fake_torch = SimpleNamespace(load=load, serialization=SimpleNamespace(safe_globals=safe_globals))
        with patch.object(probe.importlib, "import_module", return_value=fake_torch):
            result = probe.run_probe("synthetic-not-a-checkpoint.pt")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(loads), 2)
        self.assertTrue(all(c["weights_only"] is True for c in loads))
        self.assertEqual(allowed, [probe.argparse.Namespace, SimpleNamespace])

    def test_missing_state_key_is_rejected_by_strict_loading(self):
        probe = importlib.import_module("probe")
        runtime = RuntimeStub(missing_key=True)
        with patch.object(probe.importlib, "import_module", side_effect=runtime.import_module):
            result = probe.run_probe("synthetic-not-a-checkpoint.pt")
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["qualified"])
        self.assertEqual(result["error"]["stage"], "strict_load")
        self.assertEqual(runtime.strict_calls, [True])
        self.assertEqual(runtime.forward_calls, 0)
        self.assertEqual((runtime.threads, runtime.rng), (7, 101))

    def test_valid_synthetic_control_path_stays_unqualified_and_json_safe(self):
        probe = importlib.import_module("probe")
        runtime = RuntimeStub()
        with patch.object(probe.importlib, "import_module", side_effect=runtime.import_module):
            result = probe.run_probe("synthetic-not-a-checkpoint.pt")
        self.assertEqual(result["status"], "ok", result)
        self.assertTrue(result["strict_state_equal"])
        self.assertTrue(result["strict_original_state_load"])
        self.assertFalse(result["qualified"])
        self.assertEqual((result["variant"], result["resolution"], result["threads_during_probe"]),
                         ("RFDETRSegNanoConfig", 312, 2))
        self.assertEqual(runtime.forward_calls, 1)
        self.assertEqual(result["outputs"]["pred_masks"]["shape"], [1, 100, 78, 78])
        self.assertTrue(runtime.defaults.force_no_pretrain)
        self.assertIsNone(runtime.cfg.pretrain_weights)
        self.assertEqual((runtime.threads, runtime.rng), (7, 101))
        json.dumps(result, allow_nan=False)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("rfdetr", sys.modules)

    def test_corrupt_state_and_wrong_native_outputs_are_not_accepted(self):
        probe = importlib.import_module("probe")
        cases = ["dtype", "checkpoint_nonfinite", "equality", "mask_shape", "mask_dict", "output_nonfinite"]
        for case in cases:
            with self.subTest(case=case):
                runtime = RuntimeStub()
                expected_stage, expected_calls = "state_validation", 0
                if case == "dtype":
                    runtime.saved["weight"] = TensorStub(dtype="float16")
                elif case == "checkpoint_nonfinite":
                    runtime.saved["weight"] = TensorStub(finite=False)
                elif case == "equality":
                    runtime.torch.equal = lambda a, b: False
                    expected_stage = "state_equality"
                else:
                    expected_stage, expected_calls = "output_validation", 1
                    if case == "mask_shape":
                        runtime.outputs["pred_masks"] = TensorStub((1, 100, 77, 78))
                    elif case == "mask_dict":
                        runtime.outputs["pred_masks"] = {"spatial_features": TensorStub()}
                    else:
                        runtime.outputs["pred_masks"].finite = False
                with patch.object(probe.importlib, "import_module", side_effect=runtime.import_module):
                    result = probe.run_probe("synthetic-not-a-checkpoint.pt")
                self.assertEqual(result["status"], "failed")
                self.assertFalse(result["qualified"])
                self.assertEqual(result["error"]["stage"], expected_stage)
                self.assertEqual(runtime.forward_calls, expected_calls)
                self.assertEqual((runtime.threads, runtime.rng), (7, 101))


if __name__ == "__main__":
    unittest.main()
