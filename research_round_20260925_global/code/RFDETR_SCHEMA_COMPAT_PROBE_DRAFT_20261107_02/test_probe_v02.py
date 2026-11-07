"""Synthetic v02 control tests. Reuse the frozen v01 standard-library fixture."""
import importlib.util
import contextlib
import json
import pickle
import sys
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


fixture_path = (Path(__file__).resolve().parent.parent /
                "RFDETR_CPU_MODEL_PROBE_DRAFT_20261107_01" / "test_probe.py")
fixture_spec = importlib.util.spec_from_file_location("_v01_probe_test_fixture", fixture_path)
fixture = importlib.util.module_from_spec(fixture_spec)
sys.modules[fixture_spec.name] = fixture
fixture_spec.loader.exec_module(fixture)


class SchemaTensorStub(fixture.TensorStub):
    def clone(self):
        return SchemaTensorStub(self.shape, self.token, self.dtype, self.finite)


class ParameterStub(SchemaTensorStub):
    pass


class LegacyRuntimeStub(fixture.RuntimeStub):
    def __init__(self):
        super().__init__()
        self.saved = OrderedDict(self.saved)
        self.saved._metadata = {"": {"version": 1}}
        self.original_refs = list(self.saved.items())
        self.original_metadata = self.saved._metadata
        self.schema_buffer = SchemaTensorStub((0, 0), dtype="torch.bool")
        self.expected["_kp_active_mask"] = self.schema_buffer
        self.model.use_grouppose_keypoints = False
        self.model.num_keypoints_per_class = []
        self.model.keypoint_embed = None
        self.model.get_buffer = lambda key: self.schema_buffer
        self.torch.bool = "torch.bool"
        self.torch.nn = SimpleNamespace(Parameter=ParameterStub)


class ProbeV02Tests(unittest.TestCase):
    def test_only_completed_state_loads_strictly_and_original_mapping_survives(self):
        import probe_v02
        runtime = LegacyRuntimeStub()
        with patch.object(probe_v02.importlib, "import_module", side_effect=runtime.import_module):
            result = probe_v02.run_probe("synthetic-not-a-checkpoint.pt")
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(runtime.strict_calls, [True])
        self.assertEqual(runtime.forward_calls, 1)
        self.assertFalse(result["strict_original_state_load"])
        self.assertTrue(result["strict_completed_state_load"])
        self.assertTrue(result["original_tensor_equality"])
        self.assertFalse(result["qualified"])
        self.assertEqual(result["schema_completion"]["added_keys"], ["_kp_active_mask"])
        self.assertEqual(list(runtime.saved.items()), runtime.original_refs)
        self.assertIs(runtime.saved._metadata, runtime.original_metadata)
        for name, tensor in runtime.original_refs:
            self.assertIs(runtime.saved[name], tensor)
        self.assertNotIn("_kp_active_mask", runtime.saved)
        self.assertIsNot(runtime.expected["_kp_active_mask"], runtime.schema_buffer)
        self.assertFalse(runtime.cfg.use_grouppose_keypoints)
        self.assertEqual(runtime.cfg.num_keypoints_per_class, [])
        self.assertTrue(runtime.defaults.force_no_pretrain)
        self.assertEqual((runtime.threads, runtime.rng), (7, 101))
        self.assertEqual(result["outputs"]["pred_masks"]["shape"], [1, 100, 78, 78])
        json.dumps(result, allow_nan=False)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("rfdetr", sys.modules)

    def test_extra_schema_errors_and_original_tensor_inequality_stop_before_forward(self):
        import probe_v02
        for case in ("missing", "unexpected", "inequality", "nonfinite", "loaded_nonfinite"):
            with self.subTest(case=case):
                runtime = LegacyRuntimeStub()
                expected_stage = "schema_completion"
                if case == "missing":
                    runtime.saved.pop("bias")
                elif case == "unexpected":
                    runtime.saved["unknown"] = fixture.TensorStub()
                elif case == "nonfinite":
                    runtime.saved["weight"] = fixture.TensorStub(finite=False)
                    expected_stage = "state_validation"
                else:
                    expected_stage = "state_equality"
                    strict_load = runtime.strict_load

                    def unequal_load(state, strict):
                        incompatible = strict_load(state, strict)
                        runtime.expected["weight"] = fixture.TensorStub(
                            token=99 if case == "inequality" else 1,
                            finite=case != "loaded_nonfinite",
                        )
                        return incompatible

                    runtime.strict_load = unequal_load
                with patch.object(probe_v02.importlib, "import_module", side_effect=runtime.import_module):
                    result = probe_v02.run_probe("synthetic-not-a-checkpoint.pt")
                self.assertEqual(result["status"], "failed", result)
                self.assertFalse(result["qualified"])
                self.assertFalse(result["strict_original_state_load"])
                self.assertFalse(result["original_tensor_equality"])
                self.assertEqual(result["error"]["stage"], expected_stage)
                self.assertEqual(runtime.forward_calls, 0)
                expected_strict = [True] if case in ("inequality", "loaded_nonfinite") else []
                self.assertEqual(runtime.strict_calls, expected_strict)
                self.assertEqual(result["strict_completed_state_load"], bool(expected_strict))
                self.assertEqual((runtime.threads, runtime.rng), (7, 101))

    def test_safe_load_is_cpu_weights_only_with_bounded_allowlist_retry(self):
        import probe_v02
        with patch.object(probe_v02.importlib, "import_module") as runtime_import:
            result = probe_v02.run_probe(b"synthetic-not-a-checkpoint.pt")
        self.assertEqual(result["status"], "failed")
        runtime_import.assert_not_called()
        for allow_retry in (False, True):
            with self.subTest(allow_retry=allow_retry):
                calls, allowed = [], []

                def load(path, **kwargs):
                    calls.append(kwargs)
                    raise pickle.UnpicklingError("synthetic unsupported global")

                def safe_globals(classes):
                    allowed.extend(classes)
                    return contextlib.nullcontext()

                serialization = SimpleNamespace()
                if allow_retry:
                    serialization.safe_globals = safe_globals
                torch_stub = SimpleNamespace(load=load, serialization=serialization)
                with patch.object(probe_v02.importlib, "import_module", return_value=torch_stub):
                    result = probe_v02.run_probe("synthetic-not-a-checkpoint.pt")
                self.assertEqual(result["status"], "failed")
                self.assertFalse(result["qualified"])
                self.assertEqual(len(calls), 2 if allow_retry else 1)
                self.assertTrue(all(c == {"map_location": "cpu", "weights_only": True} for c in calls))
                self.assertEqual(allowed, [probe_v02.argparse.Namespace, SimpleNamespace] if allow_retry else [])

    def test_native_wrong_shape_or_nonfinite_outputs_fail_after_one_forward(self):
        import probe_v02
        for output in (fixture.TensorStub((1, 100, 77, 78)),
                       fixture.TensorStub((1, 100, 78, 78), finite=False)):
            with self.subTest(output=output):
                runtime = LegacyRuntimeStub()
                runtime.outputs["pred_masks"] = output
                with patch.object(probe_v02.importlib, "import_module", side_effect=runtime.import_module):
                    result = probe_v02.run_probe("synthetic-not-a-checkpoint.pt")
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["error"]["stage"], "output_validation")
                self.assertFalse(result["qualified"])
                self.assertEqual(runtime.strict_calls, [True])
                self.assertEqual(runtime.forward_calls, 1)
                self.assertEqual((runtime.threads, runtime.rng), (7, 101))


if __name__ == "__main__":
    unittest.main()
