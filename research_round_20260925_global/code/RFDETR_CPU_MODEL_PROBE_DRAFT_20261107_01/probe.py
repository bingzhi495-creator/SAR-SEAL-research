"""A CPU-only real-model probe; execution qualification belongs to the caller.

Importing this module imports only the standard library. No command-line runner,
download, training, real-image read, checkpoint rewrite, or resource worker lives here.
"""
import argparse
import importlib
import os
import pickle
import time
from dataclasses import replace
from types import SimpleNamespace


def _safe_load(torch, path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (RuntimeError, pickle.UnpicklingError):
        safe_globals = getattr(getattr(torch, "serialization", None), "safe_globals", None)
        if safe_globals is None:
            raise
        with safe_globals([argparse.Namespace, SimpleNamespace]):
            return torch.load(path, map_location="cpu", weights_only=True)


def run_probe(checkpoint_path, seed=5201):
    """Return JSON-safe evidence. ``qualified`` is always False, including on success."""
    started = time.perf_counter()
    result = {"status": "failed", "qualified": False, "device": "cpu", "seed": None,
              "variant": "RFDETRSegNanoConfig", "resolution": 312,
              "strict_original_state_load": False, "strict_state_equal": False,
              "forward_executed": False, "phase_seconds": {}}
    stage = "arguments"
    path = ""
    torch = None
    previous_threads = previous_rng = None
    try:
        path = os.fspath(checkpoint_path)
        if not isinstance(path, str) or not path or "://" in path:
            raise ValueError("checkpoint_path must name a local checkpoint file")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        result["seed"] = seed
        stage = "torch_import"
        phase = time.perf_counter()
        torch = importlib.import_module("torch")
        result["phase_seconds"][stage] = time.perf_counter() - phase
        stage = "checkpoint_load"
        phase = time.perf_counter()
        checkpoint = _safe_load(torch, path)
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
            raise ValueError("checkpoint must contain the original model state dictionary")
        state = checkpoint["model"]
        result["phase_seconds"][stage] = time.perf_counter() - phase
        stage = "cpu_policy"
        previous_threads, previous_rng = torch.get_num_threads(), torch.get_rng_state()
        torch.set_num_threads(2)
        torch.random.default_generator.manual_seed(seed)  # CPU generator only; never torch.manual_seed/CUDA.
        result["threads_during_probe"] = 2
        stage = "model_build"
        phase = time.perf_counter()
        configs = importlib.import_module("rfdetr.config")
        models = importlib.import_module("rfdetr.models")
        cfg = configs.RFDETRSegNanoConfig(
            pretrain_weights=None, device="cpu", num_classes=90, num_channels=3,
            resolution=312, compile=False, cuda_graphs=False,
            gradient_checkpointing=False, backbone_lora=False,
        )
        defaults = replace(models.MODEL_DEFAULTS, force_no_pretrain=True)
        net = models.build_model_from_config(cfg, defaults=defaults)
        result["phase_seconds"][stage] = time.perf_counter() - phase
        stage = "state_validation"
        expected = net.state_dict()
        for key, tensor in state.items():
            if not isinstance(key, str) or not torch.is_tensor(tensor):
                raise ValueError("original model state must contain named tensors only")
            if tensor.device.type != "cpu":
                raise ValueError("checkpoint tensor is not on CPU: " + key)
            if not bool(torch.isfinite(tensor).all().item()):
                raise ValueError("non-finite checkpoint tensor: " + key)
            if key in expected:
                if tuple(tensor.shape) != tuple(expected[key].shape) or tensor.dtype != expected[key].dtype:
                    raise ValueError("checkpoint/model shape or dtype mismatch: " + key)
        stage = "strict_load"
        phase = time.perf_counter()
        incompatible = net.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError("strict loader reported incompatible keys")
        result["phase_seconds"][stage] = time.perf_counter() - phase
        stage = "state_equality"
        phase = time.perf_counter()
        loaded = net.state_dict()
        if set(loaded) != set(state):
            raise ValueError("model/checkpoint key sets differ after strict loading")
        for key, tensor in state.items():
            actual = loaded[key]
            if actual.device.type != "cpu" or tuple(actual.shape) != tuple(tensor.shape) or actual.dtype != tensor.dtype:
                raise ValueError("loaded tensor device, shape or dtype differs: " + key)
            if not torch.equal(actual, tensor) or not bool(torch.isfinite(actual).all().item()):
                raise ValueError("loaded tensor differs or is non-finite: " + key)
        result["strict_state_equal"] = True
        result["strict_original_state_load"] = True
        result["state_tensor_count"] = len(state)
        result["state_numel"] = sum(int(t.numel()) for t in state.values())
        result["phase_seconds"][stage] = time.perf_counter() - phase
        del state, checkpoint, loaded, expected
        stage = "forward"
        net = net.to(device="cpu", dtype=torch.float32).eval()
        image01 = torch.full((1, 3, 312, 312), 0.5, device="cpu", dtype=torch.float32)
        mean = image01.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = image01.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        inputs = (image01 - mean) / std
        phase = time.perf_counter()
        with torch.inference_mode():
            outputs = net(inputs)
            result["forward_executed"] = True
            result["phase_seconds"][stage] = time.perf_counter() - phase
            stage = "output_validation"
            if not isinstance(outputs, dict):
                raise ValueError("native LWDETR forward must return a dictionary")
            shapes = {"pred_logits": (1, 100, 91), "pred_boxes": (1, 100, 4),
                      "pred_masks": (1, 100, 78, 78)}
            summaries = {}
            for key, shape in shapes.items():
                tensor = outputs.get(key)
                if not torch.is_tensor(tensor):
                    raise ValueError("native output is not a dense tensor: " + key)
                if tuple(tensor.shape) != shape or tensor.device.type != "cpu" or tensor.dtype != torch.float32:
                    raise ValueError("native output shape, device or dtype mismatch: " + key)
                finite = bool(torch.isfinite(tensor).all().item())
                if not finite:
                    raise ValueError("non-finite native output: " + key)
                summaries[key] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                                  "device": "cpu", "numel": int(tensor.numel()), "finite": finite}
            result["outputs"] = summaries
        result["status"] = "ok"
    except Exception as error:
        message = str(error)
        if isinstance(path, str) and path:
            message = message.replace(path, "<checkpoint>")
        result["error"] = {"stage": stage, "type": type(error).__name__, "message": message[:2000]}
    finally:
        cleanup_errors = []
        if torch is not None and previous_rng is not None:
            try:
                torch.set_rng_state(previous_rng)
            except Exception as error:
                cleanup_errors.append(type(error).__name__ + ": CPU RNG restoration failed")
        if torch is not None and previous_threads is not None:
            try:
                torch.set_num_threads(previous_threads)
            except Exception as error:
                cleanup_errors.append(type(error).__name__ + ": thread restoration failed")
        if cleanup_errors:
            result["status"] = "failed"
            result["cleanup_errors"] = cleanup_errors
    result["elapsed_seconds"] = time.perf_counter() - started
    return result
