"""Run the closed P87 F/S training recipe with the verified strict backward route."""
from hashlib import sha256
import os
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch


D = Path(__file__).resolve().parent.parent
P87 = D / "RFDETR_E24_OPTIMIZER_PRESERVATION_TRAIN_CANDIDATE_20261108_01"
COMBINED = D / "RFDETR_E24_COMBINED_SAMPLING_BACKWARD_PROBE_CANDIDATE_20261108_01"
STRICT = D / "RFDETR_E24_DETERMINISTIC_BACKWARD_REPLAY_CANDIDATE_20261108_02"
REFERENCE_FIRST_GRADIENT_SHA256 = "ddad103f09d69255ee35f5e7cc755d850be88b49844e2550a4f1fa0c15914f21"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def check_first_step(digest, counts):
    need(counts == (4, 36, 8), "Exact strict first-window routes before AdamW step")
    need(digest == REFERENCE_FIRST_GRADIENT_SHA256, "Exact P94 first clipped gradient before AdamW step")


def exact_bytes(path, digest):
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == digest, "Exact closed dependency " + path.name)
    return raw


def exact_module(path, digest, name):
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(exact_bytes(path, digest), str(path), "exec"), module.__dict__)
    return module


class BackwardControl:
    def __init__(self, torch, strict, combined):
        self.torch = torch
        self.strict = strict.StrictBackward(torch.Tensor.backward)
        self.combined = combined
        self.grid = combined.load_grid_probe()
        self.grid_trace = self.grid.GridBackwardTrace()
        self.resize_trace = combined.ResizeBackwardTrace()
        self.first_counts = None

    def backward(self, tensor, *args, **kwargs):
        result = self.strict(tensor, *args, **kwargs)
        if self.strict.calls == 4:
            grid, resize = self.routes()
            self.first_counts = (self.strict.calls, grid["cpu_backward_calls"], resize["cpu_backward_calls"])
        return result

    def routes(self):
        grid = self.grid_trace.summary()
        resize = self.resize_trace.summary()
        need(grid["unchanged_differentiable_cuda_bilinear_calls"] == 0
             and resize["unchanged_cuda_bilinear_antialias_calls"] == 0, "No unsupported native CUDA sampling backward")
        for route in (grid, resize):
            need(route["routed_forward_calls"] == route["cpu_backward_calls"] > 0
                 and all(x["routed_forward_calls"] == x["cpu_backward_calls"] for x in route["observed_shapes"]),
                 "Complete routed CPU backward by shape")
        return grid, resize

    def train(self, stock, *args, **kwargs):
        torch = self.torch
        f = torch.nn.functional
        need(not torch.are_deterministic_algorithms_enabled(), "Keep the native forward policy")
        def backward(tensor, *backward_args, **backward_kwargs):
            return self.backward(tensor, *backward_args, **backward_kwargs)

        with (patch.object(f, "grid_sample", self.grid.routed_grid_sample(self.grid_trace)),
              patch.object(f, "interpolate", self.combined.routed_interpolate(self.resize_trace)),
              patch.object(torch.Tensor, "backward", backward)):
            result = stock(*args, **kwargs)
        grid, resize = self.routes()
        need(self.strict.calls == result["microbatches"] == 1488
             and not torch.are_deterministic_algorithms_enabled(), "Exact full strict backward count and policy restoration")
        result["backward_control"] = dict(strict_calls=self.strict.calls, strict_seconds=self.strict.seconds,
            first_counts=self.first_counts, first_gradient_sha256=REFERENCE_FIRST_GRADIENT_SHA256,
            grid_sample=grid, bilinear_interpolate=resize, cublas_workspace_override=None)
        return result


def main():
    import torch

    need(torch.__version__ == "2.10.0+cu130"
         and torch.version.git_version == "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0"
         and os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None, "Exact P94 runtime and workspace policy")
    exact_bytes(P87 / "state_preservation.py", "8dda41131a2930aacf00c51dbe8cfa4817c0a3cd4b02c6bb606948db77341678")
    worker = exact_module(P87 / "run_state_arm.py", "f870227ecce5da591c1bdc9b975591fff720b36540e0f01818277105d3c2176c",
                          "_stock_p87_state_arm")
    sys.addaudithook(worker.offline)
    combined = exact_module(COMBINED / "combined_sampling_backward_probe.py",
                           "b90dcaf0cc408b078c2fcad2adb2c646f0e2632fe034f6a4cf50792cff8822e2",
                           "_closed_combined_sampling")
    strict = exact_module(STRICT / "deterministic_sampling_replay.py",
                         "7939d724a8b7b5ff66c9ab7fa7022efe272382f7db31b427875cc02d81c7a4fa",
                         "_closed_strict_backward")
    control = BackwardControl(torch, strict, combined)
    stock_source = worker.p77_source

    def controlled_source():
        original = stock_source()
        train = original.train_epochs
        original.train_epochs = lambda *args, **kwargs: control.train(train, *args, **kwargs)
        return original

    class CheckedFirstStep(worker.FirstStepTrace):
        def _before(self, optimizer, args, kwargs):
            super()._before(optimizer, args, kwargs)
            check_first_step(self.first["clipped_grad_sha256"], control.first_counts)

    with (patch.object(worker, "p77_source", controlled_source),
          patch.object(worker, "FirstStepTrace", CheckedFirstStep)):
        worker.main()


if __name__ == "__main__":
    main()

