"""Probe native bilinear CUDA sampling with only the backward operator moved to CPU."""
import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import os
import re
import sys
from time import perf_counter
from types import ModuleType
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable


HERE = Path(__file__).resolve().parent
REPLAY = HERE.parent / "RFDETR_E24_FIRST_GRAD_REPLAY_CANDIDATE_20261108_01/first_gradient_replay.py"
REPLAY_SHA256 = "b35340218f6b557a915417a352ce6ca5270a368375a94e2f6b0b3178d2545314"
STOCK_GRID_SAMPLE = F.grid_sample
BILINEAR = 0
PADDING_MODE = {"zeros": 0, "border": 1}
CPU_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


class GridBackwardTrace:
    def __init__(self):
        self.forward = Counter()
        self.backward = Counter()
        self.unchanged_calls = 0
        self.unchanged_differentiable_cuda_bilinear_calls = 0
        self.cpu_backward_seconds = 0.0
        self.cpu_kernel_seconds = 0.0
        self.cpu_input_bytes = 0
        self.returned_gradient_bytes = 0

    def summary(self):
        shapes = []
        for shape, count in self.forward.items():
            input_shape, grid_shape, input_dtype, grid_dtype, dtype, padding, align = shape
            shapes.append(dict(input_shape=input_shape, grid_shape=grid_shape, input_dtype=input_dtype,
                grid_dtype=grid_dtype, effective_dtype=dtype, padding_mode=padding, align_corners=align,
                routed_forward_calls=count, cpu_backward_calls=self.backward[shape]))
        return dict(routed_forward_calls=sum(self.forward.values()), cpu_backward_calls=sum(self.backward.values()),
                    unchanged_calls=self.unchanged_calls,
                    unchanged_differentiable_cuda_bilinear_calls=self.unchanged_differentiable_cuda_bilinear_calls,
                    cpu_backward_seconds=self.cpu_backward_seconds, cpu_kernel_seconds=self.cpu_kernel_seconds,
                    cpu_input_bytes=self.cpu_input_bytes, returned_gradient_bytes=self.returned_gradient_bytes,
                    observed_shapes=shapes)


class _CpuGridSampleBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, grid, padding_mode, align_corners, trace):
        output = STOCK_GRID_SAMPLE(data, grid, mode="bilinear", padding_mode=padding_mode,
                                   align_corners=align_corners)
        if output.dtype not in CPU_DTYPES:
            raise ValueError("No exact-effective-dtype CPU backward for " + str(output.dtype))
        ctx.save_for_backward(data, grid)
        ctx.padding_mode = padding_mode
        ctx.align_corners = bool(align_corners)  # torch.grid_sample uses False for its None default.
        ctx.dtype = output.dtype
        ctx.trace = trace
        ctx.shape = (tuple(data.shape), tuple(grid.shape), str(data.dtype), str(grid.dtype), str(output.dtype),
                     padding_mode, ctx.align_corners)
        trace.forward[ctx.shape] += 1
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        started = perf_counter()
        data, grid = ctx.saved_tensors
        trace = ctx.trace
        needs_input, needs_grid = ctx.needs_input_grad[:2]
        with torch.autocast(device_type="cpu", enabled=False):
            cpu_grad = grad_output.to(device="cpu", dtype=ctx.dtype)
            cpu_data = data.to(device="cpu", dtype=ctx.dtype)
            cpu_grid = grid.to(device="cpu", dtype=ctx.dtype)
            kernel_started = perf_counter()
            grad_data, grad_grid = torch.ops.aten.grid_sampler_2d_backward.default(
                cpu_grad, cpu_data, cpu_grid, BILINEAR, PADDING_MODE[ctx.padding_mode],
                ctx.align_corners, (needs_input, needs_grid),
            )
            trace.cpu_kernel_seconds += perf_counter() - kernel_started
        trace.cpu_input_bytes += sum(x.numel() * x.element_size() for x in (cpu_grad, cpu_data, cpu_grid))
        restored = tuple(
            grad.to(device=original.device, dtype=original.dtype) if needed else None
            for grad, original, needed in ((grad_data, data, needs_input), (grad_grid, grid, needs_grid))
        )
        trace.returned_gradient_bytes += sum(x.numel() * x.element_size() for x in restored if x is not None)
        trace.backward[ctx.shape] += 1
        trace.cpu_backward_seconds += perf_counter() - started
        return *restored, None, None, None


def cpu_backward_grid_sample(data, grid, padding_mode, align_corners, trace):
    """Use native sampling on the input device and the native CPU derivative."""
    return _CpuGridSampleBackward.apply(data, grid, padding_mode, align_corners, trace)


def routed_grid_sample(trace):
    def route(input, grid, mode="bilinear", padding_mode="zeros", align_corners=None):
        eligible = (mode == "bilinear" and input.is_cuda and grid.device == input.device
                    and input.ndim == grid.ndim == 4 and torch.is_grad_enabled()
                    and (input.requires_grad or grid.requires_grad))
        if eligible and padding_mode in PADDING_MODE:
            return cpu_backward_grid_sample(input, grid, padding_mode, align_corners, trace)
        trace.unchanged_calls += 1
        trace.unchanged_differentiable_cuda_bilinear_calls += int(eligible)
        return STOCK_GRID_SAMPLE(input, grid, mode, padding_mode, align_corners)
    return route


def load_replay():
    source = REPLAY.read_bytes()
    if sha256(source).hexdigest() != REPLAY_SHA256:
        raise ValueError("Exact closed first-window source required")
    module = ModuleType("_pinned_first_window_replay")
    module.__file__ = str(REPLAY)
    sys.modules[module.__name__] = module
    exec(compile(source, str(REPLAY), "exec"), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True, choices=("F0", "F1"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--protocol-commit", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.protocol_commit):
        parser.error("Expected exact prospective protocol commit")
    replay = load_replay()
    stock_window = replay.trace_first_window
    trace = GridBackwardTrace()

    def probe_window(*window_args, **window_kwargs):
        with patch.object(F, "grid_sample", routed_grid_sample(trace)):
            rows, clip = stock_window(*window_args, **window_kwargs)
        if not trace.backward:
            raise ValueError("No differentiable native CUDA bilinear backward was routed")
        if trace.unchanged_differentiable_cuda_bilinear_calls:
            raise ValueError("An unsupported padding mode left a CUDA bilinear backward on its native route")
        return rows, clip

    with patch.object(replay, "trace_first_window", probe_window):
        replay.run(args)
    result_path = args.output_dir / "RESULT.json"
    result = dict(status="STOCK_CUDA_GRID_FORWARD_CPU_BILINEAR_BACKWARD_PROBE_COMPLETE_NO_UPDATE",
                  protocol_commit=args.protocol_commit, attempt=args.attempt, replay_source_sha256=REPLAY_SHA256,
                  native_replay_result_sha256=sha256(result_path.read_bytes()).hexdigest(),
                  optimizer_updates=0, ema_updates=0, DEV=False, TEST=False, AP=False, **trace.summary())
    with (args.output_dir / "GRID_BACKWARD_PROBE_PRIVATE.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=result["status"], attempt=args.attempt,
                          routed_forward_calls=result["routed_forward_calls"],
                          cpu_backward_calls=result["cpu_backward_calls"],
                          cpu_backward_seconds=result["cpu_backward_seconds"],
                          optimizer_updates=0, ema_updates=0)), flush=True)


if __name__ == "__main__":
    main()
