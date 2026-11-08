"""Replay the native FIT first window with grid and bilinear-resize backward on CPU."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from types import ModuleType
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable


HERE = Path(__file__).resolve().parent
GRID = HERE.parent / "RFDETR_E24_GRID_BACKWARD_PROBE_CANDIDATE_20261108_01/grid_backward_probe.py"
GRID_SHA256 = "cebf8410b9511a446674119171c82f7fde090c72304eb7645cda1c5fa7c121a6"
STOCK_INTERPOLATE = F.interpolate
CPU_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


class ResizeBackwardTrace:
    def __init__(self):
        self.forward = Counter()
        self.backward = Counter()
        self.unchanged_calls = 0
        self.unchanged_cuda_bilinear_antialias_calls = 0
        self.cpu_backward_seconds = 0.0
        self.cpu_kernel_seconds = 0.0
        self.cpu_input_bytes = 0
        self.returned_gradient_bytes = 0

    def summary(self):
        shapes = []
        for key, count in self.forward.items():
            input_shape, output_shape, input_dtype, dtype, align, scales, recompute = key
            shapes.append(dict(input_shape=input_shape, output_shape=output_shape, input_dtype=input_dtype,
                effective_dtype=dtype, align_corners=align, effective_scales=scales,
                recompute_scale_factor=recompute, routed_forward_calls=count, cpu_backward_calls=self.backward[key]))
        return dict(routed_forward_calls=sum(self.forward.values()), cpu_backward_calls=sum(self.backward.values()),
                    unchanged_calls=self.unchanged_calls,
                    unchanged_cuda_bilinear_antialias_calls=self.unchanged_cuda_bilinear_antialias_calls,
                    cpu_backward_seconds=self.cpu_backward_seconds, cpu_kernel_seconds=self.cpu_kernel_seconds,
                    cpu_input_bytes=self.cpu_input_bytes, returned_gradient_bytes=self.returned_gradient_bytes,
                    observed_shapes=shapes)


class _CpuBilinearResizeBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, size, scale_factor, align_corners, recompute_scale_factor, trace):
        output = STOCK_INTERPOLATE(data, size=size, scale_factor=scale_factor, mode="bilinear",
                                   align_corners=align_corners, recompute_scale_factor=recompute_scale_factor,
                                   antialias=False)
        if output.dtype not in CPU_DTYPES:
            raise ValueError("No exact-effective-dtype CPU resize backward for " + str(output.dtype))

        # Native interpolate clears scales for an explicit size or a recomputed size.
        if size is not None or recompute_scale_factor:
            scales = (None, None)
        elif isinstance(scale_factor, (list, tuple)):
            scales = tuple(float(scale) for scale in scale_factor)
        else:
            scales = (float(scale_factor), float(scale_factor))
        ctx.input_shape = tuple(data.shape)
        ctx.output_shape = tuple(output.shape)
        ctx.device = data.device
        ctx.input_dtype = data.dtype
        ctx.effective_dtype = output.dtype
        ctx.align_corners = bool(align_corners)
        ctx.scales = scales
        ctx.trace = trace
        ctx.key = (ctx.input_shape, ctx.output_shape, str(data.dtype), str(output.dtype),
                   ctx.align_corners, scales, recompute_scale_factor)
        trace.forward[ctx.key] += 1
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        started = perf_counter()
        trace = ctx.trace
        with torch.autocast(device_type="cpu", enabled=False):
            grad = grad_output.to(device="cpu", dtype=ctx.effective_dtype)
            kernel_started = perf_counter()
            grad_input = torch.ops.aten.upsample_bilinear2d_backward.default(
                grad, ctx.output_shape[-2:], ctx.input_shape, ctx.align_corners, *ctx.scales,
            )
            trace.cpu_kernel_seconds += perf_counter() - kernel_started
        restored = grad_input.to(device=ctx.device, dtype=ctx.input_dtype)
        trace.cpu_input_bytes += grad.numel() * grad.element_size()
        trace.returned_gradient_bytes += restored.numel() * restored.element_size()
        trace.backward[ctx.key] += 1
        trace.cpu_backward_seconds += perf_counter() - started
        return restored, None, None, None, None, None


def cpu_backward_interpolate(data, size=None, scale_factor=None, align_corners=None,
                             recompute_scale_factor=None, *, trace):
    """Keep native bilinear forward on the input device and compute its derivative on CPU."""
    return _CpuBilinearResizeBackward.apply(data, size, scale_factor, align_corners, recompute_scale_factor, trace)


def routed_interpolate(trace):
    def route(input, size=None, scale_factor=None, mode="nearest", align_corners=None,
              recompute_scale_factor=None, antialias=False):
        eligible = (input.is_cuda and input.ndim == 4 and mode == "bilinear"
                    and torch.is_grad_enabled() and input.requires_grad)
        if eligible and not antialias:
            return cpu_backward_interpolate(input, size, scale_factor, align_corners, recompute_scale_factor,
                                            trace=trace)
        trace.unchanged_calls += 1
        trace.unchanged_cuda_bilinear_antialias_calls += int(eligible and antialias)
        return STOCK_INTERPOLATE(input, size=size, scale_factor=scale_factor, mode=mode,
                                 align_corners=align_corners, recompute_scale_factor=recompute_scale_factor,
                                 antialias=antialias)
    return route


def load_grid_probe():
    source = GRID.read_bytes()
    if sha256(source).hexdigest() != GRID_SHA256:
        raise ValueError("Exact closed grid-only probe source required")
    module = ModuleType("_pinned_grid_backward_probe")
    module.__file__ = str(GRID)
    sys.modules[module.__name__] = module
    exec(compile(source, str(GRID), "exec"), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True, choices=("F0", "F1"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--protocol-commit", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.protocol_commit):
        parser.error("Expected exact prospective protocol commit")

    grid = load_grid_probe()
    replay = grid.load_replay()
    stock_window = replay.trace_first_window
    grid_trace = grid.GridBackwardTrace()
    resize_trace = ResizeBackwardTrace()

    def probe_window(*window_args, **window_kwargs):
        if torch.are_deterministic_algorithms_enabled():
            raise ValueError("Expected the original CUDA interpolation forward route")
        with (patch.object(F, "grid_sample", grid.routed_grid_sample(grid_trace)),
              patch.object(F, "interpolate", routed_interpolate(resize_trace))):
            rows, clip = stock_window(*window_args, **window_kwargs)
        if not grid_trace.backward or not resize_trace.backward:
            raise ValueError("Both registered CPU backward families must run on the real first window")
        if grid_trace.unchanged_differentiable_cuda_bilinear_calls:
            raise ValueError("An unsupported padding mode left a CUDA bilinear grid backward on its native route")
        return rows, clip

    started = perf_counter()
    with patch.object(replay, "trace_first_window", probe_window):
        replay.run(args)
    result_path = args.output_dir / "RESULT.json"
    result = dict(status="STOCK_CUDA_FORWARD_CPU_GRID_AND_RESIZE_BACKWARD_PROBE_COMPLETE_NO_UPDATE",
                  protocol_commit=args.protocol_commit, attempt=args.attempt,
                  grid_probe_source_sha256=GRID_SHA256, replay_source_sha256=grid.REPLAY_SHA256,
                  native_replay_result_sha256=sha256(result_path.read_bytes()).hexdigest(),
                  grid_sample=grid_trace.summary(), bilinear_interpolate=resize_trace.summary(),
                  replay_elapsed_seconds=perf_counter()-started,
                  optimizer_updates=0, ema_updates=0, DEV=False, TEST=False, AP=False)
    with (args.output_dir / "COMBINED_SAMPLING_BACKWARD_PROBE_PRIVATE.json").open(
            "x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=result["status"], attempt=args.attempt,
                          grid_cpu_backward_calls=result["grid_sample"]["cpu_backward_calls"],
                          resize_cpu_backward_calls=result["bilinear_interpolate"]["cpu_backward_calls"],
                          replay_elapsed_seconds=result["replay_elapsed_seconds"],
                          optimizer_updates=0, ema_updates=0)), flush=True)


if __name__ == "__main__":
    main()
