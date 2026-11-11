"""Bounded stock GB10 CUDA operator probe, with no model or optimizer."""
import argparse
import csv
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys

GIB = 1024**3
MIB = 1024**2
HOST_FLOOR = 40 * GIB
CUDA_PROBE_FLOOR = GIB
ALLOCATOR_CAP = 256 * MIB


def host_available():
    rows = [row.split() for row in Path("/proc/meminfo").read_text().splitlines() if row.startswith("MemAvailable:")]
    row, = rows
    assert row[2] == "kB"
    return int(row[1]) * 1024


def competition(lease_paths, allowed_services):
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name",
                             "--format=csv,noheader,nounits"], check=True, timeout=8,
                            capture_output=True, text=True)
    external = []
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if not row:
            continue
        pid, name = row
        if int(pid) != os.getpid():
            external.append(Path(name).name)
    lock_nodes = set()
    for path in lease_paths:
        st = path.stat()
        lock_nodes.add((os.major(st.st_dev), os.minor(st.st_dev), st.st_ino))
    assert len(lock_nodes) == len(lease_paths)
    held = set()
    for line in Path("/proc/locks").read_text().splitlines():
        for field in line.split():
            if field.count(":") != 2:
                continue
            a, b, inode = field.split(":")
            node = (int(a, 16), int(b, 16), int(inode))
            if node in lock_nodes:
                held.add(node)
    return dict(approved_existing_contexts=sum(name in allowed_services for name in external),
                other_compute_contexts=sum(name not in allowed_services for name in external),
                checked_project_lease_files=len(lock_nodes), held_project_leases=len(held))


def emit(result):
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


def skip_if_unsafe(result, state, available):
    result.update(competition=state, host_available_bytes=available)
    return available < HOST_FLOOR or state["other_compute_contexts"] or state["held_project_leases"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--lease-file", type=Path, action="append", required=True)
    parser.add_argument("--allow-existing-compute-basename", action="append", default=[])
    args = parser.parse_args(argv)
    lease_paths = tuple(args.lease_file)
    allowed_services = set(args.allow_existing_compute_basename)
    assert len(lease_paths) == len(set(lease_paths)) == 2
    assert all(path.is_absolute() and path.resolve(strict=True) == path and path.is_file() for path in lease_paths)
    assert all(name and Path(name).name == name for name in allowed_services)
    result = dict(status="GB10_STOCK_TORCH_OPERATOR_PROBE_SKIPPED", training_started=False,
                  model_forward_calls=0, optimizer_updates=0, allocator_cap_bytes=ALLOCATOR_CAP,
                  host_floor_bytes=HOST_FLOOR, probe_cuda_floor_bytes=CUDA_PROBE_FLOOR,
                  full_training_cuda_floor_bytes=12 * GIB, service_changed=False, lease_acquired=False)
    assert sys.flags.isolated == sys.flags.utf8_mode == 1 and sys.dont_write_bytecode
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    assert os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    assert version("torch") == "2.10.0+cu130" and version("rfdetr") == "1.12.0"
    if skip_if_unsafe(result, competition(lease_paths, allowed_services), host_available()):
        emit(result)
        return 0

    import torch
    import torch.nn.functional as functional

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert torch.get_num_threads() == torch.get_num_interop_threads() == 1
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1 and not torch.cuda.is_initialized()
    free, total = torch.cuda.mem_get_info(0)
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    arches = torch.cuda.get_arch_list()
    assert name == "NVIDIA GB10" and capability == (12, 1)
    result.update(cuda_free_bytes=free, cuda_total_bytes=total, torch=torch.__version__,
                  torch_build_cuda=torch.version.cuda, rfdetr=version("rfdetr"), device_name=name,
                  device_capability=list(capability), torch_arch_list=arches,
                  exact_sm121_listed="sm_121" in arches,
                  full_training_cuda_headroom_admitted=free >= 12 * GIB)
    if skip_if_unsafe(result, competition(lease_paths, allowed_services), host_available()) or free < CUDA_PROBE_FLOOR:
        emit(result)
        return 0

    torch.cuda.set_per_process_memory_fraction(ALLOCATOR_CAP / total, 0)
    torch.cuda.reset_peak_memory_stats(0)
    assert torch.cuda.is_bf16_supported(including_emulation=False)
    result["native_bf16"] = True
    device = torch.device("cuda:0")
    outcomes = {}

    def bf16_gemm():
        x = (torch.arange(1024, dtype=torch.float32).reshape(32, 32).remainder(7) - 3) / 8
        y = (torch.arange(1024, dtype=torch.float32).reshape(32, 32).remainder(5) - 2) / 8
        ref = x @ y
        a = x.to(device=device, dtype=torch.bfloat16)
        b = y.to(device=device, dtype=torch.bfloat16)
        got = a @ b
        assert got.device == device and got.dtype == torch.bfloat16 and got.shape == (32, 32)
        torch.cuda.synchronize()
        actual = got.float().cpu()
        error = (actual - ref).abs().max().item()
        assert torch.isfinite(actual).all() and error <= 0.01
        return dict(shape=list(got.shape), dtype=str(got.dtype), max_abs_error_vs_fp32_cpu=error)

    def grid_backward():
        x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4) / 8
        g = torch.tensor([[[[-0.4, 0.25], [0., 0.]], [[0.6, -0.4], [1.1, 0.3]]]], dtype=torch.float32)
        cpu_x, cpu_g = x.clone().requires_grad_(True), g.clone().requires_grad_(True)
        gpu_x, gpu_g = x.to(device).requires_grad_(True), g.to(device).requires_grad_(True)
        kw = dict(mode="bilinear", padding_mode="zeros", align_corners=False)
        expected = functional.grid_sample(cpu_x, cpu_g, **kw)
        expected.square().sum().backward()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            got = functional.grid_sample(gpu_x, gpu_g, **kw)
            loss = got.square().sum()
        loss.backward()
        torch.cuda.synchronize()
        assert got.device == gpu_x.grad.device == gpu_g.grad.device == device
        actuals = (got.detach().cpu(), gpu_x.grad.cpu(), gpu_g.grad.cpu())
        refs = (expected.detach(), cpu_x.grad, cpu_g.grad)
        errors = [(a - b).abs().max().item() for a, b in zip(actuals, refs, strict=True)]
        assert all(torch.isfinite(a).all() for a in actuals) and max(errors) <= 1e-5
        return dict(output_dtype=str(got.dtype), shape=list(got.shape), autocast="bfloat16",
                    forward_max_abs_error=errors[0], input_gradient_max_abs_error=errors[1],
                    grid_gradient_max_abs_error=errors[2], cpu_reference_dtype="torch.float32")

    for name, operation in (("bf16_gemm", bf16_gemm), ("grid_sample_forward_backward", grid_backward)):
        try:
            outcomes[name] = dict(passed=True, **operation())
        except Exception as error:
            outcomes[name] = dict(passed=False, failure_type=type(error).__name__, failure=str(error)[:500])

    result.update(status="GB10_STOCK_TORCH_OPERATOR_PROBE_COMPLETE", operators=outcomes,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(0), torch_threads=1)
    emit(result)
    return 0 if all(item["passed"] for item in outcomes.values()) else 3


if __name__ == "__main__":
    sys.exit(main())

