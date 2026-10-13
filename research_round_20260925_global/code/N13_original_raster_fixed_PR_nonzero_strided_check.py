"""UNBOUND, synthetic-only check of a supplied original-raster PR draft.

This targeted CPU check imports only the explicit --adapter path and host
libraries. It never reads a scientific source, GT, checkpoint, cache, receipt,
SPEC, Q, or qualification chain. Its reference samples each pixel explicitly
with F[:, y, x] and evaluates an independently constructed nn.Module one point
at a time. It does not reuse the adapter's raster reshape/gather or PR function.
The check establishes no actual SAR provenance, renderer, or runtime qualification.

Run with python -B and an explicit --adapter path. No local path is embedded.
All checks raise explicit exceptions and remain active under Python optimization.
"""

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time


RTOL = 1e-6
ATOL = 1e-7
SEED = 2026101402
EXPECTED_SHAPES = (
    (91, 257), (91,), (91, 92), (91,),
    (91, 92), (91,), (1, 92), (1,),
)


def check(condition, message):
    if not bool(condition):
        raise RuntimeError("synthetic check failed: " + message)


def reference_module(torch, weights):
    class PointReference(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.LayerNorm(
                256, eps=1e-5, elementwise_affine=False)
            self.first = torch.nn.Linear(257, 91)
            self.second = torch.nn.Linear(92, 91)
            self.third = torch.nn.Linear(92, 91)
            self.last = torch.nn.Linear(92, 1)
            with torch.no_grad():
                for layer, offset in ((self.first, 0), (self.second, 2),
                                      (self.third, 4), (self.last, 6)):
                    layer.weight.copy_(weights[offset])
                    layer.bias.copy_(weights[offset + 1])

        def forward(self, feature, native_logit_scalar):
            coarse = native_logit_scalar.clamp(-8.0, 8.0)
            coarse = coarse.unsqueeze(0).unsqueeze(1) / 8.0
            hidden = self.norm(feature)
            hidden = torch.relu(self.first(torch.cat((hidden, coarse), dim=1)))
            hidden = torch.relu(self.second(torch.cat((hidden, coarse), dim=1)))
            hidden = torch.relu(self.third(torch.cat((hidden, coarse), dim=1)))
            return self.last(torch.cat((hidden, coarse), dim=1))[0, 0]

    return PointReference().eval()


def reference_partition(torch, module, F, L, domain, rule):
    """Independent row-major enumeration; no flat sampling/reshape/gather."""
    height, width = L.shape
    indices, residuals, refined, binaries, coordinates = [], [], [], [], []
    with torch.no_grad():
        for y in range(height):
            for x in range(width):
                if bool(domain[y, x].item()):
                    u = module(F[:, y, x].unsqueeze(0), L[y, x])
                    updated = L[y, x] + u
                    candidate = updated > 0 if rule == "strict_gt_zero" \
                        else updated >= 0
                    indices.append(y * width + x)
                    residuals.append(u)
                    refined.append(updated)
                    binaries.append(candidate)
                    coordinates.append((y, x))
    return {
        "indices": torch.tensor(indices, dtype=torch.int64, device="cpu"),
        "residual": torch.stack(residuals) if residuals else
            torch.empty(0, dtype=torch.float32),
        "refined": torch.stack(refined) if refined else
            torch.empty(0, dtype=torch.float32),
        "binary": torch.stack(binaries) if binaries else
            torch.empty(0, dtype=torch.bool),
        "coordinates": coordinates,
    }


def compare_partition(torch, actual, expected, name):
    check(torch.equal(actual["flat_indices"], expected["indices"]),
          name + " ascending row-major identities")
    maximum_errors = {}
    for actual_key, expected_key in (("raw_residual", "residual"),
                                     ("refined_logit", "refined")):
        observed = actual[actual_key]
        target = expected[expected_key]
        check(tuple(observed.shape) == tuple(target.shape),
              name + " " + actual_key + " shape")
        check(observed.dtype == torch.float32 and observed.device.type == "cpu",
              name + " " + actual_key + " FP32 CPU")
        check(torch.allclose(observed, target, rtol=RTOL, atol=ATOL),
              name + " " + actual_key + " independent per-point reference")
        maximum_errors[actual_key] = float((observed - target).abs().max().item()) \
            if observed.numel() else 0.0
    check(torch.equal(actual["binary_candidate"], expected["binary"]),
          name + " exact binary candidates")
    check(actual["chunk_count"] == (expected["indices"].numel() + 2047) // 2048,
          name + " fixed chunk count")
    return maximum_errors


def compare_case(torch, adapter, module, weights, F, L, old, added, rule, name):
    actual = adapter.fixed_pr_candidates(
        torch, weights, F, L, old, added,
        refined_binary_rule=rule, return_merged=True)
    old_reference = reference_partition(torch, module, F, L, old, rule)
    added_reference = reference_partition(torch, module, F, L, added, rule)
    errors = {
        "originalLOG": compare_partition(
            torch, actual["originalLOG"], old_reference, name + " originalLOG"),
        "addedHalo": compare_partition(
            torch, actual["addedHalo"], added_reference, name + " addedHalo"),
    }
    check(torch.equal(actual["native_support_L_gt_zero"], L > 0),
          name + " native support uses strict L>0")
    expected_merged = (L > 0).contiguous().clone()
    for reference in (old_reference, added_reference):
        for ordinal, (y, x) in enumerate(reference["coordinates"]):
            expected_merged[y, x] = reference["binary"][ordinal]
    check(torch.equal(actual["merged_C"], expected_merged),
          name + " exact merged_C from explicit y,x reference")
    check(torch.equal(actual["merged_C"][~(old | added)], (L > 0)[~(old | added)]),
          name + " outside-domain native preservation")
    return actual, {
        "name": name,
        "rule": rule,
        "old_points": int(old.sum().item()),
        "added_points": int(added.sum().item()),
        "maximum_absolute_errors": errors,
        "binary_and_merged_exact": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True,
                        help="explicit path of the UNBOUND adapter to check")
    arguments = parser.parse_args()
    check(sys.dont_write_bytecode, "python -B is required")
    started = time.perf_counter()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import psutil
    process = psutil.Process()
    process.cpu_affinity([0, 1])
    check(sorted(process.cpu_affinity()) == [0, 1], "actual CPU affinity [0,1]")
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    check(torch.get_num_threads() == 1 and torch.get_num_interop_threads() == 1,
          "actual one intra/inter-op thread")
    check(not torch.cuda.is_initialized(), "CUDA is not initialized")

    adapter_path = Path(arguments.adapter).resolve(strict=True)
    check(adapter_path.is_file() and adapter_path.suffix == ".py",
          "explicit adapter is an existing Python file")
    adapter_bytes = adapter_path.read_bytes()
    module_spec = importlib.util.spec_from_file_location(
        "unbound_fixed_pr_adapter_for_targeted_strided_check", str(adapter_path))
    check(module_spec is not None and module_spec.loader is not None,
          "adapter loader exists")
    adapter = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(adapter)
    check(adapter.STATUS == "UNBOUND_DRAFT_SYNTHETIC_ONLY",
          "adapter remains explicitly UNBOUND synthetic-only")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    weights = tuple(torch.randn(shape, generator=generator, dtype=torch.float32)
                    * 0.06 for shape in EXPECTED_SHAPES)
    weights[-1].fill_(0.17)
    check(len(weights) == 8 and sum(w.numel() for w in weights) == 40497,
          "eight independently specified shapes / 40497 parameters")
    check(all(bool((weight != 0).any().item()) for weight in weights),
          "all eight weight tensors are nonzero")
    reference = reference_module(torch, weights)
    check(sum(parameter.numel() for parameter in reference.parameters()) == 40497,
          "independent nn.Module parameter count")

    base_F = torch.randn((256, 7, 5), generator=generator, dtype=torch.float32)
    base_L = torch.linspace(-3.0, 3.0, 35, dtype=torch.float32).reshape(7, 5)
    base_old = base_L.abs() <= 2.0
    base_added_large = ~base_old
    base_added_small = base_added_large.clone()
    for y in range(7):
        for x in range(5):
            base_added_small[y, x] = bool(base_added_large[y, x].item()) \
                and (y * 5 + x) % 3 == 0
    F = base_F.transpose(1, 2)
    L = base_L.transpose(0, 1)
    old = base_old.transpose(0, 1)
    added_small = base_added_small.transpose(0, 1)
    added_large = base_added_large.transpose(0, 1)
    empty = torch.zeros((7, 5), dtype=torch.bool).transpose(0, 1)
    check(all(not tensor.is_contiguous()
              for tensor in (F, L, old, added_small, added_large, empty)),
          "F/L/all supplied domains are transposed, strided, noncontiguous")
    check(not bool((old & added_large).any().item()), "disjoint domains")
    check(int(added_small.sum().item()) < int(added_large.sum().item()),
          "added domain expansion is nontrivial")
    snapshots = [(tensor, tensor.detach().clone()) for tensor in
                 weights + (F, L, old, added_small, added_large, empty)]

    cases = []
    extension_checks = []
    for rule in ("strict_gt_zero", "inclusive_ge_zero"):
        small, small_report = compare_case(
            torch, adapter, reference, weights, F, L, old, added_small,
            rule, "strided_small_halo")
        large, large_report = compare_case(
            torch, adapter, reference, weights, F, L, old, added_large,
            rule, "strided_expanded_halo")
        cases.extend((small_report, large_report))
        for key in ("flat_indices", "raw_residual", "refined_logit",
                    "binary_candidate"):
            check(torch.equal(small["originalLOG"][key], large["originalLOG"][key]),
                  "old partition unchanged when added halo expands: " + key)
        check(small["originalLOG"]["chunk_count"] ==
              large["originalLOG"]["chunk_count"],
              "old chunk count unchanged under halo expansion")
        check(bool((large["originalLOG"]["raw_residual"] != 0).any().item()),
              "test uses nonzero PR residual on the old domain")
        check(float(large["originalLOG"]["raw_residual"].max().item() -
                    large["originalLOG"]["raw_residual"].min().item()) > 1e-5,
              "nonconstant residual exercises pixel-specific input dependence")
        extension_checks.append({"rule": rule, "old_values_bitwise_unchanged": True})
    for name, empty_old, empty_added in (
            ("both_domains_empty", empty, empty),
            ("old_domain_empty", empty, added_small),
            ("added_domain_empty", old, empty)):
        _, report = compare_case(
            torch, adapter, reference, weights, F, L, empty_old, empty_added,
            "strict_gt_zero", name)
        cases.append(report)
    for tensor, before in snapshots:
        check(torch.equal(tensor.detach(), before), "all supplied tensors unchanged")
    check(not torch.cuda.is_initialized(), "CUDA remains uninitialized")

    report = {
        "status": "PASS_SYNTHETIC_ONLY_NONZERO_STRIDED_REFERENCE",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "seed": SEED,
        "python_version": platform.python_version(),
        "host_torch_version": str(torch.__version__),
        "psutil_version": str(psutil.__version__),
        "documented_science_torch_version_not_qualified": "2.0.1+cu118",
        "device": "cpu",
        "CPU_affinity": process.cpu_affinity(),
        "intra_op_threads": torch.get_num_threads(),
        "inter_op_threads": torch.get_num_interop_threads(),
        "CUDA_initialized": torch.cuda.is_initialized(),
        "bytecode_disabled": bool(sys.dont_write_bytecode),
        "adapter_basename": adapter_path.name,
        "adapter_bytes": len(adapter_bytes),
        "adapter_sha256": hashlib.sha256(adapter_bytes).hexdigest(),
        "weight_tensors": 8,
        "parameter_scalars": 40497,
        "all_eight_weight_tensors_nonzero": True,
        "F_shape": list(F.shape),
        "F_stride": list(F.stride()),
        "L_shape": list(L.shape),
        "L_stride": list(L.stride()),
        "all_inputs_noncontiguous": True,
        "reference_sampling": "explicit F[:,y,x]; independent Module; one point at a time",
        "float_reference_rtol": RTOL,
        "float_reference_atol": ATOL,
        "cases": cases,
        "extension_checks": extension_checks,
        "all_supplied_tensors_unchanged": True,
        "check_uses_optimization_removable_asserts": False,
        "actual_SAR_or_renderer_or_source_mapping_qualified": False,
        "actual_science_runtime_qualified": False,
        "multi_chunk_noncontiguous_reference_tested_here": False,
        "large_raster_resource_envelope_established": False,
        "actual_exporter_binary_rule_established": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"status": "FAIL_SYNTHETIC_ONLY_NO_RETRY",
                          "error_type": type(error).__name__,
                          "error": str(error),
                          "actual_science_qualified": False}, sort_keys=True),
              file=sys.stderr)
        raise
