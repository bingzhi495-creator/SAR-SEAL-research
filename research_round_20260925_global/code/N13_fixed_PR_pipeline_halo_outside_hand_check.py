"""One independent synthetic halo/outside connection check; no scientific input.

All four unbound modules must be supplied explicitly on the command line. This
checks one literal eight-pixel case and one rejection. It does not qualify a
scientific runtime, source, renderer, width2, weights, pairing, or endpoint.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from types import SimpleNamespace


STATUS = "UNBOUND_DRAFT_SYNTHETIC_ONLY"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hand_check(args):
    if not __debug__:
        raise RuntimeError("synthetic checks require normal Python mode")
    thread_keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    for key in thread_keys:
        if os.environ.get(key) != "1":
            raise RuntimeError("thread environment must equal 1: " + key)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or not sys.dont_write_bytecode:
        raise RuntimeError("CUDA must be empty and bytecode disabled")
    import psutil
    process = psutil.Process()
    process.cpu_affinity([0, 1])
    if process.cpu_affinity() != [0, 1]:
        raise RuntimeError("CPU affinity must be [0,1]")
    import torch
    import numpy as np
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    pipeline = load_module("supplied_unbound_pipeline", args.pipeline)
    adapter = load_module("supplied_unbound_adapter", args.adapter)
    statistics = load_module("supplied_unbound_statistics", args.statistics)
    oracle = load_module("supplied_unbound_oracle", args.oracle)
    checks = []

    def check(condition, description):
        if not bool(condition):
            raise RuntimeError("independent hand check failed: " + description)
        checks.append(description)

    # Expected predictions and measurements below are independent literals.
    L = torch.tensor([[-1.0, 0.0, 1.0, 3.0], [-3.0, -0.25, -4.0, 4.0]], dtype=torch.float32)
    F = torch.zeros((256, 2, 4), dtype=torch.float32)
    weights = tuple(torch.zeros(shape, dtype=torch.float32) for shape in (
        (91, 257), (91,), (91, 92), (91,), (91, 92), (91,), (1, 92), (1,)))
    weights[-1].fill_(5.0)
    old = torch.tensor([[True, True, True, False], [False, True, False, False]])
    halo = torch.tensor([[False, False, False, False], [True, False, False, False]])
    literal_A = torch.tensor([[False, False, True, True], [False, False, False, True]])
    expected_old = torch.tensor([[True, True, True, True], [False, True, False, True]])
    expected_C = torch.tensor([[True, True, True, True], [True, True, False, True]])
    outside = torch.tensor([[False, False, False, True], [False, False, True, True]])
    G = np.array([[False, True, True, False], [True, False, False, False]], dtype=np.bool_)
    metadata = {"input_kind": "SYNTHETIC_ONLY", "reference": "independent literal eight-pixel original LOG prediction"}
    tensors = (L, F, old, halo, expected_old, expected_C, literal_A, outside) + weights
    snapshots = tuple(item.clone() for item in tensors)
    G_snapshot = G.copy()
    metadata_snapshot = json.loads(json.dumps(metadata))
    calls = {"statistics": 0}

    def observed_statistics(supplied_oracle, **kwargs):
        calls["statistics"] += 1
        return statistics.diagnose_instance(supplied_oracle, **kwargs)

    statistics_spy = SimpleNamespace(diagnose_instance=observed_statistics)
    inputs = dict(weight_tensors=weights, F_original=F, native_L=L,
                  old_LOG=old, new_halo=halo, G=G, image_id=19, gt_id=37,
                  query_id=3, expected_original_LOG_binary=expected_old,
                  metadata=metadata)
    check(torch.equal(old, L.abs() <= 2), "old literal is exactly abs(L)<=2")
    check(torch.equal(L > 0, literal_A), "native support equals independent literal A")
    result = pipeline.diagnose_fixed_pr_pair(torch, adapter, statistics_spy, oracle, **inputs)
    check(torch.equal(result["candidate"], expected_C), "C equals independent literal {0,1,2,3,4,5,7}")
    check(torch.equal(result["original_LOG_binary"], expected_old), "original reconstruction equals independent literal {0,1,2,3,5,7}")
    check(not torch.equal(result["candidate"], result["original_LOG_binary"]), "full C and original reconstruction are different")
    check(not literal_A[1, 0].item() and result["candidate"][1, 0].item(), "halo index4 candidate actually flips native 0 to 1")
    check(not result["original_LOG_binary"][1, 0].item(), "original reconstruction preserves native false at halo index4")
    check((L[1, 2].item() + 5.0) > 0 and not literal_A[1, 2].item(), "outside index6 would turn positive under an erroneous residual write")
    check(not result["candidate"][1, 2].item(), "candidate preserves native false at outside index6")
    check(torch.equal(result["candidate"][outside], literal_A[outside]), "candidate preserves every nonempty outside-Omega position")
    check(torch.equal(result["original_LOG_binary"][~old], literal_A[~old]), "original reconstruction preserves all positions outside old_LOG")
    check(torch.equal(result["candidate_partitions"]["originalLOG"]["flat_indices"], torch.tensor([0, 1, 2, 5])), "old candidate indices are literal [0,1,2,5]")
    check(torch.equal(result["candidate_partitions"]["addedHalo"]["flat_indices"], torch.tensor([4])), "halo candidate index is literal [4]")
    diag = result["diagnostic"]
    check(diag["K"] == 4 and diag["Omega_count"] == 5, "K4 and Omega5")
    # Tuple order: TP, FP, FN, union, selected, written, flipped, repairs, harms, noops.
    hand_counts = {
        "native": (1, 2, 2, 5, 0, 0, 0, 0, 0, 0),
        "original_selection_fixed_candidate": (2, 4, 1, 7, 4, 4, 3, 1, 2, 1),
        "exact_k_fixed_candidate_oracle": (3, 3, 0, 6, 4, 4, 3, 2, 1, 1),
        "exact_k_gt_action_oracle": (3, 2, 0, 5, 4, 4, 2, 2, 0, 2),
        "unlimited_eligible_gt_oracle": (3, 2, 0, 5, 5, 5, 2, 2, 0, 3),
        "at_most_k_fixed_candidate_repair_oracle": (3, 2, 0, 5, 2, 2, 2, 2, 0, 0),
    }
    count_fields = ("tp", "fp", "fn", "union", "selected_count", "written_count",
                    "flip_count", "repairs", "harms", "noops")
    for name, values in hand_counts.items():
        control = diag["controls"][name]
        check(control["iou"] == values[0] / values[3], name + " independent hand IoU")
        for field, wanted in zip(count_fields, values):
            check(control[field] == wanted, name + " independent hand " + field)
    check(diag["minimum_unavoidable_exact_K_harms"] == 1, "minimum unavoidable exact-K harm is one")
    check(diag["actual_exact_K_oracle_harms"] == 1, "fixed-C exact-K oracle actually has one harm")
    check(diag["identity"] == {"image_id": 19, "gt_id": 37, "query_id": 3}, "explicit synthetic identity preserved")
    check(diag["metadata"] == metadata_snapshot, "metadata preserved without source certification")
    check(result["scope"]["scientific_execution_qualified"] is False and
          result["scope"]["actual_input_reference_and_pair_identity_certified"] is False and
          result["scope"]["candidate_independence_certified"] is False, "scientific/source/candidate provenance remains uncertified")
    before = calls["statistics"]
    try:
        pipeline.diagnose_fixed_pr_pair(torch, adapter, statistics_spy, oracle,
                                       **dict(inputs, expected_original_LOG_binary=expected_C))
    except ValueError as exc:
        check("original LOG reconstruction" in str(exc), "full C used as original reference rejected with reconstruction reason")
    else:
        raise RuntimeError("full C incorrectly accepted as original prediction reference")
    check(calls["statistics"] == before == 1, "full-C reference rejected before statistics; exactly one passing statistics call")
    for index, (item, snapshot) in enumerate(zip(tensors, snapshots)):
        check(torch.equal(item, snapshot), "tensor input unchanged " + str(index))
    check(np.array_equal(G, G_snapshot), "supplied G unchanged")
    check(metadata == metadata_snapshot, "input metadata unchanged")
    check(torch.get_num_threads() == 1 and torch.get_num_interop_threads() == 1
          and not torch.cuda.is_initialized(), "host one/one threads and CUDA uninitialized")
    return {
        "status": STATUS,
        "small_example_set": "one eight-pixel hand case and one full-C-reference rejection",
        "checks_passed": len(checks), "checks": checks,
        "elapsed_check_seconds": time.perf_counter() - started,
        "literal_native_A": [[0, 0, 1, 1], [0, 0, 0, 1]],
        "literal_expected_original_LOG_binary": [[1, 1, 1, 1], [0, 1, 0, 1]],
        "literal_expected_C": [[1, 1, 1, 1], [1, 1, 0, 1]],
        "independent_hand_counts": {name: dict(zip(count_fields, values)) for name, values in hand_counts.items()},
        "diagnostic": diag, "reconstruction_check": result["reconstruction_check"],
        "scope": result["scope"], "statistics_call_count": calls["statistics"],
        "reference_kind": "independent literal synthetic prediction; no scientific output read",
        "scientific_files_loaded": False,
        "runtime": {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
                    "psutil": psutil.__version__, "cpu_affinity": process.cpu_affinity(),
                    "intraop_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                    "cuda_initialized": torch.cuda.is_initialized(), "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
                    "dont_write_bytecode": sys.dont_write_bytecode,
                    "thread_environment": {key: os.environ[key] for key in thread_keys}},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", required=True, help="explicit ordinary unbound pipeline module path")
    parser.add_argument("--adapter", required=True, help="explicit ordinary unbound adapter module path")
    parser.add_argument("--statistics", required=True, help="explicit ordinary unbound statistics module path")
    parser.add_argument("--oracle", required=True, help="explicit ordinary unbound oracle module path")
    print(json.dumps(hand_check(parser.parse_args()), sort_keys=True, allow_nan=False))
