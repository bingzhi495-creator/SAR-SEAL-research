"""One literal distance-band passthrough check; no scientific inputs.

Supply four ordinary unbound modules explicitly. The masks below are declared
synthetic bands, not a geometric distance formula or certified distance source.
This checks only the added optional interface and its reconstruction gate.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from types import SimpleNamespace


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hand_check(args):
    if not __debug__:
        raise RuntimeError("hand checks require normal Python mode")
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
            raise RuntimeError("distance passthrough hand check failed: " + description)
        checks.append(description)

    # Literals are indexed row-major, without a distance construction step.
    L = torch.tensor([[-1.0, 0.0, 1.0, 3.0], [-3.0, -0.25, -4.0, 4.0]], dtype=torch.float32)
    F = torch.zeros((256, 2, 4), dtype=torch.float32)
    weights = tuple(torch.zeros(shape, dtype=torch.float32) for shape in (
        (91, 257), (91,), (91, 92), (91,), (91, 92), (91,), (1, 92), (1,)))
    weights[-1].fill_(5.0)
    old = torch.tensor([[True, True, True, False], [False, True, False, False]])
    halo = torch.tensor([[False, False, False, False], [True, False, False, False]])
    original = torch.tensor([[True, True, True, True], [False, True, False, True]])
    literal_C = torch.tensor([[True, True, True, True], [True, True, False, True]])
    G = np.array([[False, True, True, False], [True, False, False, False]], dtype=np.bool_)
    bands = {
        "declared_A": {"mask": np.array([[True, True, False, False], [True, False, False, False]], dtype=np.bool_),
                       "source": "synthetic declared indices [0,1,4]; no geometric distance formula"},
        "declared_B": {"mask": np.array([[False, False, True, True], [False, False, True, False]], dtype=np.bool_),
                       "source": "synthetic declared indices [2,3,6]; no geometric distance formula"},
    }
    band_snapshots = {name: item["mask"].copy() for name, item in bands.items()}
    calls = []

    def observed_statistics(supplied_oracle, **kwargs):
        calls.append(kwargs.get("distance_bands"))
        return statistics.diagnose_instance(supplied_oracle, **kwargs)

    observed = SimpleNamespace(diagnose_instance=observed_statistics)
    inputs = dict(weight_tensors=weights, F_original=F, native_L=L, old_LOG=old,
                  new_halo=halo, G=G, image_id=19, gt_id=37, query_id=3,
                  expected_original_LOG_binary=original,
                  metadata={"input_kind": "SYNTHETIC_ONLY", "distance_sources": "explicit synthetic declarations"})
    with_bands = pipeline.diagnose_fixed_pr_pair(
        torch, adapter, observed, oracle, **inputs, distance_bands=bands)
    check(len(calls) == 1 and calls[0] is bands, "explicit distance_bands reaches real statistics unchanged")
    without_bands = pipeline.diagnose_fixed_pr_pair(torch, adapter, observed, oracle, **inputs)
    check(len(calls) == 2 and calls[1] is None, "omitted optional parameter reaches statistics as None")
    check(torch.equal(with_bands["candidate"], literal_C), "candidate matches independent literal")
    check(torch.equal(with_bands["original_LOG_binary"], original), "original LOG matches independent literal")
    check(torch.equal(with_bands["candidate"], without_bands["candidate"]), "distance bands do not change candidate")
    check(torch.equal(with_bands["original_LOG_binary"], without_bands["original_LOG_binary"]), "distance bands do not change original LOG reconstruction")
    check(with_bands["reconstruction_check"] == without_bands["reconstruction_check"], "reconstruction check is unchanged")
    check(with_bands["scope"] == without_bands["scope"], "pipeline scope remains unchanged")
    diag = with_bands["diagnostic"]
    bare_diag = without_bands["diagnostic"]
    check(bare_diag["distance_bands"] == {} and bare_diag["distance_bands_uncovered_pixels"] == 8,
          "omitting bands retains empty bands and eight uncovered pixels")
    check(diag["distance_bands_uncovered_pixels"] == 2, "literal uncovered indices [5,7] count two")
    # Count tuple: pixels, eligible, native TP/FP/FN/errors, repair FN/FP,
    # noop, add FP, delete TP. Native errors globally are FP2+FN2=4.
    fields = ("pixels", "candidate_eligible_count", "native_tp", "native_fp", "native_fn",
              "native_errors", "repair_fn", "repair_fp", "noop", "add_fp", "delete_tp")
    hand = {
        "declared_A": {"all": (3, 3, 0, 0, 2, 2, 2, 0, 0, 1, 0),
                       "old_LOG": (2, 2, 0, 0, 1, 1, 1, 0, 0, 1, 0),
                       "new_halo": (1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0),
                       "outside_Omega": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)},
        "declared_B": {"all": (3, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0),
                       "old_LOG": (1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0),
                       "new_halo": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                       "outside_Omega": (2, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0)},
    }
    for name, wanted in hand.items():
        actual = diag["distance_bands"][name]
        check(actual["source"] == bands[name]["source"] and actual["source_registered"] is False,
              name + " source declaration preserved without certification")
        for region, counts in wanted.items():
            measured = actual["statistics"] if region == "all" else actual["by_region"][region]
            check({key: measured[key] for key in fields} == dict(zip(fields, counts)),
                  name + "/" + region + " counts match independent literals")
            expected_ratios = (counts[5] / 4, counts[3] / 2, counts[4] / 2)
            check(tuple(measured[key] for key in ("native_error_coverage", "native_fp_coverage", "native_fn_coverage")) == expected_ratios,
                  name + "/" + region + " coverage matches independent 4/2/2 denominators")
    other_with = {key: value for key, value in diag.items() if key not in ("distance_bands", "distance_bands_uncovered_pixels")}
    other_without = {key: value for key, value in bare_diag.items() if key not in ("distance_bands", "distance_bands_uncovered_pixels")}
    check(other_with == other_without, "only distance-band diagnostic fields change")
    before = len(calls)
    try:
        pipeline.diagnose_fixed_pr_pair(
            torch, adapter, observed, oracle,
            **dict(inputs, expected_original_LOG_binary=literal_C), distance_bands=bands)
    except ValueError as exc:
        check("original LOG reconstruction" in str(exc), "wrong original reference is rejected for reconstruction")
    else:
        raise RuntimeError("wrong original prediction reference was accepted")
    check(len(calls) == before == 2, "wrong reference is refused before statistics receives bands or G")
    for name, snapshot in band_snapshots.items():
        check(np.array_equal(bands[name]["mask"], snapshot), name + " supplied mask unchanged")
    check(diag["scope"]["actual_pairing_and_distance_sources_registered"] is False,
          "distance and pairing identity remain unregistered")
    check(torch.get_num_threads() == 1 and torch.get_num_interop_threads() == 1
          and not torch.cuda.is_initialized(), "one/one host threads and CUDA uninitialized")
    return {
        "status": "UNBOUND_DRAFT_SYNTHETIC_ONLY", "checks_passed": len(checks), "checks": checks,
        "small_example_set": "one 2x4 literal passthrough case, omitted optional argument, one wrong-reference rejection",
        "elapsed_check_seconds": time.perf_counter() - started,
        "hand_count_field_order": list(fields), "independent_hand_counts": hand,
        "distance_bands": diag["distance_bands"], "uncovered_pixels": diag["distance_bands_uncovered_pixels"],
        "statistics_call_count": len(calls), "pipeline_scope": with_bands["scope"], "statistics_scope": diag["scope"],
        "scientific_files_loaded": False, "distance_formula_constructed": False,
        "runtime": {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
                    "cpu_affinity": process.cpu_affinity(), "intraop_threads": torch.get_num_threads(),
                    "interop_threads": torch.get_num_interop_threads(), "cuda_initialized": torch.cuda.is_initialized(),
                    "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
                    "dont_write_bytecode": sys.dont_write_bytecode,
                    "thread_environment": {key: os.environ[key] for key in thread_keys}},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--statistics", required=True)
    parser.add_argument("--oracle", required=True)
    print(json.dumps(hand_check(parser.parse_args()), sort_keys=True, allow_nan=False))
