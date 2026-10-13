"""UNBOUND_DRAFT / SYNTHETIC_ONLY: one hand-counted 2+1 population.

Imports only the explicit --statistics and --oracle pure-module paths. No old
self-test, exhaustive oracle check, scientific file or real dataset is invoked.
"""
import argparse
import copy
import importlib.util
import json
import math
import os
import sys


def load_explicit(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("explicit pure module cannot be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", required=True)
    parser.add_argument("--oracle", required=True)
    args = parser.parse_args()
    if sys.flags.optimize or not sys.dont_write_bytecode:
        raise RuntimeError("hand check requires optimize=0 and -B")
    thread_keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or any(os.environ.get(key) != "1" for key in thread_keys):
        raise RuntimeError("explicit CPU-only single-thread limits are required")
    import psutil
    psutil.Process().cpu_affinity([0, 1])
    if psutil.Process().cpu_affinity() != [0, 1]:
        raise RuntimeError("CPU affinity must equal [0, 1]")
    import numpy as np
    stats = load_explicit("unbound_explicit_statistics", args.statistics)
    oracle = load_explicit("unbound_explicit_oracle", args.oracle)
    checks = 0

    def require(condition, label):
        nonlocal checks
        checks += 1
        if not condition:
            raise RuntimeError("precomputed population hand check failed: " + label)

    def equal(actual, expected, label):
        if isinstance(expected, float):
            require(isinstance(actual, (int, float)) and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-15), label)
        else:
            require(actual == expected, label)

    def raster(size, indices):
        mask = np.zeros((1, size), dtype=np.bool_)
        mask[0, list(indices)] = True
        return mask

    def fixture(number, size, a, c, g, old, halo):
        old_mask = raster(size, old)
        first = raster(size, range(size // 2))
        return {"A": raster(size, a), "C": raster(size, c), "G": raster(size, g),
                "old_LOG": old_mask, "new_halo": raster(size, halo), "S": old_mask.copy(),
                "K": len(old), "image_id": 1100 + number, "gt_id": 1200 + number,
                "query_id": 0, "candidate_independent_of_selection": True,
                "metadata": {"label": "SYNTHETIC_ONLY", "hand_fixture": number},
                "distance_bands": {
                    "first_half": {"mask": first, "source": "synthetic explicit first half; not a real distance"},
                    "second_half": {"mask": ~first, "source": "synthetic explicit second half; not a real distance"}}}

    # Hand sets: pair1 native IoU=1/4, original=2/3; pair2 native=1/2,
    # original=3/4. Original pooled TP/FP/FN=5/1/1, unlike macro 17/24.
    fixtures = [fixture(1, 4, (0, 1, 3), (0, 2, 3), (0, 2), (1, 2), (3,)),
                fixture(2, 6, (0, 1), (0, 1, 2), (0, 1, 2, 3), (2,), (3, 4))]
    unmatched = [{"image_id": 1103, "gt_id": 1203}]
    metadata = {"label": "SYNTHETIC_ONLY", "hand_population": "2+1"}
    lock = {"matched_count": 2, "unmatched_gt_count": 1, "total_gt_count": 3}
    rows = [stats.diagnose_instance(oracle, **item) for item in fixtures]
    row_snapshot = json.dumps(rows, sort_keys=True, allow_nan=False)
    from_arrays = stats.diagnose_population(oracle, matched_instances=fixtures,
        unmatched_gt=unmatched, metadata=metadata, locked_population=lock)
    original_instance = stats.diagnose_instance

    def forbidden_recompute(*unused_args, **unused_kwargs):
        raise RuntimeError("precomputed rows must not call diagnose_instance")

    stats.diagnose_instance = forbidden_recompute
    try:
        from_rows = stats.diagnose_population(oracle, matched_diagnostics=rows,
            unmatched_gt=unmatched, metadata=metadata, locked_population=lock)
    finally:
        stats.diagnose_instance = original_instance
    equal(from_rows, from_arrays, "precomputed and array paths have identical full JSON")
    equal(json.dumps(rows, sort_keys=True, allow_nan=False), row_snapshot, "input JSON rows unchanged")
    original = from_rows["controls"]["original_selection_fixed_candidate"]
    equal(original["matched_iou_mean"], 17 / 24, "independent macro IoU")
    equal(original["pooled_pixel_iou"], 5 / 7, "independent pooled IoU")
    equal(from_rows["matched_iou_denominator"], 2, "matched denominator")
    equal(from_rows["threshold_total_GT_denominator"], 3, "full GT denominator")
    for key, value in {"tp": 5, "fp": 1, "fn": 1, "union": 7,
                       "selected_count": 3, "written_count": 3, "flip_count": 3,
                       "repairs": 3, "harms": 0, "noops": 0}.items():
        equal(original["pixel_and_action_totals"][key], value, "original hand total " + key)
    for label, count in (("0.50", 2), ("0.75", 1), ("0.90", 0), ("0.95", 0)):
        threshold = original["thresholds"][label]
        equal(threshold["matched_instances_at_or_above"], count, "threshold count " + label)
        equal(threshold["fraction_of_matched"], count / 2, "matched fraction " + label)
        equal(threshold["fraction_of_total_GT_without_crediting_unmatched"], count / 3, "total GT fraction " + label)
    for region, errors, total, macro in (("old_LOG", 3, 3 / 5, 7 / 12),
            ("new_halo", 2, 2 / 5, 5 / 12), ("outside_Omega", 0, 0.0, 0.0)):
        values = from_rows["regions"][region]
        equal(values["pixel_count_totals"]["native_errors"], errors, "region errors " + region)
        equal(values["total_pixel_error_coverage"], total, "region pixel coverage " + region)
        equal(values["instance_coverage_means"]["native_error_coverage"]["mean_over_defined_instances"], macro, "region macro coverage " + region)
    for band, error_total, macro in (("first_half", 2, 5 / 12), ("second_half", 3, 7 / 12)):
        values = from_rows["distance_bands"][band]
        equal(values["statistics"]["pixel_count_totals"]["native_errors"], error_total, "band error total " + band)
        equal(values["statistics"]["instance_coverage_means"]["native_error_coverage"]["mean_over_defined_instances"], macro, "band macro " + band)
        equal(values["source_registered"], False, "band remains unregistered " + band)
    for name in ("candidate_independence_certified", "scientific_execution_qualified", "ap_upper_bound_claim", "deployable_gain_claim"):
        equal(from_rows["scope"][name], False, "scope remains false " + name)
    rejected_labels = []

    def reject(call, label):
        try:
            call()
        except (TypeError, ValueError):
            require(True, "public input refusal " + label)
            rejected_labels.append(label)
        except Exception as error:
            raise RuntimeError("precomputed validation tracer missing: " + label) from error
        else:
            raise RuntimeError("precomputed validation tracer missing: " + label)

    def population_with(provided_rows, provided_lock=lock):
        return stats.diagnose_population(oracle, matched_diagnostics=provided_rows,
            unmatched_gt=unmatched, metadata=metadata, locked_population=provided_lock)

    reject(lambda: stats.diagnose_population(oracle, matched_instances=fixtures,
        matched_diagnostics=rows, unmatched_gt=unmatched, metadata=metadata), "ambiguous dual path")
    bad = copy.deepcopy(rows)
    bad[1]["identity"]["gt_id"] = bad[0]["identity"]["gt_id"]
    reject(lambda: population_with(bad), "duplicate GT identity")
    bad = copy.deepcopy(rows)
    bad[1]["identity"]["image_id"] = bad[0]["identity"]["image_id"]
    reject(lambda: population_with(bad), "duplicate image query identity")
    bad = copy.deepcopy(rows)
    del bad[0]["controls"]["native"]
    reject(lambda: population_with(bad), "missing required branch")
    bad = copy.deepcopy(rows)
    bad[0]["controls"]["native"]["iou"] = float("nan")
    reject(lambda: population_with(bad), "nonfinite JSON value")
    reject(lambda: population_with(rows, {"matched_count": 1,
        "unmatched_gt_count": 1, "total_gt_count": 2}), "locked population mismatch")
    bad = copy.deepcopy(rows)
    bad[0]["controls"]["native"]["tp"] = -1
    reject(lambda: population_with(bad), "negative basic count")
    bad = copy.deepcopy(rows)
    bad[0]["identity"]["query_id"] = True
    reject(lambda: population_with(bad), "Boolean identity")
    bad = copy.deepcopy(rows)
    bad[0]["scope"]["scientific_execution_qualified"] = True
    reject(lambda: population_with(bad), "caller forged qualification")
    bad = copy.deepcopy(rows)
    del bad[1]["distance_bands"]["first_half"]
    reject(lambda: population_with(bad), "inconsistent distance band keys")
    equal(json.dumps(rows, sort_keys=True, allow_nan=False), row_snapshot, "rejection paths leave source rows unchanged")
    require("torch" not in sys.modules, "Torch is not imported")
    print(json.dumps({"status": "SYNTHETIC_ONLY_PASS", "check": "precomputed population one independent hand fixture",
        "checked_assertions": checks, "population": lock, "array_and_precomputed_full_JSON_equal": True,
        "precomputed_diagnose_instance_calls": 0, "original_macro_iou": original["matched_iou_mean"],
        "original_pooled_iou": original["pooled_pixel_iou"], "original_pixel_action_totals": original["pixel_and_action_totals"],
        "invalid_input_rejections": rejected_labels,
        "self_tests_or_enumeration_invoked": False, "scientific_files_loaded": False,
        "scientific_execution_qualified": False, "candidate_independence_certified": False,
        "AP_or_efficacy_claim": False, "runtime_binding_certified": False,
        "python": sys.version.split()[0], "numpy": np.__version__, "cpu_affinity": psutil.Process().cpu_affinity(),
        "thread_environment": {key: os.environ[key] for key in thread_keys}, "CUDA_VISIBLE_DEVICES": "", "torch_imported": False},
        sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
