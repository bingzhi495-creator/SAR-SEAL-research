"""UNBOUND_DRAFT / SYNTHETIC_ONLY: four independent, hand-counted fixtures.

Only explicit --statistics and --oracle pure-module paths are imported.
No scientific files, model, renderer, GT file, qualification chain or Git is used.
This check does not invoke either module's self-test or exhaustive enumeration.
The fixtures check Boolean arithmetic, not SAR efficacy, AP or runtime binding.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys


def load_explicit_module(name, path):
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
        raise RuntimeError("independent check requires optimize=0 and -B")
    thread_keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be empty")
    if any(os.environ.get(key) != "1" for key in thread_keys):
        raise RuntimeError("all explicit host thread limits must be 1")
    import psutil
    psutil.Process().cpu_affinity([0, 1])
    if psutil.Process().cpu_affinity() != [0, 1]:
        raise RuntimeError("CPU affinity must equal [0, 1]")
    import numpy as np
    statistics = load_explicit_module("explicit_unbound_statistics", args.statistics)
    oracle = load_explicit_module("explicit_unbound_pure_oracle", args.oracle)
    checked = 0

    def require(condition, description):
        nonlocal checked
        checked += 1
        if not condition:
            raise RuntimeError("independent hand-count check failed: " + description)

    def equal(actual, expected, description):
        if isinstance(expected, float):
            require(isinstance(actual, (int, float)) and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-15), description)
        else:
            require(actual == expected, description)

    def raster(size, indices):
        out = np.zeros((1, size), dtype=np.bool_)
        out[0, list(indices)] = True
        return out

    def fixture(number, size, a, c, g, old, halo):
        old_mask = raster(size, old)
        return {"A": raster(size, a), "C": raster(size, c), "G": raster(size, g),
                "old_LOG": old_mask, "new_halo": raster(size, halo), "S": old_mask.copy(),
                "K": len(old), "image_id": 900 + number, "gt_id": 1000 + number,
                "query_id": 0, "candidate_independent_of_selection": True,
                "metadata": {"label": "SYNTHETIC_ONLY", "hand_fixture": number}}

    # Sets and all expectations below were independently hand-counted before
    # this script was written. No oracle ranking or action masks generate them.
    fixtures = [
        fixture(1, 4, (0,1,2,3), (1,3), (0,1), (0,1), (2,)),
        fixture(2, 5, (0,1,3), (0,1,2), (0,1,2), (0,2,3), (4,)),
        fixture(3, 3, (0,), (1,), (0,), (0,), (1,)),
        fixture(4, 3, (0,), (0,1), (0,1), (), (0,1)),
    ]
    snapshots = [{key: value.copy() for key, value in item.items() if isinstance(value, np.ndarray)}
                 for item in fixtures]
    branch_names = ("native", "unlimited_eligible_gt_oracle", "exact_k_gt_action_oracle",
                    "exact_k_fixed_candidate_oracle", "original_selection_fixed_candidate",
                    "at_most_k_fixed_candidate_repair_oracle")
    # Tuple: TP, FP, FN, selected/written, binary flips,
    #        repair_FN, repair_FP, add_FP, delete_TP, selected noops.
    expected_counts = [
        [(2,2,0,0,0,0,0,0,0,0), (2,1,0,3,1,0,1,0,0,2),
         (2,1,0,2,1,0,1,0,0,1), (2,1,0,2,1,0,1,0,0,1),
         (1,2,1,2,1,0,0,0,1,1), (2,1,0,1,1,0,1,0,0,0)],
        [(2,1,1,0,0,0,0,0,0,0), (3,0,0,4,2,1,1,0,0,2),
         (3,0,0,3,2,1,1,0,0,1), (3,0,0,3,2,1,1,0,0,1),
         (3,0,0,3,2,1,1,0,0,1), (3,0,0,2,2,1,1,0,0,0)],
        [(1,0,0,0,0,0,0,0,0,0), (1,0,0,2,0,0,0,0,0,2),
         (1,0,0,1,0,0,0,0,0,1), (1,1,0,1,1,0,0,1,0,0),
         (0,0,1,1,1,0,0,0,1,0), (1,0,0,0,0,0,0,0,0,0)],
        [(1,0,1,0,0,0,0,0,0,0), (2,0,0,2,1,1,0,0,0,1),
         (1,0,1,0,0,0,0,0,0,0), (1,0,1,0,0,0,0,0,0,0),
         (1,0,1,0,0,0,0,0,0,0), (1,0,1,0,0,0,0,0,0,0)],
    ]
    expected_ious = [(1/2,2/3,2/3,2/3,1/4,2/3), (1/2,1,1,1,1,1),
                     (1,1,1,1/2,0,1), (1/2,1,1/2,1/2,1/2,1/2)]
    rows = [statistics.diagnose_instance(oracle, **item) for item in fixtures]
    for number, row in enumerate(rows):
        for branch, values, iou in zip(branch_names, expected_counts[number], expected_ious[number]):
            tp, fp, fn, selected, flips, repair_fn, repair_fp, add_fp, delete_tp, noops = values
            expected = {"tp": tp, "fp": fp, "fn": fn, "union": tp+fp+fn,
                        "selected_count": selected, "written_count": selected, "flip_count": flips,
                        "repair_fn": repair_fn, "repair_fp": repair_fp, "repairs": repair_fn+repair_fp,
                        "add_fp": add_fp, "delete_tp": delete_tp, "harms": add_fp+delete_tp,
                        "noops": noops, "iou": iou}
            for key, value in expected.items():
                equal(row["controls"][branch][key], value, f"fixture {number+1} {branch} {key}")

    original, fixed, at_most = branch_names[4], branch_names[3], branch_names[5]
    equal(rows[0]["threshold_crossings"][original]["native"]["0.50"],
          {"upward": False, "downward": True}, "original falls below .50")
    for label in ("0.75", "0.90", "0.95"):
        equal(rows[1]["threshold_crossings"][original]["native"][label],
              {"upward": True, "downward": False}, "repair upward crossing " + label)
    equal(rows[2]["minimum_unavoidable_exact_K_harms"], 1, "forced harm lower bound")
    equal(rows[2]["actual_exact_K_oracle_harms"], 1, "forced harm actual count")
    equal(rows[2]["controls"][at_most]["selected_count"], 0, "at-most-K rejects every harmful action")
    equal(rows[2]["controls"][at_most]["iou"], 1, "at-most-K preserves perfect native output")
    for branch in branch_names[2:]:
        for key in ("selected_count", "written_count", "flip_count", "repairs", "harms", "noops"):
            equal(rows[3]["controls"][branch][key], 0, "K=0 " + branch + " " + key)
    equal(rows[3]["controls"][branch_names[1]]["iou"], 1, "unlimited GT remains a distinct K=0 condition")
    equal(rows[0]["regions"]["outside_Omega"]["native_fp"], 1, "outside native FP retained in region")
    equal(rows[0]["controls"][original]["fp"], 2, "original retains outside FP")
    equal(rows[0]["controls"][fixed]["fp"], 1, "fixed oracle cannot repair outside FP")
    for branch in branch_names:
        for key, value in rows[0]["controls"][branch]["selected_actions_by_region"]["outside_Omega"].items():
            equal(value, 0, "outside action count " + branch + " " + key)
    for region in ("old_LOG", "new_halo", "outside_Omega"):
        for coverage in ("native_error_coverage", "native_fp_coverage", "native_fn_coverage"):
            equal(rows[2]["regions"][region][coverage], None, "zero native errors make coverage undefined")

    population = statistics.diagnose_population(oracle,
        matched_instances=[fixtures[0], fixtures[1], fixtures[3]],
        unmatched_gt=[{"image_id": 905, "gt_id": 1005}],
        metadata={"label": "SYNTHETIC_ONLY", "independent_hand_population": "3+1"},
        locked_population={"matched_count": 3, "unmatched_gt_count": 1, "total_gt_count": 4})
    equal(population["population"], {"matched_count": 3, "unmatched_gt_count": 1, "total_gt_count": 4}, "3+1 identities")
    equal(population["matched_iou_denominator"], 3, "matched mean denominator")
    equal(population["threshold_total_GT_denominator"], 4, "full GT threshold denominator")
    equal(population["unmatched_gt"], [{"image_id": 905, "gt_id": 1005, "metadata": {}}], "unmatched remains identity only")
    summary = population["controls"][original]
    equal(summary["matched_iou_sum"], 7/4, "matched IoU sum")
    equal(summary["matched_iou_mean"], 7/12, "matched macro IoU")
    equal(summary["pooled_pixel_iou"], 5/9, "pooled pixel IoU")
    require(not math.isclose(summary["matched_iou_mean"], summary["pooled_pixel_iou"]), "macro differs from pooled IoU")
    for key, value in {"tp": 5, "fp": 2, "fn": 2, "union": 9, "selected_count": 5,
                       "written_count": 5, "flip_count": 3, "repairs": 2, "harms": 1, "noops": 2}.items():
        equal(summary["pixel_and_action_totals"][key], value, "original population " + key)
    threshold = summary["thresholds"]["0.50"]
    equal(threshold["matched_instances_at_or_above"], 2, ".50 matched count")
    equal(threshold["fraction_of_matched"], 2/3, ".50 matched denominator")
    equal(threshold["fraction_of_total_GT_without_crediting_unmatched"], 1/2, ".50 total GT denominator")
    equal(threshold["crossings"]["native"], {"upward": 0, "downward": 1}, "population .50 downward crossing")
    for label in ("0.75", "0.90", "0.95"):
        equal(summary["thresholds"][label]["matched_instances_at_or_above"], 1, label + " matched count")
        equal(summary["thresholds"][label]["fraction_of_matched"], 1/3, label + " matched denominator")
        equal(summary["thresholds"][label]["fraction_of_total_GT_without_crediting_unmatched"], 1/4, label + " total GT denominator")
        equal(summary["thresholds"][label]["crossings"]["native"], {"upward": 1, "downward": 0}, label + " population repair crossing")
    # Pixel-error totals and macro coverage have deliberately different weights.
    coverage_expected = {
        "old_LOG": (2, 2/5, 1/3, 1/2, 1/2),
        "new_halo": (2, 2/5, 1/2, 1/4, 1/2),
        "outside_Omega": (1, 1/5, 1/6, 1/4, 0),
    }
    for region, (errors, pooled, macro, fp_macro, fn_macro) in coverage_expected.items():
        value = population["regions"][region]
        equal(value["pixel_count_totals"]["native_errors"], errors, region + " errors")
        equal(value["total_pixel_error_coverage"], pooled, region + " pixel error coverage")
        for field, denominator, expected_mean in (("native_error_coverage", 3, macro),
                ("native_fp_coverage", 2, fp_macro), ("native_fn_coverage", 2, fn_macro)):
            mean = value["instance_coverage_means"][field]
            equal(mean["defined_instance_count"], denominator, region + " " + field + " denominator")
            equal(mean["mean_over_defined_instances"], expected_mean, region + " " + field + " mean")
    equal(population["controls"]["native"]["pixel_and_action_totals"]["fp"] +
          population["controls"]["native"]["pixel_and_action_totals"]["fn"], 5, "native error coverage denominator")
    for fixture_input, saved in zip(fixtures, snapshots):
        for key, before in saved.items():
            require(np.array_equal(fixture_input[key], before), "input immutability " + key)
    require("torch" not in sys.modules, "Torch was not imported")
    print(json.dumps({"status": "SYNTHETIC_ONLY_PASS", "check": "four independent hand-counted fixtures",
        "hand_fixture_count": 4, "checked_assertions": checked,
        "matched_count": 3, "unmatched_gt_count": 1, "total_gt_count": 4,
        "original_matched_iou_mean": summary["matched_iou_mean"],
        "original_pooled_pixel_iou": summary["pooled_pixel_iou"],
        "original_below_0_50_from_native": threshold["crossings"]["native"]["downward"],
        "original_population_action_totals": summary["pixel_and_action_totals"],
        "region_coverage": {name: population["regions"][name] for name in coverage_expected},
        "all_branch_hand_counts_checked": True, "forced_harm_and_at_most_K_checked": True,
        "K_zero_checked": True, "outside_error_retention_checked": True,
        "input_arrays_unchanged": True, "self_tests_and_enumeration_invoked": False,
        "python": sys.version.split()[0], "numpy": np.__version__,
        "cpu_affinity": psutil.Process().cpu_affinity(),
        "thread_environment": {key: os.environ[key] for key in thread_keys},
        "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"], "torch_imported": False,
        "scientific_files_loaded": False, "scientific_execution_qualified": False,
        "AP_or_efficacy_claim": False, "runtime_binding_certified": False}, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
