"""UNBOUND_DRAFT / SYNTHETIC_ONLY: fixed-PR conditional statistics.

The caller supplies Boolean original-raster masks, actual identities and a pure
oracle module. No model, GT file, endpoint, renderer, width2, dataset, scientific
SOURCE, qualification chain or Git is loaded. C must be generated independently
of final selection: the explicit caller declaration is preserved, not certified.
This concrete protocol fixes S=old_LOG, K=count(S), and Omega=old_LOG|new_halo.
Distance bands are supplied disjoint masks with source descriptions; definitions
remain unregistered. Four primary branches and at-most-K remain distinct.
Matched means use only matched instances. Unmatched GT has identity only; no
query/domain is invented. Threshold counts over total GT are not AP or recall.
Conditional IoU gaps are neither an AP upper bound nor a causal decomposition,
and privileged oracles are not deployable gains. All production checks raise
explicit exceptions; normal import loads standard-library modules only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys

STATUS = "UNBOUND_DRAFT_SYNTHETIC_ONLY"
THRESHOLDS = (0.50, 0.75, 0.90, 0.95)
BRANCHES = (
    "unlimited_eligible_gt_oracle",
    "exact_k_gt_action_oracle",
    "exact_k_fixed_candidate_oracle",
    "original_selection_fixed_candidate",
    "at_most_k_fixed_candidate_repair_oracle",
)
ACTIONS = ("repair_fn", "repair_fp", "noop", "add_fp", "delete_tp")
COUNT_KEYS = ("tp", "fp", "fn", "union", "selected_count", "written_count",
              "flip_count", "repair_fn", "repair_fp", "repairs", "add_fp",
              "delete_tp", "harms", "noops")
REGION_KEYS = ("pixels", "candidate_eligible_count", "native_tp", "native_fp",
               "native_fn", "native_errors") + ACTIONS
SCOPE = {
    "fixed_matched_instances_only": True,
    "candidate_independence_certified": False,
    "actual_pairing_and_distance_sources_registered": False,
    "scientific_execution_qualified": False,
    "ap_upper_bound_claim": False,
    "causal_decomposition_claim": False,
    "deployable_gain_claim": False,
}


def _np():
    import numpy as np
    return np


def _identity(value, name):
    np = _np()
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(name + " must be an actual nonnegative integer identity")
    if int(value) < 0:
        raise ValueError(name + " must be nonnegative")
    return int(value)


def _json_snapshot(value, name):
    if not isinstance(value, dict):
        raise TypeError(name + " must be a JSON dictionary")
    def check(item):
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise TypeError(name + " dictionary keys must be strings")
            for val in item.values():
                check(val)
        elif isinstance(item, (list, tuple)):
            for val in item:
                check(val)
        elif item is None or isinstance(item, (str, bool, int)):
            pass
        elif isinstance(item, float) and math.isfinite(item):
            pass
        else:
            raise TypeError(name + " must contain only finite JSON metadata")
    check(value)
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def _arrays(**arrays):
    np = _np()
    shape = None
    for name, arr in arrays.items():
        if not isinstance(arr, np.ndarray) or arr.dtype != np.dtype(np.bool_):
            raise TypeError(name + " must be an explicit NumPy Boolean array")
        if arr.ndim != 2 or min(arr.shape) <= 0:
            raise ValueError(name + " must have a nonempty two-axis original raster")
        if shape is None:
            shape = arr.shape
        elif arr.shape != shape:
            raise ValueError(name + " has a different raster shape")
    return shape


def _count(mask):
    return int(_np().count_nonzero(mask))


def _ratio(num, den):
    return num / den if den else None


def _actions(A, C, G, selected):
    return {
        "repair_fn": _count(selected & ~A & C & G),
        "repair_fp": _count(selected & A & ~C & ~G),
        "noop": _count(selected & (A == C)),
        "add_fp": _count(selected & ~A & C & ~G),
        "delete_tp": _count(selected & A & ~C & G),
    }


def _measure(A, C, G, selected, output):
    np = _np()
    counts = _actions(A, C, G, selected)
    tp = _count(output & G)
    fp = _count(output & ~G)
    fn = _count(~output & G)
    selected_count = _count(selected)
    flips = _count(output != A)
    if sum(counts.values()) != selected_count:
        raise RuntimeError("action classes do not partition selected positions")
    if flips != selected_count - counts["noop"]:
        raise RuntimeError("write/flip accounting is inconsistent")
    if np.any((output != A) & ~selected):
        raise RuntimeError("branch changed pixels outside its selection")
    return {
        "iou": tp / (tp + fp + fn),
        "tp": tp, "fp": fp, "fn": fn, "union": tp + fp + fn,
        "selected_count": selected_count,
        # Literal replacements include unchanged assignments. They are not flips.
        "written_count": selected_count, "flip_count": flips,
        "repair_fn": counts["repair_fn"], "repair_fp": counts["repair_fp"],
        "repairs": counts["repair_fn"] + counts["repair_fp"],
        "add_fp": counts["add_fp"], "delete_tp": counts["delete_tp"],
        "harms": counts["add_fp"] + counts["delete_tp"], "noops": counts["noop"],
    }


def _region(A, C, G, domain, Omega):
    eligible = domain & Omega
    total_fp, total_fn = _count(A & ~G), _count(~A & G)
    fp, fn = _count(domain & A & ~G), _count(domain & ~A & G)
    return dict({
        "pixels": _count(domain), "candidate_eligible_count": _count(eligible),
        "native_tp": _count(domain & A & G), "native_fp": fp,
        "native_fn": fn, "native_errors": fp + fn,
        "native_error_coverage": _ratio(fp + fn, total_fp + total_fn),
        "native_fp_coverage": _ratio(fp, total_fp),
        "native_fn_coverage": _ratio(fn, total_fn),
    }, **_actions(A, C, G, eligible))


def _difference(left, right):
    numerator = left["tp"] * right["union"] - right["tp"] * left["union"]
    if numerator < 0:
        raise RuntimeError("required conditional IoU ordering failed")
    return numerator / (left["union"] * right["union"])


def diagnose_instance(oracle, *, A, C, G, old_LOG, new_halo, S, K,
                      image_id, gt_id, query_id,
                      candidate_independent_of_selection, metadata,
                      distance_bands=None):
    """Return JSON-safe statistics without modifying any supplied input.

    distance_bands: optional {name: {mask: bool[H,W], source: nonempty str}}.
    Masks must be pairwise disjoint; coverage need not be exhaustive. These
    descriptions are caller declarations, not validation of a distance formula.
    """
    np = _np()
    shape = _arrays(A=A, C=C, G=G, old_LOG=old_LOG, new_halo=new_halo, S=S)
    identity = {"image_id": _identity(image_id, "image_id"),
                "gt_id": _identity(gt_id, "gt_id"),
                "query_id": _identity(query_id, "query_id")}
    meta = _json_snapshot(metadata, "metadata")
    if candidate_independent_of_selection is not True:
        raise ValueError("explicit independent-of-selection C declaration required")
    if not np.any(G):
        raise ValueError("matched instance G must be nonempty")
    if np.any(old_LOG & new_halo):
        raise ValueError("old_LOG and new_halo must be disjoint")
    if not np.array_equal(S, old_LOG):
        raise ValueError("this protocol fixes S exactly to old_LOG")
    K = _identity(K, "K")
    if K != _count(S):
        raise ValueError("this protocol fixes K=count(S)=count(old_LOG)")
    Omega = old_LOG | new_halo
    regions = {"old_LOG": old_LOG, "new_halo": new_halo, "outside_Omega": ~Omega}
    if distance_bands is None:
        distance_bands = {}
    if not isinstance(distance_bands, dict):
        raise TypeError("distance_bands must be an explicit dictionary")
    band_union = np.zeros(shape, dtype=np.bool_)
    bands = {}
    for name, item in distance_bands.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("distance-band name must be nonempty")
        if not isinstance(item, dict) or set(item) != {"mask", "source"}:
            raise ValueError("distance-band requires exactly mask and source")
        _arrays(A=A, band=item["mask"])
        if not isinstance(item["source"], str) or not item["source"].strip():
            raise ValueError("distance-band source description must be explicit")
        if np.any(band_union & item["mask"]):
            raise ValueError("distance bands must be pairwise disjoint")
        band_union |= item["mask"]
        bands[name] = {
            "source": item["source"], "source_registered": False,
            "statistics": _region(A, C, G, item["mask"], Omega),
            "by_region": {key: _region(A, C, G, mask & item["mask"], Omega)
                          for key, mask in regions.items()},
        }
    if not callable(getattr(oracle, "diagnose_fixed_candidate", None)):
        raise TypeError("caller must supply the pure Boolean oracle API")
    result = oracle.diagnose_fixed_candidate(A, C, G, Omega, S, K)
    controls = {}
    selections = {}
    for name in ("native",) + BRANCHES:
        branch = result["native"] if name == "native" else result["branches"][name]
        selected, output = branch["selected_mask"], branch["output_mask"]
        _arrays(A=A, selected=selected, output=output)
        candidate = G if name in BRANCHES[:2] else (A if name == "native" else C)
        expected = A.copy()
        expected[selected] = candidate[selected]
        if not np.array_equal(output, expected) or np.any(selected & ~Omega):
            raise RuntimeError("oracle branch output/eligibility mismatch")
        if name == "native" and np.any(selected):
            raise RuntimeError("native baseline must not select or write")
        if name == BRANCHES[0] and not np.array_equal(selected, Omega):
            raise RuntimeError("unlimited GT branch must select all Omega")
        if name in BRANCHES[1:4] and _count(selected) != K:
            raise RuntimeError("main exact-K branch did not preserve K")
        if name == BRANCHES[3] and not np.array_equal(selected, S):
            raise RuntimeError("original branch did not preserve S")
        if name == BRANCHES[4] and _count(selected) > K:
            raise RuntimeError("at-most-K branch exceeded K")
        measured = _measure(A, candidate, G, selected, output)
        measured["selected_actions_by_region"] = {
            key: dict({"selected_count": _count(selected & mask)},
                      **_actions(A, candidate, G, selected & mask))
            for key, mask in regions.items()}
        controls[name] = measured
        selections[name] = selected
    at_most = controls[BRANCHES[4]]
    if at_most["harms"] or at_most["noops"]:
        raise RuntimeError("at-most-K branch must select repairs only")
    gt_unlimited, gt_same, fixed_same, original = [controls[key] for key in BRANCHES[:4]]
    _difference(gt_same, controls["native"])
    gaps = {"budget": _difference(gt_unlimited, gt_same),
            "candidate": _difference(gt_same, fixed_same),
            "selection": _difference(fixed_same, original)}
    eligible_actions = _actions(A, C, G, Omega)
    unavoidable = max(0, K - eligible_actions["repair_fn"] - eligible_actions["repair_fp"]
                      - eligible_actions["noop"])
    if fixed_same["harms"] != unavoidable:
        raise RuntimeError("exact-K unavoidable harm accounting failed")
    crossing = {}
    for name, value in controls.items():
        crossing[name] = {}
        for ref_name, ref in (("native", controls["native"]),
                              ("original", original)):
            crossing[name][ref_name] = {
                format(t, ".2f"): {
                    "upward": ref["iou"] < t <= value["iou"],
                    "downward": value["iou"] < t <= ref["iou"],
                } for t in THRESHOLDS}
    return {
        "status": STATUS, "scope": dict(SCOPE), "identity": identity,
        "metadata": meta, "shape": list(shape),
        "candidate_independence_caller_declared": True,
        "S_equals_old_LOG": True, "K": K, "Omega_count": _count(Omega),
        "candidate_evaluated_points": _count(Omega),
        "distance_bands": bands, "distance_bands_uncovered_pixels": _count(~band_union),
        "regions": {key: _region(A, C, G, mask, Omega) for key, mask in regions.items()},
        "controls": controls, "threshold_crossings": crossing,
        "conditional_iou_gaps": gaps,
        "minimum_unavoidable_exact_K_harms": unavoidable,
        "actual_exact_K_oracle_harms": fixed_same["harms"],
        "at_most_K_is_separate_constraint_diagnosis": True,
    }


def _region_summary(items):
    sums = {key: sum(item[key] for item in items) for key in REGION_KEYS}
    fractions = {}
    for field in ("native_error_coverage", "native_fp_coverage", "native_fn_coverage"):
        defined = [item[field] for item in items if item[field] is not None]
        fractions[field] = {"defined_instance_count": len(defined),
                            "mean_over_defined_instances": _ratio(sum(defined), len(defined))}
    return {"pixel_count_totals": sums, "instance_coverage_means": fractions}


def diagnose_population(oracle, *, matched_instances, unmatched_gt,
                        metadata, locked_population=None):
    """Diagnose explicit input records and preserve actual GT/query identities.

    Each matched record contains diagnose_instance's keyword inputs. Unmatched
    GT records contain image_id, gt_id and optional JSON metadata ONLY. A lock,
    when supplied, must specify matched_count, unmatched_gt_count,total_gt_count.
    Global GT IDs must be unique; query IDs are unique within their actual image.
    This does not certify native .50 matching; ROOT registers actual provenance.
    """
    if not isinstance(matched_instances, (list, tuple)) or not isinstance(unmatched_gt, (list, tuple)):
        raise TypeError("explicit matched and unmatched record lists required")
    meta = _json_snapshot(metadata, "population metadata")
    gt_seen, query_seen = set(), set()
    rows, unmatched = [], []
    for record in matched_instances:
        if not isinstance(record, dict):
            raise TypeError("matched instance must be an explicit dictionary")
        row = diagnose_instance(oracle, **record)
        ids = row["identity"]
        gt_identity = ids["gt_id"]
        query_identity = (ids["image_id"], ids["query_id"])
        if gt_identity in gt_seen or query_identity in query_seen:
            raise ValueError("duplicate GT or image/query identity in fixed population")
        gt_seen.add(gt_identity)
        query_seen.add(query_identity)
        rows.append(row)
    for record in unmatched_gt:
        if not isinstance(record, dict) or set(record) not in (
            {"image_id", "gt_id"}, {"image_id", "gt_id", "metadata"}):
            raise ValueError("unmatched GT accepts actual identity only, plus optional metadata")
        item = {"image_id": _identity(record["image_id"], "image_id"),
                "gt_id": _identity(record["gt_id"], "gt_id"),
                "metadata": _json_snapshot(record.get("metadata", {}), "unmatched metadata")}
        if item["gt_id"] in gt_seen:
            raise ValueError("duplicate GT identity across matched/unmatched population")
        gt_seen.add(item["gt_id"])
        unmatched.append(item)
    population = {"matched_count": len(rows), "unmatched_gt_count": len(unmatched),
                  "total_gt_count": len(rows) + len(unmatched)}
    if locked_population is not None:
        if not isinstance(locked_population, dict) or set(locked_population) != set(population):
            raise ValueError("population lock requires exact three count fields")
        lock = {key: _identity(value, key) for key, value in locked_population.items()}
        if lock != population:
            raise ValueError("actual identities do not satisfy locked population counts")
    else:
        lock = None
    n, total = len(rows), population["total_gt_count"]
    band_schema = ({key: val["source"] for key, val in rows[0]["distance_bands"].items()}
                   if rows else {})
    for row in rows:
        if {key: val["source"] for key, val in row["distance_bands"].items()} != band_schema:
            raise ValueError("distance-band names and source descriptions must match across instances")
    summary = {}
    for name in ("native",) + BRANCHES:
        values = [row["controls"][name] for row in rows]
        sums = {key: sum(value[key] for value in values) for key in COUNT_KEYS}
        ious = [value["iou"] for value in values]
        thresholds = {}
        for t in THRESHOLDS:
            label = format(t, ".2f")
            passed = sum(value >= t for value in ious)
            thresholds[label] = {
                "matched_instances_at_or_above": passed,
                "fraction_of_matched": _ratio(passed, n),
                "fraction_of_total_GT_without_crediting_unmatched": _ratio(passed, total),
                "crossings": {ref: {
                    direction: sum(row["threshold_crossings"][name][ref][label][direction] for row in rows)
                    for direction in ("upward", "downward")}
                    for ref in ("native", "original")},
            }
        summary[name] = {
            "matched_iou_sum": sum(ious), "matched_iou_mean": _ratio(sum(ious), n),
            "matched_iou_min": min(ious) if ious else None,
            "matched_iou_max": max(ious) if ious else None,
            "pooled_pixel_iou": _ratio(sums["tp"], sums["union"]),
            "pixel_and_action_totals": sums, "thresholds": thresholds,
            "selected_actions_by_region": {
                region: {key: sum(value["selected_actions_by_region"][region][key] for value in values)
                         for key in ("selected_count",) + ACTIONS}
                for region in ("old_LOG", "new_halo", "outside_Omega")},
        }
    region_summaries = {region: _region_summary([row["regions"][region] for row in rows])
                        for region in ("old_LOG", "new_halo", "outside_Omega")}
    native_errors = summary["native"]["pixel_and_action_totals"]["fp"] + summary["native"]["pixel_and_action_totals"]["fn"]
    for region in region_summaries.values():
        region["total_pixel_error_coverage"] = _ratio(region["pixel_count_totals"]["native_errors"], native_errors)
    return {
        "status": STATUS, "scope": dict(SCOPE), "metadata": meta,
        "population": population, "locked_population": lock,
        "matched_iou_denominator": n, "threshold_total_GT_denominator": total,
        "unmatched_GT_has_no_assigned_query_or_domain": True,
        "threshold_counts_are_not_AP_or_recall": True,
        "unmatched_gt": unmatched, "instances": rows, "controls": summary,
        "regions": region_summaries,
        "distance_bands": {name: {
            "source": source, "source_registered": False,
            "statistics": _region_summary([row["distance_bands"][name]["statistics"] for row in rows]),
            "by_region": {region: _region_summary([row["distance_bands"][name]["by_region"][region] for row in rows])
                          for region in ("old_LOG", "new_halo", "outside_Omega")},
        } for name, source in band_schema.items()},
        "conditional_iou_gaps": {name: {
            "matched_sum": sum(row["conditional_iou_gaps"][name] for row in rows),
            "matched_mean": _ratio(sum(row["conditional_iou_gaps"][name] for row in rows), n),
        } for name in ("budget", "candidate", "selection")},
        "exact_K_harm_constraint": {
            "minimum_unavoidable_harms_total": sum(row["minimum_unavoidable_exact_K_harms"] for row in rows),
            "actual_oracle_harms_total": sum(row["actual_exact_K_oracle_harms"] for row in rows),
            "instances_with_unavoidable_harm": sum(row["minimum_unavoidable_exact_K_harms"] > 0 for row in rows),
        },
        "at_most_K_is_separate_constraint_diagnosis": True,
    }


def _self_test(oracle_path):
    if sys.flags.optimize:
        raise RuntimeError("synthetic self-test requires optimize=0; -O is explicitly refused")
    if not sys.dont_write_bytecode:
        raise RuntimeError("synthetic self-test must be invoked with -B")
    import os
    import psutil
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("synthetic self-test requires CUDA_VISIBLE_DEVICES empty")
    psutil.Process().cpu_affinity([0, 1])
    if psutil.Process().cpu_affinity() != [0, 1]:
        raise RuntimeError("synthetic self-test CPU affinity mismatch")
    spec = importlib.util.spec_from_file_location("explicit_unbound_oracle", oracle_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("explicit pure oracle could not be imported")
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    np = _np()
    def mask(values):
        return np.array([values], dtype=np.bool_)
    def record(i, A, C, G, old, halo):
        old_mask = mask(old)
        return {"A": mask(A), "C": mask(C), "G": mask(G), "old_LOG": old_mask,
                "new_halo": mask(halo), "S": old_mask.copy(), "K": _count(old_mask),
                "image_id": 100 + i, "gt_id": 200 + i, "query_id": 0,
                "candidate_independent_of_selection": True,
                "metadata": {"example": "synthetic hand-counted", "pairing_registered": False}}
    a = record(0, [1,1,0,0], [1,0,1,1], [1,0,1,0], [0,1,0,1], [0,0,1,0])
    b = record(1, [1,1,0,0], [0,0,1,1], [1,1,0,0], [1,0,1,0], [0,1,0,1])
    c = record(2, [0,0], [1,1], [1,0], [0,0], [1,1])
    for item in (a, b, c):
        item["distance_bands"] = {
            "explicit_first_half": {"mask": np.arange(item["A"].size).reshape(item["A"].shape) < item["A"].size // 2,
                                    "source": "synthetic explicit first-half mask; no distance formula"},
            "explicit_second_half": {"mask": np.arange(item["A"].size).reshape(item["A"].shape) >= item["A"].size // 2,
                                     "source": "synthetic explicit second-half mask; no distance formula"},
        }
    snapshots = [{key: val.copy() for key, val in item.items() if isinstance(val, np.ndarray)}
                 for item in (a, b, c)]
    unmatched = [{"image_id": 103, "gt_id": 203}]
    result = diagnose_population(oracle, matched_instances=[a, b, c], unmatched_gt=unmatched,
        metadata={"label": "SYNTHETIC_ONLY"},
        locked_population={"matched_count": 3, "unmatched_gt_count": 1, "total_gt_count": 4})
    def require(condition, description):
        if not condition:
            raise RuntimeError("synthetic check failed: " + description)
    rows = result["instances"]
    expected = {
        "native": (1/3, 1, 0),
        BRANCHES[0]: (1, 1, 1), BRANCHES[1]: (1, 1, 0),
        BRANCHES[2]: (1, 0.5, 0), BRANCHES[3]: (1/3, 1/3, 0),
        BRANCHES[4]: (1, 1, 0),
    }
    for name, values in expected.items():
        require(all(math.isclose(row["controls"][name]["iou"], val, rel_tol=0, abs_tol=1e-15)
                    for row, val in zip(rows, values)), "independent hand-counted branch IoU " + name)
        require(math.isclose(result["controls"][name]["matched_iou_mean"], sum(values)/3,
                             rel_tol=0, abs_tol=1e-15), "matched mean denominator")
    require(rows[0]["controls"][BRANCHES[2]]["repairs"] == 2, "two actual repairs")
    require(rows[1]["controls"][BRANCHES[2]]["harms"] == 2, "exact-K forced harm")
    require(rows[1]["minimum_unavoidable_exact_K_harms"] == 2, "harm lower bound")
    require(rows[1]["controls"][BRANCHES[4]]["selected_count"] == 0, "at-most-K separate rejection")
    require(all(rows[2]["controls"][name]["selected_count"] == 0 for name in BRANCHES[1:]), "K=0 exact/at-most selections")
    require(rows[2]["conditional_iou_gaps"]["budget"] == 1, "budget gap")
    require(rows[1]["conditional_iou_gaps"]["candidate"] == 0.5, "candidate gap")
    require(math.isclose(rows[0]["conditional_iou_gaps"]["selection"], 2/3), "selection gap")
    fixed_thresholds = result["controls"][BRANCHES[2]]["thresholds"]
    require(fixed_thresholds["0.50"]["crossings"]["native"] == {"upward": 1, "downward": 0}, "inclusive .50 crossing")
    for label in ("0.75", "0.90", "0.95"):
        require(fixed_thresholds[label]["crossings"]["native"] == {"upward": 1, "downward": 1}, "two-direction crossing")
    require(fixed_thresholds["0.50"]["fraction_of_matched"] == 2/3, "matched threshold denominator")
    require(fixed_thresholds["0.50"]["fraction_of_total_GT_without_crediting_unmatched"] == 0.5, "total GT threshold denominator")
    require(result["unmatched_gt"] == [{"image_id": 103, "gt_id": 203, "metadata": {}}], "unmatched identity preserved")
    require(sum(result["regions"][r]["pixel_count_totals"]["native_errors"] for r in result["regions"]) == 3, "regional error partition")
    require(result["regions"]["new_halo"]["pixel_count_totals"]["repair_fn"] == 2, "candidate halo repair count")
    for item, snapshot in zip((a, b, c), snapshots):
        for key, value in snapshot.items():
            require(np.array_equal(item[key], value), "input immutability " + key)
    invalid_count = 0
    def rejected(call):
        nonlocal invalid_count
        try:
            call()
        except (TypeError, ValueError):
            invalid_count += 1
        else:
            raise RuntimeError("invalid synthetic input was accepted")
    for change in (
        {"K": True}, {"K": 1}, {"S": np.zeros_like(a["S"])},
        {"new_halo": a["old_LOG"].copy()}, {"G": np.zeros_like(a["G"])},
        {"A": a["A"].astype(np.int64)}, {"candidate_independent_of_selection": False},
        {"metadata": {"bad": float("nan")}}, {"query_id": -1},
        {"distance_bands": {"missing_source": {"mask": a["A"]}}},
    ):
        bad = dict(a)
        bad.update(change)
        rejected(lambda bad=bad: diagnose_instance(oracle, **bad))
    rejected(lambda: diagnose_population(oracle, matched_instances=[a, a], unmatched_gt=[], metadata={}))
    rejected(lambda: diagnose_population(oracle, matched_instances=[a], unmatched_gt=[{"image_id": 100, "gt_id": 200}], metadata={}))
    rejected(lambda: diagnose_population(oracle, matched_instances=[a], unmatched_gt=[{"image_id": 101, "gt_id": 999, "query_id": 7}], metadata={}))
    rejected(lambda: diagnose_population(oracle, matched_instances=[a], unmatched_gt=[], metadata={},
        locked_population={"matched_count": 1, "unmatched_gt_count": 1, "total_gt_count": 2}))
    print(json.dumps({"status": "SYNTHETIC_ONLY_PASS", "examples": 3,
        "invalid_inputs_rejected": invalid_count, "independent_hand_counted_IoUs": expected,
        "matched_count": 3, "unmatched_gt_count": 1, "total_gt_count": 4,
        "population_and_bidirectional_crossings_checked": True,
        "three_conditional_gaps_and_forced_harm_checked": True,
        "K_zero_and_at_most_K_separation_checked": True,
        "input_arrays_unchanged": True, "scope": dict(SCOPE),
        "python": sys.version.split()[0], "numpy": np.__version__,
        "affinity": psutil.Process().cpu_affinity(),
        "thread_environment": {key: os.environ.get(key) for key in
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_imported": "torch" in sys.modules,
        "scientific_files_loaded": False,
    }, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--oracle", help="explicit path to the ordinary unbound pure oracle draft")
    args = parser.parse_args()
    if not args.self_test or not args.oracle:
        parser.error("only explicit --self-test --oracle PATH is supported as a CLI")
    _self_test(args.oracle)
