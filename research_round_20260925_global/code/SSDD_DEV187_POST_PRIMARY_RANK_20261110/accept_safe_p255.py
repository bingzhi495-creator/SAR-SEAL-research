"""Independently recalculate the closed P255 aggregate from one safe receipt."""
from hashlib import sha256
import json
from math import isclose, isfinite
from pathlib import Path
from statistics import fmean
import sys


DIGEST = "f51f68cf67ce36e4105999c40f842bdf9c0b8a7deebb291f4af6d55ecb6f488e"
ARMS = ("native", "center")
SEEDS = ("5201", "5202", "5203")
METRICS = ("AP", "AP50", "AP75", "AP90", "APs")
THRESHOLDS = ("0.75", "0.90")
CONTRASTS = {
    "own_scores_center_minus_native": ("native", "native", "center", "center"),
    "score_on_native_geometry": ("native", "native", "native", "center"),
    "score_on_center_geometry": ("center", "native", "center", "center"),
    "geometry_with_native_scores": ("native", "native", "center", "native"),
    "geometry_with_center_scores": ("native", "center", "center", "center"),
}
FIXED = {
    "status": "P255_P250_DEV187_POST_PRIMARY_RANK_COVERAGE_COMPLETE_V1",
    "evaluation_protocol": "ad2cb6182b37f9db1dc24908220dd5c923a1257c",
    "accepted_primary_score_sha256": "5c703c813647b6f7d2f3e21d653500a46d26db4b46b59022b2b642ebeb99d15d",
    "closed_plan_sha256": "fc7ac31ba4d121f4329a6536ebca2be4f22552c140cf9af47b8f5ae60d8a4256",
    "original_integer_gt_sha256": "5657600d6ea198f79bd5d557d2a1aadce5bec031d2ecf72a98716248d1183a55",
    "image_manifest_sha256": "18df007e15e3e838e607f178f14e7a26f2b37b992ee39bf091e0b74cf084fb13",
    "images": 187, "annotations": 380, "primary_score_unchanged": True,
    "within_image_matching_unchanged_across_donors": True,
    "score_blind_iou_thresholds": [.75, .90], "device": "cpu",
    "package_versions": {"numpy": "1.26.4", "pycocotools": "2.0.11", "scipy": "1.15.3"},
    "GT": "INTEGER_PRIMARY", "DEV": True, "official_TEST_opened": False,
    "model_forward_calls": 0, "training": False, "target_adaptation": False,
    "geometry_scope": "original mask contents and within-image native ordering at equal nonempty prefix counts",
    "donor_scope": "replace only cross-image score template, with exact per-image matches and ignore flags held",
    "coverage_scope": "one-to-one maximum cardinality using original-mask IoU on the identical selected pools",
}


def check(condition, why):
    if not condition:
        raise ValueError(why)


def keys(value, expected):
    check(type(value) is dict and set(value) == set(expected), f"Unexpected keys, expected {tuple(expected)}")


def finite(value, lower=-100., upper=100.):
    return type(value) in (int, float) and isfinite(value) and lower <= value <= upper


def equivalent(actual, expected):
    if type(expected) is dict:
        keys(actual, expected)
        for name, value in expected.items():
            equivalent(actual[name], value)
    elif type(expected) is float:
        check(finite(actual) and isclose(actual, expected, abs_tol=1e-12, rel_tol=0.),
              f"Numeric readback differs: {actual!r} vs {expected!r}")
    else:
        check(type(actual) is type(expected) and actual == expected,
              f"Exact readback differs: {actual!r} vs {expected!r}")


def count(value, upper):
    check(type(value) is int and 0 <= value <= upper, "Invalid count")
    return value


def unique(pairs):
    result = dict(pairs)
    check(len(result) == len(pairs), "Duplicate field in safe readback")
    return result


def reject_constant(value):
    raise ValueError("Nonfinite safe readback JSON value " + value)


def read_safe(path):
    check(path.is_absolute() and path.resolve(strict=True) == path, "Canonical safe readback required")
    with path.open("rb") as stream:
        raw = stream.read(512 * 1024 + 1)
    check(0 < len(raw) <= 512 * 1024 and sha256(raw).hexdigest() == DIGEST, "Pinned safe receipt SHA differs")
    return json.loads(raw, object_pairs_hook=unique, parse_constant=reject_constant)


def primary_summary(value):
    keys(value, ("by_seed", "mean_center_minus_native_points"))
    rows = value["by_seed"]
    check(type(rows) is list and len(rows) == len(SEEDS), "Three ordered primary pairs required")
    fields = (*METRICS, "AR100")
    paired = []
    for row, seed in zip(rows, SEEDS, strict=True):
        keys(row, ("seed", "center_minus_native_points"))
        equivalent(row["seed"], int(seed))
        deltas = row["center_minus_native_points"]
        keys(deltas, fields)
        check(all(finite(deltas[name]) for name in fields), "Finite embedded primary deltas required")
        paired.append(deltas)
    equivalent(value["mean_center_minus_native_points"], {
        name: fmean(pair[name] for pair in paired) for name in fields
    })
    return [pair["AP"] for pair in paired]


def summarize_seed(row):
    keys(row, ("cells", "contrasts_points", "matching_witness", "prefix_counts",
               "score_blind_coverage", "score_blind_maximum_center_minus_native"))
    prefix, cells, witnesses = (row[name] for name in ("prefix_counts", "cells", "matching_witness"))
    keys(prefix, (*ARMS, "retained_per_arm", "images_with_prefix", "images_without_prefix"))
    keys(cells, ARMS)
    keys(witnesses, ARMS)
    kept = count(prefix["retained_per_arm"], 18700)
    with_rows, without_rows = (count(prefix[name], 187) for name in ("images_with_prefix", "images_without_prefix"))
    check(with_rows + without_rows == 187 and with_rows <= kept <= 100 * with_rows, "Invalid retained population")
    for arm in ARMS:
        census = prefix[arm]
        keys(census, ("mapped", "nonempty", "mapped_empty", "discarded_nonempty_after_count_matching"))
        mapped, nonempty, empty, discarded = (count(census[key], 18700) for key in
                                               ("mapped", "nonempty", "mapped_empty", "discarded_nonempty_after_count_matching"))
        check(mapped == nonempty + empty and nonempty == kept + discarded, "Prefix counts do not reconcile")
        witness = witnesses[arm]
        keys(witness, ("exact_across_score_donors", "all_area_image_records", "all_area_nonnull_records",
                       "all_area_detection_rows", "all_area_dt_matches_cells", "all_area_dt_ignore_cells",
                       "all_area_ranges_records"))
        nonnull = count(witness["all_area_nonnull_records"], 187)
        check(nonnull >= with_rows, "Missing matching witness record")
        equivalent(witness, {
            "exact_across_score_donors": True, "all_area_image_records": 187, "all_area_nonnull_records": nonnull,
            "all_area_detection_rows": kept, "all_area_dt_matches_cells": 10 * kept,
            "all_area_dt_ignore_cells": 10 * kept, "all_area_ranges_records": 4 * 187,
        })
        keys(cells[arm], ARMS)
        for donor in ARMS:
            cell = cells[arm][donor]
            keys(cell, (*METRICS, "recall75", "recall90"))
            check(all(finite(value, 0., 1.) for value in cell.values()), "Invalid actual COCO result fraction")
        for recall in ("recall75", "recall90"):
            check(cells[arm]["native"][recall] == cells[arm]["center"][recall],
                  "Fixed-geometry donor recall differs")

    contrasts = {}
    for name, (old_geometry, old_donor, new_geometry, new_donor) in CONTRASTS.items():
        contrasts[name] = {metric: (cells[new_geometry][new_donor][metric]
                                    - cells[old_geometry][old_donor][metric]) * 100.
                           for metric in METRICS}
    equivalent(row["contrasts_points"], contrasts)

    coverage = row["score_blind_coverage"]
    keys(coverage, ("arms", "eligible_gt", "excluded_gt"))
    eligible = count(coverage["eligible_gt"], 380)
    keys(coverage["excluded_gt"], ("crowd", "empty", "ignore"))
    excluded = sum(count(value, 380) for value in coverage["excluded_gt"].values())
    check(eligible > 0 and eligible + excluded == 380, "Coverage GT accounting differs")
    keys(coverage["arms"], ARMS)
    numbers, same = {}, True
    for arm in ARMS:
        keys(coverage["arms"][arm], THRESHOLDS)
        numbers[arm] = {}
        for threshold in THRESHOLDS:
            values = coverage["arms"][arm][threshold]
            keys(values, ("any_candidate", "maximum_one_to_one", "maximum_recall"))
            any_hit = count(values["any_candidate"], eligible)
            maximum = count(values["maximum_one_to_one"], any_hit)
            equivalent(values["maximum_recall"], maximum / eligible)
            same &= not excluded and any_hit == maximum and isclose(
                cells[arm]["native"]["recall" + threshold[-2:]] * eligible, maximum, abs_tol=1e-10,
            )
            numbers[arm][threshold] = maximum
        check(numbers[arm]["0.90"] <= numbers[arm]["0.75"], "High-IoU matching is not nested in low-IoU matching")

    delta = {}
    for threshold in THRESHOLDS:
        gain = numbers["center"][threshold] - numbers["native"][threshold]
        delta[threshold] = {"count": gain, "points": 100. * gain / eligible}
    equivalent(row["score_blind_maximum_center_minus_native"], delta)
    return contrasts, delta, numbers, kept, same


def validate(value):
    keys(value, (*FIXED, "accepted_primary_paired_summary", "by_seed", "mean_prefix_contrasts_points",
                 "mean_score_blind_maximum_center_minus_native"))
    equivalent({name: value[name] for name in FIXED}, FIXED)
    primary = primary_summary(value["accepted_primary_paired_summary"])
    keys(value["by_seed"], SEEDS)
    summaries = {seed: summarize_seed(value["by_seed"][seed]) for seed in SEEDS}
    contrasts = {name: {metric: fmean(item[0][name][metric] for item in summaries.values()) for metric in METRICS}
                 for name in CONTRASTS}
    coverage = {threshold: {field: fmean(item[1][threshold][field] for item in summaries.values())
                             for field in ("count", "points")} for threshold in THRESHOLDS}
    equivalent(value["mean_prefix_contrasts_points"], contrasts)
    equivalent(value["mean_score_blind_maximum_center_minus_native"], coverage)
    prefix = [item[0]["own_scores_center_minus_native"]["AP"] for item in summaries.values()]
    return {
        "status": "P255_INDEPENDENT_SAFE_AGGREGATE_READBACK_VERIFIED_V1",
        "result_sha256": DIGEST,
        "primary_AP_delta_by_seed_points": primary,
        "primary_AP_direction_gate": all(item > 0 for item in primary) and fmean(primary) >= 1,
        "prefix_AP_delta_by_seed_points": prefix,
        "prefix_AP_positive_seeds": sum(item > 0 for item in prefix),
        "prefix_count_by_seed": {seed: summary[3] for seed, summary in summaries.items()},
        "mean_prefix_contrasts_points": contrasts,
        "score_blind_max_counts_by_seed": {seed: summary[2] for seed, summary in summaries.items()},
        "mean_score_blind_maximum_center_minus_native": coverage,
        "any_candidate_maximum_and_coco_recall_agree_all_cells": all(item[4] for item in summaries.values()),
        "raw_evaluator_arrays_or_target_assets_opened": False,
    }


if __name__ == "__main__":
    print(json.dumps(validate(read_safe(Path(sys.argv[1]))), sort_keys=True, allow_nan=False))
