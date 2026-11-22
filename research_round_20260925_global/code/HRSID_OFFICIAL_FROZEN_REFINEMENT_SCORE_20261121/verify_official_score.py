"""Independently reconcile the accepted HRSID official score aggregate."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re


SEEDS = ("5217", "5218", "5219")
ARMS = ("SF", "SU=F", "SU", "LW=S")
CONTRASTS = (
    ("SU", "N"),
    ("SU=F", "SF"),
    ("SU", "SU=F"),
    ("SU", "LW=S"),
)
POPULATION = {
    "official": (1962, 5922, 1, 0, 187796),
    "empty_raster_sensitivity": (1962, 5918, 1, 0, 187796),
    "image_identity_sensitivity": (1959, 5891, 1, 0, 187498),
}
COUNTS = (
    "images", "annotations", "images_without_annotations",
    "images_without_predictions", "mapped_predictions",
)
METRICS = (
    "AP", "AP50", "AP75", "AP90", "APs", "APm", "APl",
    "AR1", "AR10", "AR100", "ARs", "ARm", "ARl",
)
BASELINE_SHA = "0b5759262f96214457064735b0680a7c2f020b1503223613dcf56cc7072eed05"
REFINEMENT_SCORE_SHA = "f047034705e6048230764d7f5712419546a387cd1d3c3a92ec00c57561dea385"
FIT_SHA = "476b0f3fa748d1d3fd2393bdeb1af2d83e1532995d58b12625d57222e3f2638a"
STATUS = "HRSID_OFFICIAL_TEST_FROZEN_REFINEMENT_SCORE_COMPLETE"
READOUT_STATUS = "HRSID_OFFICIAL_TEST_FROZEN_REFINEMENT_READOUT_INDEPENDENT_ACCEPTED"
READOUT_ASSERTIONS = (
    "exact_candidate_identity", "exact_native_per_image", "exact_per_candidate_budgets",
    "actual_XOR_area_RLE_verified",
)


def check(ok, reason):
    if not ok:
        raise ValueError(reason)


def unique_values(pairs):
    out = {}
    for key, value in pairs:
        check(key not in out, f"Duplicate object key: {key}")
        out[key] = value
    return out


def is_digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def load_confirmed(path, expected_digest):
    check(is_digest(expected_digest), f"Expected SHA-256 for {path.name}")
    payload = path.read_bytes()
    check(sha256(payload).hexdigest() == expected_digest, f"Digest mismatch: {path.name}")
    return json.loads(payload, object_pairs_hook=unique_values)


def check_population(populations):
    check(set(populations) == set(POPULATION), "All three original test cohorts")
    for view, values in populations.items():
        check(
            len(values) == len(COUNTS) + 2
            and all(type(values.get(key)) is int for key in COUNTS)
            and tuple(values[key] for key in COUNTS) == POPULATION[view],
            f"Original population mismatch: {view}",
        )
        metrics = values["metrics"]
        check(set(metrics) == set(METRICS), f"Missing or additional COCO metrics: {view}")
        check(
            all(type(v) in (int, float) and (v == -1 or 0 <= v <= 1) for v in metrics.values()),
            f"Invalid COCO fraction or undefined sentinel: {view}",
        )
        check(is_digest(values.get("ordered_prediction_block_digests_sha256")), f"Block identity: {view}")
    digest_key = "ordered_prediction_block_digests_sha256"
    check(
        populations["official"][digest_key] == populations["empty_raster_sensitivity"][digest_key],
        "Dropping four empty GT rasters must preserve prediction bytes on every image",
    )


def check_expected(actual, expected, key):
    matches = (
        actual is None if expected is None else
        type(actual) in (int, float) and math.isclose(actual, expected, rel_tol=0, abs_tol=2e-12)
    )
    check(matches, f"Contrast arithmetic mismatch: {key}, got {actual}, expected {expected}")


def audit(result, baseline, readout, readout_sha):
    check(
        result.get("status") == STATUS
        and all(result.get(key) is True for key in ("historical_test_selection_exposure", "TEST", "AP"))
        and all(type(result.get(key)) is int and result[key] == 0
                for key in ("model_forward_calls", "optimizer_updates")),
        "Scoring-only historical held allocation",
    )
    check(result.get("metric_units") == "COCO fraction; -1 is undefined"
          and result.get("contrast_units") == "percentage points; null is undefined", "Original units")
    check(result.get("arms") == ["N", *ARMS] and result.get("seeds") == [int(s) for s in SEEDS],
          "Common N plus four arms on the three frozen point heads")
    check(result.get("refinement_score_source_sha256") == REFINEMENT_SCORE_SHA
          and result.get("native_score_sha256") == BASELINE_SHA, "Original scorer and frozen native result")
    baseline_fields = {
        "cohort_sha256": "cohorts_sha256",
        "gt_sha256": "gt_sha256",
        "native_predictions_sha256": "predictions_sha256",
        "endpoint_sha256": "endpoint_sha256",
        "runtime": "runtime_versions",
        "score_source_sha256": "score_source_sha256",
        "adapter_sha256": "adapter_sha256",
    }
    for score_key, baseline_key in baseline_fields.items():
        check(result.get(score_key) == baseline[baseline_key], f"Changed native identity: {score_key}")
    check(result.get("fit_accepted_sha256") == FIT_SHA and result.get("readout_sha256") == readout_sha,
          "Point-head fit and accepted independent readout")
    check(
        readout.get("status") == READOUT_STATUS
        and type(readout.get("original_exit")) is int and readout["original_exit"] == 0
        and readout.get("AP") is False and readout.get("GT_opened") is False and readout.get("TEST") is True
        and readout.get("images") == 1962 and readout.get("mapped_rows") == 187796
        and all(readout.get(key) is True for key in READOUT_ASSERTIONS),
        "Closed label-free native replay and actual edit-budget witness",
    )
    for key in ("fit_accepted_sha256", "native_predictions_sha256", "endpoint_sha256"):
        check(result[key] == readout.get(key), f"Mismatched readout identity: {key}")

    native, treatments, streams = result["native"], result["treatments"], result["streams"]
    check_population(native)
    check(native == baseline["scores"], "Native recomputation must match the closed baseline exactly")
    check(set(treatments) == set(streams) == set(readout["streams"]) == set(SEEDS), "All three seeds")
    for seed in SEEDS:
        check(set(treatments[seed]) == set(streams[seed]) == set(readout["streams"][seed]) == set(ARMS),
              f"All four arms in seed {seed}")
        for arm in ARMS:
            ref = streams[seed][arm]
            check(set(ref) == {"sha256", "bytes"} and is_digest(ref["sha256"])
                  and type(ref["bytes"]) is int and ref["bytes"] > 0, f"Stream identity: {seed}/{arm}")
            check(ref == {k: readout["streams"][seed][arm][k] for k in ("bytes", "sha256")},
                  f"Scored stream differs from accepted reconstruction: {seed}/{arm}")
            check_population(treatments[seed][arm])

    contrasts = result["paired_contrasts"]
    check(set(contrasts) == {f"{a} minus {b}" for a, b in CONTRASTS}, "All four registered contrasts")
    summary = {}
    for left, right in CONTRASTS:
        key = f"{left} minus {right}"
        check(set(contrasts[key]) == set(POPULATION), f"All views of {key}")
        summary[key] = {}
        for view, data in contrasts[key].items():
            per_seed, means = data["per_head_seed_points"], data["arithmetic_mean_points"]
            check(set(per_seed) == set(SEEDS) and set(means) == set(METRICS), f"Coverage: {key}/{view}")
            check(all(set(per_seed[s]) == set(METRICS) for s in SEEDS), f"Seed metrics: {key}/{view}")
            ap_values = []
            for metric in METRICS:
                deltas = []
                for seed in SEEDS:
                    left_value = treatments[seed][left][view]["metrics"][metric]
                    right_view = native[view] if right == "N" else treatments[seed][right][view]
                    right_value = right_view["metrics"][metric]
                    delta = None if -1 in (left_value, right_value) else 100 * (left_value - right_value)
                    check_expected(per_seed[seed][metric], delta, f"{key}/{view}/{metric}/{seed}")
                    deltas.append(delta)
                expected_mean = None if None in deltas else math.fsum(deltas) / len(deltas)
                check_expected(means[metric], expected_mean, f"{key}/{view}/{metric}/mean")
                if metric == "AP":
                    ap_values = deltas
            summary[key][view] = {
                "AP_points_by_seed": dict(zip(SEEDS, ap_values)),
                "AP_signs_by_seed": dict(zip(SEEDS, (
                    None if v is None else (v > 0) - (v < 0) for v in ap_values
                ))),
                "mean_points": {m: means[m] for m in ("AP", "AP75", "AP90", "APs", "AR100")},
            }
    return summary


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--score", type=Path, required=True)
    cli.add_argument("--score-sha256", required=True, help="Closed scorer footer digest, independently observed")
    cli.add_argument("--baseline", type=Path, required=True)
    cli.add_argument("--readout", type=Path, required=True)
    cli.add_argument("--readout-sha256", required=True)
    args = cli.parse_args()
    score = load_confirmed(args.score, args.score_sha256)
    baseline = load_confirmed(args.baseline, BASELINE_SHA)
    readout = load_confirmed(args.readout, args.readout_sha256)
    summary = audit(score, baseline, readout, args.readout_sha256)
    print(json.dumps({
        "status": "OFFICIAL_HRSID_FROZEN_SCORE_AGGREGATE_VERIFIED",
        "score_sha256": args.score_sha256,
        "readout_sha256": args.readout_sha256,
        "individual_contrasts_verified": 468,
        "mean_contrasts_verified": 156,
        "summary": summary,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
