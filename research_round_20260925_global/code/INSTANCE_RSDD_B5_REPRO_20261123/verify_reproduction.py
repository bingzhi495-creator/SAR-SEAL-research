"""Compare a closed 74-image reproduction with the published Instance-RSDD reserve result."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re


TOLERANCE_POINTS = 1e-6
CHECKPOINT = "011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11"
ROSTER = "2614e567e9ef1b2c791c1b357b03adfe7302ece53e911c1963c40e623ad405fc"
VIEWS = {"full", "date_separated", "gf3", "terrasar_x", "full_deduplicated"}
ARMS = {"raw", "fixed", "free_matched_count", "free_natural"}


def load(path):
    data = path.read_bytes()
    return json.loads(data), sha256(data).hexdigest()


def is_sha(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def compare(path, expected, actual, report):
    """Compare the published scientific tree, scaling AP fractions to percentage points."""
    def fail(reason):
        report["differences"] += 1
        if len(report["first_differences"]) < 30:
            report["first_differences"].append({"path": ".".join(path), "reason": reason})

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            fail("expected an object")
            return
        if actual.keys() != expected.keys():
            fail({"missing": sorted(expected.keys() - actual.keys()), "extra": sorted(actual.keys() - expected.keys())})
        for key in sorted(expected.keys() & actual.keys()):
            compare(path + (key,), expected[key], actual[key], report)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            fail("different or missing list length")
            return
        for idx, (a, b) in enumerate(zip(expected, actual)):
            compare(path + (str(idx),), a, b, report)
    elif isinstance(expected, float):
        report["numerical_values"] += 1
        if type(actual) not in (float, int) or not math.isfinite(actual):
            fail({"expected": expected, "actual": str(actual)})
            return
        scale = 100 if "metrics_fraction" in path else 1
        difference = abs(expected - actual) * scale
        report["maximum_absolute_difference_points"] = max(report["maximum_absolute_difference_points"], difference)
        if difference > TOLERANCE_POINTS:
            fail({"expected": expected, "actual": actual, "difference_points": difference})
    else:
        report["exact_values"] += 1
        if type(expected) is not type(actual) or expected != actual:
            fail({"expected": expected, "actual": actual})


def minifit(report):
    expected = dict(status="HRSID_MINIMAL_FIT_COMPLETE", source_dataset="HRSID",
                    source_route="fresh_png_cpu", parameters=1321, learning_rate=0.002, weight_decay=0.01,
                    source_annotations_sha256="4ac69031accd771b784f90e128f206373f7332293a0130324333154d723541fc",
                    source_images=4, seed=5217, sample_seed=20261112, optimizer="AdamW",
                    optimizer_steps=8, batch_pairs=64, detector_optimizer_updates=0,
                    target_labels_used=False, reload_predictions_exact=True, carrier_sha256=CHECKPOINT)
    checks = {k: type(report.get(k)) is type(v) and report[k] == v for k, v in expected.items()}
    for k in ("source_pairs", "source_groups", "source_matches", "fractional_endpoints"):
        checks[k] = type(report.get(k)) is int and report[k] > 0
    for k in ("gradient_norm_max", "loss_first", "loss_last", "parameter_update_l2"):
        x = report.get(k)
        checks[k] = type(x) in (int, float) and math.isfinite(x)
        if checks[k] and k in ("gradient_norm_max", "parameter_update_l2"):
            checks[k] = x > 0
    checks["checkpoint_sha256"] = is_sha(report.get("checkpoint_sha256"))
    checks["source_samples_sha256"] = is_sha(report.get("source_samples_sha256"))
    return checks

REFERENCE = "737a5de1e28a2a9729c69d8c32a5a1dfb667affab989c81561c957a901bb3b2e"
MANIFEST = "154d0d748b96df0d56ac249adcb8b6ae63275a09dfd7a0d7480839005e044338"
FROZEN_HEAD = "3a60f658149f4262451cd9118902bbe52dc83345d9fbcd23973a11dc473fcdf1"
PRIMARY = "free_matched_count minus fixed"
ORIGINAL_STREAMS = {
    "manifest_sha256": MANIFEST,
    "predictions_sha256": "41163bd418d13a05f9c52b28e7821458bfb5eafca0fcf761682436fbff75f71c",
    "audit_sha256": "bb10ca305fe08f8fb15c85db96e9b9553d3ed24949c1edcaef03739be041d5dd",
    "edits_sha256": "353362e0efabc0374eaaf0b9c27fdabbcad195402a1090230c4ef56ca2a10398",
}


def verify(inputs):
    reference = inputs["reference"]
    run = {k: inputs[k] for k in ("conversion", "capture", "replay", "score")}
    conversion, capture, replay, score = (run[k] for k in ("conversion", "capture", "replay", "score"))
    report = dict(differences=0, first_differences=[], numerical_values=0,
                  maximum_absolute_difference_points=0.0, exact_values=0)
    checks = {
        "conversion": dict(status="INSTANCE_RSDD_RESERVE74_CLEAN_CONVERSION_COMPLETE_V1", roster_sha256=ROSTER,
                           image_manifest_sha256=MANIFEST, images=74, parents=13, exact_crc_and_rgb=74,
                           sensor_counts={"GF-3": 42, "TerraSAR-X": 32}, rgb_unique=74,
                           detector_forwards=0, private_annotations_opened=0),
        "capture": dict(status="B5_RESERVE74_CAPTURE_COMPLETE", model_forward_calls=74, target_labels_opened=False,
                        counts=reference["capture_counts"]),
        "replay": dict(status="B5_RESERVE74_REPLAY_COMPLETE", independent_decisions_equal=True, target_labels_opened=False,
                       point_head_sha256=FROZEN_HEAD, counts=reference["replay_counts"]),
        "score": dict(status="B5_RESERVE74_SCORE_COMPLETE", per_candidate_fixed_matched_budget_equal=True,
                      point_head_sha256=FROZEN_HEAD),
    }
    for name in ("capture", "replay", "score"):
        checks[name].update(roster_sha256=ROSTER, manifest_sha256=MANIFEST, checkpoint_sha256=CHECKPOINT,
                            images=74, source_partition="author train", optimizer_updates=0)
    for key in ("author_train_sha256", "census_sha256", "evaluator_annotation_ids_rekeyed", "non_simple_polygons_retained_in_ap"):
        checks["score"][key] = reference["score"][key]
    for stage, expected in checks.items():
        for key, value in expected.items():
            compare((stage, key), value, run[stage].get(key), report)

    for key in ("views", "by_parent"):
        compare(("score", key), reference["score"][key], score.get(key), report)
    training = minifit(inputs["training"])
    # Numeric training state changes are checked directly, without using the short-fit head for scoring.
    compare(("training",), dict.fromkeys(training, True), training, report)

    current = {"manifest_sha256": conversion.get("image_manifest_sha256"),
               "predictions_sha256": capture.get("predictions_sha256"),
               "audit_sha256": capture.get("audit_sha256"),
               "edits_sha256": replay.get("edits_sha256")}
    stream_results = {}
    for key, observed in current.items():
        original = ORIGINAL_STREAMS[key]
        compare(("new_streams", key, "valid_digest"), True, is_sha(observed), report)
        consumers = ("capture", "replay", "score") if key == "manifest_sha256" else ("score",)
        if key in ("predictions_sha256", "audit_sha256"):
            consumers = ("replay", "score")
        for name in consumers:
            compare((name, key), observed, run[name].get(key), report)
        stream_results[key] = {"observed": observed, "original": original,
                               "equal_to_original": original == observed}
        compare(("original_streams", key), original, observed, report)
    views = score.get("views", {})
    full = views.get("full", {})
    per_parent = score.get("by_parent", {})
    compare(("cohort", "views"), sorted(VIEWS), sorted(views), report)
    compare(("cohort", "parents"), 13, len(per_parent), report)
    compare(("cohort", "arms"), sorted(ARMS), sorted(full.get("metrics_fraction", {})), report)
    compare(("cohort", "polygons"), 81, full.get("annotations"), report)
    primary = {name: {"images": view.get("images"),
                      "ap_difference_points": view.get("paired_contrasts_points", {}).get(PRIMARY, {}).get("AP"),
                      "continuous_iou": view.get("geometry", {}).get("paired_contrasts_points", {}).get(PRIMARY, {}).get("continuous_iou_points"),
                      "gates": view.get("criteria")}
               for name, view in views.items()}
    return dict(passed=report["differences"] == 0, absolute_tolerance_points=TOLERANCE_POINTS,
                scientific_comparison=report, training_checks=training, canonical_streams=stream_results,
                training_observed={key: inputs["training"].get(key) for key in training},
                parent_count=len(per_parent), reserve_views=primary)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    phases = ("reference", "conversion", "training", "capture", "replay", "score")
    for phase in phases:
        parser.add_argument("--" + phase, type=Path, required=True, help=phase + " summary JSON")
    parser.add_argument("--output", type=Path, required=True, help="A new comparison report file")
    args = parser.parse_args()
    inputs, hashes = {}, {}
    for phase in phases:
        inputs[phase], hashes[phase] = load(getattr(args, phase))
    if hashes["reference"] != REFERENCE:
        parser.error("Expected published Instance-RSDD reserve aggregate at commit 6d7c0f0")
    result = verify(inputs)
    result["summary_sha256"] = hashes
    with args.output.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(json.dumps({"passed": result["passed"], **{k: result["scientific_comparison"][k] for k in ("differences", "numerical_values", "maximum_absolute_difference_points")}}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
