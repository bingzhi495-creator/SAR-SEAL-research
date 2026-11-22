"""Compare fresh HRSID and Instance-RSDD results with their frozen published references."""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean


TOLERANCE_POINTS = 1e-6
HRSID = {
    "readout": ("HRSID_OFFICIAL_FROZEN_REFINEMENT_READOUT_ACCEPTED_20261122.json",
                "631e18d6dca60239aff9dd2b7f19b0e98b4fdefb61f60adf4f6b57369e60cf14"),
    "score": ("HRSID_OFFICIAL_FROZEN_REFINEMENT_SCORE_ACTUAL_20261122.json",
              "90ef9abb8387eb1ede4d6cd7979f08d4c57e75997347df448ab34af2bd33411f"),
}
RSDD = {
    "intake": ("INSTANCE_RSDD_FROZEN_HRSID_INTAKE_SAFE_20261122.json",
               "c8b0f3862930b1c996bbec02efe1025c859d53531506a7221a8120e268a00fca"),
    "capture": ("INSTANCE_RSDD_FROZEN_HRSID_CAPTURE_ACCEPTED_SAFE_20261122.json",
                "29357a70ceeff135e02ddb98b400b9f3695a8fe1f808f4a2af5e0ed86e4769ec"),
    "edits_5217": ("INSTANCE_RSDD_FROZEN_HRSID_EDITS_5217_ACCEPTED_SAFE_20261122.json",
                   "0796f68909a5c99ab49e1805a6cb209606cf4e5a1ee5cd135a8a672ff57a6eac"),
    "edits_5218": ("INSTANCE_RSDD_FROZEN_HRSID_EDITS_5218_ACCEPTED_SAFE_20261122.json",
                   "e66f377f4785ef4bfd8db0b30ebf0bcf7c081be8dd042c161c877ed5a1db8147"),
    "score_5217": ("INSTANCE_RSDD_FROZEN_HRSID_5217_SAFE_20261122.json",
                   "59936bb4ceb70db1af4801f8d31d1ba84f223adb688759c795d8a71e58f7da4b"),
    "score_5218": ("INSTANCE_RSDD_FROZEN_HRSID_5218_SAFE_20261122.json",
                   "a8cf7537d8a42c6c17cc08b11d4699354b3eedc3fb26cb2ee24150eef7b1d90a"),
}
HEADS = {5217: "3a60f658149f4262451cd9118902bbe52dc83345d9fbcd23973a11dc473fcdf1",
         5218: "08b39b9911f77c039b26f8bb9bc34e940f5a350390911586f47290b2c3d88869"}
MISSING = object()


def object_without_duplicates(pairs):
    data = {}
    for key, value in pairs:
        if key in data:
            raise ValueError("Duplicate JSON key: " + key)
        data[key] = value
    return data


def load(path):
    def invalid(value):
        raise ValueError("Invalid JSON number: " + value)

    data = path.read_bytes()
    return json.loads(data, object_pairs_hook=object_without_duplicates, parse_constant=invalid), sha256(data).hexdigest()


class Comparison:
    def __init__(self):
        self.numerical_values = 0
        self.exact_values = 0
        self.unavailable_values = 0
        self.maximum_absolute_difference_points = 0.0
        self.differences = []
        self.canonical_streams = {}

    def mismatch(self, path, **detail):
        self.differences.append({"path": ".".join(path), **detail})

    def compare(self, path, expected, actual):
        if actual is MISSING:
            self.mismatch(path, reason="missing")
            return
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                self.mismatch(path, reason="expected object")
                return
            for key in sorted(actual.keys() - expected.keys()):
                self.mismatch((*path, key), reason="unexpected")
            for key, value in expected.items():
                self.compare((*path, key), value, actual.get(key, MISSING))
            return
        if isinstance(expected, list):
            if not isinstance(actual, list) or len(actual) != len(expected):
                self.mismatch(path, reason="different list type or length")
                return
            for i, (left, right) in enumerate(zip(expected, actual)):
                self.compare((*path, str(i)), left, right)
            return
        fraction = "metrics_fraction" in path or "metrics" in path
        if expected is None or (fraction and expected == -1):
            self.unavailable_values += 1
            same = actual is None if expected is None else type(actual) in (float, int) and actual == -1
            if not same:
                self.mismatch(path, expected=expected, actual=actual)
            return
        if isinstance(expected, float):
            self.numerical_values += 1
            if type(actual) not in (float, int) or not math.isfinite(actual) or (fraction and not 0 <= actual <= 1):
                self.mismatch(path, reason="expected a finite defined metric", actual=actual)
                return
            delta = abs(expected - actual) * (100 if fraction else 1)
            self.maximum_absolute_difference_points = max(self.maximum_absolute_difference_points, delta)
            if delta > TOLERANCE_POINTS:
                self.mismatch(path, expected=expected, actual=actual, difference_points=delta)
            return
        self.exact_values += 1
        if type(actual) is not type(expected) or actual != expected:
            self.mismatch(path, expected=expected, actual=actual)

    def fields(self, prefix, actual, expected):
        for key, value in expected.items():
            self.compare((prefix, key), value, actual.get(key, MISSING))

    def stream(self, name, expected, actual):
        self.canonical_streams[name] = dict(expected_sha256=expected,
                                            observed_sha256=None if actual is MISSING else actual,
                                            equal=expected == actual)
        self.compare(("canonical_streams", name), expected, actual)


def select(mapping, *keys):
    return {key: mapping[key] for key in keys}


def hrsid(reference, actual, hashes, checks):
    original, before = reference["score"], reference["readout"]
    score, readout = actual["score"], actual["readout"]
    checks.fields("readout", readout, select(before, "counts", "streams", "images", "mapped_rows", "AP",
                                           "endpoint_sha256", "optimizer_updates", "GT_opened",
                                           "exact_candidate_identity", "exact_native_per_image",
                                           "exact_per_candidate_budgets", "actual_XOR_area_RLE_verified",
                                           "point_score_decisions_replayed"))
    checks.fields("readout", readout, dict(model_forward_calls=1962, device="cpu",
                 manifest_sha256="dd42cbf5ef415dc152bab9160c5eb385ea33255d697fba2dddf49917eda3ee1b",
                 point_heads_sha256={"5217": HEADS[5217], "5218": HEADS[5218],
                             "5219": "93a3be4356816f9d29560d40d9bff8cc5d8730e9e600b29db07959612f504604"}))
    checks.fields("score", score, select(original, "native", "treatments", "paired_contrasts", "streams",
                                         "gt_sha256", "cohort_sha256", "endpoint_sha256", "optimizer_updates",
                                         "model_forward_calls", "native_predictions_sha256", "score_source_sha256",
                                         "adapter_sha256", "TEST", "AP", "historical_test_selection_exposure",
                                         "metric_units", "contrast_units", "seeds", "arms"))
    checks.fields("score", score, {"readout_sha256": hashes["readout"]})
    for key in ("native_predictions_sha256", "descriptor_identity_ledger_sha256"):
        checks.stream(key, before[key], readout.get(key, MISSING))
    for seed, arms in before["streams"].items():
        for arm, target in arms.items():
            stream = readout.get("streams", {}).get(seed, {}).get(arm, {})
            checks.stream(f"{seed}/{arm}", target["sha256"], stream.get("sha256", MISSING))


def rsdd(reference, actual, hashes, checks):
    original, intake = reference["intake"], actual["intake"]
    cap, before = actual["capture"], reference["capture"]
    checks.fields("intake", intake, select(original, "test_roster_sha256", "sensor_counts", "acquisition_groups",
                                           "encoded_unique", "rgb_unique", "target_polygon_rows_opened"))
    checks.fields("intake", intake, dict(images=original["test_images"], exact_crc_and_rgb=original["rgb_decode_verified"],
                                         source_archive_directory_sha256=original["zip_central_directory_sha256"],
                                         status="RSDD740_CLEAN_IMAGE_INPUT_COMPLETE"))
    checks.stream("image_manifest", original["image_manifest_sha256"], intake.get("image_manifest_sha256", MISSING))
    checks.fields("capture", cap, select(before, "counts", "checkpoint_sha256", "images", "optimizer_updates"))
    checks.fields("capture", cap, dict(model_forward_calls=740, target_labels_opened=False, phase="capture",
                                       status="B5_RSDD740_CAPTURE_COMPLETE", source_partition="author test",
                                       manifest_sha256=original["image_manifest_sha256"]))
    for key in ("predictions_sha256", "audit_sha256"):
        checks.stream(key, before[key], cap.get(key, MISSING))
    for seed, head in HEADS.items():
        edit_key, score_key = f"edits_{seed}", f"score_{seed}"
        edits, score = actual[edit_key], actual[score_key]
        old_edits, old_score = reference[edit_key], reference[score_key]
        shared = dict(images=740, point_head_seed=seed, point_head_sha256=head,
                      source_partition="author test", optimizer_updates=0, capture_sha256=hashes["capture"],
                      manifest_sha256=original["image_manifest_sha256"], checkpoint_sha256=before["checkpoint_sha256"],
                      predictions_sha256=before["predictions_sha256"], audit_sha256=before["audit_sha256"])
        checks.fields(edit_key, edits, dict(shared, phase="replay", status="B5_RSDD740_REPLAY_COMPLETE",
                                            counts=old_edits["counts"], exact_replayed_policy_choices=True,
                                            exact_support_xor_area=True, model_forward_calls=0, target_labels_opened=False))
        checks.stream(edit_key, old_edits["edits_sha256"], edits.get("edits_sha256", MISSING))
        checks.fields(score_key, score, dict(shared, phase="score", status="B5_RSDD740_SCORE_COMPLETE",
                                             replay_sha256=hashes[edit_key], edits_sha256=old_edits["edits_sha256"]))
        checks.fields(score_key, score, select(old_score, "views", "criteria", "primary_ap_points", "author_TEST",
                                              "detector_forwards", "gt_sha256", "evaluator_annotation_ids_rekeyed",
                                              "non_simple_ap_retained", "full_replication_eligible", "replicate_5218"))
    # Each published table uses arithmetic means of the two source point-head contrasts.
    def means(values):
        result = {}
        for view, summary in reference["score_5217"]["views"].items():
            result[view] = {}
            for pair in summary["paired_contrasts_points"]:
                values_by_seed = [values[f"score_{s}"].get("views", {}).get(view, {}).get(
                    "paired_contrasts_points", {}).get(pair, {}).get("AP") for s in HEADS]
                if all(type(v) in (float, int) and math.isfinite(v) for v in values_by_seed):
                    result[view][pair] = mean(values_by_seed)
        return result

    derived = means(actual)
    checks.compare(("derived_two_head_mean_mask_AP_points",), means(reference), derived)
    return derived


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = parser.add_subparsers(dest="dataset", required=True)
    for name, sources in (("hrsid", HRSID), ("rsdd", RSDD)):
        cmd = sub.add_parser(name, allow_abbrev=False)
        cmd.add_argument("--published-results", type=Path, help="Directory containing the published reference JSONs")
        for key, (filename, _) in sources.items():
            flag = key.replace("_", "-")
            cmd.add_argument("--" + flag, type=Path, required=True, help="Fresh " + key + " summary JSON")
            cmd.add_argument("--reference-" + flag, type=Path, help="Explicit reference instead of " + filename)
        cmd.add_argument("--output", type=Path, required=True, help="New verification report JSON")
    args = parser.parse_args()
    sources = HRSID if args.dataset == "hrsid" else RSDD
    references, observed, hashes, published = {}, {}, {}, {}
    for key, (filename, pin) in sources.items():
        path = getattr(args, "reference_" + key)
        if path is None:
            if args.published_results is None:
                parser.error("Supply --published-results or each --reference-* file")
            path = args.published_results / filename
        references[key], fingerprint = load(path)
        if fingerprint != pin:
            parser.error("Published reference digest differs: " + filename)
        published[filename] = fingerprint
        observed[key], hashes[key] = load(getattr(args, key))
    checks = Comparison()
    derived = {}
    if args.dataset == "hrsid":
        hrsid(references, observed, hashes, checks)
    else:
        derived = rsdd(references, observed, hashes, checks)
    result = dict(dataset=args.dataset, passed=not checks.differences, absolute_tolerance_points=TOLERANCE_POINTS,
                  published_reference_sha256=published, fresh_summary_sha256=hashes,
                  numerical_values=checks.numerical_values, exact_values=checks.exact_values,
                  unavailable_values=checks.unavailable_values,
                  maximum_absolute_difference_points=checks.maximum_absolute_difference_points,
                  difference_count=len(checks.differences), differences=checks.differences,
                  canonical_streams=checks.canonical_streams)
    if derived:
        result["derived_two_head_mean_mask_AP_points"] = derived
    with args.output.open("x", encoding="utf-8", newline="\n") as out:
        json.dump(result, out, sort_keys=True, indent=2, allow_nan=False)
        out.write("\n")
    print(json.dumps(select(result, "passed", "difference_count", "numerical_values", "maximum_absolute_difference_points")))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
