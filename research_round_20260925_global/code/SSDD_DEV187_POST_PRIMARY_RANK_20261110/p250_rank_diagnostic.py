"""Post-primary rank and score-blind coverage diagnostic for closed P250 DEV187."""

import argparse
from builtins import __import__ as standard_import
from collections import Counter
from contextlib import redirect_stdout
from hashlib import sha256
from importlib.metadata import version
from io import StringIO
import json
from math import fsum, isfinite
import os
from pathlib import Path
import platform
import sys
from types import ModuleType


P250_SOURCE = (
    "HRSID_TO_SSDD_FROZEN_DEV187_20261110/score_frozen_hrsid_ssdd.py",
    "dc6af86e4c38fdd79499f98f96476e398e376bf5891343a66aa7901412381cac",
)
TEMPLATE_DIR = "HRSID_MASK_PHASE_REPEAT_DEV705_20261109"
TEMPLATE_PINS = {
    "dev705_group_sensitivity": "f7a004ed3d128cffd2a7114618b892d964f1c00d946377044d0be09d46a871b9",
    "dev705_paired_coverage": "2ac4abb1a0cb9e684f6bde6b2990419eb0ec209d469411b240dc7dd520e69d3d",
    "dev705_score_template": "65388e3f32388f8aea10b022e1a9c229848f10b19401fcecc292e410a48e5ee4",
}
PLAN_SHA = "fc7ac31ba4d121f4329a6536ebca2be4f22552c140cf9af47b8f5ae60d8a4256"
ACCEPTED_SHA = "5c703c813647b6f7d2f3e21d653500a46d26db4b46b59022b2b642ebeb99d15d"
PROTOCOL = "ad2cb6182b37f9db1dc24908220dd5c923a1257c"
THRESHOLDS = (.75, .90)
ARMS = ("native", "center")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def load_support(code):
    """Load ordinary pinned score-template/coverage helpers and the P250 SSDD adapter."""
    require(code.is_absolute() and code.resolve(strict=True) == code and code.is_dir(),
            "Canonical ordinary research code directory required")
    modules = {}

    def pinned_import(name, globals=None, locals=None, fromlist=(), level=0):
        if not level and name in modules:
            return modules[name]
        return standard_import(name, globals, locals, fromlist, level)

    def load(name, path, digest):
        require(path.resolve(strict=True) == path and path.is_file(), "Canonical exact diagnostic source required")
        with path.open("rb") as stream:
            source = stream.read(64 * 1024 + 1)
        require(0 < len(source) <= 64 * 1024 and sha256(source).hexdigest() == digest,
                "Pinned diagnostic source changed: " + path.name)
        module = ModuleType(name)
        module.__file__ = str(path)
        module.__dict__["__builtins__"] = dict(vars(standard_import("builtins")), __import__=pinned_import)
        exec(compile(source, str(path), "exec"), module.__dict__)
        return module

    p250 = load("_p255_original_p250_scorer", code / P250_SOURCE[0], P250_SOURCE[1])
    for name, digest in TEMPLATE_PINS.items():
        modules[name] = load(name, code / TEMPLATE_DIR / (name + ".py"), digest)
    scoring, official = p250.load_scoring()
    return p250, modules["dev705_score_template"], modules["dev705_paired_coverage"], scoring, official


def blind_coverage(gt, selected, coverage):
    """Score-blind original-mask IoU coverage on exactly the two selected pools."""
    import numpy as np
    from pycocotools import mask
    from pycocotools.coco import COCO

    images = gt["images"]
    grouped = {mode: coverage.group_rows(selected[mode], images) for mode in ARMS}
    graph_counts = coverage.graph_counts_fn()
    coco = COCO()
    coco.dataset = gt
    with redirect_stdout(StringIO()):
        coco.createIndex()

    excluded = Counter(empty=0, crowd=0, ignore=0)
    targets_total = 0
    counts = {mode: np.zeros((len(THRESHOLDS), 2), dtype=np.int64) for mode in ARMS}
    for i, image in enumerate(images):
        targets, _ = coverage.targets_for_image(coco, image, excluded)
        targets_total += len(targets)
        for mode in ARMS:
            candidates = grouped[mode][i]
            overlap = (mask.iou(candidates, targets, [0] * len(targets)) if candidates and targets
                       else np.zeros((len(candidates), len(targets))))
            require(overlap.shape == (len(candidates), len(targets)) and np.isfinite(overlap).all()
                    and (overlap >= 0).all() and (overlap <= 1).all(), "Finite original-mask IoU matrix required")
            for threshold_index, threshold in enumerate(THRESHOLDS):
                any_candidate, maximum, _ = graph_counts(overlap, threshold)
                counts[mode][threshold_index] += (any_candidate, maximum)
    require(targets_total + sum(excluded.values()) == len(gt["annotations"]),
            "Complete original GT coverage accounting required")
    arms = {}
    for mode in ARMS:
        arms[mode] = {}
        for threshold, (any_candidate, maximum) in zip(THRESHOLDS, counts[mode].tolist(), strict=True):
            arms[mode][f"{threshold:.2f}"] = dict(
                any_candidate=any_candidate, maximum_one_to_one=maximum,
                maximum_recall=maximum / targets_total if targets_total else None,
            )
    return dict(eligible_gt=targets_total, excluded_gt=dict(excluded), arms=arms)


def matching_witness(matching, image_count, selected_count):
    """Summarize the exact all-area record comparisons without publishing rows."""
    require(len(matching) == 4 * image_count, "Exact one-category COCO area/image record population required")
    records = [item for item in matching[:image_count] if item is not None]
    detections = sum(len(item["dtIds"]) for item in records)
    require(detections == selected_count
            and all(item["dtMatches"].shape == item["dtIgnore"].shape == (10, len(item["dtIds"]))
                    for item in records), "Exact all-area detection and match/ignore accounting required")
    return dict(exact_across_score_donors=True, all_area_image_records=image_count,
                all_area_nonnull_records=len(records), all_area_detection_rows=detections,
                all_area_dt_matches_cells=sum(item["dtMatches"].size for item in records),
                all_area_dt_ignore_cells=sum(item["dtIgnore"].size for item in records),
                all_area_ranges_records=len(matching))


def analyze_pair(gt, native, center, support):
    _, template, coverage, _, official = support
    images = gt["images"]
    image_ids = [image["id"] for image in images]
    require(len(set(image_ids)) == len(images) > 0, "Unique ordered original image identities required")
    rows = dict(native=native, center=center)
    nonempty = {mode: template.nonempty_by_image(rows[mode], images) for mode in ARMS}
    selected, counts = template.prefix_rows(nonempty["native"], nonempty["center"])
    kept = sum(counts)
    prefix_counts = dict(retained_per_arm=kept, images_with_prefix=sum(count > 0 for count in counts),
                         images_without_prefix=sum(count == 0 for count in counts))
    for mode in ARMS:
        count = sum(map(len, nonempty[mode]))
        prefix_counts[mode] = dict(mapped=len(rows[mode]), nonempty=count, mapped_empty=len(rows[mode]) - count,
                                   discarded_nonempty_after_count_matching=count - kept)

    cells, layouts, witnesses = {}, {}, {}
    for geometry in ARMS:
        cells[geometry] = {}
        reference = None
        for donor in ARMS:
            donated = template.score_template(selected[geometry], selected[donor])
            metrics, matching, layout = template.evaluate_cell(gt, donated, image_ids, official)
            if donor in layouts:
                require(layout == layouts[donor], "Same donor must induce the exact same global rank layout")
            else:
                layouts[donor] = layout
            if reference is None:
                reference = matching
            else:
                template.require_same_matching(reference, matching)
                require(all(metrics[key] == cells[geometry]["native"][key] for key in ("recall75", "recall90")),
                        "Fixed-geometry recalls must be exactly equal across score donors")
                witnesses[geometry] = matching_witness(matching, len(images), kept)
            cells[geometry][donor] = metrics
    a, b = (cells[mode] for mode in ARMS)
    differences = template.differences
    return dict(
        prefix_counts=prefix_counts, cells=cells, matching_witness=witnesses,
        contrasts_points=dict(
            own_scores_center_minus_native=differences(a["native"], b["center"]),
            score_on_native_geometry=differences(a["native"], a["center"]),
            score_on_center_geometry=differences(b["native"], b["center"]),
            geometry_with_native_scores=differences(a["native"], b["native"]),
            geometry_with_center_scores=differences(a["center"], b["center"]),
        ),
        score_blind_coverage=blind_coverage(gt, selected, coverage),
    )


def bound_prediction_refs(p250, plan, accepted):
    """Cross-bind exact accepted source identities to prediction refs without opening other arm files."""
    require(type(plan["arms"]) is type(accepted["arms"]) is list
            and len(plan["arms"]) == len(accepted["arms"]) == 6, "Expected all six closed source arms")
    keys = {(seed, mode) for seed in p250.TRAIN_PROTOCOLS for mode in p250.MODES}
    accepted_arms = {}
    for arm in accepted["arms"]:
        require(type(arm) is dict and type(arm.get("seed")) is int and type(arm.get("mode")) is str,
                "Malformed accepted score arm")
        key = arm["seed"], arm["mode"]
        require(key in keys and key not in accepted_arms, "Unexpected or duplicate accepted source arm")
        metrics = arm["metrics"]
        require(type(metrics) is dict and set(metrics) == {*p250.METRICS, "all12"}
                and type(metrics["all12"]) is list and len(metrics["all12"]) == 12
                and all(type(value) in (int, float) and isfinite(value) and -1 <= value <= 1
                        for value in [*(metrics[name] for name in p250.METRICS), *metrics["all12"]])
                and arm["source_training_protocol"] == p250.TRAIN_PROTOCOLS[key[0]]
                and p250.bounded_int(arm["mapped_rows"], 18700)
                and p250.bounded_int(arm["empty_image_predictions"], 187), "Exact finite accepted scoring census required")
        accepted_arms[key] = arm

    result, paths = {}, set()
    for entry in plan["arms"]:
        identity = ("seed", "mode", "source_commit", "checkpoint_sha256", "source_pair_receipt_sha256",
                    "source_arm_receipt_sha256")
        require(p250.exact(entry, {}, (*identity, "terminal", "report", "predictions"))
                and type(entry["seed"]) is int and type(entry["mode"]) is str, "Malformed closed-plan arm")
        key = entry["seed"], entry["mode"]
        require(key in keys and key not in result, "Unexpected or duplicate closed-plan source arm")
        arm = accepted_arms[key]
        require(all(arm[name] == entry[name] for name in identity), "Closed source arm differs from accepted score")
        for role, field in (("terminal", "export_terminal_sha256"), ("report", "export_report_sha256"),
                            ("predictions", "predictions_sha256")):
            require(type(entry[role]) is dict and arm[field] == entry[role].get("sha256"),
                    "Closed arm digests differ from independently accepted score")
        ref = entry["predictions"]
        path = p250.ref_path(ref, 64 * p250.MIB, p250.PREDICTION_NAME)
        require(path not in paths, "Every admitted export needs its distinct prediction path")
        paths.add(path)
        result[key] = ref
    require(result.keys() == accepted_arms.keys() == keys
            and len({arm["predictions_sha256"] for arm in accepted_arms.values()}) == len(keys),
            "The same complete and distinct six-arm population must be admitted")
    return result, accepted_arms


def _block_network(event, _):
    if event in ("socket.connect", "socket.getaddrinfo"):
        raise RuntimeError("The closed-prediction CPU diagnostic must stay offline")


def _mean(values):
    return fsum(values) / len(values) if all(value is not None for value in values) else None


def _coverage_delta(result):
    values = result["score_blind_coverage"]["arms"]
    a, b = (values[mode] for mode in ARMS)
    return {threshold: dict(
        count=b[threshold]["maximum_one_to_one"] - a[threshold]["maximum_one_to_one"],
        points=(None if a[threshold]["maximum_recall"] is None else
                100 * (b[threshold]["maximum_recall"] - a[threshold]["maximum_recall"])),
    ) for threshold in a}


def diagnose(code, plan_path, plan_sha, accepted_path, accepted_sha):
    require(plan_sha == PLAN_SHA and accepted_sha == ACCEPTED_SHA,
            "Exact pre-existing independently accepted P250 plan and safe score SHA required")
    require(platform.system() == "Linux" and platform.machine() == "aarch64"
            and sys.version_info[:3] == (3, 12, 14) and sys.flags.isolated == sys.flags.utf8_mode == 1
            and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
            "Original isolated Linux DGX CPU scoring runtime required")
    packages = {name: version(name) for name in ("numpy", "pycocotools", "scipy")}
    require(packages["numpy"] == "1.26.4" and packages["pycocotools"] == "2.0.11",
            "Exact original NumPy and COCO packages required")
    sys.addaudithook(_block_network)
    support = load_support(code)
    p250, _, _, scoring, _ = support

    plan = p250.document(p250.stable_file(plan_path, plan_sha, p250.MIB))
    accepted = p250.document(p250.stable_file(accepted_path, accepted_sha, p250.MIB))
    require(p250.exact(plan, dict(status=p250.PLAN_STATUS, evaluation_protocol=PROTOCOL),
                       ("image_manifest", "integer_gt", "arms")),
            "Exact closed P250 score plan and protocol required")
    fixed = dict(
        status="FROZEN_HRSID_TO_SSDD_DEV187_INDEPENDENTLY_ACCEPTED_V1",
        closed_plan_sha256=plan_sha, evaluation_protocol=PROTOCOL, images=187, annotations=380,
        distinct_jpeg_sha256=183, device="cpu", primary_GT="INTEGER_PRIMARY", max_dets=[1, 10, 100],
        image_manifest_sha256=p250.IMAGE_MANIFEST_SHA, integer_gt_sha256=p250.INTEGER_GT_SHA,
        independent_recompute=True, units="fraction", AP=True, DEV=True,
        target_adaptation=False, training=False, official_TEST_opened=False,
    )
    extras = ("arms", "image_zip_sha256", "long_export_admission_sha256", "long_export_owner_sha256",
              "original_cpu_terminal_sha256", "original_score_sha256", "paired_summary")
    require(p250.exact(accepted, fixed, extras) and all(p250.hex_digest(accepted[key])
            for key in extras if key.endswith("_sha256")), "Exact independently accepted CPU scoreboard required")
    source_ref, gt_ref = plan["image_manifest"], plan["integer_gt"]
    require(source_ref["sha256"] == p250.IMAGE_MANIFEST_SHA and gt_ref["sha256"] == p250.INTEGER_GT_SHA
            and gt_ref["bytes"] == p250.INTEGER_GT_BYTES, "Exact fixed DEV187 image manifest and integer GT required")
    refs, arms = bound_prediction_refs(p250, plan, accepted)
    _, ids, items = scoring.fixed_images(scoring.Artifact(source_ref, p250.read_ref(source_ref, p250.MIB)))
    require(len({item["image_ref"]["sha256"] for item in items}) == accepted["distinct_jpeg_sha256"],
            "Exact accepted number of distinct source JPEG identities required")
    gt = p250.document(p250.read_ref(gt_ref, p250.MIB))
    scoring.validate_gt(gt, ids, items)

    results = {}
    for seed in p250.TRAIN_PROTOCOLS:
        rows = {}
        for mode in p250.MODES:
            key = seed, mode
            arm = arms[key]
            report_census = dict(mapped_rows=arm["mapped_rows"], unmapped_slot1_rows=18700 - arm["mapped_rows"],
                                 empty_image_predictions=arm["empty_image_predictions"])
            rows[mode] = p250.score_rows(p250.read_ref(refs[key], 64 * p250.MIB), items, report_census)
        result = analyze_pair(gt, rows["native"], rows["center"], support)
        result["score_blind_maximum_center_minus_native"] = _coverage_delta(result)
        results[str(seed)] = result
    summary = list(results.values())
    contrasts = {contrast: {metric: _mean([r["contrasts_points"][contrast][metric] for r in summary])
                            for metric in metrics}
                 for contrast, metrics in summary[0]["contrasts_points"].items()}
    coverages = {threshold: {field: _mean([r["score_blind_maximum_center_minus_native"][threshold][field]
                                           for r in summary]) for field in ("count", "points")}
                 for threshold in summary[0]["score_blind_maximum_center_minus_native"]}
    return dict(
        status="P255_P250_DEV187_POST_PRIMARY_RANK_COVERAGE_COMPLETE_V1", evaluation_protocol=PROTOCOL,
        accepted_primary_score_sha256=accepted_sha, closed_plan_sha256=plan_sha,
        original_integer_gt_sha256=p250.INTEGER_GT_SHA, image_manifest_sha256=p250.IMAGE_MANIFEST_SHA,
        images=187, annotations=380, primary_score_unchanged=True,
        accepted_primary_paired_summary=accepted["paired_summary"], by_seed=results,
        mean_prefix_contrasts_points=contrasts, mean_score_blind_maximum_center_minus_native=coverages,
        within_image_matching_unchanged_across_donors=True, score_blind_iou_thresholds=list(THRESHOLDS),
        geometry_scope="original mask contents and within-image native ordering at equal nonempty prefix counts",
        donor_scope="replace only cross-image score template, with exact per-image matches and ignore flags held",
        coverage_scope="one-to-one maximum cardinality using original-mask IoU on the identical selected pools",
        device="cpu", package_versions=packages, GT="INTEGER_PRIMARY", DEV=True, official_TEST_opened=False,
        model_forward_calls=0, training=False, target_adaptation=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--accepted-score", type=Path, required=True)
    parser.add_argument("--accepted-score-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    require(output.is_absolute() and output.parent.resolve(strict=True) == output.parent
            and not output.exists() and not output.is_symlink(), "Fresh canonical diagnostic output required")
    report = diagnose(args.code, args.plan, args.plan_sha256, args.accepted_score, args.accepted_score_sha256)
    payload = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with output.open("xb") as stream:
        require(stream.write(payload) == len(payload), "Complete safe diagnostic report required")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=report["status"], result_sha256=sha256(payload).hexdigest(),
                          images=report["images"], seeds=len(report["by_seed"])), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
