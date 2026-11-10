"""Diagnose bounded mask translations on the accepted SSDD DEV187 fixed pairs."""

from argparse import ArgumentParser
from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version
from io import StringIO
import json
from math import fsum
import os
from pathlib import Path
import sys
from types import ModuleType


P299_SOURCE = ("SSDD_RETRAINED_DEV187_DIAGNOSTIC_20261111/ssdd_retrained_dev187_diagnostic.py",
               "ebefd0eab1394484219cda2991b45f3c3b6a93462938e3e4b0c8297b16785930")
P299_SAFE_SHA = "c96087f644e13f8199fc956f52af503bb726da0c003c064385c732dc074dc98d"
STATUS = "P310_SSDD_FIXED_MATCH_TRANSLATION_COMPLETE_PENDING_INDEPENDENT_ACCEPTANCE"
ARMS = ("native", "center")
RADIUS = 2
OFFSETS = {
    "raw": ((0, 0),),
    "vertical": tuple((dy, 0) for dy in range(-RADIUS, RADIUS + 1)),
    "horizontal": tuple((0, dx) for dx in range(-RADIUS, RADIUS + 1)),
    "joint": tuple((dy, dx) for dy in range(-RADIUS, RADIUS + 1) for dx in range(-RADIUS, RADIUS + 1)),
}
THRESHOLDS = {"0.75": (3, 4), "0.90": (9, 10)}
SCOPE = dict(FIT=False, DEV=True, TEST=False, official_TEST_opened=False, training=False, model_forward_calls=0, AP=False)
MIB = 1024**2


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def load_source(path, digest, name):
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(),
            "Exact canonical ordinary source required")
    with path.open("rb") as stream:
        raw = stream.read(64 * 1024 + 1)
    require(0 < len(raw) <= 64 * 1024 and sha256(raw).hexdigest() == digest, "Exact ordinary source digest required")
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def measure_masks(rles):
    """Decode one GT and its two fixed masks. Shift only overlap, keeping all areas."""
    import numpy as np
    from pycocotools import mask

    rles = list(rles)
    height, width = rles[0]["size"]
    require(len(rles) == 3 and all(row["size"] == [height, width] for row in rles), "Exact original common RLE canvas")
    areas = tuple(map(int, mask.area(rles)))
    require(min(areas) > 0, "All three fixed masks must be nonempty")
    boxes = mask.toBbox(rles)
    bounds = boxes.astype(np.int64)
    require(boxes.shape == (3, 4) and np.array_equal(boxes, bounds) and (bounds >= 0).all(),
            "Exact original integer mask bounding boxes")
    x0, y0 = bounds[:, :2].min(axis=0)
    x1, y1 = (bounds[:, :2] + bounds[:, 2:]).max(axis=0)
    require(0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height, "Exact in-canvas union mask support")
    crop = mask.decode(rles)[y0:y1, x0:x1, :].astype(bool, copy=False)
    require(tuple(map(int, np.count_nonzero(crop, axis=(0, 1)))) == areas,
            "Union crop must retain all original GT and prediction pixels")
    target = crop[:, :, 0]
    h, w = target.shape
    measured = {}
    for index, mode in enumerate(ARMS, start=1):
        pred = np.pad(crop[:, :, index], RADIUS)
        overlaps = {}
        for dy, dx in OFFSETS["joint"]:
            top, left = RADIUS - dy, RADIUS - dx
            overlaps[dy, dx] = int(np.count_nonzero(target & pred[top:top + h, left:left + w]))
        best = {name: max(overlaps[offset] for offset in family) for name, family in OFFSETS.items()}
        base = best["raw"]
        area_sum = areas[0] + areas[index]
        require(2 * base >= area_sum - base and max(best.values()) <= min(areas[0], areas[index]),
                "The unchanged official IoU .50 match and exact fixed areas must reproduce")
        measured[mode] = dict(gt_area=areas[0], pred_area=areas[index], intersection=best)
    return measured


def match_state(native, center):
    if native and center:
        return "both"
    if native:
        return "native_only"
    return "center_only" if center else "neither"


def iou(record, shift):
    overlap = record["intersection"][shift]
    return overlap / (record["gt_area"] + record["pred_area"] - overlap)


def threshold_pass(record, shift, ratio):
    intersection = record["intersection"][shift]
    union = record["gt_area"] + record["pred_area"] - intersection
    numerator, denominator = ratio
    return denominator * intersection >= numerator * union


def summarize(rows):
    """Keep each original fixed GT pair in all translation denominators."""
    count = len(rows)
    require(count > 0, "Nonempty fixed-match population required")
    baseline = {}
    for mode in ARMS:
        base = [row[mode] for row in rows]
        baseline[mode] = dict(
            gt_area_pixels_total=sum(r["gt_area"] for r in base),
            prediction_area_pixels_total=sum(r["pred_area"] for r in base),
            fp_pixels_total=sum(r["pred_area"] - r["intersection"]["raw"] for r in base),
            fn_pixels_total=sum(r["gt_area"] - r["intersection"]["raw"] for r in base),
        )
    result = dict(both_matched=count, fixed_raw_pixel_totals=baseline)
    raw_delta = fsum(iou(row["center"], "raw") - iou(row["native"], "raw") for row in rows) / count
    for shift in OFFSETS:
        mean = {mode: fsum(iou(row[mode], shift) for row in rows) / count for mode in ARMS}
        gain = {mode: fsum(iou(row[mode], shift) - iou(row[mode], "raw") for row in rows) / count
                for mode in ARMS}
        delta = mean["center"] - mean["native"]
        require(min(gain.values()) >= 0 and abs(delta + gain["native"] - gain["center"] - raw_delta) < 1e-12,
                "Identity inclusion and exact paired translation decomposition required")
        thresholds = {}
        for label, ratio in THRESHOLDS.items():
            states = Counter()
            gained = Counter()
            for row in rows:
                matched = [threshold_pass(row[mode], shift, ratio) for mode in ARMS]
                states[match_state(*matched)] += 1
                for mode, new in zip(ARMS, matched, strict=True):
                    old = threshold_pass(row[mode], "raw", ratio)
                    require(not old or new, "Identity inclusion preserves fixed-pair threshold passes")
                    gained[mode] += int(new and not old)
            thresholds[label] = dict(census={key: states[key] for key in ("both", "native_only", "center_only", "neither")},
                                     gain_from_raw=dict(gained))
        result[shift] = dict(mean_iou=mean, center_minus_native_iou_points=100 * delta,
                             mean_gain_from_raw_points={mode: 100 * value for mode, value in gain.items()},
                             native_minus_center_gain_from_raw_points=100 * (gain["native"] - gain["center"]),
                             thresholds=thresholds)
    return result


def paired_measurement(gt, rows, helpers, prior):
    _, _, template, _, official, geometry = helpers
    from pycocotools import mask
    from pycocotools.coco import COCO

    images = gt["images"]
    ids = [row["id"] for row in images]
    annotation_images = {row["id"]: row["image_id"] for row in gt["annotations"]}
    require(len(annotation_images) == len(gt["annotations"]), "Exact unique annotation identities")
    selected, counts = template.prefix_rows(*(template.nonempty_by_image(rows[mode], images) for mode in ARMS))
    require(sum(counts) == prior["prefix_rows_per_arm"] and sum(count > 0 for count in counts) == prior["prefix_images"],
            "Exact original P299 equal nonempty candidate quota")
    matches = {}
    for mode in ARMS:
        _, records, _ = template.evaluate_cell(gt, selected[mode], ids, official)
        matches[mode] = geometry.match_official(records, selected[mode], annotation_images)
        del records
    coco = COCO()
    coco.dataset = deepcopy(gt)
    with redirect_stdout(StringIO()):
        coco.createIndex()

    exclusions = dict(empty=0, crowd=0, ignore=0)
    groups = ("all", "small")
    census = {group: Counter() for group in groups}
    observed = {group: [] for group in groups}
    for annotation in gt["annotations"]:
        aid = annotation["id"]
        target = coco.annToRLE(coco.anns[aid])
        if mask.area(target) == 0:
            exclusions["empty"] += 1
        elif annotation.get("iscrowd", 0):
            exclusions["crowd"] += 1
        elif annotation.get("ignore", 0):
            exclusions["ignore"] += 1
        else:
            strata = groups if annotation["area"] < 1024 else ("all",)
            pair = [matches[mode].get(aid) for mode in ARMS]
            state = match_state(*(row is not None for row in pair))
            for group in strata:
                census[group][state] += 1
            if state == "both":
                record = measure_masks((target, *pair))
                for group in strata:
                    observed[group].append(record)
    require(exclusions == prior["exclusions"]
            and sum(exclusions.values()) + census["all"].total() == len(gt["annotations"]),
            "Exact P299 eligible GT and exclusion accounting")
    results = {}
    for group in groups:
        results[group] = summarize(observed[group])
        previous = prior["strata"][group]
        current = dict(eligible_gt=census[group].total(), both_matched=census[group]["both"],
                       **{key: census[group][key] for key in ("native_only", "center_only", "neither")})
        require(all(previous[key] == value for key, value in current.items())
                and results[group]["both_matched"] == current["both_matched"],
                "Exact P299 same-GT match and stratum census")
        for mode in ARMS:
            reference = previous["paired"][mode]
            require(abs(results[group]["raw"]["mean_iou"][mode] - reference["mean_iou"]) < 1e-12
                    and all(reference[key] == value for key, value in results[group]["fixed_raw_pixel_totals"][mode].items()),
                    "Exact P299 zero-shift IoU and original fixed pixel totals")
        results[group]["official_iou50_match_census"] = current
    return results


def diagnose(args):
    packages = {name: version(name) for name in ("numpy", "pycocotools", "scipy")}
    require(packages == dict(numpy="1.26.4", pycocotools="2.0.11", scipy="1.15.3"),
            "Exact original Windows scoring packages required")
    p299 = load_source(args.code / P299_SOURCE[0], P299_SOURCE[1], "_p310_exact_p299")
    receipt = p299.decode(p299.pinned_bytes(args.accepted, p299.ACCEPTED_SHA256, MIB, "P295 safe"), dict)
    prior = p299.decode(p299.pinned_bytes(args.prior, P299_SAFE_SHA, MIB, "P299 independent safe"), dict)
    require(receipt["status"] == "P286_SSDD_DEV187_INDEPENDENTLY_ACCEPTED"
            and receipt["DEV"] is True and receipt["TEST"] is receipt["FIT"] is receipt["training"] is False
            and receipt["primary_GT"] == "INTEGER_PRIMARY" and receipt["floating_gt_sha256"] is None
            and receipt["coordinator_closed"] is receipt["coordinator_original_os_wait"] is True
            and type(receipt["coordinator_original_os_exit"]) is int and receipt["coordinator_original_os_exit"] == 0
            and prior["status"] == "SSDD_RETRAINED_DEV187_CPU_DIAGNOSTIC_INDEPENDENTLY_READBACK"
            and prior["original_launcher_wait"] is True and type(prior["original_launcher_exit"]) is int
            and prior["original_launcher_exit"] == 0 and prior["original_stderr_empty"] is True,
            "Original closed and independently accepted P295 and P299 integer DEV187 required")
    prior = prior["report"]
    require(prior["status"] == p299.STATUS and prior["training_seed"] == 5201 and prior["DEV"] is True
            and prior["TEST"] is prior["training"] is False and prior["model_forward_calls"] == 0
            and prior["package_versions"] == packages and prior["accepted_p295_sha256"] == p299.ACCEPTED_SHA256,
            "Exact prior P299 pairing and runtime evidence required")
    path = args.code / p299.GEOMETRY_WRAPPER
    helper = load_source(path, p299.GEOMETRY_SHA256, "_p310_exact_geometry")
    helpers = helper.load_helpers(args.code, args.code / p299.GEOMETRY_CORE)
    _, _, template, scoring, official, _ = helpers
    require(receipt["integer_gt_sha256"] == prior["integer_gt_sha256"] == scoring.INTEGER_GT_SHA,
            "Exact original integer GT digest required")
    gt = p299.decode(p299.pinned_bytes(args.integer_gt, scoring.INTEGER_GT_SHA, MIB, "integer GT",
                                     exact_size=scoring.INTEGER_GT_BYTES), dict)
    ids = [image["id"] for image in gt["images"]]
    require(len(ids) == len(set(ids)) == receipt["images"] == scoring.DEV_IMAGES
            and len(gt["annotations"]) == receipt["annotations"] == scoring.DEV_ANNOTATIONS,
            "Exact original integer DEV187 image and annotation census")
    require(args.native != args.center and args.native.name == args.center.name == "PREDICTIONS.json",
            "Two distinct original P295 prediction artifacts required")
    rows, predictions = {}, {}
    for mode in ARMS:
        expected = receipt["arms"][mode]
        digest = predictions[mode] = expected["prediction_sha256"]
        require(digest == prior["prediction_sha256"][mode], "Exact original P295/P299 arm digest")
        rows[mode] = p299.decode(p299.pinned_bytes(getattr(args, mode), digest, 64 * MIB, mode + " predictions"), list)
        nonempty = sum(map(len, template.nonempty_by_image(rows[mode], gt["images"])))
        require(len(rows[mode]) == expected["prediction_count"] and nonempty == expected["nonempty_masks"]
                and len(rows[mode]) - nonempty == expected["all_zero_masks"], "Exact accepted candidate population")
    require(scoring.paired_scores(official, gt, rows, ids) == receipt["primary"] == prior["primary"],
            "Exact full original primary score readback required before fixed-pair diagnostics")
    geometry = prior["fixed_official_iou50_common_gt_geometry"]
    result = paired_measurement(gt, rows, helpers, geometry)
    return dict(status=STATUS, **SCOPE, device="CPU", posthoc_development_diagnostic=True,
                training_seed=prior["training_seed"], training_protocol=receipt["training_protocol"],
                evaluation_protocol=receipt["evaluation_protocol"], package_versions=packages,
                original_primary_scores_reproduced=True, original_fixed_pair_means_and_pixel_totals_reproduced=True,
                accepted_p295_sha256=p299.ACCEPTED_SHA256, prior_p299_sha256=P299_SAFE_SHA,
                integer_gt_sha256=scoring.INTEGER_GT_SHA, predictions_sha256=predictions,
                image_count=len(ids), original_annotation_count=len(gt["annotations"]),
                prefix_rows_per_arm=geometry["prefix_rows_per_arm"], prefix_images=geometry["prefix_images"],
                primary_translation_family="vertical", translation_offsets_yx={key: list(value) for key, value in OFFSETS.items()},
                translation_oracle_scope="fixed GT matching and original mask areas; moved pixels outside the image still count as FP",
                small_definition="original COCO annotation area less than 1024",
                strata=result)


def deny_network(event, _):
    if event.startswith("socket."):
        raise PermissionError("Offline fixed-mask translation diagnostic required")


def main(argv=None):
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    for flag in ("code", "accepted", "prior", "integer-gt", "native", "center", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args(argv)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and sys.flags.isolated == sys.flags.utf8_mode == 1
            and sys.dont_write_bytecode, "Exact isolated Windows CPU flags required")
    require(args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent
            and not args.output.exists() and not args.output.is_symlink(), "Fresh canonical output required")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    sys.addaudithook(deny_network)
    report = diagnose(args)
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with args.output.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=STATUS, result_sha256=sha256(raw).hexdigest()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
