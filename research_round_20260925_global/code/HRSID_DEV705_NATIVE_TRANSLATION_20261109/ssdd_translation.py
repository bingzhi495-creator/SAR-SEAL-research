"""Apply the HRSID historical translation diagnostic to frozen SSDD DEV187."""

import argparse
from collections import defaultdict
from contextlib import redirect_stdout
from hashlib import sha256
from importlib.metadata import version
import io
import json
import math
from pathlib import Path
import re
import sys

from pycocotools import mask
from pycocotools.coco import COCO


ROOT = Path(__file__).resolve().parent
COMMON_SHA = "c0c076b65452a888b2f4e5370882f2dccac4a76fd3efff06ef40ecb80035809d"
CORE_SHA = "c713975658bb3d96261bc6f3d4eb84ee723e375cfde4cbc2f429b66bd14fbbec"
GT_SHA = "5657600d6ea198f79bd5d557d2a1aadce5bec031d2ecf72a98716248d1183a55"
PRED_SHA = "7d6239e87ccc6aed4a012e491007ce4c48edff053d9edf8f04f7177f5e138321"
GT_BYTES, PRED_BYTES = 118319, 2502350
IMAGES, ANNOTATIONS, PREDICTIONS, EMPTY_PREDICTIONS = 187, 380, 15434, 5180
THRESHOLDS = (0.50, 0.75, 0.90)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_common():
    path = ROOT / "score_translation.py"
    raw = path.read_bytes()
    require(0 < len(raw) <= 64 * 1024 and sha256(raw).hexdigest() == COMMON_SHA, "Pinned HRSID diagnostic helpers")
    namespace = dict(__name__="_ssdd_historical_translation", __file__=str(path))
    exec(compile(raw, str(path), "exec"), namespace)
    return namespace


def pinned_json(path, size, digest):
    require(
        path.is_absolute() and path.resolve(strict=True) == path and path.is_file()
        and path.stat().st_size == size,
        "Exact canonical closed input size required",
    )
    with path.open("rb") as stream:
        raw = stream.read(size + 1)
    require(len(raw) == size and sha256(raw).hexdigest() == digest, "Pinned closed input SHA-256 mismatch")
    return json.loads(raw)


def group_predictions(images, rows):
    dimensions = {image["id"]: [image["height"], image["width"]] for image in images}
    positions = {image["id"]: index for index, image in enumerate(images)}
    grouped = defaultdict(list)
    previous_position, previous_score = -1, 1.0
    for row in rows:
        require(
            isinstance(row, dict) and row.keys() == {"image_id", "category_id", "score", "segmentation"}
            and type(row["image_id"]) is int and row["image_id"] in dimensions
            and type(row["category_id"]) is int and row["category_id"] == 1,
            "Exact native mapped ship row required",
        )
        image_id = row["image_id"]
        position = positions[image_id]
        require(position >= previous_position, "Preserve original flat image order")
        if position != previous_position:
            previous_score = 1.0
        score = row["score"]
        require(
            type(score) in (int, float) and math.isfinite(score) and 0 <= score <= previous_score,
            "Preserve per-image native descending score order",
        )
        rle = row["segmentation"]
        require(
            isinstance(rle, dict) and rle.keys() == {"size", "counts"} and rle["size"] == dimensions[image_id]
            and isinstance(rle["counts"], str) and re.fullmatch(r"[0-o]+", rle["counts"]) is not None,
            "Exact original variable canvas and compressed mask required",
        )
        require(len(grouped[image_id]) < 100, "No more than 100 mapped native candidates per image")
        grouped[image_id].append(rle)
        previous_position, previous_score = position, score
    return grouped


def iou_stratum(base):
    tp = base["tp"]
    union = tp + base["fp"] + base["fn"]
    if 100 * tp < 80 * union:
        return "0.75-0.80"
    if 100 * tp < 85 * union:
        return "0.80-0.85"
    return "0.85-0.90"


def summary(gt, predictions, measure, common):
    images, annotations = gt["images"], gt["annotations"]
    require(
        [category["id"] for category in gt["categories"]] == [1]
        and all(type(image[key]) is int for image in images for key in ("id", "height", "width"))
        and all(image["height"] > 0 and image["width"] > 0 for image in images),
        "Original positive SSDD image shapes and one ship category required",
    )
    coco = COCO()
    coco.dataset = gt
    with redirect_stdout(io.StringIO()):
        coco.createIndex()
    require(
        len(coco.imgs) == len(images) and len(coco.anns) == len(annotations)
        and all(annotation["image_id"] in coco.imgs and annotation["category_id"] == 1 for annotation in annotations),
        "Original unique SSDD image and annotation links required",
    )
    grouped = group_predictions(images, predictions)
    summarize = common["shape_and_pixel_summary"]
    strict = common["strict"]
    coverage = dict.fromkeys(THRESHOLDS, 0)
    fixed, near = [], []
    valid_gt = empty_predictions = 0
    for image in images:
        image_id = image["id"]
        rows = grouped.get(image_id, [])
        if rows:
            empty_predictions += int((mask.area(rows) == 0).sum())
        targets, areas = [], []
        for annotation in coco.imgToAnns[image_id]:
            rle = coco.annToRLE(annotation)
            require(rle["size"] == [image["height"], image["width"]], "Expected original target canvas")
            if mask.area(rle) and not annotation.get("iscrowd", 0) and not annotation.get("ignore", 0):
                targets.append(rle)
                areas.append(annotation["area"])
        valid_gt += len(targets)
        if not rows or not targets:
            continue
        overlap = mask.iou(rows, targets, [0] * len(targets))
        best = overlap.max(axis=0)
        for threshold in THRESHOLDS:
            coverage[threshold] += int((best >= threshold).sum())
        for pred_index, gt_index in common["original_pairs"](overlap):
            values = measure(mask.decode(rows[pred_index]).astype(bool, copy=False),
                             mask.decode(targets[gt_index]).astype(bool, copy=False))
            base = values["base"]
            union = base["tp"] + base["fp"] + base["fn"]
            require(abs(base["tp"] / union - overlap[pred_index, gt_index]) <= 1e-15,
                    "Exact fixed native mask intersection and union")
            row = dict(values, group=image_id, small=areas[gt_index] < 1024, elongated=values["elongation"] >= 3)
            fixed.append(row)
            if best[gt_index] < .90:
                require(not strict(base), "Near-strict target cannot have an original strict candidate")
                near.append(row)

    fixed_shifts = {name: common["fixed_summary"](fixed, name) for name in ("fixed_up", "fixed_down")}
    return dict(
        images=len(images), annotations=len(annotations), valid_gt=valid_gt, mapped_predictions=len(predictions),
        zero_prediction_images=len(images) - len(grouped), empty_predictions=empty_predictions,
        independent_native_coverage={f"{key:.2f}": value for key, value in coverage.items()},
        fixed_pairs=len(fixed), strict_on_fixed_pairs=fixed_shifts["fixed_up"]["initial_strict"],
        group_unit="logical_images", near=summarize(near),
        strata_by_area={name: summarize([row for row in near if row["small"] is flag])
                        for name, flag in (("small", True), ("non_small", False))},
        strata_by_shape={name: summarize([row for row in near if row["elongated"] is flag])
                         for name, flag in (("elongated", True), ("non_elongated", False))},
        strata_by_initial_iou={name: summarize([row for row in near if iou_stratum(row["base"]) == name])
                               for name in ("0.75-0.80", "0.80-0.85", "0.85-0.90")},
        fixed_one_pixel=fixed_shifts,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    require(
        out.is_absolute() and out.parent.resolve(strict=True) == out.parent and not out.exists() and not out.is_symlink(),
        "Fresh canonical output required",
    )
    require(version("numpy") == "1.26.4" and version("pycocotools") == "2.0.11", "Canonical CPU mask runtime")
    common = load_common()
    sys.addaudithook(common["no_network"])
    measure = common["load_code"](ROOT / "translation_probe.py", CORE_SHA)["measure"]
    predictions = pinned_json(args.predictions, PRED_BYTES, PRED_SHA)
    gt = pinned_json(args.gt, GT_BYTES, GT_SHA)
    require(len(gt["images"]) == IMAGES and len(gt["annotations"]) == ANNOTATIONS
            and len(predictions) == PREDICTIONS, "Exact historical SSDD DEV187 population")
    result = summary(gt, predictions, measure, common)
    require(
        result["valid_gt"] == ANNOTATIONS and result["zero_prediction_images"] == 0
        and result["empty_predictions"] == EMPTY_PREDICTIONS
        and result["independent_native_coverage"] == {"0.50": 377, "0.75": 351, "0.90": 103},
        "Historical native SSDD candidate coverage must reconcile",
    )
    result.update(
        status="SSDD_DEV187_HISTORICAL_TRANSLATION_COMPLETE",
        original_gt_sha256=GT_SHA, native_predictions_sha256=PRED_SHA,
        common_diagnostic_sha256=COMMON_SHA, translation_core_sha256=CORE_SHA,
        native_top_rows=IMAGES * 100, unmapped_slot1_rows=IMAGES * 100 - PREDICTIONS,
        dev_role="internal_development", official_TEST_opened=False, AP=False,
    )
    data = (json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n").encode()
    with out.open("xb") as stream:
        stream.write(data)


if __name__ == "__main__":
    main()
