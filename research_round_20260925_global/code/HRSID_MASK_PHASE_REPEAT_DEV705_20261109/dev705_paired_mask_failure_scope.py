"""Locate fixed DEV705 paired-mask pixel errors by original size and GT edge."""

from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
from heapq import nlargest
from importlib.metadata import version
from io import StringIO
from math import fsum, isfinite

from pycocotools import mask
from pycocotools.coco import COCO

from dev705_group_sensitivity import load_official_adapter
from dev705_paired_coverage import ARMS, CANVAS, DEV_IMAGES, EMPTY_COUNT, GT_COUNT, require
from dev705_paired_mask_geometry import match_official
from dev705_score_template import evaluate_cell, nonempty_by_image, prefix_rows


STRATA = tuple(f"{size}_{location}" for size in ("small", "medium", "large")
               for location in ("edge", "interior"))


def error_counts(target, predicted, gt_area):
    pred_area = int(mask.area(predicted))
    intersection = int(mask.area(mask.merge([target, predicted], intersect=True)))
    union = gt_area + pred_area - intersection
    require(0 < intersection <= min(gt_area, pred_area) and 2 * intersection >= union,
            "Original fixed nonignored .50 match must reproduce on the grid")
    return dict(fp_pixels=pred_area - intersection, fn_pixels=gt_area - intersection, iou=intersection / union)


def compare_masks(target, native_rle, center_rle, gt_area):
    native = error_counts(target, native_rle, gt_area)
    center = error_counts(target, center_rle, gt_area)
    return dict(gt_area_pixels=gt_area, native=native, center=center,
                center_minus_native={key: center[key] - native[key] for key in native})


def aggregate_errors(records, arm, gt_area):
    result = {}
    for kind in ("fp", "fn"):
        field = f"{kind}_pixels"
        pixels = sum(row[arm][field] for row in records)
        result[field] = pixels
        result[f"micro_{kind}_per_gt_area"] = pixels / gt_area if gt_area else None
        result[f"macro_{kind}_per_gt_area"] = (
            fsum(row[arm][field] / row["gt_area_pixels"] for row in records) / len(records)
            if records else None
        )
    return result


def summarize(records):
    gt_area = sum(row["gt_area_pixels"] for row in records)
    result = {arm: aggregate_errors(records, arm, gt_area) for arm in (*ARMS, "center_minus_native")}
    delta = result["center_minus_native"]
    delta["total_error_pixels"] = delta["fp_pixels"] + delta["fn_pixels"]
    delta["mean_iou_change"] = (
        fsum(row["center_minus_native"]["iou"] for row in records) / len(records) if records else None
    )
    increases = sum(row["center_minus_native"]["fn_pixels"] > 0 for row in records)
    decreases = sum(row["center_minus_native"]["fn_pixels"] < 0 for row in records)
    return dict(paired_gt=len(records), gt_area_pixels=gt_area, **result,
                fn_delta_counts=dict(increase=increases, decrease=decreases, tie=len(records) - increases - decreases))


def concentration(records):
    deltas = [row["center_minus_native"]["fn_pixels"] for row in records]
    positive = [delta for delta in deltas if delta > 0]
    positive_total = sum(positive)
    negative_magnitude = -sum(delta for delta in deltas if delta < 0)
    net = positive_total - negative_magnitude
    largest_one = max(positive, default=0)
    largest_five = sum(nlargest(5, positive))
    return dict(paired_gt=len(records), increased_fn_gt=len(positive),
                decreased_fn_gt=sum(delta < 0 for delta in deltas),
                total_positive_fn_delta_pixels=positive_total,
                total_negative_fn_delta_magnitude_pixels=negative_magnitude, net_fn_delta_pixels=net,
                largest_one_positive_fn_delta_pixels=largest_one, largest_five_positive_fn_delta_pixels=largest_five,
                net_without_largest_one_positive_fn_delta_pixels=net - largest_one,
                net_without_largest_five_positive_fn_delta_pixels=net - largest_five)


def stratum(target, area):
    require(type(area) in (int, float) and isfinite(area) and area >= 0,
            "Original COCO GT area must be finite")
    size = "small" if area < 1024 else "medium" if area < 9216 else "large"
    height, width = CANVAS
    x, y, w, h = mask.toBbox(target)
    right, bottom = x + w, y + h
    require(target["size"] == [height, width] and 0 <= x < right <= width and 0 <= y < bottom <= height,
            "Nonempty in-canvas original GT RLE bounds required")
    edge = x == 0 or y == 0 or right == width or bottom == height
    return f"{size}_{'edge' if edge else 'interior'}"


def evaluate_pair(gt, native_rows, center_rows):
    """Aggregate center-minus-native errors on the common official .50 match cohort."""
    require(version("pycocotools") == "2.0.11" and version("numpy") == "1.26.4",
            "Exact original DEV705 CPU COCO runtime required")
    images, annotations = gt["images"], gt["annotations"]
    image_ids = [image["id"] for image in images]
    known_ids = set(image_ids)
    require(len(images) == len(known_ids) == DEV_IMAGES and len(annotations) == GT_COUNT
            and [category["id"] for category in gt["categories"]] == [1]
            and all(type(image["id"]) is int and (image["height"], image["width"]) == CANVAS for image in images),
            "Exact original DEV705 population and canvas required")
    annotation_images = {row["id"]: row["image_id"] for row in annotations}
    require(len(annotation_images) == GT_COUNT
            and all(type(row["id"]) is int and type(row["image_id"]) is int and row["image_id"] in known_ids
                    and type(row["category_id"]) is int and row["category_id"] == 1 for row in annotations),
            "Exact original DEV705 annotation identities required")
    selected, counts = prefix_rows(nonempty_by_image(native_rows, images), nonempty_by_image(center_rows, images))
    adapter = load_official_adapter()
    matches = {}
    for arm in ARMS:
        _, eval_images, _ = evaluate_cell(gt, selected[arm], image_ids, adapter)
        matches[arm] = match_official(eval_images, selected[arm], annotation_images)
        del eval_images

    coco = COCO()
    coco.dataset = deepcopy(gt)
    with redirect_stdout(StringIO()):
        coco.createIndex()
    exclusions = Counter(empty=0, crowd=0, ignore=0)
    groups = {key: [] for key in STRATA}
    for row in annotations:
        aid = row["id"]
        target = coco.annToRLE(coco.anns[aid])
        gt_area = int(mask.area(target))
        if gt_area == 0:
            exclusions["empty"] += 1
        elif row.get("iscrowd", 0):
            exclusions["crowd"] += 1
        elif row.get("ignore", 0):
            exclusions["ignore"] += 1
        elif aid in matches["native"] and aid in matches["center"]:
            key = stratum(target, row["area"])
            groups[key].append(compare_masks(target, matches["native"][aid], matches["center"][aid], gt_area))
    require(exclusions["empty"] == EMPTY_COUNT, "Exact original DEV705 empty raster count required")
    all_records = [record for group in groups.values() for record in group]
    non_small = [record for key, group in groups.items() if not key.startswith("small_") for record in group]
    strata = dict(all=summarize(all_records), **{key: summarize(rows) for key, rows in groups.items()})
    return dict(status="HRSID_DEV705_EQUAL_NONEMPTY_PAIRED_MASK_FAILURE_SCOPE_COMPLETE",
                images=DEV_IMAGES, annotations=GT_COUNT, retained_prefix_rows=sum(counts),
                images_with_prefix_rows=sum(count > 0 for count in counts),
                images_without_prefix_rows=sum(count == 0 for count in counts),
                exclusions=dict(exclusions), strata=strata,
                fn_concentration=dict(all=concentration(all_records), non_small=concentration(non_small)))
