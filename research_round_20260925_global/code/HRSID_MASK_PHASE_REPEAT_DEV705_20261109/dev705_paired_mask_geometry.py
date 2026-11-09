"""Measure closed DEV705 mask geometry on fixed official .50 matches."""

from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
from importlib.metadata import version
from io import StringIO
from math import fsum, isfinite

import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO

from dev705_group_sensitivity import load_official_adapter
from dev705_paired_coverage import ARMS, CANVAS, DEV_IMAGES, EMPTY_COUNT, GT_COUNT, require
from dev705_score_template import evaluate_cell, nonempty_by_image, prefix_rows


AREAS = ("all", "small")
DIRECTIONS = ("up", "down")
MEANS = ("iou", "gt_area_pixels", "prediction_area_pixels", "pred_to_gt_area",
         "fp_per_gt_area", "fn_per_gt_area", "signed_dx_pixels", "signed_dy_pixels",
         "abs_dx_pixels", "abs_dy_pixels")
TOTALS = ("gt_area_pixels", "prediction_area_pixels", "fp_pixels", "fn_pixels")


def match_official(eval_images, selected, annotation_images):
    """Resolve official all-area detection IDs to the unchanged admitted prefix rows."""
    assigned, seen_gt, seen_dt = {}, set(), set()
    for record in eval_images:
        if record is None or list(record["aRng"]) != [0, 1e10]:
            continue
        iid = record["image_id"]
        gt_ids, dt_ids = record["gtIds"], record["dtIds"]
        require(record["category_id"] == 1 and record["maxDet"] == 100
                and len(gt_ids) == len(record["gtIgnore"]) and len(dt_ids) == len(record["dtScores"])
                and record["dtMatches"].shape == record["dtIgnore"].shape == (10, len(dt_ids))
                and record["gtMatches"].shape == (10, len(gt_ids)),
                "Exact all-area original COCO match tensor shapes required")
        gt_position = {}
        for i, aid in enumerate(gt_ids):
            require(aid not in seen_gt and annotation_images.get(aid) == iid,
                    "Exact original COCO annotation image and identity required")
            seen_gt.add(aid)
            gt_position[aid] = i
        for i, did in enumerate(dt_ids):
            require(isinstance(did, (int, np.integer)) and 1 <= did <= len(selected) and did not in seen_dt,
                    "Exact original COCO loadRes detection ordinal required")
            seen_dt.add(did)
            row = selected[did - 1]
            require(row["image_id"] == iid and float(row["score"]).hex() == float(record["dtScores"][i]).hex(),
                    "Original matching ordinal must preserve image and score")
            value = record["dtMatches"][0, i]
            if value == 0 or record["dtIgnore"][0, i]:
                continue
            require(isfinite(value) and value == int(value) and int(value) in gt_position,
                    "Exact official .50 nonignored match required")
            aid = int(value)
            position = gt_position[aid]
            require(not record["gtIgnore"][position] and record["gtMatches"][0, position] == did
                    and aid not in assigned, "Exact mutual official nonignored .50 assignment required")
            assigned[aid] = row["segmentation"]
    require(seen_gt == set(annotation_images) and len(seen_dt) == len(selected),
            "Original all-area COCO matching must account for every GT and prefix detection")
    return assigned


def centroid(bitmap):
    y, x = np.nonzero(bitmap)
    return float(x.mean()), float(y.mean())


def measure_masks(rles):
    """Decode only the three matched RLEs and measure a one-pixel padded union crop."""
    height, width = rles[0]["size"]
    require(all(rle["size"] == [height, width] for rle in rles), "Exact shared original RLE canvas required")
    boxes = mask.toBbox(rles)
    bounds = boxes.astype(np.int64)
    require(boxes.shape == (3, 4) and np.array_equal(boxes, bounds) and (bounds[:, :2] >= 0).all()
            and (bounds[:, 2:] > 0).all(), "Nonempty exact integer RLE bounding boxes required")
    left, top = bounds[:, :2].min(axis=0)
    rights, bottoms = (bounds[:, :2] + bounds[:, 2:]).T
    require((rights <= width).all() and (bottoms <= height).all(), "Exact in-canvas RLE bounds required")
    crop = mask.decode(rles)[max(0, top-1):min(height, int(bottoms.max())+1), left:int(rights.max()), :]
    target, native, center = (crop[..., i].astype(bool, copy=False) for i in range(3))
    areas = tuple(map(int, mask.area(rles)))
    require(tuple(np.count_nonzero(view) for view in (target, native, center)) == areas and min(areas) > 0,
            "Exact original RLE areas retained by union crop required")
    shared_interior = bool((bounds[1:, 1] > 0).all() and (bottoms[1:] < height).all())
    target_x, target_y = centroid(target)
    metrics, directional = {}, {}
    for mode, prediction, pred_area in zip(ARMS, (native, center), areas[1:], strict=True):
        gt_area = areas[0]
        intersection = int(np.count_nonzero(prediction & target))
        union = gt_area + pred_area - intersection
        require(2 * intersection >= union, "Original fixed nonignored .50 match must reproduce on the grid")
        fp, fn = pred_area - intersection, gt_area - intersection
        pred_x, pred_y = centroid(prediction)
        dx, dy = pred_x - target_x, pred_y - target_y
        iou = intersection / union
        metrics[mode] = dict(iou=iou, gt_area_pixels=gt_area, prediction_area_pixels=pred_area,
                             pred_to_gt_area=pred_area / gt_area, fp_pixels=fp, fn_pixels=fn,
                             fp_per_gt_area=fp / gt_area, fn_per_gt_area=fn / gt_area,
                             signed_dx_pixels=dx, signed_dy_pixels=dy,
                             abs_dx_pixels=abs(dx), abs_dy_pixels=abs(dy))
        if shared_interior:
            require(not prediction[0].any() and not prediction[-1].any(), "Both shifts must preserve foreground")
            overlaps = (np.count_nonzero(prediction[1:] & target[:-1]),
                        np.count_nonzero(prediction[:-1] & target[1:]))
            directions = {}
            base_strict = 10 * intersection >= 9 * union
            for direction, shifted_intersection in zip(DIRECTIONS, map(int, overlaps), strict=True):
                shifted_union = gt_area + pred_area - shifted_intersection
                shifted_iou = shifted_intersection / shifted_union
                shifted_strict = 10 * shifted_intersection >= 9 * shifted_union
                directions[direction] = dict(base_iou=iou, shifted_iou=shifted_iou, delta_iou=shifted_iou-iou,
                                             base_strict90=base_strict, shifted_strict90=shifted_strict,
                                             gains_strict90=shifted_strict and not base_strict,
                                             losses_strict90=base_strict and not shifted_strict)
            directional[mode] = directions
    return metrics, directional


def average(rows, key):
    return fsum(row[key] for row in rows) / len(rows) if rows else None


def summarize(coverage, records):
    metrics = {}
    directional = dict(shared_interior=sum(bool(shifts) for _, shifts in records))
    directional["boundary_excluded"] = len(records) - directional["shared_interior"]
    for mode in ARMS:
        rows = [values[mode] for values, _ in records]
        metrics[mode] = {**{f"mean_{key}": average(rows, key) for key in MEANS},
                         **{f"{key}_total": sum(row[key] for row in rows) for key in TOTALS}}
        directions = {}
        for direction in DIRECTIONS:
            shifted = [shifts[mode][direction] for _, shifts in records if shifts]
            directions[direction] = {
                **{f"mean_{key}": average(shifted, key) for key in ("base_iou", "shifted_iou", "delta_iou")},
                **{key: sum(row[key] for row in shifted)
                   for key in ("base_strict90", "shifted_strict90", "gains_strict90", "losses_strict90")},
            }
        directional[mode] = directions
    both, native_only, center_only, neither = (coverage[key] for key in ("both", "native", "center", "neither"))
    require(both == len(records), "Common official matches and paired measurements must reconcile")
    return dict(eligible_gt=coverage.total(), native_matched=both + native_only, center_matched=both + center_only,
                both_matched=both, native_only=native_only, center_only=center_only, neither=neither,
                paired=metrics, directional=directional)


def evaluate_pair(gt, native_rows, center_rows):
    """Compare paired masks and fixed vertical translations on the original pixel grid."""
    require(version("pycocotools") == "2.0.11" and version("numpy") == "1.26.4",
            "Exact original DEV705 CPU COCO runtime required")
    images, annotations = gt["images"], gt["annotations"]
    require(len(images) == DEV_IMAGES and len(annotations) == GT_COUNT
            and [category["id"] for category in gt["categories"]] == [1]
            and all(type(image["id"]) is int and (image["height"], image["width"]) == CANVAS for image in images),
            "Exact original DEV705 population and canvas required")
    image_ids = [image["id"] for image in images]
    known_ids = set(image_ids)
    require(len(known_ids) == DEV_IMAGES and len({row["id"] for row in annotations}) == GT_COUNT
            and all(type(row["id"]) is int and type(row["image_id"]) is int and row["image_id"] in known_ids
                    and type(row["category_id"]) is int and row["category_id"] == 1 for row in annotations),
            "Exact original DEV705 annotation identities required")
    annotation_images = {row["id"]: row["image_id"] for row in annotations}
    selected, counts = prefix_rows(nonempty_by_image(native_rows, images), nonempty_by_image(center_rows, images))
    adapter = load_official_adapter()
    matches = {}
    for mode in ARMS:
        _, eval_images, _ = evaluate_cell(gt, selected[mode], image_ids, adapter)
        matches[mode] = match_official(eval_images, selected[mode], annotation_images)
        del eval_images

    coco = COCO()
    coco.dataset = deepcopy(gt)
    with redirect_stdout(StringIO()):
        coco.createIndex()
    exclusions = Counter(empty=0, crowd=0, ignore=0)
    coverage = {key: Counter() for key in AREAS}
    pairs = {key: [] for key in AREAS}
    for row in annotations:
        aid, area = row["id"], row["area"]
        target = coco.annToRLE(coco.anns[aid])
        if mask.area(target) == 0:
            exclusions["empty"] += 1
        elif row.get("iscrowd", 0):
            exclusions["crowd"] += 1
        elif row.get("ignore", 0):
            exclusions["ignore"] += 1
        else:
            require(type(area) in (int, float) and isfinite(area) and area >= 0,
                    "Original COCO GT area must be finite")
            strata = AREAS if area < 1024 else ("all",)
            left, right = (matches[mode].get(aid) for mode in ARMS)
            state = "both" if left is not None and right is not None else (
                "native" if left is not None else "center" if right is not None else "neither")
            for key in strata:
                coverage[key][state] += 1
            if state == "both":
                measurements = measure_masks([target, left, right])
                for key in strata:
                    pairs[key].append(measurements)
    require(exclusions["empty"] == EMPTY_COUNT and coverage["all"].total() + exclusions.total() == GT_COUNT,
            "Exact original eligible and excluded DEV705 GT counts required")
    return dict(status="HRSID_DEV705_EQUAL_NONEMPTY_PAIRED_MASK_GEOMETRY_COMPLETE",
                images=DEV_IMAGES, annotations=GT_COUNT, retained_prefix_rows=sum(counts),
                images_with_prefix_rows=sum(count > 0 for count in counts),
                images_without_prefix_rows=sum(count == 0 for count in counts), exclusions=dict(exclusions),
                strata={key: summarize(coverage[key], pairs[key]) for key in AREAS})

