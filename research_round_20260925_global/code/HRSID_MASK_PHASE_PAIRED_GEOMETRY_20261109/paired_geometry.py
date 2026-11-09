"""CPU geometry diagnostics for two independent native-ranked COCO mask streams."""

import io
import math
from collections import Counter
from contextlib import redirect_stdout

import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO


POINT_FIVE = 0.50
STRICT = (0.75, 0.90)
STATES = ("unmatched", "matched_below", "strict")
ARMS = ("native", "center")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def predictions_for(image, block):
    rows, ranks = block["coco"], block["native_ranks"]
    require(
        type(block["raw_query_count"]) is int
        and block["raw_query_count"] >= 100
        and type(block["native_count"]) is int
        and block["native_count"] == 100
        and type(block["unmapped_slot1_count"]) is int
        and block["unmapped_slot1_count"] >= 0
        and block["unmapped_slot1_count"] + len(rows) == 100,
        "Expected original top-100 row accounting",
    )
    require(
        len(rows) == len(ranks) <= 100
        and all(type(rank) is int and 1 <= rank <= 100 for rank in ranks)
        and all(a < b for a, b in zip(ranks, ranks[1:]))
        and all(
            row["image_id"] == image["id"]
            and row["category_id"] == 1
            and math.isfinite(row["score"])
            and 0 <= row["score"] <= 1
            for row in rows
        )
        and all(a["score"] >= b["score"] for a, b in zip(rows, rows[1:])),
        "Expected original ship candidates in native rank order",
    )
    size = [image["height"], image["width"]]
    predictions = [row["segmentation"] for row in rows]
    require(all(rle["size"] == size for rle in predictions), "Expected original prediction canvas")
    return predictions


def targets_for(coco, image, exclusions):
    size = [image["height"], image["width"]]
    targets = []
    for row in coco.imgToAnns[image["id"]]:
        require(row["category_id"] == 1, "Expected original ship annotation")
        rle = coco.annToRLE(row)
        require(rle["size"] == size, "Expected original annotation canvas")
        if mask.area(rle) == 0:
            exclusions["empty"] += 1
        elif row.get("iscrowd", 0):
            exclusions["crowd"] += 1
        elif row.get("ignore", 0):
            exclusions["ignore"] += 1
        else:
            targets.append(rle)
    return targets


def match(predictions, targets):
    """Assign native rows greedily at .50. Equal IoUs prefer later original GT."""
    assigned = [None] * len(targets)
    if not predictions or not targets:
        return assigned
    overlap = mask.iou(predictions, targets, [0] * len(targets))
    require(
        overlap.shape == (len(predictions), len(targets))
        and np.isfinite(overlap).all()
        and (overlap >= 0).all()
        and (overlap <= 1).all(),
        "Expected finite ordinary mask IoU",
    )
    taken = np.zeros(len(targets), dtype=bool)
    for prediction, ious in enumerate(overlap):
        available = np.flatnonzero((ious >= POINT_FIVE) & ~taken)[::-1]
        if available.size:
            target = available[np.argmax(ious[available])]
            taken[target] = True
            assigned[target] = (prediction, float(ious[target]))
    return assigned


def strict_state(assignment, threshold):
    if assignment is None:
        return "unmatched"
    return "strict" if assignment[1] >= threshold else "matched_below"


def geometry(prediction, target, target_y, target_x, iou):
    pixels = mask.decode(prediction).astype(bool, copy=False)
    y, x = np.nonzero(pixels)
    dx, dy = float(x.mean() - target_x), float(y.mean() - target_y)
    return dict(
        fp_pixels=int(np.count_nonzero(pixels & ~target)),
        fn_pixels=int(np.count_nonzero(target & ~pixels)),
        dx_pixels=dx,
        dy_pixels=dy,
        abs_dx_pixels=abs(dx),
        abs_dy_pixels=abs(dy),
        centroid_distance_pixels=math.hypot(dx, dy),
        iou=iou,
    )


def summarize_geometry(sums, count):
    result = {key: sums[key] for key in ("fp_pixels", "fn_pixels")}
    for key in (
        "fp_pixels", "fn_pixels", "dx_pixels", "dy_pixels",
        "abs_dx_pixels", "abs_dy_pixels", "centroid_distance_pixels", "iou",
    ):
        result["mean_" + key] = sums[key] / count if count else None
    return result


def evaluate_pair(coco_gt, native_blocks, center_blocks):
    """Return aggregate geometry diagnostics for the same GT in the two arms.

    Each arm retains its exported rank order. Strict thresholds use the fixed
    .50 assignment. Transition rows are native states and columns are center
    states. Geometry means weight each both-matched GT equally. Signed offsets
    are prediction minus GT, with x rightward and y downward.
    """
    coco = COCO()
    coco.dataset = coco_gt
    with redirect_stdout(io.StringIO()):
        coco.createIndex()
    images = coco_gt["images"]
    image_ids = [image["id"] for image in images]
    require(
        image_ids == [block["image_id"] for block in native_blocks]
        and image_ids == [block["image_id"] for block in center_blocks]
        and len(coco.imgs) == len(images)
        and len(coco.anns) == len(coco_gt["annotations"])
        and [category["id"] for category in coco_gt["categories"]] == [1],
        "Expected both streams in original image order with unique COCO IDs and one ship category",
    )

    exclusions = Counter(empty=0, crowd=0, ignore=0)
    point_five = Counter(both_unmatched=0, native_only=0, center_only=0, both_matched=0)
    strict = {f"{threshold:.2f}": {a: dict.fromkeys(STATES, 0) for a in STATES} for threshold in STRICT}
    totals = {arm: Counter() for arm in ARMS}
    candidates = {arm: dict(mapped=0, empty=0) for arm in ARMS}

    for image, native_block, center_block in zip(images, native_blocks, center_blocks, strict=True):
        targets = targets_for(coco, image, exclusions)
        streams = (predictions_for(image, native_block), predictions_for(image, center_block))
        for arm, stream in zip(ARMS, streams, strict=True):
            candidates[arm]["mapped"] += len(stream)
            if stream:
                candidates[arm]["empty"] += int(np.count_nonzero(mask.area(stream) == 0))
        native_matches, center_matches = (match(stream, targets) for stream in streams)
        for target, native, center in zip(targets, native_matches, center_matches, strict=True):
            states = (native is not None, center is not None)
            point_five[{
                (False, False): "both_unmatched",
                (True, False): "native_only",
                (False, True): "center_only",
                (True, True): "both_matched",
            }[states]] += 1
            for threshold, matrix in zip(STRICT, strict.values(), strict=True):
                matrix[strict_state(native, threshold)][strict_state(center, threshold)] += 1
            if all(states):
                pixels = mask.decode(target).astype(bool, copy=False)
                y, x = np.nonzero(pixels)
                target_y, target_x = y.mean(), x.mean()
                for arm, stream, (prediction, iou) in zip(ARMS, streams, (native, center), strict=True):
                    totals[arm].update(geometry(stream[prediction], pixels, target_y, target_x, iou))

    eligible_gt = point_five.total()
    annotation_count = len(coco_gt["annotations"])
    require(eligible_gt + exclusions.total() == annotation_count, "Expected all annotations to reconcile")
    require(
        all(sum(sum(row.values()) for row in matrix.values()) == eligible_gt for matrix in strict.values()),
        "Expected all strict transitions to reconcile",
    )
    common_count = point_five["both_matched"]
    pair_metrics = {arm: summarize_geometry(totals[arm], common_count) for arm in ARMS}
    pair_metrics["center_minus_native"] = {
        key: None if value is None else pair_metrics["center"][key] - value
        for key, value in pair_metrics["native"].items()
    }
    return dict(
        images=len(images),
        annotations=annotation_count,
        eligible_gt=eligible_gt,
        exclusions=dict(exclusions),
        candidates=candidates,
        point_five=dict(point_five),
        strict=strict,
        common_pair_geometry=dict(count=common_count, **pair_metrics),
    )
