"""Compare emitted native-order DEV705 candidate sets on identical GT."""

from collections import Counter
from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
from pathlib import Path
import re
from types import ModuleType

import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO


SOURCE = Path(__file__).resolve().parent.parent / "HRSID_DEV705_NATIVE_CANDIDATE_COVERAGE_20261109/candidate_coverage.py"
SOURCE_SHA = "de18b2d993af7549e437add80c72b20633a3e596170f6b4789188539c7cc25fa"
DEV_IMAGES, GT_COUNT, EMPTY_COUNT = 705, 1782, 4
CANVAS = (800, 800)
NATIVE_ROWS = 100
ARMS = ("native", "center")
SCOPES = ("full", "equal_mapped_prefix", "equal_nonempty_prefix")
AREA_KEYS = ("all", "small", "non_small")
TRANSITION_KEYS = ("neither", "center_only", "native_only", "both")
THRESHOLDS = (.50, .75, .90)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def graph_counts_fn():
    require(SOURCE.resolve(strict=True) == SOURCE, "Exact canonical historical coverage source required")
    limit = 64 * 1024
    with SOURCE.open("rb") as stream:
        raw = stream.read(limit + 1)
    require(0 < len(raw) <= limit and sha256(raw).hexdigest() == SOURCE_SHA,
            "Exact immutable historical coverage source required")
    module = ModuleType("_paired_dev705_coverage_matching")
    exec(compile(raw, str(SOURCE), "exec"), module.__dict__)
    return module.graph_counts


def group_rows(rows, images):
    """Validate, then retain the given per-image native order including ties."""
    require(type(rows) is list, "Expected the admitted flat native prediction list")
    image_index = {image["id"]: i for i, image in enumerate(images)}
    groups = [[] for _ in images]
    previous_index, previous_score = -1, 1
    for row in rows:
        require(type(row) is dict and row.keys() == {"image_id", "category_id", "score", "segmentation"}
                and type(row["image_id"]) is int and row["image_id"] in image_index
                and type(row["category_id"]) is int and row["category_id"] == 1,
                "Exact mapped original ship prediction rows required")
        index = image_index[row["image_id"]]
        score, rle = row["score"], row["segmentation"]
        require(index >= previous_index and type(score) in (int, float) and 0 <= score <= 1
                and (index != previous_index or score <= previous_score) and len(groups[index]) < NATIVE_ROWS,
                "Original image and nonincreasing native score order required")
        image = images[index]
        require(type(rle) is dict and rle.keys() == {"size", "counts"}
                and rle["size"] == [image["height"], image["width"]] and type(rle["counts"]) is str
                and re.fullmatch(r"[0-o]+", rle["counts"]) is not None,
                "Exact compressed original-canvas prediction RLE required")
        groups[index].append(rle)
        previous_index, previous_score = index, score
    return groups


def targets_for_image(coco, image, exclusions):
    targets, small = [], []
    for annotation in coco.imgToAnns[image["id"]]:
        rle = coco.annToRLE(annotation)
        require(rle["size"] == [image["height"], image["width"]], "Exact original GT raster canvas required")
        if mask.area(rle) == 0:
            exclusions["empty"] += 1
        elif annotation.get("iscrowd", 0):
            exclusions["crowd"] += 1
        elif annotation.get("ignore", 0):
            exclusions["ignore"] += 1
        else:
            area = annotation["area"]
            require(type(area) in (int, float) and 0 <= area < float("inf"), "Finite original COCO GT area required")
            targets.append(rle)
            small.append(area < 1024)
    return targets, np.asarray(small, dtype=bool)


def candidate_subsets(predictions, targets):
    overlaps, empties = {}, {}
    for mode, rows in predictions.items():
        overlaps[mode] = (mask.iou(rows, targets, [0] * len(targets)) if rows and targets
                          else np.zeros((len(rows), len(targets))))
        empties[mode] = mask.area(rows) == 0 if rows else np.zeros(0, dtype=bool)
        require(overlaps[mode].shape == (len(rows), len(targets)) and np.isfinite(overlaps[mode]).all()
                and (overlaps[mode] >= 0).all() and (overlaps[mode] <= 1).all(), "Finite ordinary original-grid IoU required")
    mapped = min(map(len, overlaps.values()))
    nonempty = min(int(np.count_nonzero(~empty)) for empty in empties.values())
    return {
        "full": {mode: (overlaps[mode], empties[mode]) for mode in ARMS},
        "equal_mapped_prefix": {mode: (overlaps[mode][:mapped], empties[mode][:mapped]) for mode in ARMS},
        "equal_nonempty_prefix": {mode: (overlaps[mode][~empties[mode]][:nonempty], np.zeros(nonempty, dtype=bool))
                                  for mode in ARMS},
    }


def report_threshold(independent, maximum, greedy, eligible_gt):
    return dict(independent=independent, maximum_one_to_one=maximum, native_greedy=greedy,
                candidate_quality_gap=eligible_gt - independent, shared_candidate_gap=independent - maximum,
                rank_or_assignment_gap=maximum - greedy)


def evaluate_pair(gt, native_rows, center_rows):
    """Compute aggregate candidate-coverage diagnostics, with the population fixed in both arms."""
    images, annotations = gt["images"], gt["annotations"]
    require(len(images) == DEV_IMAGES and len(annotations) == GT_COUNT
            and [cat["id"] for cat in gt["categories"]] == [1]
            and all(type(image["id"]) is int and (image["width"], image["height"]) == CANVAS for image in images),
            "Exact fixed original DEV705 population and canvas required")
    image_ids = {image["id"] for image in images}
    require(len(image_ids) == len(images) and len({row["id"] for row in annotations}) == len(annotations)
            and all(type(row["id"]) is int and type(row["image_id"]) is int and row["image_id"] in image_ids
                    and type(row["category_id"]) is int and row["category_id"] == 1 for row in annotations),
            "Exact original GT IDs and ship category required")
    grouped = dict(native=group_rows(native_rows, images), center=group_rows(center_rows, images))
    coco = COCO()
    coco.dataset = gt
    with redirect_stdout(StringIO()):
        coco.createIndex()
    graph_counts = graph_counts_fn()
    exclusions = Counter(empty=0, crowd=0, ignore=0)
    eligible_gt = small_gt = 0
    thresholds = tuple(f"{value:.2f}" for value in THRESHOLDS)
    counts = {scope: {mode: np.zeros((len(THRESHOLDS), 3), dtype=np.int64) for mode in ARMS} for scope in SCOPES}
    transition = {scope: np.zeros((len(THRESHOLDS), len(AREA_KEYS), 4), dtype=np.int64) for scope in SCOPES}
    population = {scope: {mode: Counter(predictions=0, empty_predictions=0, images_without_candidates=0)
                          for mode in ARMS} for scope in SCOPES}
    for index, image in enumerate(images):
        targets, small = targets_for_image(coco, image, exclusions)
        eligible_gt += len(targets)
        small_gt += int(small.sum())
        subsets = candidate_subsets({mode: grouped[mode][index] for mode in ARMS}, targets)
        for scope, candidates in subsets.items():
            for mode, (overlap, empty) in candidates.items():
                population[scope][mode].update(predictions=len(overlap), empty_predictions=int(empty.sum()),
                                               images_without_candidates=int(len(overlap) == 0))
            for t, threshold in enumerate(THRESHOLDS):
                for mode in ARMS:
                    counts[scope][mode][t] += graph_counts(candidates[mode][0], threshold)
                native, center = ((candidates[mode][0] >= threshold).any(axis=0) for mode in ARMS)
                codes = 2 * native.astype(np.int8) + center
                for a, subset in enumerate((codes, codes[small], codes[~small])):
                    transition[scope][t, a] += np.bincount(subset, minlength=4)
    require(exclusions["empty"] == EMPTY_COUNT and eligible_gt + sum(exclusions.values()) == len(annotations),
            "Exact DEV705 GT exclusions and denominator required")

    by_area = dict(small=small_gt, non_small=eligible_gt - small_gt)
    results = {}
    for scope in SCOPES:
        matrix = transition[scope]
        require((matrix.sum(axis=2) == [eligible_gt, *by_area.values()]).all()
                and (matrix[:, 0] == matrix[:, 1] + matrix[:, 2]).all()
                and all((matrix[:, 0, arm_only] + matrix[:, 0, 3] == counts[scope][mode][:, 0]).all()
                        for mode, arm_only in (("native", 2), ("center", 1))),
                "Paired candidate transitions and marginal counts must reconcile")
        results[scope] = dict(
            arms={mode: dict(population[scope][mode], thresholds={
                name: report_threshold(*counts[scope][mode][t].tolist(), eligible_gt) for t, name in enumerate(thresholds)
            }) for mode in ARMS},
            any_candidate_transitions={name: {
                key: dict(zip(TRANSITION_KEYS, matrix[t, a].tolist(), strict=True)) for a, key in enumerate(AREA_KEYS)
            } for t, name in enumerate(thresholds)},
        )
        if scope != "full":
            native_count, center_count = (population[scope][mode]["predictions"] for mode in ARMS)
            require(native_count == center_count, "Matched candidate prefix counts must agree")
            results[scope]["rows_per_arm"] = native_count
    return dict(status="HRSID_DEV705_PAIRED_CANDIDATE_COVERAGE_COMPLETE", images=len(images),
                annotations=len(annotations), eligible_gt=eligible_gt, exclusions=dict(exclusions),
                eligible_gt_by_coco_area=by_area, **results, COCO_AP=False, official_TEST_opened=False,
                candidate_scope="score-selected mapped native top100 rows, preserving original order",
                prefix_scope="per-image minimum mapped or nonempty count with original arm order and native-rank cutoffs",
                assignment_scope="per-image diagnostic at each fixed IoU on eligible GT only")
