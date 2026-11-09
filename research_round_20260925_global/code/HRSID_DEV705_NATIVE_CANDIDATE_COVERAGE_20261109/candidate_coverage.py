"""Describe fixed candidate coverage without changing the baseline COCO score."""
import argparse
from collections import Counter
from contextlib import redirect_stdout
from hashlib import sha256
import io
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

THRESHOLDS = (.50, .75, .90)
BIN_LABELS = ("zero", "(0,.50)", "[.50,.75)", "[.75,.90)", "[.90,1]")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def graph_counts(overlap, threshold):
    """Return eligible GT, exact maximum cardinality, and ranked greedy matches."""
    edges = overlap >= threshold
    eligible = int(edges.any(axis=0).sum())
    if not eligible:
        return 0, 0, 0
    matching = maximum_bipartite_matching(csr_matrix(edges), perm_type="column")
    maximum = int(np.count_nonzero(matching >= 0))
    taken = np.zeros(edges.shape[1], dtype=bool)
    for index, row in enumerate(edges):
        available = np.flatnonzero(row & ~taken)[::-1]
        if available.size:
            selected = available[np.argmax(overlap[index, available])]
            taken[selected] = True
    greedy = int(taken.sum())
    require(0 <= greedy <= maximum <= eligible, "Invalid coverage decomposition")
    return eligible, maximum, greedy


def bins(best):
    # Bin zero separately from positive overlap below .50.
    result = np.searchsorted(THRESHOLDS, best, side="right") + (best > 0)
    return np.bincount(result, minlength=len(BIN_LABELS))


def census(gt, blocks):
    coco = COCO()
    coco.dataset = gt
    with redirect_stdout(io.StringIO()):
        coco.createIndex()
    images = gt["images"]
    require([image["id"] for image in images] == [b["image_id"] for b in blocks]
            and len(coco.imgs) == len(images) and len(coco.anns) == len(gt["annotations"])
            and [category["id"] for category in gt["categories"]] == [1],
            "Exact original image order, unique IDs, and one ship category")
    exclusions = Counter(empty=0, crowd=0, ignore=0)
    counts = np.zeros((len(THRESHOLDS), 3), dtype=np.int64)
    area_bins = {key: np.zeros(len(BIN_LABELS), dtype=np.int64) for key in ("small", "non_small")}
    annotation_count = prediction_count = empty_predictions = without_predictions = multiple_gt = 0

    for image, block in zip(images, blocks, strict=True):
        image_id = image["id"]
        annotations = coco.imgToAnns[image_id]
        annotation_count += len(annotations)
        rows = block["coco"]
        ranks = block["native_ranks"]
        require(len(ranks) == len(rows) <= 100 and ranks == sorted(set(ranks))
                and all(type(rank) is int and 1 <= rank <= 100 for rank in ranks)
                and all(row["image_id"] == image_id and row["category_id"] == 1
                        and math.isfinite(row["score"]) and 0 <= row["score"] <= 1 for row in rows)
                and all(a["score"] >= b["score"] for a, b in zip(rows, rows[1:])),
                "Exact mapped candidate identity and native rank order")
        predictions = [row["segmentation"] for row in rows]
        prediction_count += len(rows)
        without_predictions += not rows
        size = [image["height"], image["width"]]
        require(all(rle["size"] == size for rle in predictions), "Original prediction canvas")
        if predictions:
            empty_predictions += int(np.count_nonzero(mask.area(predictions) == 0))

        targets, small = [], []
        for row in annotations:
            require(row["category_id"] == 1, "Original ship GT")
            rle = coco.annToRLE(row)
            require(rle["size"] == size, "Original GT canvas")
            if mask.area(rle) == 0:
                exclusions["empty"] += 1
            elif row.get("iscrowd", 0):
                exclusions["crowd"] += 1
            elif row.get("ignore", 0):
                exclusions["ignore"] += 1
            else:
                require(math.isfinite(row["area"]) and row["area"] >= 0, "Original COCO area")
                targets.append(rle)
                small.append(row["area"] < 1024)
        multiple_gt += len(targets) >= 2

        overlap = (mask.iou(predictions, targets, [0] * len(targets))
                   if predictions and targets else np.zeros((len(predictions), len(targets))))
        require(overlap.shape == (len(predictions), len(targets))
                and np.isfinite(overlap).all() and (overlap >= 0).all() and (overlap <= 1).all(),
                "Finite ordinary mask IoU")
        best = overlap.max(axis=0, initial=0)
        small = np.asarray(small, dtype=bool)
        area_bins["small"] += bins(best[small])
        area_bins["non_small"] += bins(best[~small])
        for index, threshold in enumerate(THRESHOLDS):
            counts[index] += graph_counts(overlap, threshold)

    best_bins = area_bins["small"] + area_bins["non_small"]
    eligible_gt = int(best_bins.sum())
    require(eligible_gt + sum(exclusions.values()) == annotation_count == len(gt["annotations"]),
            "Exact GT population reconciliation")
    result = {}
    for index, (independent, maximum, greedy) in enumerate(counts):
        require(int(best_bins[index + 2:].sum()) == independent, "Bins and threshold eligibility differ")
        result[f"{THRESHOLDS[index]:.2f}"] = {
            "independent": int(independent),
            "maximum_one_to_one": int(maximum),
            "native_greedy": int(greedy),
            "candidate_quality_gap": eligible_gt - int(independent),
            "shared_candidate_gap": int(independent - maximum),
            "rank_or_assignment_gap": int(maximum - greedy),
        }
    return dict(images=len(images), annotations=annotation_count, eligible_gt=eligible_gt,
                exclusions=dict(exclusions), predictions=prediction_count, empty_predictions=empty_predictions,
                images_without_candidates=without_predictions, images_with_multiple_eligible_gt=multiple_gt,
                bin_labels=BIN_LABELS, best_iou_bins=best_bins.tolist(),
                best_iou_bins_by_coco_area={key: value.tolist() for key, value in area_bins.items()},
                thresholds=result)


def read_pinned(path, digest, limit):
    require(0 < path.stat().st_size <= limit, "Bounded closed input")
    raw = path.read_bytes()
    require(0 < len(raw) <= limit and sha256(raw).hexdigest() == digest, "Exact closed input SHA-256")
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--gt-sha256", required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--predictions-sha256", required=True)
    args = parser.parse_args()
    started = perf_counter()
    gt = json.loads(read_pinned(args.gt, args.gt_sha256, 4 * 1024**2))
    blocks = [json.loads(row) for row in
              read_pinned(args.predictions, args.predictions_sha256, 32 * 1024**2).splitlines()]
    require(len(gt["images"]) == len(blocks) == 705 and len(gt["annotations"]) == 1782,
            "Exact frozen HRSID DEV population")
    require(all(row["raw_query_count"] == 164 and row["native_count"] == 100
                and row["unmapped_slot1_count"] + len(row["coco"]) == 100 for row in blocks),
            "Exact frozen native candidate population")
    result = census(gt, blocks)
    require(result["predictions"] == 62536 and result["exclusions"]["empty"] == 4,
            "Exact closed predictions and empty annotation counts")
    result.update(status="ACTUAL_HRSID_DEV705_NATIVE_CANDIDATE_COVERAGE", gt_sha256=args.gt_sha256,
                  predictions_sha256=args.predictions_sha256, elapsed_seconds=perf_counter() - started,
                  forward_calls=0, AP=False, official_test_opened=False)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
