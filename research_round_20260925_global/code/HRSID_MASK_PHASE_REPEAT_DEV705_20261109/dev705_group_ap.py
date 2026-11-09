"""Reaccumulate canonical COCO mask AP under DEV705 image-group multiplicities."""

from dataclasses import dataclass
import math

import numpy as np


METRICS = ("AP", "AP75")
MAX_IMAGES = 705
MAX_GROUPS = 28
MAX_DETECTIONS = 100
IOU_THRESHOLDS = np.linspace(.50, .95, 10)
RECALL_THRESHOLDS = np.linspace(0., 1., 101)
AP75_INDEX = 5


def require(ok, message):
    if not ok:
        raise ValueError(message)


@dataclass(frozen=True)
class PositiveBlocks:
    """False-positive counts needed at each true positive under image cloning."""

    group: np.ndarray
    tp_start: np.ndarray
    tp_length: np.ndarray
    fp_by_group_before_block: np.ndarray
    fp_in_block: np.ndarray
    fp_before_tp_inside_block: np.ndarray

    def precision(self, multiplicities, gt_count):
        copies = multiplicities[self.group]
        block = np.repeat(np.arange(len(copies), dtype=np.int32), copies)
        if block.size == 0:
            return np.zeros(len(RECALL_THRESHOLDS), dtype=np.float64)
        clone = np.arange(block.size, dtype=np.int32) - np.repeat(np.cumsum(copies) - copies, copies)
        length = self.tp_length[block]
        destination = np.cumsum(length) - length
        tp_index = np.repeat(self.tp_start[block] - destination, length) + np.arange(int(length.sum()))
        earlier = self.fp_by_group_before_block @ multiplicities
        fp_count = (np.repeat(earlier[block] + clone * self.fp_in_block[block], length)
                    + self.fp_before_tp_inside_block[tp_index])
        tp_count = np.arange(1, len(tp_index) + 1, dtype=np.float64)
        precision = tp_count / (tp_count + fp_count + np.spacing(1))
        # Between true positives, precision can only stay constant or decrease.
        envelope = np.maximum.accumulate(precision[::-1])[::-1]
        positions = np.searchsorted(tp_count / gt_count, RECALL_THRESHOLDS, side="left")
        return np.append(envelope, 0.)[positions]


@dataclass(frozen=True)
class GroupAP:
    groups: tuple
    image_groups: tuple
    image_gt_counts: tuple
    gt_by_group: np.ndarray
    thresholds: tuple

    def scores(self, group_multiplicities):
        """Return AP and AP75 fractions in the sorted `groups` order."""
        values = np.asarray(group_multiplicities)
        require(values.shape == (len(self.groups),) and np.issubdtype(values.dtype, np.integer)
                and np.all((values >= 0) & (values <= MAX_GROUPS)),
                "Expected a nonnegative integer multiplicity for each DEV image group")
        weights = values.astype(np.int32)
        require(1 <= int(weights.sum()) <= MAX_GROUPS, "DEV group draw must contain 1 to 28 groups")
        gt_count = int(self.gt_by_group @ weights)
        if gt_count == 0:
            return dict.fromkeys(METRICS, -1.0)
        precision = np.stack([
            np.zeros(len(RECALL_THRESHOLDS), dtype=np.float64) if row is None
            else row.precision(weights, gt_count)
            for row in self.thresholds
        ])
        return dict(AP=float(precision.mean()), AP75=float(precision[AP75_INDEX].mean()))


def positive_blocks(tp, fp, ranked_group, block_ids, starts, ends, group_count):
    positions = np.flatnonzero(tp)
    if positions.size == 0:
        return None
    blocks, tp_start, tp_length = np.unique(block_ids[positions], return_index=True, return_counts=True)
    block_first = starts[blocks]
    fp_prefix = np.empty(len(fp) + 1, dtype=np.int32)
    fp_prefix[0] = 0
    np.cumsum(fp, dtype=np.int32, out=fp_prefix[1:])
    fp_positions = np.flatnonzero(fp)
    fp_groups = ranked_group[fp_positions]
    earlier = np.column_stack([
        np.searchsorted(fp_positions[fp_groups == group], block_first, side="left")
        for group in range(group_count)
    ]).astype(np.int32)
    return PositiveBlocks(ranked_group[block_first], tp_start, tp_length, earlier,
                          fp_prefix[ends[blocks]] - fp_prefix[block_first],
                          fp_prefix[positions] - fp_prefix[starts[block_ids[positions]]])


def prepare_group_ap(evaluator, image_groups):
    """Cache one evaluated pycocotools segm arm and immutable group membership.

    With repeated images, equal-score detection order follows ascending original
    image ID, copy ordinal and the cached within-image detection order.
    """
    params = evaluator._paramsEval
    require(params.iouType == "segm" and params.useCats == 1 and list(params.catIds) == [1]
            and list(params.maxDets) == [1, 10, MAX_DETECTIONS]
            and np.array_equal(params.iouThrs, IOU_THRESHOLDS)
            and np.array_equal(params.recThrs, RECALL_THRESHOLDS),
            "Expected canonical single-category COCO segmentation AP settings")
    labels = list(params.areaRngLbl)
    require(labels.count("all") == 1 and len(labels) == len(params.areaRng),
            "Expected canonical COCO all-area block")
    area_index = labels.index("all")
    require(list(params.areaRng[area_index]) == [0, 1e10], "Unexpected COCO all-area range")

    image_ids = tuple(params.imgIds)
    require(1 <= len(image_ids) <= MAX_IMAGES
            and all(isinstance(i, (int, np.integer)) and not isinstance(i, bool) and i >= 0 for i in image_ids)
            and tuple(sorted(set(image_ids))) == image_ids and set(image_groups) == set(image_ids),
            "Expected a sorted unique COCO image roster and matching group membership")
    membership = tuple((int(i), image_groups[i]) for i in image_ids)
    require(all(type(group) is str and group for _, group in membership), "Expected named image groups")
    groups = tuple(sorted({group for _, group in membership}))
    require(1 <= len(groups) <= MAX_GROUPS, "DEV evaluation is bounded to 28 image groups")
    group_index = {group: index for index, group in enumerate(groups)}

    require(len(evaluator.evalImgs) == len(labels) * len(image_ids),
            "Expected canonical category-area-image cache dimensions")
    offset = area_index * len(image_ids)
    scores, owners, tp_flags, fp_flags = [], [], [], []
    image_gt_counts = []
    gt_by_group = np.zeros(len(groups), dtype=np.int64)
    for position, (image_id, group) in enumerate(membership):
        row = evaluator.evalImgs[offset + position]
        if row is None:
            image_gt_counts.append(0)
            continue
        require(row["image_id"] == image_id and row["category_id"] == 1 and row["maxDet"] == MAX_DETECTIONS
                and list(row["aRng"]) == [0, 1e10], "Cached COCO image/category/area identity mismatch")
        ranked = np.asarray(row["dtScores"], dtype=np.float64)
        matches = np.asarray(row["dtMatches"])
        ignored = np.asarray(row["dtIgnore"])
        gt_ignored = np.asarray(row["gtIgnore"])
        require(ranked.ndim == 1 and ranked.size <= MAX_DETECTIONS
                and np.isfinite(ranked).all() and np.all(ranked[:-1] >= ranked[1:])
                and matches.shape == ignored.shape == (len(IOU_THRESHOLDS), ranked.size)
                and matches.dtype.kind in "iuf" and np.isfinite(matches).all() and (matches >= 0).all()
                and (matches == np.floor(matches)).all() and ignored.dtype == np.bool_ and gt_ignored.ndim == 1
                and np.all((gt_ignored == 0) | (gt_ignored == 1)),
                "Invalid cached COCO scores, matches, or ignore flags")
        positive = int(np.count_nonzero(gt_ignored == 0))
        image_gt_counts.append(positive)
        gt_by_group[group_index[group]] += positive
        tp = (matches > 0) & ~ignored
        require(np.all(tp.sum(axis=1) <= positive), "Cached COCO true positives exceed the image's valid GT")
        scores.append(ranked)
        owners.append(np.full(ranked.size, position, dtype=np.int32))
        tp_flags.append(tp)
        fp_flags.append((matches == 0) & ~ignored)

    thresholds = (None,) * len(IOU_THRESHOLDS)
    if any(value.size for value in scores):
        flat_scores = np.concatenate(scores)
        order = np.argsort(-flat_scores, kind="mergesort")
        sorted_scores = flat_scores[order]
        image_owner = np.concatenate(owners)[order]
        image_group_index = np.asarray([group_index[group] for _, group in membership], dtype=np.int32)
        ranked_group = image_group_index[image_owner]
        boundary = np.r_[True, (sorted_scores[1:] != sorted_scores[:-1]) | (image_owner[1:] != image_owner[:-1])]
        block_ids = np.cumsum(boundary, dtype=np.int32) - 1
        starts = np.flatnonzero(boundary)
        ends = np.r_[starts[1:], len(order)]
        tps = np.concatenate(tp_flags, axis=1)[:, order]
        fps = np.concatenate(fp_flags, axis=1)[:, order]
        thresholds = tuple(positive_blocks(tp, fp, ranked_group, block_ids, starts, ends, len(groups))
                           for tp, fp in zip(tps, fps, strict=True))

    prepared = GroupAP(groups, membership, tuple(image_gt_counts), gt_by_group, thresholds)
    if "precision" in evaluator.eval:
        precision = evaluator.eval["precision"][:, :, 0, area_index, -1]
        values = dict(zip(METRICS, (precision, precision[AP75_INDEX]), strict=True))
        expected = {key: float(valid.mean()) if (valid := row[row > -1]).size else -1.
                    for key, row in values.items()}
        assert_all_ones(prepared, expected)
    return prepared


def assert_all_ones(prepared, expected):
    """Verify cache scoring against the full-population original COCO result."""
    actual = prepared.scores(np.ones(len(prepared.groups), dtype=np.int32))
    require(all(math.isclose(actual[key], float(expected[key]), rel_tol=0.0, abs_tol=1e-12) for key in METRICS),
            "All-ones DEV AP/AP75 disagree with the original COCO score")
    return actual


def score_pair(native, center, group_multiplicities):
    """Evaluate both arms with the exact same image-group multiplicities."""
    require(native.image_groups == center.image_groups and native.image_gt_counts == center.image_gt_counts,
            "Paired DEV cache image membership and valid GT counts must agree")
    scores_a = native.scores(group_multiplicities)
    scores_b = center.scores(group_multiplicities)
    delta = {key: scores_b[key] - scores_a[key] if scores_a[key] >= 0 and scores_b[key] >= 0 else None
             for key in METRICS}
    return dict(native=scores_a, center=scores_b, center_minus_native=delta)

