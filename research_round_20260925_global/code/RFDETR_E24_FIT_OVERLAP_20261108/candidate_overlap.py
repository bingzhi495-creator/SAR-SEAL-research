"""Separate overlaps between distinct matched instances from unmatched overlaps."""
from itertools import combinations


def distinct_pair_cases(predicted_iou, match, adjacent_truth):
    matched = [i for i, gt_id in enumerate(match) if gt_id is not None]
    indices = set()
    result = dict(distinct_matched_pairs=0, adjacent_distinct_matched_pairs=0,
                  distinct_matched_overlap_pairs=0, adjacent_distinct_matched_overlap_pairs=0,
                  unmatched_overlaps_matched_ge50=0)
    for i, j in combinations(matched, 2):
        result["distinct_matched_pairs"] += 1
        adjacent = frozenset((match[i], match[j])) in adjacent_truth
        overlaps = bool(predicted_iou[i, j] > 0)
        result["adjacent_distinct_matched_pairs"] += adjacent
        result["distinct_matched_overlap_pairs"] += overlaps
        if adjacent and overlaps:
            result["adjacent_distinct_matched_overlap_pairs"] += 1
            indices.update((i, j))
    result["unmatched_overlaps_matched_ge50"] = sum(
        any(predicted_iou[i, j] >= 0.5 for j in matched)
        for i, gt_id in enumerate(match) if gt_id is None)
    result["adjacent_distinct_matched_overlap_indices"] = sorted(indices)
    return result
