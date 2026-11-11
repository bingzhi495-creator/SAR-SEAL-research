"""Optimize continuous IoU over unit-pixel edits with locked support."""

from dataclasses import dataclass
from itertools import accumulate
from operator import index

import numpy as np


@dataclass(frozen=True)
class EditChoice:
    iou: float
    intersection: float
    pred_area: float
    adds: tuple[int, ...]
    deletes: tuple[int, ...]

    @property
    def flips(self):
        return len(self.adds) + len(self.deletes)


def _prefer(candidate, current):
    """Compare (intersection, union, adds, deletes) using exact ratios."""
    if current is None:
        return True
    cross_product_delta = candidate[0] * current[1] - current[0] * candidate[1]
    if cross_product_delta:
        return cross_product_delta > 0
    candidate_counts = candidate[2:]
    current_counts = current[2:]
    return (sum(candidate_counts), candidate_counts) < (sum(current_counts), current_counts)


def solve_edit_oracles(pred_area, truth_area, intersection, occupancy, inside, max_pairs):
    """Return balanced cap, flexible cap, and flexible at balanced-used flips.

    Support must consist of distinct disjoint unit cells of a fixed mask grid.
    Each occupancy is the continuous truth intersection area of one cell.
    The solver treats the supplied finite floating areas as exact dyadic
    numbers for objective comparisons and returns original support indices.
    """
    max_pairs = index(max_pairs)
    if max_pairs < 0:
        raise ValueError("max_pairs must be nonnegative")
    occupancy = np.asarray(occupancy, dtype=np.float64)
    inside = np.asarray(inside)
    if occupancy.ndim != 1 or inside.shape != occupancy.shape or inside.dtype != np.bool_:
        raise ValueError("occupancy and Boolean inside must be aligned vectors")
    if not np.all(np.isfinite(occupancy) & (0 <= occupancy) & (occupancy <= 1)):
        raise ValueError("occupancy must be in [0, 1]")

    pred_area, truth_area, intersection = map(float, (pred_area, truth_area, intersection))
    if not np.all(np.isfinite((pred_area, truth_area, intersection))):
        raise ValueError("areas must be finite")
    if truth_area <= 0 or not 0 <= intersection <= min(pred_area, truth_area):
        raise ValueError("areas must describe a valid mask and positive-area truth")
    if pred_area < np.count_nonzero(inside):
        raise ValueError("inside support exceeds predicted area")

    max_flips = min(2 * max_pairs, occupancy.size)
    outside_indices = np.flatnonzero(~inside)
    inside_indices = np.flatnonzero(inside)
    adds = outside_indices[np.argsort(-occupancy[outside_indices], kind="stable")][:max_flips]
    deletes = inside_indices[np.argsort(occupancy[inside_indices], kind="stable")][:max_flips]

    values = (pred_area, truth_area, intersection, *occupancy[adds], *occupancy[deletes])
    ratios = [float(value).as_integer_ratio() for value in values]
    scale = max(denominator for _, denominator in ratios)
    area, truth, overlap, *cell_areas = (
        numerator * (scale // denominator) for numerator, denominator in ratios
    )
    add_prefix = list(accumulate(cell_areas[:len(adds)], initial=0))
    delete_prefix = list(accumulate(cell_areas[len(adds):], initial=0))

    balanced = None
    flexible = None
    best_at_count = [None] * (max_flips + 1)
    for add_count, added in enumerate(add_prefix):
        delete_limit = min(len(deletes), max_flips - add_count)
        for delete_count in range(delete_limit + 1):
            flips = add_count + delete_count
            edited_overlap = overlap + added - delete_prefix[delete_count]
            edited_area = area + scale * (add_count - delete_count)
            union = edited_area + truth - edited_overlap
            candidate = edited_overlap, union, add_count, delete_count
            if add_count == delete_count and _prefer(candidate, balanced):
                balanced = candidate
            if _prefer(candidate, flexible):
                flexible = candidate
            if _prefer(candidate, best_at_count[flips]):
                best_at_count[flips] = candidate

    def describe(candidate):
        edited_overlap, union, add_count, delete_count = candidate
        return EditChoice(
            iou=edited_overlap / union,
            intersection=edited_overlap / scale,
            pred_area=(area + scale * (add_count - delete_count)) / scale,
            adds=tuple(map(int, adds[:add_count])),
            deletes=tuple(map(int, deletes[:delete_count])),
        )

    balanced_flips = sum(balanced[2:])
    return describe(balanced), describe(flexible), describe(best_at_count[balanced_flips])
