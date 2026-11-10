"""Exact-in-model horizontal-run overlaps and paired source-image bootstrap."""

import numpy as np
import shapely
from shapely.geometry import Polygon, box


BOOTSTRAP_SEED = 251114
BOOTSTRAP_DRAWS = 4096
TIE_TOLERANCE = 1e-10


def foreground_runs(mask):
    """Return (row, inclusive x0, exclusive x1) for disjoint horizontal runs."""
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    starts = np.r_[0, np.flatnonzero((rows[1:] != rows[:-1]) | (columns[1:] != columns[:-1] + 1)) + 1]
    ends = np.r_[starts[1:] - 1, rows.size - 1]
    return np.column_stack((rows[starts], columns[starts], columns[ends] + 1))


def run_footprint_metrics(rings, hw, candidates):
    """The pilot's continuous polygon-area IoU with disjoint cell runs."""
    if not candidates or any(mask.shape != tuple(hw) or mask.dtype != np.bool_ for mask in candidates.values()):
        raise ValueError("Boolean masks on the same final grid required")
    height, width = hw
    reference = shapely.union_all([Polygon(ring) for ring in rings]).intersection(box(0, 0, width, height))
    visible = float(reference.area)
    runs = {name: foreground_runs(mask) for name, mask in candidates.items()}
    unique, inverse = np.unique(np.concatenate(tuple(runs.values()), axis=0), axis=0, return_inverse=True)
    row, left, right = unique.T
    overlap = np.asarray(shapely.area(shapely.intersection(shapely.box(left, row, right, row + 1), reference)),
                         dtype=np.float64)[inverse]
    spans = right - left
    if (not np.isfinite(overlap).all() or np.any(overlap < -1e-10)
            or np.any(overlap > spans[inverse] + 1e-10)):
        raise ArithmeticError("Invalid continuous polygon-run intersection area")
    result = {}
    offset = 0
    for name, arm_runs in runs.items():
        area = int(np.sum(arm_runs[:, 2] - arm_runs[:, 1], dtype=np.int64))
        next_offset = offset + len(arm_runs)
        intersect = float(overlap[offset:next_offset].sum(dtype=np.float64))
        offset = next_offset
        if intersect > min(area, visible) + 1e-8:
            raise ArithmeticError("Polygon intersection exceeds a corresponding support")
        intersect = min(intersect, area, visible)
        union = area + visible - intersect
        result[name] = dict(
            iou=intersect / union if union else None, fp_area=area - intersect, fn_area=visible - intersect,
        )
    return dict(visible_polygon_area=visible, by_arm=result)


def bootstrap_draws(images, *, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """Draw the same source-image cluster multiplicities for all readouts."""
    if type(images) is not int or images < 1 or type(draws) is not int or draws < 1:
        raise ValueError("Positive source-image cluster and draw counts required")
    generator = np.random.Generator(np.random.PCG64(seed))
    samples = generator.integers(0, images, size=(draws, images))
    return np.asarray([np.bincount(row, minlength=images) for row in samples], dtype=np.int64)


def paired_cluster_stats(values, multiplicities):
    """Summarize paired differences without treating views as independent.

    values contains one (source_image_ordinal, candidate_minus_reference)
    tuple for each row with both values defined. Absent images contribute zero
    numerator and denominator in the common full-population bootstrap.
    """
    cluster_count = multiplicities.shape[1]
    groups = np.asarray([image for image, _ in values], dtype=np.int64)
    delta = np.asarray([value for _, value in values], dtype=np.float64)
    if np.any(groups < 0) or np.any(groups >= cluster_count) or not np.isfinite(delta).all():
        raise ValueError("Expected finite paired differences on the frozen image population")
    counts = np.bincount(groups, minlength=cluster_count)
    sums = np.bincount(groups, weights=delta, minlength=cluster_count)
    present = counts > 0
    image_count = int(np.count_nonzero(present))
    means = np.divide(sums, counts, out=np.zeros(cluster_count, dtype=np.float64), where=present)
    row_denominator = multiplicities @ counts
    image_denominator = multiplicities @ present.astype(np.int64)
    valid = image_denominator > 0
    undefined = int(np.count_nonzero(~valid))

    def interval(total, denominator):
        if image_count < 2 or not valid.any():
            return None
        values = (multiplicities @ total)[valid] / denominator[valid]
        return np.quantile(values, [0.025, 0.975]).tolist()

    return dict(
        paired_rows=int(delta.size), source_images=image_count,
        min_rows_per_contributing_image=int(counts[present].min()) if image_count else None,
        max_rows_per_contributing_image=int(counts[present].max()) if image_count else None,
        wins=int(np.count_nonzero(delta > TIE_TOLERANCE)),
        losses=int(np.count_nonzero(delta < -TIE_TOLERANCE)),
        ties=int(np.count_nonzero(np.abs(delta) <= TIE_TOLERANCE)),
        row_weighted_mean_delta=float(delta.mean()) if delta.size else None,
        image_equal_mean_delta=float(means[present].mean()) if image_count else None,
        row_weighted_image_cluster_percentile95=interval(sums, row_denominator),
        image_equal_image_cluster_percentile95=interval(means, image_denominator),
        undefined_draws=undefined,
    )
