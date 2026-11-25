"""Compare fixed-native-mask boundary scores with paired acquisition bootstrap."""

from collections import defaultdict

import numpy as np


WIDTHS = (1, 2)
N_DRAWS = 4096
SEED = 5217


def binary(mask):
    mask = np.asarray(mask)
    if mask.ndim != 2 or not all(mask.shape):
        raise ValueError("Expected a native two-dimensional mask")
    if mask.dtype.kind != "b" and not np.all((mask == 0) | (mask == 1)):
        raise ValueError("Expected binary mask labels")
    return mask.astype(bool, copy=False)


def summarize_groups(group_values, names, blocks):
    group_order = sorted(group_values)
    if not group_order:
        return {}
    mat = np.asarray([group_values[group] for group in group_order], dtype=np.float64) * 100
    draws = np.random.Generator(np.random.PCG64(SEED)).integers(len(mat), size=(N_DRAWS, len(mat)))
    intervals = np.percentile(mat[draws].mean(axis=1), [2.5, 97.5], axis=0)
    summary = {
        name: {"mean_points": float(mat[:, i].mean()), "paired_95_interval_points": intervals[:, i].tolist()}
        for i, name in enumerate(names)
    }
    if blocks is not None:
        if not set(group_order) <= set(blocks):
            raise ValueError("Missing declared grouping block for an acquisition")
        block_values = defaultdict(list)
        for group, row in zip(group_order, mat, strict=True):
            block_values[blocks[group]].append(row)
        bundles = [np.asarray(rows) for _, rows in sorted(block_values.items())]
        totals = np.asarray([rows.sum(axis=0) for rows in bundles])
        sizes = np.asarray([len(rows) for rows in bundles])
        draws = np.random.Generator(np.random.PCG64(SEED)).integers(len(totals), size=(N_DRAWS, len(totals)))
        paired = totals[draws].sum(axis=1) / sizes[draws].sum(axis=1)[:, None]
        block_ci = np.percentile(paired, [2.5, 97.5], axis=0)
        for i, name in enumerate(names):
            summary[name].update(resample_blocks=len(bundles), block_95_interval_points=block_ci[:, i].tolist())
    return summary


class Audit:
    """Retain exactly one paired score vector per eligible original reference anchor."""

    def __init__(self, boundary_metric, arms, contrasts):
        self.boundary_metric = boundary_metric
        self.arms = ("native", *arms)
        if len(set(self.arms)) != len(self.arms) or not contrasts:
            raise ValueError("Expected distinct immutable arms and declared comparisons")
        self.comparisons = {}
        for name, pairs in contrasts.items():
            indexes = tuple((self.arms.index(treatment), self.arms.index(control)) for treatment, control in pairs)
            if not indexes or any(treatment == control for treatment, control in indexes):
                raise ValueError("Expected at least one valid paired comparison per contrast")
            self.comparisons[name] = indexes
        self.rows = defaultdict(list)

    def add_rles(self, group, reference, native, variants):
        from pycocotools import mask as cocomask

        self.add(group, cocomask.decode(reference), cocomask.decode(native),
                 {name: cocomask.decode(rle) for name, rle in variants.items()})

    def add(self, group, reference, native, variants):
        if set(variants) != set(self.arms[1:]):
            raise ValueError("Mask variants must match all and only declared arms")
        ref = binary(reference)
        masks = (binary(native), *(binary(variants[name]) for name in self.arms[1:]))
        if not ref.any() or not masks[0].any() or any(mask.shape != ref.shape for mask in masks):
            raise ValueError("An eligible native anchor needs a nonempty reference and native mask on the same grid")
        # Cropping at the shared foreground envelope is algebraically equivalent to the full-grid zero-padded ring.
        support = ref.copy()
        for mask in masks:
            support |= mask
        h, w = np.flatnonzero(support.any(axis=1)), np.flatnonzero(support.any(axis=0))
        extent = np.s_[h[0]:h[-1]+1, w[0]:w[-1]+1]
        ref = ref[extent]
        masks = tuple(mask[extent] for mask in masks)
        ref_area = int(ref.sum())
        areas = [int(mask.sum()) for mask in masks]
        scores, saturated, both = [], [], []
        for width in WIDTHS:
            reference_ring = self.boundary_metric.boundary_ring(ref, width)
            rings = [self.boundary_metric.boundary_ring(mask, width) for mask in masks]
            values = [self.boundary_metric.ring_iou(ring, reference_ring) for ring in rings]
            status = [ref_area == int(reference_ring.sum()),
                      *(area > 0 and area == int(ring.sum()) for area, ring in zip(areas, rings, strict=True))]
            scores.append(values)
            saturated.append(status)
            both.append([status[0] and current for current in status[1:]])
        self.rows[str(group)].append((scores, saturated, both, [not area for area in areas]))

    def delta(self, scores):
        return np.column_stack([
            np.column_stack([scores[:, i] - scores[:, j] for i, j in pairs]).mean(axis=1)
            for pairs in self.comparisons.values()
        ])

    def report(self, author_groups=None, blocks=None):
        if not self.rows:
            raise ValueError("At least one original eligible matched authority is needed")
        if author_groups is not None:
            author_groups = {str(key): count for key, count in author_groups.items()}
            if (any(type(count) is not int or count <= 0 for count in author_groups.values())
                    or any(author_groups.get(group, 0) < len(rows) for group, rows in self.rows.items())):
                raise ValueError("Author denominator must cover all original eligible anchors")
        if blocks is not None:
            blocks = {str(group): str(block) for group, block in blocks.items()}
        census = {"anchors": sum(map(len, self.rows.values())), "acquisition_groups": len(self.rows),
                  "bootstrap_draws": N_DRAWS, "bootstrap_seed": SEED}
        empties = np.asarray([row[3] for rows in self.rows.values() for row in rows]).sum(axis=0)
        census["empty_masks"] = {name: int(value) for name, value in zip(self.arms, empties, strict=True)}
        widths = {}
        for i, width in enumerate(WIDTHS):
            deltas, non_saturated, author_sums = {}, {}, {}
            means, sat, all_sat = [], [], []
            informative = 0
            for group in sorted(self.rows):
                rows = self.rows[group]
                scores = np.asarray([row[0][i] for row in rows])
                saturated = np.asarray([row[1][i] for row in rows])
                if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
                    raise ValueError("Boundary IoU on nonempty references must lie in [0,1]")
                differences = self.delta(scores)
                criterion = ~(saturated[:, 0] | saturated[:, 1])
                informative += int(criterion.sum())
                if criterion.any():
                    non_saturated[group] = differences[criterion].mean(axis=0)
                means.append(scores.mean(axis=0))
                deltas[group] = differences.mean(axis=0)
                author_sums[group] = differences.sum(axis=0)
                sat.extend(saturated)
                all_sat.extend(row[2][i] for row in rows)
            sat_count = np.asarray(sat).sum(axis=0)
            both_count = np.asarray(all_sat).sum(axis=0)
            saturation = {
                "reference_whole_mask": int(sat_count[0]),
                "arms_whole_mask": {arm: int(n) for arm, n in zip(self.arms, sat_count[1:], strict=True)},
                "both_reference_and_arm_whole_mask": {arm: int(n) for arm, n in
                                                      zip(self.arms, both_count, strict=True)},
            }
            row = {
                "mean_boundary_iou_points": dict(zip(self.arms, (100 * np.mean(means, axis=0)).tolist(), strict=True)),
                "contrasts": summarize_groups(deltas, self.comparisons, blocks),
                "saturation": saturation,
                "native_reference_non_saturated": {
                    "anchors": informative, "acquisition_groups": len(non_saturated),
                    "contrasts": summarize_groups(non_saturated, self.comparisons, blocks),
                },
            }
            if author_groups is not None:
                zero = np.zeros(len(self.comparisons))
                inclusive = {group: author_sums.get(group, zero) / count for group, count in author_groups.items()}
                row["all_author_zero_delta"] = {
                    "anchors": sum(author_groups.values()), "acquisition_groups": len(author_groups),
                    "contrasts": summarize_groups(inclusive, self.comparisons, blocks),
                }
            widths[str(width)] = row
        return {**census, "widths": widths}
