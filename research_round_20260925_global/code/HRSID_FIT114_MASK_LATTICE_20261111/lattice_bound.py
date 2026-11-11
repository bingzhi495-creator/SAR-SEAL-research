"""A FIT-only lower bound on errors of a thresholded bilinear mask lattice.

Inside a bilinear cell, fixing one coordinate makes the logit an affine
function of the other. Its sign along a row or column can change at most once.
Allow each run its own optimal sign transition. Summed over rows, or separately
over columns, the relaxed cost is a lower bound. Their maximum remains a lower
bound, and output centers belong to disjoint cells, so cell bounds can be added.

For ground truth area A and IoU target t=p/q, q*I-p*U is
(q-p)*A-q*FN-p*FP. Both selected fractions have q-p=1. A lower bound on the
weighted errors strictly above A proves IoU below t for every exact bilinear
logit lattice on that grid. Bounds at or below A do not certify attainability.
This is an exact-coordinate, real-arithmetic statement, not an FP32 guarantee
for sign evaluations of nearly zero logits.
"""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from types import ModuleType

import numpy as np

WEIGHTS = {"75": (4, 3), "90": (10, 9)}
NODES = (144, 288)
SIZE = 800
P317_SHA256 = "261f533956651b6cd3868ebfa91624ed5a6b8efa7b40b69cb26254d3b09dd05d"


def axis_cell_ids(nodes, pixels):
    """Partition output centers by interpolation interval using exact arithmetic."""
    if nodes < 2 or pixels < nodes:
        raise ValueError("A coarse axis must have at least two nodes and no more than pixels")
    # IDs -1 and nodes-1 denote the left and right constant-clamped intervals.
    centers = 2 * np.arange(pixels, dtype=np.int64) + 1
    return (centers * nodes - pixels) // (2 * pixels)


def one_transition_cost(truth, axis, fn_weight, fp_weight):
    """Minimize weighted error over prefixes and suffixes of predicted ones."""
    all_background = fn_weight * np.count_nonzero(truth, axis=axis)
    gain_per_foreground = np.where(truth, fn_weight, -fp_weight)
    prefix_gain = np.cumsum(gain_per_foreground, axis=axis, dtype=np.int32)
    suffix_gain = np.take(prefix_gain, -1, axis=axis) - prefix_gain.min(axis=axis, initial=0)
    best_gain = np.maximum(prefix_gain.max(axis=axis, initial=0), suffix_gain)
    return all_background - best_gain


def covering_runs(nodes, pixels, first, last):
    """Return pixel indices of complete cells intersecting an axis-aligned span."""
    _, starts, lengths = np.unique(axis_cell_ids(nodes, pixels), return_index=True, return_counts=True)
    selected = (starts <= last) & (starts + lengths > first)
    starts, lengths = starts[selected], lengths[selected]
    return [starts[lengths == length, None] + np.arange(length)
            for length in np.unique(lengths)]


def cell_cost_bounds(truth, nodes):
    """Sum disjoint-cell maxima of horizontal and vertical error lower bounds."""
    truth = np.asarray(truth, dtype=bool)
    rows = np.flatnonzero(truth.any(axis=1))
    cols = np.flatnonzero(truth.any(axis=0))
    totals = dict.fromkeys(WEIGHTS, 0)
    if rows.size == 0:
        return totals
    y_runs = covering_runs(nodes, truth.shape[0], rows[0], rows[-1])
    x_runs = covering_runs(nodes, truth.shape[1], cols[0], cols[-1])
    for ys in y_runs:
        for xs in x_runs:
            cells = truth[ys[:, None, :, None], xs[None, :, None, :]]
            for threshold, (fn, fp) in WEIGHTS.items():
                row_cost = one_transition_cost(cells, -1, fn, fp).sum(axis=-1)
                col_cost = one_transition_cost(cells, -2, fn, fp).sum(axis=-1)
                totals[threshold] += int(np.maximum(row_cost, col_cost).sum())
    return totals


def summarize(records):
    """Summarize paired certificates for all and decoded-pixel size strata."""
    out = {}
    for size, lower, upper in (("all", 1, SIZE**2 + 1), ("small", 1, 32**2),
                               ("medium", 32**2, 96**2), ("large", 96**2, SIZE**2 + 1)):
        rows = [(group, area, costs) for group, area, costs in records if lower <= area < upper]
        entry = out[size] = dict(instances=len(rows), source_groups=len({group for group, _, _ in rows}))
        for nodes in NODES:
            entry[str(nodes)] = {}
            for threshold in WEIGHTS:
                ratios = [costs[nodes][threshold] / area for _, area, costs in rows]
                affected_groups = {group for group, area, costs in rows if costs[nodes][threshold] > area}
                entry[str(nodes)][threshold] = dict(
                    positive_cost_lower_bound=sum(costs[nodes][threshold] > 0 for _, _, costs in rows),
                    certified_below_threshold=sum(costs[nodes][threshold] > area for _, area, costs in rows),
                    groups_with_certificate=len(affected_groups),
                    bound_over_area_mean=float(np.mean(ratios)) if ratios else None,
                    bound_over_area_max=max(ratios) if ratios else None,
                )
        comparison = entry["paired_288_vs_144"] = {}
        for threshold in WEIGHTS:
            pairs = [(area, costs[144][threshold], costs[288][threshold]) for _, area, costs in rows]
            comparison[threshold] = dict(
                bound_reduced=sum(fine < coarse for _, coarse, fine in pairs),
                bound_equal=sum(fine == coarse for _, coarse, fine in pairs),
                bound_increased=sum(fine > coarse for _, coarse, fine in pairs),
                certified_144_only=sum(coarse > area >= fine for area, coarse, fine in pairs),
                certified_288_only=sum(fine > area >= coarse for area, coarse, fine in pairs),
                certified_both=sum(min(coarse, fine) > area for area, coarse, fine in pairs),
            )
    return out


def run(p317_source, bundle):
    source = p317_source.read_bytes()
    if sha256(source).hexdigest() != P317_SHA256:
        raise ValueError("Exact published P317 FIT-only loader source required")
    p317 = ModuleType("p317_fit_loader")
    exec(compile(source, str(p317_source), "exec"), p317.__dict__)
    cohort, index = p317.load_scope(bundle)
    roles, groups = p317.split_roles(cohort)
    if p317.split_digest(roles, groups) != p317.ROLE_SHA:
        raise ValueError("Original FIT114 role digest differs")
    masks = p317.ground_truth(index, cohort)
    core = [row["image_id"] for row in cohort if row["core"]]
    records = []
    for image_id in core:
        for rle in masks[image_id]:
            mask = p317.codec.decode(rle).astype(bool, copy=False)
            area = int(np.count_nonzero(mask))
            if mask.shape != (SIZE, SIZE) or area != int(p317.codec.area(rle)):
                raise ValueError("Decoded FIT mask geometry differs from the pinned loader")
            records.append((groups[image_id], area, {nodes: cell_cost_bounds(mask, nodes) for nodes in NODES}))
    return dict(
        status="HRSID_FIT114_BILINEAR_LATTICE_CAPACITY_LOWER_BOUND_COMPLETE_V1",
        loader_sha256=P317_SHA256,
        fit_bundle_sha256=p317.BUNDLE_SHA,
        fit_index_sha256=p317.INDEX_SHA,
        fit_cohort_sha256=p317.COHORT_SHA,
        fit_role_sha256=p317.ROLE_SHA,
        population=dict(core_images=len(core), stress_images_excluded=len(cohort) - len(core),
                        core_source_groups=len({groups[image_id] for image_id in core})),
        geometry=dict(output_shape=[SIZE, SIZE], input_axes=list(NODES), mode="bilinear",
                      align_corners=False, arithmetic="exact_coordinate_real_logit_relaxation",
                      size_bins="decoded_foreground_area_at_32_squared_and_96_squared",
                      per_threshold_fn_fp_weights=WEIGHTS),
        scope=dict(FIT=True, DEV=False, TEST=False, AP=False, predictions=False,
                   image_members_decoded=False, model_forward_calls=0, detector_optimizer_updates=0),
        strata=summarize(records),
    )


def main():
    cli = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("p317-source", "bundle", "output"):
        cli.add_argument("--" + name, type=Path, required=True)
    args = cli.parse_args()
    for path in (args.p317_source, args.bundle):
        if not path.is_absolute() or path.resolve(strict=True) != path or not path.is_file():
            raise ValueError("Canonical input file required")
    if (not args.output.is_absolute() or args.output.parent.resolve(strict=True) != args.output.parent
            or args.output.exists() or args.output.is_symlink()):
        raise ValueError("Fresh canonical output file required")
    result = run(args.p317_source, args.bundle)
    payload = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with args.output.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=result["status"], result_sha256=sha256(payload).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
