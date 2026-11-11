"""Construct masks on two exact-coordinate bilinear grids from FIT ground truth.

Each signed EDT is sampled at the centers of a candidate coarse grid and
resampled to the original image. Subtracting a fixed threshold at every
coarse node is equivalent to subtracting it after exact bilinear interpolation
because the weights sum to one. Every threshold therefore describes a valid
lattice field. A successful guarded IoU is evidence of representability for
that target. An unsuccessful construction leaves feasibility unknown unless
the separate analytical bound certifies otherwise.
"""

import argparse
from fractions import Fraction
from hashlib import sha256
import json
import os
from pathlib import Path
from types import ModuleType

import numpy as np
from scipy.ndimage import distance_transform_edt

THRESHOLDS = (-1.0, -0.5, 0.0, 0.5, 1.0)
FLOAT_GUARD = 1e-9
P320_SHA256 = "1eed8181c07cec38209c5e41c2bbd4b27600cf0da58eb07b41b4daf731a81374"


def resize_axis(values, count, axis):
    """Resize using integer center-coordinate numerators and clamped borders."""
    values = np.moveaxis(values, axis, 0)
    old = values.shape[0]
    den = 2 * count
    num = np.clip((2 * np.arange(count) + 1) * old - count, 0, den * (old - 1))
    left, weight = np.divmod(num, den)
    right = np.minimum(left + 1, old - 1)
    resized = (values[left] * (den - weight[:, None]) + values[right] * weight[:, None]) / den
    return np.moveaxis(resized, 0, axis)


def resize_bilinear(values, shape):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(shape) != 2 or min(shape) < 1 or min(values.shape) < 1:
        raise ValueError("Positive two-dimensional shapes required")
    return resize_axis(resize_axis(values, shape[1], 1), shape[0], 0)


def signed_distance(truth):
    """Distance inside minus distance outside, in original image pixel units."""
    truth = np.asarray(truth, dtype=bool)
    if truth.ndim != 2 or not truth.any():
        raise ValueError("A nonempty 2D ground truth mask is required")
    inside = distance_transform_edt(np.pad(truth, 1))[1:-1, 1:-1]
    return inside - distance_transform_edt(~truth)


def threshold_witnesses(truth, field):
    """Return conservative intersection, union and sign-ambiguity counts."""
    truth = np.asarray(truth, dtype=bool)
    if truth.shape != field.shape or not np.isfinite(field).all():
        raise ValueError("Aligned finite field required")
    inside = field[truth]
    outside = field[~truth]
    witnesses = {}
    for threshold in THRESHOLDS:
        lower = threshold - FLOAT_GUARD
        upper = threshold + FLOAT_GUARD
        intersection = int(np.count_nonzero(inside > upper))
        false_positive = int(np.count_nonzero(outside >= lower))
        ambiguous = (inside.size - intersection - np.count_nonzero(inside < lower)
                     + false_positive - np.count_nonzero(outside > upper))
        witnesses[threshold] = (intersection, inside.size + false_positive, int(ambiguous))
    return witnesses


def summarize(records, grids, weights):
    """Pair constructive witnesses with the analytical states on the same GTs."""
    summary = {}
    for name, lo, hi in (("all", 1, np.inf), ("small", 1, 32**2),
                         ("medium", 32**2, 96**2), ("large", 96**2, np.inf)):
        rows = [(group, area, bounds, results) for group, area, bounds, results in records if lo <= area < hi]
        entry = summary[name] = dict(instances=len(rows), source_groups=len({row[0] for row in rows}))
        states = {}
        for grid in grids:
            entry[str(grid)] = {}
            best = [max((Fraction(i, u) for i, u, _ in results[grid].values()))
                    for _, _, _, results in rows]
            for target, (q, p) in weights.items():
                goal = Fraction(p, q)
                impossible = [bounds[grid][target] > area for _, area, bounds, _ in rows]
                attainable = [iou >= goal for iou in best]
                if any(yes and no for yes, no in zip(attainable, impossible)):
                    raise ValueError("Constructive witness contradicts the analytical certificate")
                states[grid, target] = (attainable, impossible)
                per_threshold = {}
                for threshold in THRESHOLDS:
                    fixed = [results[grid][threshold] for _, _, _, results in rows]
                    per_threshold[str(threshold)] = dict(
                        witnesses=sum(q * i >= p * u for i, u, _ in fixed),
                        ambiguous_pixels=sum(ambiguous for _, _, ambiguous in fixed),
                    )
                entry[str(grid)][target] = dict(
                    constructive_witnesses=sum(attainable),
                    analytic_certified_below=sum(impossible),
                    unknown=len(rows) - sum(attainable) - sum(impossible),
                    zero_threshold_witnesses=per_threshold["0.0"]["witnesses"],
                    groups_with_witness=len({row[0] for row, yes in zip(rows, attainable) if yes}),
                    mean_best_guarded_iou=float(np.mean([float(iou) for iou in best])) if best else None,
                    fixed_thresholds=per_threshold,
                )
        comparison = entry["paired_288_vs_144"] = {}
        for target in weights:
            coarse, coarse_fail = states[144, target]
            fine, fine_fail = states[288, target]
            comparison[target] = dict(
                witnessed_both=sum(a and b for a, b in zip(coarse, fine)),
                witnessed_144_only=sum(a and not b for a, b in zip(coarse, fine)),
                witnessed_288_only=sum(b and not a for a, b in zip(coarse, fine)),
                witnessed_neither=sum(not a and not b for a, b in zip(coarse, fine)),
                **{"144_impossible_288_witness": sum(a and b for a, b in zip(coarse_fail, fine)),
                   "288_impossible_144_witness": sum(a and b for a, b in zip(fine_fail, coarse))},
            )
    return summary


def load_pinned(path, expected_sha, label):
    source = path.read_bytes()
    if sha256(source).hexdigest() != expected_sha:
        raise ValueError(f"Exact {label} source required")
    module = ModuleType(label)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def run(p320_source, p317_source, bundle):
    bound = load_pinned(p320_source, P320_SHA256, "P320")
    p317 = load_pinned(p317_source, bound.P317_SHA256, "P317")
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
            if mask.shape != (bound.SIZE, bound.SIZE) or area != int(p317.codec.area(rle)):
                raise ValueError("Decoded FIT mask geometry differs from the pinned loader")
            distance = signed_distance(mask)
            bounds = {}
            witnesses = {}
            for nodes in bound.NODES:
                bounds[nodes] = bound.cell_cost_bounds(mask, nodes)
                field = resize_bilinear(resize_bilinear(distance, (nodes, nodes)), mask.shape)
                witnesses[nodes] = threshold_witnesses(mask, field)
            records.append((groups[image_id], area, bounds, witnesses))
    return dict(
        status="HRSID_FIT114_BILINEAR_LATTICE_EDT_WITNESSES_COMPLETE_V1",
        p320_source_sha256=P320_SHA256,
        p317_source_sha256=bound.P317_SHA256,
        fit_bundle_sha256=p317.BUNDLE_SHA,
        fit_index_sha256=p317.INDEX_SHA,
        fit_cohort_sha256=p317.COHORT_SHA,
        fit_role_sha256=p317.ROLE_SHA,
        population=dict(core_images=len(core), stress_images_excluded=len(cohort) - len(core),
                        core_source_groups=len({groups[image_id] for image_id in core})),
        geometry=dict(output_shape=[bound.SIZE, bound.SIZE], input_axes=list(bound.NODES),
                      mode="bilinear", align_corners=False, threshold_rule="strict_positive",
                      threshold_native_pixels=THRESHOLDS, float64_ambiguity_guard=FLOAT_GUARD,
                      signed_distance="foreground_edt_minus_background_edt_with_exterior_background",
                      size_bins="decoded_foreground_area_at_32_squared_and_96_squared"),
        scope=dict(FIT=True, DEV=False, TEST=False, AP=False, predictions=False,
                   image_members_decoded=False, model_forward_calls=0, detector_optimizer_updates=0,
                   masks_constructed_directly_from_ground_truth=True,
                   threshold_sweep_oracle=True, training_or_inference_gain_measured=False),
        strata=summarize(records, bound.NODES, bound.WEIGHTS),
    )


def main():
    cli = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("p320-source", "p317-source", "bundle", "output"):
        cli.add_argument("--" + name, type=Path, required=True)
    args = cli.parse_args()
    for path in (args.p320_source, args.p317_source, args.bundle):
        if not path.is_absolute() or path.resolve(strict=True) != path or not path.is_file():
            raise ValueError("Canonical input file required")
    if (not args.output.is_absolute() or args.output.parent.resolve(strict=True) != args.output.parent
            or args.output.exists() or args.output.is_symlink()):
        raise ValueError("Fresh canonical output file required")
    result = run(args.p320_source, args.p317_source, args.bundle)
    payload = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with args.output.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=result["status"], result_sha256=sha256(payload).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
