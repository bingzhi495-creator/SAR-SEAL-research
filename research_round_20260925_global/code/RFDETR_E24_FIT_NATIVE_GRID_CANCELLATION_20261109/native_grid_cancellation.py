"""Same-support native mask-grid cancellation witness on frozen SSDD FIT labels."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
import hashlib
import os
from pathlib import Path
import re
import sys
import time
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIGNED = ROOT / "RFDETR_E24_FIT_SIGNED_BOUNDARY_EVIDENCE_CANDIDATE_20261109_01/signed_evidence.py"
SIGNED_SHA = "291d1e6a1175fc2a06fdfcdf6468e0f3dab1e406c62166cb839a87cfe067f1e4"
CAPTURE = ROOT / "RFDETR_E24_FIT741_NATIVE_WINDOWS_CUDA_ACTUAL_PRIVATE_20261108_01"
CAPTURE_SPEC_SHA = "1566a1fd376355e3eec14ee34b8253ec683c7decfa4374701bd9fa2ec193e777"
CAPTURE_CORE_SHA = "459115f5f279a4e7d1c598dca410bb8db117e97694be329fa74480be8d85ed13"
GRIDS = (144, 288)
SCOPES = ("all_original_matched_small", "top10_original_matched_small")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def grid_axis(index, extent, size):
    """Half-pixel source position, exactly grouped and clipped at constant edges."""
    denominator = 2 * extent
    numerator = np.clip((2 * index + 1) * size - extent, 0, (size - 1) * denominator)
    lower, remainder = np.divmod(numerator, denominator)
    return lower, np.minimum(lower + 1, size - 1), remainder / denominator


def interpolation(positions, hw, size):
    y, x = np.divmod(positions.astype(np.int64), hw[1])
    y0, y1, a = grid_axis(y, hw[0], size)
    x0, x1, b = grid_axis(x, hw[1], size)
    return np.column_stack((y0 * size + x0, y0 * size + x1, y1 * size + x0, y1 * size + x1)), (
        np.column_stack(((1 - a) * (1 - b), (1 - a) * b, a * (1 - b), a * b)))


def gather(nodes, indices, weights):
    return (nodes[indices] * weights).sum(axis=1)


def scatter(pixel_update, indices, weights, size):
    return np.bincount(indices.ravel(), weights=(pixel_update[:, None] * weights).ravel(),
                       minlength=size * size)


def cancellation(pixel_update, indices, weights, size):
    mass = np.abs(pixel_update).sum()
    if not mass:
        return None
    node_mass = np.abs(scatter(pixel_update, indices, weights, size)).sum()
    need(node_mass <= mass * (1 + 1e-12), "Positive bilinear weights conserve absolute input mass")
    return float(max(0., 1. - node_mass / mass))


def targets(logits, signs, flip_correct):
    inside = logits > 0
    need(np.isfinite(logits).all() and np.array_equal(signs, np.where(inside, -1, 1)),
         "Frozen original float logits and edit signs agree")
    return inside ^ flip_correct


def nonmonotone_scanlines(positions, labels, hw, size):
    """Count local GT reversals in a single ideal-bilinear scanline cell only."""
    y, x = np.divmod(positions.astype(np.int64), hw[1])
    y_cell, _, _ = grid_axis(y, hw[0], size)
    x_cell, _, _ = grid_axis(x, hw[1], size)
    counts = []
    for key, along in ((y * size + x_cell, x), (x * size + y_cell, y)):
        order = np.lexsort((along, key))
        line, label = key[order], labels[order]
        same_line = line[1:] == line[:-1]
        flips = same_line & (label[1:] != label[:-1])
        _, groups = np.unique(line[:-1][flips], return_counts=True)
        counts.append(int((groups > 1).sum()))
    return tuple(counts)


def fraction(mask):
    return float(mask.mean()) if len(mask) else None


def instance_report(positions, logits, signs, correct_edit, hw, size):
    desired = targets(logits, signs, correct_edit)
    direction = np.where(desired, 1., -1.)
    z = logits.astype(np.float64)
    full_gradient = direction * np.exp(-np.logaddexp(0., direction * z))
    error_gradient = np.where(correct_edit, full_gradient, 0.)
    unit_error = signs * correct_edit
    indices, weights = interpolation(positions, hw, size)
    update = gather(scatter(full_gradient, indices, weights, size), indices, weights)
    signed_update = direction * update
    stats = dict(
        error_BCE_cancellation=cancellation(error_gradient, indices, weights, size),
        error_unit_cancellation=cancellation(unit_error, indices, weights, size),
        full_BCE_cancellation=cancellation(full_gradient, indices, weights, size),
        full_BCE_nonhelpful_on_errors=fraction(signed_update[correct_edit] <= 0),
        full_BCE_nonhelpful_on_add_errors=fraction(signed_update[correct_edit & (signs == 1)] <= 0),
        full_BCE_nonhelpful_on_remove_errors=fraction(signed_update[correct_edit & (signs == -1)] <= 0),
        full_BCE_adverse_on_correct=fraction(signed_update[~correct_edit] < 0),
    )
    gt_h, gt_v = nonmonotone_scanlines(positions, desired, hw, size)
    pred_h, pred_v = nonmonotone_scanlines(positions, logits > 0, hw, size)
    witnesses = dict(gt_horizontal=gt_h, gt_vertical=gt_v,
                     observed_prediction_horizontal=pred_h, observed_prediction_vertical=pred_v)
    return stats, witnesses


def compare(images, signed):
    """Paired image bootstrap, with identical valid instances in both grid arms."""
    pairs = defaultdict(list)
    witnesses = {grid: Counter() for grid in GRIDS}
    support = Counter(images=len(images))
    for image in images:
        if image:
            support["images_with_instances"] += 1
        if any(item["both_error_directions"] for item, _, _ in image):
            support["images_with_bidirectional_instance_errors"] += 1
        local = defaultdict(lambda: ([], []))
        for item, a, b in image:
            support.update(item)
            witnesses[GRIDS[0]].update(a[1])
            witnesses[GRIDS[1]].update(b[1])
            for key in a[0]:
                av, bv = a[0][key], b[0][key]
                need((av is None) == (bv is None), "Both arms use identical labels/support")
                if av is not None:
                    local[key][0].append(av)
                    local[key][1].append(bv)
        for key, value in local.items():
            pairs[key].append(value)
    return dict(
        support=dict(support),
        scanline_local_counts={str(grid): dict(counts) for grid, counts in witnesses.items()},
        native_minus_fine={
            key: dict(signed.paired_image_means(value), instances=sum(len(a) for a, _ in value))
            for key, value in pairs.items()
        },
    )


def load_helper():
    need(SIGNED.resolve(strict=True) == SIGNED, "Canonical signed helper")
    raw = SIGNED.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == SIGNED_SHA, "Pinned original FIT helper")
    signed = types.ModuleType("_frozen_native_grid_fit")
    signed.__file__ = SIGNED.as_posix()
    exec(compile(raw, signed.__file__, "exec"), signed.__dict__)
    return signed


def run(protocol, output_dir):
    need(sys.version_info[:3] == (3, 10, 13) and np.__version__ == "1.26.4"
         and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and "torch" not in sys.modules,
         "CPU NumPy-only frozen FIT diagnostic")
    need(re.fullmatch(r"[0-9a-f]{40}", protocol), "Exact approved protocol commit")
    need(output_dir.is_absolute() and output_dir.parent.resolve(strict=True) == ROOT
         and output_dir.name.startswith("RFDETR_E24_FIT_NATIVE_GRID_CANCELLATION_ACTUAL_PRIVATE_")
         and not output_dir.exists(), "Fresh private result directory")
    start = time.perf_counter()
    signed = load_helper()
    api, scout, _, _, manifest, index, capture, gt, refs = signed.load_inputs()
    capture_spec_ref = signed.file_pin(CAPTURE / "SPEC.json", CAPTURE_SPEC_SHA)
    capture_spec = api.doc(capture_spec_ref)
    core = capture_spec["sources"]["core"]
    need(core["sha256"] == CAPTURE_CORE_SHA and capture_spec["DEV"] is capture_spec["TEST"] is False,
         "Pinned original FIT interpolation implementation")
    raw_core = api.checked(core)
    need(b"144, 144" in raw_core and b"mode='bilinear', align_corners=False" in raw_core,
         "Original 144 bilinear no-align capture mapping")
    refs.update(signed_helper=signed.file_pin(SIGNED, SIGNED_SHA),
                capture_spec=capture_spec_ref, capture_core=core)
    small_ids = {a["id"] for a in gt["annotations"] if not a["iscrowd"] and a["area"] < 1024}
    groups = {scope: [] for scope in SCOPES}
    population = Counter()
    for item, meta, native in zip(manifest, index, capture, strict=True):
        need(all(item[k] == meta[k] == native[k] for k in ("image_id", "dataset_index", "partition")),
             "Closed original matching, sparse bands and native identity")
        if item["partition"] != "head_holdout":
            continue
        rows, arrays = scout.load_image(np, meta)
        offsets, label_offsets = arrays["offsets"], arrays["label_offsets"]
        sizes = np.diff(offsets)
        eligible = signed.applied_rows(rows, sizes)
        population.update(images=1, retained=len(rows), top10_nonempty=len(eligible),
                          labeled_pixels=len(arrays["y"]), correct_labeled_pixels=int(arrays["y"].sum()))
        samples = {scope: [] for scope in SCOPES}
        for i, row in enumerate(rows):
            if row["status"] != "matched" or row["GT_annotation_id"] not in small_ids:
                continue
            lo, hi = map(int, offsets[i:i + 2])
            yl, yh = map(int, label_offsets[i:i + 2])
            need((lo, hi) == (row["start"], row["stop"]) and hi - lo == yh - yl,
                 "Identical original label and point support")
            positions, logits, signs = (arrays[name][lo:hi] for name in ("positions", "logits", "edit_sign"))
            labels = arrays["y"][yl:yh]
            add = int((labels & (signs == 1)).sum())
            remove = int((labels & (signs == -1)).sum())
            info = dict(instances=1, pixels=len(labels), additions=add, removals=remove,
                        both_error_directions=int(bool(add and remove)))
            args = positions, logits, signs, labels, tuple(item["original_hw"])
            reports = [instance_report(*args, grid) for grid in GRIDS]
            measured = (info, *reports)
            samples[SCOPES[0]].append(measured)
            if i in eligible:
                samples[SCOPES[1]].append(measured)
        for scope, sampled in samples.items():
            groups[scope].append(sampled)
    need(population == Counter(images=111, retained=9083, top10_nonempty=929,
                               labeled_pixels=149923, correct_labeled_pixels=24813),
         "Reproduced accepted SSDD FIT holdout population")
    comparisons = {scope: compare(images, signed) for scope, images in groups.items()}
    need(comparisons[SCOPES[0]]["support"]["instances"] == 248
         and comparisons[SCOPES[1]]["support"]["instances"] == 229,
         "Reproduced prior original matched small scopes")
    output_dir.mkdir()
    result = dict(
        status="FIT_ONLY_NATIVE_MASK_GRID_CANCELLATION_COMPLETE", protocol_commit=protocol, grids=GRIDS,
        same_sparse_points_and_BCE_residuals=True, DEV=False, TEST=False, GPU=False,
        coarse_logits_or_spatial_activations_read=False, weights_read=False, native_mask_changed=False,
        training=False, AP=False, population=dict(population), comparisons=comparisons,
        input_refs=refs, source=api.file_ref(Path(__file__).resolve()),
        bootstrap=dict(draws=signed.BOOTSTRAPS, seed=signed.BOOTSTRAP_SEED),
        seconds=time.perf_counter() - start,
    )
    print(api.put(output_dir / "RESULT.json", result), flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run(args.protocol, args.output_dir)


if __name__ == "__main__":
    main()
