"""Localize frozen common-support HRSID A/B margin differences by label phase."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
from scipy.ndimage import distance_transform_cdt, distance_transform_edt


SIZE = 800
RESIZE = 576
DRAWS = 4096
SEED = 3645202
P360_SHA = "224dd916d4d5672d8aaa9707f8b9233926d4590043ab754ff151b25ce5546acc"
STATUS = "HRSID_COMMON_DEV705_SIGNED_VIRTUAL_LABEL_PHASE_ACTUAL"


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def stratified_alignment(margin, sign, strata):
    """Compare signed alignment to the exact matched random-placement mean."""
    need(len(margin) == len(sign) == len(strata), "Aligned phase values required")
    _, ids = np.unique(strata, return_inverse=True)
    sizes = np.bincount(ids)
    sum_margin = np.bincount(ids, weights=margin)
    positive, negative = sign > 0, sign < 0
    plus = np.bincount(ids, weights=positive)
    minus = np.bincount(ids, weights=negative)
    expected_plus = float(np.dot(sum_margin / sizes, plus))
    expected_minus = float(np.dot(sum_margin / sizes, minus))
    observed_plus = int(margin[positive].sum())
    observed_minus = int(margin[negative].sum())
    movable = (plus < sizes) & (minus < sizes)
    return dict(
        exposed=int(positive.sum() + negative.sum()),
        exposed_positive=int(positive.sum()), exposed_negative=int(negative.sum()),
        exchangeable_exposed=int((plus[movable] + minus[movable]).sum()),
        signed_observed=observed_plus - observed_minus,
        signed_expected=expected_plus - expected_minus,
        correct_observed=observed_plus, correct_expected=expected_plus,
        harmful_observed=observed_minus, harmful_expected=expected_minus,
        pixels=len(margin), prediction_flips=int(np.count_nonzero(margin)),
    )


def phase_indices():
    target = np.arange(RESIZE, dtype=np.int64)
    output = np.arange(SIZE, dtype=np.int64)
    return (target * SIZE // RESIZE,
            (2 * target + 1) * SIZE // (2 * RESIZE),
            (2 * output + 1) * RESIZE // (2 * SIZE))


def stock_legacy_indices(rational):
    # Independently measured on original RF-DETR Boolean 800x800 masks.
    ties = [90, 162, 180, 306, 324, 342, 360]
    legacy = rational.copy()
    legacy[ties] -= 1
    return legacy, ties


def occupancy_axis(index):
    # A 576-cell spans 25 integer subunits, a source pixel 18.
    left = index * 25
    pixels = left[:, None] // 18 + np.arange(3, dtype=np.int64)
    overlap = np.minimum(left[:, None] + 25, (pixels + 1) * 18) - np.maximum(left[:, None], pixels * 18)
    return np.minimum(pixels, SIZE - 1), np.maximum(overlap, 0)


def occupancy_stratum(truth, inverse_y, inverse_x):
    y, wy = occupancy_axis(inverse_y)
    x, wx = occupancy_axis(inverse_x)
    count = (truth[y[:, :, None], x[:, None, :]] * wy[:, :, None] * wx[:, None, :]).sum(axis=(1, 2))
    need(((0 <= count) & (count <= 625)).all(), "Exact fixed footprint occupancy outside [0,625]")
    return np.digitize(count, [1, 157, 313, 469, 625])


def boundary_geometry(truth, bbox):
    x, y, w, h = map(int, bbox)
    y0, x0 = max(0, y - 2), max(0, x - 2)
    local = np.pad(truth[y0:min(SIZE, y + h + 2), x0:min(SIZE, x + w + 2)], 1)
    interior = distance_transform_cdt(local, metric="chessboard")
    exterior = distance_transform_cdt(~local, metric="chessboard")
    ring = interior + exterior
    rr, cc = np.nonzero((ring[1:-1, 1:-1] >= 1) & (ring[1:-1, 1:-1] <= 2))
    rr, cc = rr + 1, cc + 1
    fg = local[rr, cc]
    opposite_bg = distance_transform_edt(local, return_distances=False, return_indices=True)
    opposite_fg = distance_transform_edt(~local, return_distances=False, return_indices=True)
    dy = np.where(fg, opposite_bg[0, rr, cc] - rr, rr - opposite_fg[0, rr, cc])
    dx = np.where(fg, opposite_bg[1, rr, cc] - cc, cc - opposite_fg[1, rr, cc])
    normal = 3 * (np.sign(dy) + 1) + (np.sign(dx) + 1)
    need((normal != 4).all(), "Each band pixel needs a nearest-opposite outward direction")
    strata = (2 * fg.astype(np.int64) + ring[rr, cc] - 1) * 9 + normal
    return rr + y0 - 1, cc + x0 - 1, fg, strata


def measurements(truth, a, b, bbox, indices):
    y, x, fg, strata = boundary_geometry(truth, bbox)
    forward_a, forward_b, inverse, rational = indices
    iy, ix = inverse[y], inverse[x]
    teacher_a = truth[forward_a[iy], forward_a[ix]]
    teacher_b = truth[forward_b[iy], forward_b[ix]]
    weight = np.where(fg, 10, -9)
    teacher = np.sign(weight * (teacher_b.astype(np.int64) - teacher_a))
    native = a[y, x]
    center = b[y, x].astype(np.int64)
    margin = weight * (center - native)
    occupancy = occupancy_stratum(truth, iy, ix)
    base = stratified_alignment(margin, teacher, strata)
    rational_a = truth[rational[iy], rational[ix]]
    rational_teacher = np.sign(weight * (teacher_b.astype(np.int64) - rational_a))
    results = dict(
        primary=base,
        footprint_matched=stratified_alignment(margin, teacher, strata * 6 + occupancy),
        rational_legacy=stratified_alignment(margin, rational_teacher, strata),
    )
    if not a[0].any():
        # A(y+1,x) shifts the original prediction upwards by one.
        shifted = np.zeros_like(native)
        allowed = y < SIZE - 1
        shifted[allowed] = a[y[allowed] + 1, x[allowed]]
        results.update(
            no_clip_original=base,
            fixed_a_up_one=stratified_alignment(weight * (center - shifted), teacher, strata),
        )
    return results


def summarize(by_group, name):
    grouped = [policies[name] for policies in by_group.values() if name in policies]
    active = [values for values in grouped if values["exposed"]]
    need(len(active) >= 2, "Phase comparison lacks two exposed source groups")
    totals = Counter()
    for group in grouped:
        totals.update(group)
    delta = np.array([(g["signed_observed"] - g["signed_expected"]) / g["exposed"] for g in active])
    positive = [g for g in active if g["exposed_positive"]]
    harmful = [g for g in active if g["exposed_negative"]]
    rng = np.random.default_rng(SEED)
    draw = rng.integers(len(active), size=(DRAWS, len(active)))
    lower, upper = np.percentile(delta[draw].mean(axis=1), (2.5, 97.5))
    leave = (delta.sum() - delta) / (len(delta) - 1)
    return dict(
        totals=dict(totals), groups=len(grouped), groups_with_exposure=len(active),
        groups_with_positive_excess=int((delta > 0).sum()),
        group_equal_signed_excess_per_exposed_pixel=float(delta.mean()),
        pooled_signed_excess_per_exposed_pixel=(totals["signed_observed"] - totals["signed_expected"]) / totals["exposed"],
        group_equal_correct_excess_per_positive_pixel=float(np.mean([
            (g["correct_observed"] - g["correct_expected"]) / g["exposed_positive"] for g in positive])),
        group_equal_actual_margin_excess_per_harmful_pixel=float(np.mean([
            (g["harmful_observed"] - g["harmful_expected"]) / g["exposed_negative"] for g in harmful])),
        group_bootstrap_95=[float(lower), float(upper)],
        leave_one_group_out_range=[float(leave.min()), float(leave.max())],
    )


def load(path, pin, name):
    data = path.read_bytes()
    need(sha256(data).hexdigest() == pin, "Exact ordinary closed-analysis source required")
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(data, str(path), "exec"), module.__dict__)
    return module


def run(a):
    from pycocotools import mask
    from pycocotools.coco import COCO
    from contextlib import redirect_stdout
    from io import StringIO

    p360 = load(a.p360_source, P360_SHA, "_p364_p360")
    need((a.pair_sha256, a.training_protocol, a.accepted_result_sha256) ==
         (p360.PAIR_SHA, p360.TRAIN_PROTOCOL, p360.PRIMARY_SHA), "Exact closed P360 lineage required")
    base_data = a.prior_result.read_bytes()
    need(sha256(base_data).hexdigest() == p360.BASELINE_SHA, "Exact P282 safe baseline required")
    base = json.loads(base_data)
    closed = p360.load(a.code_root / p360.P282_PATH, p360.P282_SHA, "_p364_closed")
    group = p360.load(a.code_root / p360.GROUP_PATH, p360.GROUP_SHA, "_p364_groups")
    driver = group.load_driver()
    primary = driver.read_json(a.accepted_result, a.accepted_result_sha256)
    need(primary["training_seed"] == 5202 and primary["official_TEST_opened"] is False
         and primary["status"] == "HRSID_MASK_COMMON_FIT_E24_DEV705_INDEPENDENTLY_ACCEPTED_V1",
         "Exact independently accepted DEV pair required")
    gt, native, center, proof = closed.verified_inputs(a.code_root, a.pair, a.pair_sha256,
        native_arm_receipt=a.native_arm_receipt, center_arm_receipt=a.center_arm_receipt,
        training_protocol=a.training_protocol)
    need(proof == base["provenance"], "Closed paired prediction provenance differs from P282")
    pair = driver.read_json(a.pair, a.pair_sha256)
    group_gt, _, groups = group.load_population(driver, dict(path=a.image_manifest.as_posix(), sha256=p360.MANIFEST_SHA),
                                               pair["gt"])
    need(group_gt == gt and len(set(groups.values())) == 28, "Exact common GT and 28 source proxies required")
    modules = closed.load_diagnostics(a.code_root)
    failure, score = modules["dev705_paired_mask_failure_scope"], modules["dev705_score_template"]
    selected, counts = score.prefix_rows(score.nonempty_by_image(native, gt["images"]),
                                        score.nonempty_by_image(center, gt["images"]))
    need(sum(counts) == 41482, "Exact per-arm equal nonempty prefix required")
    adapter = failure.load_official_adapter()
    matches = {arm: p360.fixed_matches(gt, selected[arm], adapter, failure.match_official) for arm in p360.ARMS}
    coco = COCO()
    coco.dataset = gt
    with redirect_stdout(StringIO()):
        coco.createIndex()
    rational, center_axis, inverse = phase_indices()
    legacy, legacy_ties = stock_legacy_indices(rational)
    need(np.count_nonzero(legacy != center_axis) == 391
         and np.count_nonzero(legacy[inverse] != center_axis[inverse]) == 519,
         "Exact stock direct phase donor census differs")
    indices = legacy, center_axis, inverse, rational
    by_size = {size: defaultdict(lambda: defaultdict(Counter)) for size in ("all", "small", "non_small")}
    support = Counter()
    for ann in gt["annotations"]:
        truth_rle = coco.annToRLE(ann)
        if ann.get("iscrowd", 0) or ann.get("ignore", 0) or not mask.area(truth_rle):
            continue
        aid = ann["id"]
        shared = all(aid in matches[arm] for arm in p360.ARMS)
        support.update(eligible=1, shared=int(shared), native=int(aid in matches["native"]),
                       center=int(aid in matches["center"]))
        if not shared:
            continue
        truth = mask.decode(truth_rle).astype(bool, copy=False)
        left, right = (mask.decode(matches[arm][aid]).astype(bool, copy=False) for arm in p360.ARMS)
        need(truth.shape == left.shape == right.shape == (SIZE, SIZE), "Exact original 800 canvas required")
        values = measurements(truth, left, right, mask.toBbox(truth_rle), indices)
        for size in ("all", "small" if ann["area"] < 1024 else "non_small"):
            for policy, result in values.items():
                result = {**result, "instances": 1}
                by_size[size][groups[ann["image_id"]]][policy].update(result)
    need(support["eligible"] == 1778 and support["shared"] == 1620, "Original eligible and paired counts differ")
    policies = ("primary", "footprint_matched", "rational_legacy", "no_clip_original", "fixed_a_up_one")
    results = {size: {name: summarize(grouped, name) for name in policies} for size, grouped in by_size.items()}
    main = results["all"]["primary"]
    gates = dict(
        main_excess_at_least_point05=main["group_equal_signed_excess_per_exposed_pixel"] >= .05,
        main_bootstrap_lower_positive=main["group_bootstrap_95"][0] > 0,
        footprint_matched_excess_positive=results["all"]["footprint_matched"]["group_equal_signed_excess_per_exposed_pixel"] > 0,
        fixed_shift_excess_positive=results["all"]["fixed_a_up_one"]["group_equal_signed_excess_per_exposed_pixel"] > 0,
    )
    return dict(
        status=STATUS, matched_support=dict(support), strata=results, gate=gates, all_gate_conditions=all(gates.values()),
        exact_closed_pair_sha256=a.pair_sha256, original_dev_gt_sha256=proof["original_dev_gt_sha256"],
        provenance=dict(p360_source_sha256=P360_SHA, p282_result_sha256=p360.BASELINE_SHA,
                        accepted_result_sha256=p360.PRIMARY_SHA, group_map_sha256=group.GROUP_MAP_SHA256),
        test_design=dict(bootstrap_draws=DRAWS, bootstrap_seed=SEED, equal_prefix_per_arm=41482,
                         original_match_iou=.5, source_size=SIZE, direct_label_size=RESIZE,
                         gt_band_chebyshev_rings=[1, 2], same_label_phase_inverse=True,
                         footprint_quantized_strata=[0, 1, 157, 313, 469, 625],
                         signed_normals="nearest opposite EDT, outward direction by integer sign pair",
                         virtual_integer_label_phase=True, runtime_training_label_replay=False,
                         primary_legacy_direct_cpu_axis_ties_below_integer=legacy_ties,
                         primary_legacy_uses_independently_verified_stock_bool_cpu_axis_corrections=True,
                         fixed_up_one_refers_to_native_prediction=True, evidence="descriptive mechanism diagnostic"),
        scope=dict(DEV=True, FIT=False, TEST=False, COCO_AP=False, model_forward_calls=0,
                   optimizer_updates=0, edited_predictions_saved=False, unmatched_used_for_phase=False),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("p360-source", "code-root", "pair", "native-arm-receipt", "center-arm-receipt",
                 "accepted-result", "prior-result", "image-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("pair-sha256", "training-protocol", "accepted-result-sha256"):
        parser.add_argument("--" + name, required=True)
    a = parser.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and os.environ.get("OMP_NUM_THREADS") == "1"
         and a.output.is_absolute() and a.output.parent.resolve(strict=True) == a.output.parent,
         "Fixed isolated CPU DEV diagnostic required")

    def offline(event, _):
        if event in ("socket.connect", "socket.getaddrinfo"):
            raise PermissionError("Closed offline diagnostic only")

    sys.addaudithook(offline)
    report = run(a)
    data = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with a.output.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=STATUS, result_sha256=sha256(data).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
