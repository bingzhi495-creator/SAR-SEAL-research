"""Measure strict-mask coverage and pixel-edit bounds on the closed HRSID pair."""
import argparse
from collections import Counter
from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
import json
import os
from pathlib import Path
from statistics import mean, median
from types import ModuleType

import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO


ARMS = ("native", "center")
STRATA = ("all", "small", "medium", "large")
SWITCH_BANDS = ("below_050", "050_to_075", "075_to_090")
STAGES = ("no_075_edge", "has_075_but_no_090_edge", "loss_to_shared_090_candidates",
          "loss_from_native_order_greedy", "native_order_greedy_090_matches")
PRIMARY_STATUS = "HRSID_MASK_COMMON_FIT_E24_DEV705_INDEPENDENTLY_ACCEPTED_V1"
PRIMARY_SHA = "581a0ca8fc9966388dd6cb07a729f51726932c81bd5c037887b25c0b303dff56"
PAIR_SHA = "d080a27ade36369a8fc8d2cb003c37cc3670d75860e83b30e365bd47e77543c0"
TRAINING_PROTOCOL = "04308c52bd2b4730f1474335c6316f6b8a11762e"
P282 = "HRSID_COMMON_DEV705_LARGE_FAILURE_20261111/closed_common_diagnostics.py"
P282_SHA = "15e924e60c8573a48c8ef523be8636a04588f922d685d4d74ca410271497fa7e"
COVERAGE = "HRSID_MASK_PHASE_REPEAT_DEV705_20261109/dev705_paired_coverage.py"
COVERAGE_SHA = "2ac4abb1a0cb9e684f6bde6b2990419eb0ec209d469411b240dc7dd520e69d3d"
ANCHOR_ALL = {"native": (573, 1071, 0, 0, 134), "center": (442, 1065, 0, 0, 271)}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def exact_edit_budget(pred, gt, intersection):
    """Return the smallest count of truth-assisted foreground bit edits for IoU .90."""
    require(type(pred) is type(gt) is type(intersection) is int and min(pred, gt) > 0
            and 0 <= intersection <= min(pred, gt), "Nonempty exact integer mask areas required")
    fp, fn = pred - intersection, gt - intersection
    union = intersection + fp + fn
    deficit = max(0, 9 * union - 10 * intersection)
    add_only = (deficit + 9) // 10
    delete_only = (deficit + 8) // 9
    # An addition pays 10 units of deficit and a deletion pays 9, so spending
    # available true-positive additions first minimizes the number of edits.
    add = min(fn, add_only)
    delete = (max(0, deficit - 10 * add) + 8) // 9
    require(delete <= fp and 10 * (intersection + add) >= 9 * (union - delete),
            "Exact pixel-edit arithmetic failed")
    return dict(
        gt_area=gt, prediction_area=pred, intersection=intersection, fp=fp, fn=fn, iou=intersection / union,
        area_ratio=min(pred, gt) / max(pred, gt), area_ratio_can_reach_090=10 * min(pred, gt) >= 9 * max(pred, gt),
        add_only_edits=add_only if add_only <= fn else None,
        delete_only_edits=delete_only if delete_only <= fp else None, min_edits=add + delete,
        min_edits_over_gt=(add + delete) / gt,
    )


def strict_stages(overlaps, graph_counts):
    """Partition each eligible GT among four losses and a greedy match."""
    targets = overlaps.shape[1]
    has_075 = int((overlaps >= .75).any(axis=0).sum())
    has_090, maximum, greedy = graph_counts(overlaps, .90)
    require(0 <= greedy <= maximum <= has_090 <= has_075 <= targets, "Inconsistent strict coverage stages")
    return targets - has_075, has_075 - has_090, has_090 - maximum, maximum - greedy, greedy


def image_targets(coco, image, exclusions):
    rles, strata = [], []
    for ann in coco.imgToAnns[image["id"]]:
        rle = coco.annToRLE(ann)
        require(rle["size"] == [image["height"], image["width"]], "Original GT canvas required")
        if mask.area(rle) == 0:
            exclusions["empty"] += 1
        elif ann.get("iscrowd", 0):
            exclusions["crowd"] += 1
        elif ann.get("ignore", 0):
            exclusions["ignore"] += 1
        else:
            area = ann["area"]
            require(type(area) in (int, float) and 0 <= area < float("inf"), "Finite original COCO GT area required")
            rles.append(rle)
            strata.append("small" if area < 1024 else "medium" if area < 9216 else "large")
    return rles, np.array(strata, dtype="U6")


def eligible_prefix(candidates):
    return [rle for rle, area in zip(candidates, mask.area(candidates), strict=True) if int(area) > 0] if candidates else []


def fixed_edit(pred_rle, gt_rle, expected_overlap):
    pred, gt = (int(mask.area(rle)) for rle in (pred_rle, gt_rle))
    intersection = int(mask.area(mask.merge([pred_rle, gt_rle], intersect=True)))
    row = exact_edit_budget(pred, gt, intersection)
    require(abs(row["iou"] - expected_overlap) <= 1e-15 and row["iou"] < .90 and row["min_edits"] > 0,
            "Selected best-mask IoU must agree with independent integer RLE intersection")
    return row


def summarize_arm_edits(rows):
    return dict(
        area_ratio_can_reach_090=sum(row["area_ratio_can_reach_090"] for row in rows),
        area_ratio_cannot_reach_090=sum(not row["area_ratio_can_reach_090"] for row in rows),
        zero_overlap_targets=sum(row["intersection"] == 0 for row in rows),
        prediction_smaller_than_gt=sum(row["prediction_area"] < row["gt_area"] for row in rows),
        prediction_larger_than_gt=sum(row["prediction_area"] > row["gt_area"] for row in rows),
        add_only_feasible=sum(row["add_only_edits"] is not None for row in rows),
        delete_only_feasible=sum(row["delete_only_edits"] is not None for row in rows),
        both_add_only_and_delete_only_feasible=sum(
            row["add_only_edits"] is not None and row["delete_only_edits"] is not None for row in rows),
        min_edits_mean=mean(row["min_edits"] for row in rows) if rows else None,
        min_edits_median=median(row["min_edits"] for row in rows) if rows else None,
        min_edits_over_gt_mean=mean(row["min_edits_over_gt"] for row in rows) if rows else None,
        min_edits_over_gt_median=median(row["min_edits_over_gt"] for row in rows) if rows else None,
        mean_area_ratio=mean(row["area_ratio"] for row in rows) if rows else None,
        mean_iou=mean(row["iou"] for row in rows) if rows else None,
        mean_fp_over_gt=mean(row["fp"] / row["gt_area"] for row in rows) if rows else None,
        mean_fn_over_gt=mean(row["fn"] / row["gt_area"] for row in rows) if rows else None,
    )


def summarize_edits(pairs):
    by_arm = {arm: summarize_arm_edits([pair[arm] for pair in pairs]) for arm in ARMS}
    shifts = [pair["center"]["min_edits"] - pair["native"]["min_edits"] for pair in pairs]
    return dict(
        same_gt_pairs=len(pairs), arms=by_arm,
        center_fewer_min_edits=sum(value < 0 for value in shifts),
        center_equal_min_edits=sum(value == 0 for value in shifts),
        center_more_min_edits=sum(value > 0 for value in shifts),
        center_minus_native_min_edits_over_gt_mean=mean(
            shift / pair["native"]["gt_area"] for shift, pair in zip(shifts, pairs, strict=True)) if pairs else None,
    )


def census(gt, native_rows, center_rows, coverage):
    """Use the exact prior prediction order, without rescoring or fitting any mask."""
    coco = COCO()
    coco.dataset = gt
    with redirect_stdout(StringIO()):
        coco.createIndex()
    images = gt["images"]
    require(len(coco.imgs) == len(images) and len(coco.anns) == len(gt["annotations"])
            and [row["id"] for row in gt["categories"]] == [1], "Unique original GT image, annotation and category IDs")
    grouped = dict(native=coverage.group_rows(native_rows, images), center=coverage.group_rows(center_rows, images))
    graph_counts = coverage.graph_counts_fn()
    exclusions = Counter(empty=0, crowd=0, ignore=0)
    counts = {stratum: {arm: np.zeros(5, dtype=np.int64) for arm in ARMS} for stratum in STRATA}
    near = {stratum: [] for stratum in STRATA}
    switches = {stratum: {arm: {band: [] for band in SWITCH_BANDS} for arm in ARMS} for stratum in STRATA}
    prefix_rows = 0
    for index, image in enumerate(images):
        targets, sizes = image_targets(coco, image, exclusions)
        ranked = {arm: eligible_prefix(grouped[arm][index]) for arm in ARMS}
        quota = min(len(rows) for rows in ranked.values())
        prefix_rows += quota
        prefix = {arm: ranked[arm][:quota] for arm in ARMS}
        overlaps = {
            arm: mask.iou(rows, targets, [0] * len(targets)) if rows and targets else np.zeros((quota, len(targets)))
            for arm, rows in prefix.items()
        }
        require(all(value.shape == (quota, len(targets)) and np.isfinite(value).all()
                    and (value >= 0).all() and (value <= 1).all() for value in overlaps.values()),
                "Finite original-grid IoU matrices required")
        for stratum in STRATA:
            selected = slice(None) if stratum == "all" else sizes == stratum
            for arm in ARMS:
                counts[stratum][arm] += strict_stages(overlaps[arm][:, selected], graph_counts)
        if quota and targets:
            best = {arm: value.argmax(axis=0) for arm, value in overlaps.items()}
            target_indices = np.arange(len(targets))
            best_iou = {arm: overlaps[arm][best[arm], target_indices] for arm in ARMS}
            eligible = np.logical_and.reduce([(values >= .75) & (values < .90) for values in best_iou.values()])
            for target_index in np.flatnonzero(eligible):
                paired = {arm: fixed_edit(prefix[arm][best[arm][target_index]], targets[target_index],
                                           best_iou[arm][target_index]) for arm in ARMS}
                require(paired["native"]["gt_area"] == paired["center"]["gt_area"], "Exact common target required")
                near["all"].append(paired)
                near[sizes[target_index]].append(paired)
            for passing, failing in (("center", "native"), ("native", "center")):
                gained = (best_iou[passing] >= .90) & (best_iou[failing] < .90)
                for target_index in np.flatnonzero(gained):
                    value = best_iou[failing][target_index]
                    band = SWITCH_BANDS[0] if value < .50 else SWITCH_BANDS[1] if value < .75 else SWITCH_BANDS[2]
                    row = fixed_edit(prefix[failing][best[failing][target_index]], targets[target_index], value)
                    switches["all"][passing][band].append(row)
                    switches[sizes[target_index]][passing][band].append(row)
    population = {
        stratum: dict(eligible_gt=int(counts[stratum]["native"].sum()), arms={
            arm: dict(zip(STAGES, counts[stratum][arm].tolist(), strict=True)) for arm in ARMS})
        for stratum in STRATA
    }
    require(all(counts[stratum]["native"].sum() == counts[stratum]["center"].sum() for stratum in STRATA)
            and population["all"]["eligible_gt"] + sum(exclusions.values()) == len(gt["annotations"])
            and sum(len(near[stratum]) for stratum in STRATA[1:]) == len(near["all"]),
            "Stage and common target populations must reconcile")
    switch_counts = {arm: sum(len(rows) for rows in switches["all"][arm].values()) for arm in ARMS}
    independent = {arm: int(counts["all"][arm][2:].sum()) for arm in ARMS}
    require(switch_counts["center"] - switch_counts["native"] == independent["center"] - independent["native"],
            "Directional independent .90 transitions must reconcile")
    directional = {}
    for stratum, strata_switches in switches.items():
        directional[stratum] = {}
        for passing, failing in (("center", "native"), ("native", "center")):
            by_band = strata_switches[passing]
            directional[stratum][passing + "_only"] = dict(
                failing_arm=failing, targets=sum(len(rows) for rows in by_band.values()),
                failing_best_iou_bands={band: dict(targets=len(rows), **summarize_arm_edits(rows))
                                        for band, rows in by_band.items()},
            )

    return dict(images=len(images), annotations=len(gt["annotations"]), exclusions=dict(exclusions),
                equal_nonempty_prefix_rows_per_arm=prefix_rows,
                strict_coverage=population, common_near_strict={stratum: summarize_edits(near[stratum]) for stratum in STRATA},
                directional_090_switches=directional)


def load_closed(code_root):
    path = code_root / P282
    require(path.is_file() and path.resolve(strict=True) == path, "Exact canonical released admission source required")
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == P282_SHA, "Exact released admission SHA-256 required")
    module = ModuleType("_strict_residual_p282")
    module.__file__ = path.as_posix()
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code-root", "pair", "native-arm-receipt", "center-arm-receipt", "accepted-result", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("pair-sha256", "training-protocol", "accepted-result-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU closed-prediction diagnostic required")
    require(args.code_root.is_absolute() and args.code_root.resolve(strict=True) == args.code_root
            and args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent
            and not args.output.exists() and not args.output.is_symlink(),
            "Canonical ordinary source and fresh private aggregate output required")
    require((args.pair_sha256, args.training_protocol, args.accepted_result_sha256) == (PAIR_SHA, TRAINING_PROTOCOL, PRIMARY_SHA),
            "Exact previously accepted common-support seed 5202 lineage required")
    closed = load_closed(args.code_root)
    coverage = closed.load_source(args.code_root / COVERAGE, COVERAGE_SHA, "_strict_residual_coverage")
    driver = closed.load_gate(args.code_root).load_driver()
    primary = driver.read_json(args.accepted_result, args.accepted_result_sha256)
    driver.require(primary["status"] == PRIMARY_STATUS and primary["images"] == 705 and primary["primary_annotations"] == 1782
                   and primary["official_TEST_opened"] is False and primary["training_protocol"] == TRAINING_PROTOCOL
                   and primary["training_seed"] == 5202, "Exact prior independent primary DEV705 acceptance required")
    gt, native, center, provenance = closed.verified_inputs(
        args.code_root, args.pair, args.pair_sha256, native_arm_receipt=args.native_arm_receipt,
        center_arm_receipt=args.center_arm_receipt, training_protocol=args.training_protocol,
    )
    require(provenance["training_pair_receipt_sha256"] == primary["independent_training_pair_sha256"]
            and provenance["evaluation_protocol"] == primary["evaluation_protocol"]
            and provenance["original_dev_gt_sha256"] == primary["original_dev_gt_sha256"],
            "Exact matched existing primary evaluation lineage required")
    result = census(gt, native, center, coverage)
    require(result["images"] == 705 and result["annotations"] == 1782
            and result["exclusions"] == dict(empty=4, ignore=0, crowd=0)
            and result["equal_nonempty_prefix_rows_per_arm"] == 41482
            and result["strict_coverage"]["all"]["eligible_gt"] == 1778
            and all(tuple(result["strict_coverage"]["all"]["arms"][arm].values()) == ANCHOR_ALL[arm] for arm in ARMS),
            "Exact P282 population, equal nonempty quota and strict coverage anchors required")
    report = dict(
        status="HRSID_COMMON_E24_DEV705_STRICT_RESIDUAL_COMPLETE_V1", accepted_result_sha256=PRIMARY_SHA,
        provenance=provenance, source_sha256={"closed_common_diagnostics.py": P282_SHA, "dev705_paired_coverage.py": COVERAGE_SHA,
                                              "strict_residual.py": sha256(Path(__file__).read_bytes()).hexdigest()},
        closed_predictions_only=True, official_TEST_opened=False, COCO_AP=False, model_forward_calls=0,
        assignment_scope="per-image matching with GT restricted to the listed size stratum",
        paired_scope="same GT with per-arm highest IoU in [0.75,0.90), ties use earlier native rank",
        directional_switch_scope="same GT with independent .90 candidate in exactly one arm, edit bound on the failing arm",
        pixel_edit_scope="truth-assisted pixel budget bound on fixed masks with no morphology or score changes",
        **result,
    )
    raw = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=report["status"], bytes=len(raw), sha256=sha256(raw).hexdigest())), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
