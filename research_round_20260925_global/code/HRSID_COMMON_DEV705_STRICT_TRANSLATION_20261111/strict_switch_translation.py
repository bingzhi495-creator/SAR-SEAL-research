"""Measure fixed-mask integer translations on the closed HRSID strict coverage switches."""

import argparse
from collections import Counter
from contextlib import redirect_stdout
from hashlib import sha256
from importlib.metadata import version
from io import StringIO
import json
import os
from pathlib import Path
from statistics import mean
import sys
from types import ModuleType


P311_SOURCE_SHA = "6fbefd4e62e7715518c1e460db9b0c32f5c407f11adef8de6cd5737b65346b54"
P311_RESULT_SHA = "280ae9abe55e67d1e85c0e3a22406e5489cf295f1a37c81068644d21ea8c7e7d"
STATUS = "P312_HRSID_COMMON_STRICT_SWITCH_TRANSLATION_COMPLETE_PENDING_INDEPENDENT_ACCEPTANCE"
RADIUS = 2
FAMILIES = ("vertical", "horizontal", "joint")
ARMS = ("native", "center")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def pinned(path, digest, maximum):
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(),
            "Exact canonical P311 source or result required")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    require(0 < len(raw) <= maximum and sha256(raw).hexdigest() == digest,
            "Exact P311 source or closed result digest differs")
    return raw


def load_p311(path):
    raw = pinned(path, P311_SOURCE_SHA, 64 * 1024)
    module = ModuleType("_p312_exact_closed_p311")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def translate_best(pred, gt):
    """Maximize shared foreground while all original prediction pixels count in the union."""
    import numpy as np

    require(pred.ndim == 2 and pred.shape == gt.shape and pred.dtype == gt.dtype == np.bool_,
            "Exact Boolean masks on the same original canvas required")
    y, x = np.nonzero(pred)
    pred_area, gt_area = len(y), int(gt.sum())
    require(min(pred_area, gt_area) > 0, "Exact two nonempty masks required")
    height, width = gt.shape
    intersections = {}
    for dy in range(-RADIUS, RADIUS + 1):
        for dx in range(-RADIUS, RADIUS + 1):
            new_y, new_x = y + dy, x + dx
            inside = (0 <= new_y) & (new_y < height) & (0 <= new_x) & (new_x < width)
            intersections[dy, dx] = int(gt[new_y[inside], new_x[inside]].sum())
    return dict(pred_area=pred_area, gt_area=gt_area,
                area_impossible=10 * min(pred_area, gt_area) < 9 * max(pred_area, gt_area),
                overlaps=dict(raw=intersections[0, 0],
                              vertical=max(value for (_, dx), value in intersections.items() if dx == 0),
                              horizontal=max(value for (dy, _), value in intersections.items() if dy == 0),
                              joint=max(intersections.values())))


def pass_090(row, family):
    intersection = row["overlaps"][family]
    return 10 * intersection >= 9 * (row["pred_area"] + row["gt_area"] - intersection)


def iou(row, family):
    intersection = row["overlaps"][family]
    return intersection / (row["pred_area"] + row["gt_area"] - intersection)


def summarize(rows):
    impossible = sum(row["area_impossible"] for row in rows)
    by_family = {}
    for family in FAMILIES:
        recovered = sum(pass_090(row, family) for row in rows)
        by_family[family] = dict(
            recovered=recovered, feasible_not_recovered=len(rows) - impossible - recovered,
            mean_best_iou=mean(iou(row, family) for row in rows) if rows else None,
            mean_gain_iou_points=100 * mean(iou(row, family) - iou(row, "raw") for row in rows) if rows else None)
    both_axes_individually = sum(pass_090(row, "horizontal") and pass_090(row, "vertical") for row in rows)
    joint_only = sum(pass_090(row, "joint") and not (pass_090(row, "horizontal") or pass_090(row, "vertical"))
                     for row in rows)
    require(by_family["joint"]["recovered"] == by_family["horizontal"]["recovered"] +
            by_family["vertical"]["recovered"] - both_axes_individually + joint_only,
            "Exact nested translation families must reconcile")
    return dict(targets=len(rows), area_impossible=impossible, area_feasible=len(rows) - impossible,
                mean_raw_iou=mean(iou(row, "raw") for row in rows) if rows else None,
                both_axes_individually_recovered=both_axes_individually, joint_only_recovered=joint_only,
                by_family=by_family)


def measure(pred_rle, gt_rle, value, p311):
    from pycocotools import mask

    expected = p311.fixed_edit(pred_rle, gt_rle, value)
    row = translate_best(mask.decode(pred_rle).astype(bool), mask.decode(gt_rle).astype(bool))
    require((row["pred_area"], row["gt_area"], row["overlaps"]["raw"], row["area_impossible"])
            == (expected["prediction_area"], expected["gt_area"], expected["intersection"],
                not expected["area_ratio_can_reach_090"]),
            "Exact P311 original intersection and mask areas required")
    require(not pass_090(row, "raw")
            and (not row["area_impossible"] or not pass_090(row, "joint")),
            "Original strict failure and area ceiling must be preserved")
    return row, expected


def census(gt, native_rows, center_rows, p311, coverage, previous):
    """Reproduce P311 candidate identity and transitions before reporting new translation metrics."""
    import numpy as np
    from pycocotools import mask
    from pycocotools.coco import COCO

    coco = COCO()
    coco.dataset = gt
    with redirect_stdout(StringIO()):
        coco.createIndex()
    images = gt["images"]
    grouped = dict(native=coverage.group_rows(native_rows, images), center=coverage.group_rows(center_rows, images))
    exclusions = Counter(empty=0, crowd=0, ignore=0)
    rows = {stratum: {passing: {band: [] for band in p311.SWITCH_BANDS} for passing in ARMS}
            for stratum in p311.STRATA}
    prefix_total = 0
    transitions = Counter()
    for index, image in enumerate(images):
        targets, sizes = p311.image_targets(coco, image, exclusions)
        candidates = {arm: p311.eligible_prefix(grouped[arm][index]) for arm in ARMS}
        quota = min(map(len, candidates.values()))
        prefix_total += quota
        if not quota or not targets:
            transitions["neither"] += len(targets)
            continue
        prefix = {arm: candidates[arm][:quota] for arm in ARMS}
        overlaps = {arm: mask.iou(items, targets, [0] * len(targets)) for arm, items in prefix.items()}
        require(all(value.shape == (quota, len(targets)) and np.isfinite(value).all()
                    and (value >= 0).all() and (value <= 1).all() for value in overlaps.values()),
                "Exact finite original-grid IoU matrix required")
        best = {arm: value.argmax(axis=0) for arm, value in overlaps.items()}
        ids = np.arange(len(targets))
        top = {arm: overlaps[arm][best[arm], ids] for arm in ARMS}
        native_pass, center_pass = (top[arm] >= .90 for arm in ARMS)
        transitions["both"] += int((native_pass & center_pass).sum())
        transitions["neither"] += int((~native_pass & ~center_pass).sum())
        for passing, failing in (("center", "native"), ("native", "center")):
            gained = (top[passing] >= .90) & (top[failing] < .90)
            transitions[passing + "_only"] += int(gained.sum())
            for target_index in np.flatnonzero(gained):
                value = top[failing][target_index]
                band = p311.SWITCH_BANDS[0] if value < .50 else p311.SWITCH_BANDS[1] if value < .75 else p311.SWITCH_BANDS[2]
                measured = measure(prefix[failing][best[failing][target_index]], targets[target_index], value, p311)
                for stratum in ("all", sizes[target_index]):
                    rows[stratum][passing][band].append(measured)
    expected = dict(both=93, native_only=41, center_only=178, neither=1466)
    require(len(images) == previous["images"] == 705 and len(gt["annotations"]) == previous["annotations"] == 1782
            and dict(exclusions) == previous["exclusions"] == dict(empty=4, crowd=0, ignore=0)
            and prefix_total == previous["equal_nonempty_prefix_rows_per_arm"] == 41482
            and dict(transitions) == expected, "Exact P282/P311 matched population and .90 transitions required")
    strata = {}
    for stratum in p311.STRATA:
        strata[stratum] = {}
        for passing in ARMS:
            key = passing + "_only"
            prior = previous["directional_090_switches"][stratum][key]
            aggregate = []
            by_band = {}
            for band, pairs in rows[stratum][passing].items():
                measured = [pair[0] for pair in pairs]
                edit_rows = [pair[1] for pair in pairs]
                actual = dict(targets=len(pairs), **p311.summarize_arm_edits(edit_rows))
                require(actual == prior["failing_best_iou_bands"][band],
                        "Exact P311 per-band selected-mask IoU, edit and area aggregates required")
                by_band[band] = summarize(measured)
                aggregate.extend(measured)
            require(len(aggregate) == prior["targets"], "Exact P311 directional switch denominator required")
            strata[stratum][key] = dict(failing_arm=prior["failing_arm"], totals=summarize(aggregate),
                                        by_failing_raw_iou_band=by_band)
    return dict(images=len(images), annotations=len(gt["annotations"]), exclusions=dict(exclusions), eligible_gt=1778,
                equal_nonempty_prefix_rows_per_arm=prefix_total, transitions=expected, strata=strata)


def diagnose(args):
    p311 = load_p311(args.p311_source)
    require((args.pair_sha256, args.training_protocol, args.accepted_result_sha256)
            == (p311.PAIR_SHA, p311.TRAINING_PROTOCOL, p311.PRIMARY_SHA), "Exact independently accepted P311 training lineage")
    previous = json.loads(pinned(args.p311_result, P311_RESULT_SHA, 1024**2))
    require(previous["status"] == "HRSID_COMMON_E24_DEV705_STRICT_RESIDUAL_COMPLETE_V1"
            and previous["official_TEST_opened"] is previous["COCO_AP"] is False
            and previous["source_sha256"]["strict_residual.py"] == P311_SOURCE_SHA
            and previous["accepted_result_sha256"] == p311.PRIMARY_SHA, "Exact closed P311 aggregate provenance required")
    closed = p311.load_closed(args.code_root)
    coverage = closed.load_source(args.code_root / p311.COVERAGE, p311.COVERAGE_SHA, "_p312_original_coverage")
    driver = closed.load_gate(args.code_root).load_driver()
    primary = driver.read_json(args.accepted_result, args.accepted_result_sha256)
    require(primary["status"] == p311.PRIMARY_STATUS and primary["images"] == 705 and primary["primary_annotations"] == 1782
            and primary["official_TEST_opened"] is False and primary["training_protocol"] == p311.TRAINING_PROTOCOL
            and primary["training_seed"] == 5202, "Exact independently accepted original DEV705 score scope")
    gt, native, center, provenance = closed.verified_inputs(
        args.code_root, args.pair, args.pair_sha256, native_arm_receipt=args.native_arm_receipt,
        center_arm_receipt=args.center_arm_receipt, training_protocol=args.training_protocol)
    require(provenance == previous["provenance"]
            and provenance["training_pair_receipt_sha256"] == primary["independent_training_pair_sha256"]
            and provenance["evaluation_protocol"] == primary["evaluation_protocol"]
            and provenance["original_dev_gt_sha256"] == primary["original_dev_gt_sha256"],
            "Exact P311 original closed prediction and GT identity required")
    result = census(gt, native, center, p311, coverage, previous)
    return dict(status=STATUS, **result, p311_source_sha256=P311_SOURCE_SHA, p311_result_sha256=P311_RESULT_SHA,
                provenance=provenance, closed_predictions_only=True, official_TEST_opened=False, COCO_AP=False,
                model_forward_calls=0, training=False, device="CPU", training_seed=5202,
                radius_original_pixels=RADIUS, translation_families=["vertical", "horizontal", "joint"],
                area_policy="original whole mask area including translated pixels outside the canvas",
                population_scope="earliest native-rank maximum-IoU failing mask on exactly one-arm strict GT transitions")


def deny_network(event, _):
    if event.startswith("socket."):
        raise PermissionError("Offline closed-mask translation required")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code-root", "pair", "native-arm-receipt", "center-arm-receipt", "accepted-result", "p311-result", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--p311-source", type=Path, required=True)
    for name in ("pair-sha256", "training-protocol", "accepted-result-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and sys.flags.isolated == sys.flags.utf8_mode == 1
            and sys.dont_write_bytecode, "Exact offline isolated CPU invocation required")
    require(args.code_root.is_absolute() and args.code_root.resolve(strict=True) == args.code_root
            and args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent
            and not args.output.exists() and not args.output.is_symlink(), "Canonical sources and fresh output required")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    require(version("numpy") == "1.26.4" and version("pycocotools") == "2.0.11",
            "Exact original isolated CPU mask runtime required")
    sys.addaudithook(deny_network)
    result = diagnose(args)
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with args.output.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=STATUS, result_sha256=sha256(raw).hexdigest()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
