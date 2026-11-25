"""Audit raw RF-DETR proposal-box coverage on the previously scored SAR-SSP validation cohort."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType
from zipfile import ZipFile

import numpy as np


CAPTURE_SHA = "c40a133d62fafdecb9736d17abbd0ddcbf31ec42672d61eb0cbfc5d5df0b643e"
CAPTURE_TERM_SHA = "26a62e1c3eff5f79c6e34ff9988b1c6d33b059933e8e3771e7b292139c37d731"
READBACK_SHA = "5788cd43fdc1e404cf6cc3c611dec47bb8ed30c55b926b22aabb6d3ec80e3711"
READBACK_TERM_SHA = "467b63e4ce773a8648a3685f11b319e3794cb96982070dcf2a5aa5c9ff3edf8f"
SCORE_SHA = "fa2508ebde848235dada628406f095a330955f103fa611256964a3f714ea9972"
SCORE_TERM_SHA = "d09229a37a456a1f267303e03237fb54f7c813cd546e9cfaae833ae06e9cc430"
SCORE_WORKER_SHA = "d4d16cdfab0903480c7e6f3686de973886b726d54f91b7e235686c1502d44735"
NATIVE_SOURCE_SHA = "6549dd151178a17146de518fbe16a3a045d90c5b4ed8345d4d8e272db0e51c2d"
CHECKPOINT_SHA = "011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11"
MANIFEST_SHA = "16c8e6b7655e51f208c3a9708acd419a60126e55a9075cdea7d4b9e4edbb9425"
STATUS = "SAR_SSP_VAL232_RAW_Q164_BOX_COVERAGE_V1"
THRESHOLDS = (.25, .50, .75)
SETS = ("stock_mapped_joint_top100", "ship_score_top100", "ship_score_all164")
VIEWS = ("all_139_hulls", "original_stock_mask_unmatched_47_hulls")
HW = [500, 500]
RESULT = "RESULT_SAFE.json"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def pinned(path, expected):
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == expected, "Exact prior artifact bytes: " + path.name)
    return raw


def module(path, expected):
    source = ModuleType(path.stem)
    source.__file__ = str(path)
    exec(compile(pinned(path, expected), str(path), "exec"), source.__dict__)
    return source


def observed(root, phase, result_sha, terminal_sha, status):
    terminal_path = root / ("observed_" + phase) / "TERMINAL.json"
    terminal = json.loads(pinned(terminal_path, terminal_sha))
    result = json.loads(pinned(root / phase / RESULT, result_sha))
    need(terminal["phase"] == result["phase"] == phase and terminal["actual_exit"] == 0
         and terminal["output_footer_verified"] is True and terminal["result_sha256"] == result_sha
         and terminal["worker_sha256"] == result["worker_sha256"] and result["status"] == status,
         "Closed observed original phase: " + phase)
    return result


def rankings(probs, limit=100):
    ordered = np.argsort(-probs.ravel(), kind="stable")[:limit]
    joint = [(rank, int(x)//2) for rank, x in enumerate(ordered, 1) if x % 2 == 0]
    ship_all = [int(x) for x in np.argsort(-probs[:, 0], kind="stable")]
    return joint, ship_all[:limit], ship_all


def iou_matrix(gt, det):
    gt, det = np.asarray(gt, dtype=np.float64), np.asarray(det, dtype=np.float64)
    low = np.maximum(gt[:, None, :2], det[None, :, :2])
    high = np.minimum(gt[:, None, 2:], det[None, :, 2:])
    intersection = np.maximum(0, high - low).prod(axis=2)
    gt_area = np.maximum(0, gt[:, 2:] - gt[:, :2]).prod(axis=1)
    det_area = np.maximum(0, det[:, 2:] - det[:, :2]).prod(axis=1)
    union = gt_area[:, None] + det_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def greedy50(ious):
    remaining = np.ones(ious.shape[0], dtype=bool)
    hits = np.zeros(ious.shape[0], dtype=bool)
    for column in ious.T:
        options = np.where(remaining, column, -1)
        idx = int(options.argmax())
        if options[idx] >= .50:
            remaining[idx] = False
            hits[idx] = True
    return hits


def closed_inputs(args, native, scorer):
    root = args.capture
    capture = observed(root, "capture", CAPTURE_SHA, CAPTURE_TERM_SHA, "SAR_SSP_VAL232_RFDETR_NATIVE_CAPTURED_V1")
    readback = observed(root, "verify-capture", READBACK_SHA, READBACK_TERM_SHA, "SAR_SSP_VAL232_RFDETR_NATIVE_VERIFIED_V1")
    previous = observed(args.score, "score", SCORE_SHA, SCORE_TERM_SHA, "SAR_SSP_VAL232_NATIVE_RASTER_SCORE_V1")
    need(capture["checkpoint_sha256"] == readback["checkpoint_sha256"] == previous["checkpoint_sha256"] == CHECKPOINT_SHA
         and capture["intake_sha256"] == readback["intake_sha256"] == MANIFEST_SHA
         and capture["images"] == readback["images"] == previous["images"] == 232
         and capture["archive_sha256"] == readback["archive_sha256"] == previous["archive_sha256"] == scorer.ARCHIVE_SHA
         and capture["target_labels_opened"] is readback["target_labels_opened"] is previous["target_fit_or_selection"] is False
         and capture["model_forward_calls"] == 232 and readback["model_forward_calls"] == 0
         and readback["exact_native_parity"] is True and readback["capture_sha256"] == previous["capture_sha256"] == CAPTURE_SHA
         and previous["capture_readback_sha256"] == READBACK_SHA
         and capture["predictions_sha256"] == readback["predictions_sha256"] == previous["native_predictions_sha256"]
         and capture["audit_sha256"] == readback["audit_sha256"] == previous["native_audit_sha256"],
         "Source-capture/raw-output parity and linkage to observed original score")
    intake = json.loads(pinned(root / "intake/IMAGE_MANIFEST_PRIVATE.json", MANIFEST_SHA))["images"]
    preds = scorer.read_stream(root / "capture" / native.PRED, capture["predictions_sha256"])
    audits = scorer.read_stream(root / "capture" / native.AUDIT, capture["audit_sha256"])
    records, names, box_sets = {}, {}, {}
    n_retained = 0
    extra_counts = [0, 0]
    zero_counts = [0, 0, 0]
    for item, pred, row in zip(intake, preds, audits, strict=True):
        image_id = item["image_id"]
        need(item["original_hw"] == row["original_hw"] == HW
             and pred["image_id"] == row["image_id"] == image_id and image_id not in records
             and row["image_sha256"] == item["image_ref"]["sha256"] and row["rgb_sha256"] == item["rgb_sha256"]
             and row["source_image_name"] == item["source_image_name"], "Exact official intake and raw capture image context")
        raw = np.asarray(row["raw_boxes_cxcywh"], dtype=np.float32)
        probs = np.asarray(row["raw_query_probabilities"], dtype=np.float32)
        need(raw.shape == (164, 4) and probs.shape == (164, 2)
             and np.isfinite(raw).all() and np.isfinite(probs).all()
             and np.all(raw[:, 2:] >= 0) and np.all((0 <= probs) & (probs <= 1)),
             "Exact complete normalized raw Q164 decoder heads")
        joint, ship, all_ids = rankings(probs)
        ranks, ids = [r for r, _ in joint], [q for _, q in joint]
        boxes = native.native_xyxy(raw, HW)
        need(row["query_ids"] == ids and row["native_ranks"] == pred["native_ranks"] == ranks
             and pred["raw_query_count"] == 164 and pred["native_count"] == 100
             and pred["unmapped_slot1_count"] == 100 - len(ids)
             and row["native_boxes_xyxy"] == boxes[ids].tolist()
             and set(ids) <= set(ship) <= set(all_ids) and len(ship) == 100 and len(all_ids) == 164,
             "Exact original rank/query/clipped-box parity and nested analytic sets")
        for position, (rank, ident, candidate, stock) in enumerate(zip(
                ranks, ids, row["candidates"], pred["coco"], strict=True), 1):
            need(candidate["position"] == position and candidate["native_rank"] == rank and candidate["query_id"] == ident
                 and candidate["box_xyxy"] == boxes[ident].tolist()
                 and stock["image_id"] == image_id and stock["category_id"] == 1 and stock["segmentation"]["size"] == HW
                 and candidate["score"] == stock["score"] == float(probs[ident, 0])
                 and int(scorer.codec.area(stock["segmentation"])) == candidate["mask_area"],
                 "Every published stock candidate keeps identical native box, class, confidence and mask")
        sets = tuple(boxes[order] for order in (ids, ship, all_ids))
        for index, block in enumerate(sets):
            zero_counts[index] += int(np.count_nonzero(np.any(block[:, :2] == block[:, 2:], axis=1)))
        n_retained += len(ids)
        extra_counts[0] += len(ship) - len(ids)
        extra_counts[1] += len(all_ids) - len(ship)
        box_sets[image_id] = sets
        names[image_id] = (item["source_image_name"], item["image_ref"]["sha256"])
        records[image_id] = [dict(native=stock["segmentation"], score=stock["score"], masks={})
                             for stock in pred["coco"]]
    need(len(records) == 232 and n_retained == capture["counts"]["mapped"] == previous["candidates"] == 22741,
         "Complete unchanged original retained candidates before annotation access")
    census = dict(retained_stock_boxes=n_retained, ship_top100_boxes=23200, all_raw_boxes=38048,
                  ship_top100_added_to_stock=extra_counts[0], raw_q164_added_to_ship_top100=extra_counts[1],
                  zero_area_boxes=dict(zip(SETS, zero_counts, strict=True)))
    return records, names, box_sets, previous, census


def targets(archive, scorer, names):
    with archive.open("rb") as stream:
        digest = sha256()
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
        need(digest.hexdigest() == scorer.ARCHIVE_SHA, "Exact previously scored author archive")
        with ZipFile(stream) as package:
            return scorer.exact_ground_truth(package, names)


def coverage(labels, maxima, greedy, *, groups):
    names = [names for names in groups.values() if names]
    oracle, greedy_count = {}, {}
    for index, key in enumerate(SETS):
        oracle[key] = {}
        for threshold in THRESHOLDS:
            hit = maxima[:, index] >= threshold
            oracle[key][f"{threshold:.2f}"] = dict(
                hulls=int(hit.sum()), groups_with_any=int(sum(hit[ids].any() for ids in names)),
                hull_coverage_pct=100. * float(hit.mean()),
                group_balanced_coverage_pct=100. * float(np.mean([hit[ids].mean() for ids in names])))
        greedy_count[key] = dict(hulls=int(greedy[:, index].sum()),
                                 groups_with_any=int(sum(greedy[ids, index].any() for ids in names)))
    additional = {}
    for name, left, right in (("ship_top100_minus_stock",1,0), ("all164_minus_ship_top100",2,1),
                              ("all164_minus_stock",2,0)):
        additional[name] = {}
        for threshold in THRESHOLDS:
            new = (maxima[:, left] >= threshold) & (maxima[:, right] < threshold)
            additional[name][f"{threshold:.2f}"] = dict(
                hulls=int(new.sum()), groups_with_any=int(sum(new[ids].any() for ids in names)))
    return dict(hulls=len(labels), source_groups=len(names), per_hull_box_oracle=oracle,
                greedy_one_to_one_iou50=greedy_count, additional_covered_hulls=additional)


def diagnose(truth, groups, boxes, matches):
    annotation_by_image = defaultdict(list)
    for ann in truth["annotations"]:
        annotation_by_image[ann["image_id"]].append(ann)
    labels, maxima, greedy = [], [], []
    for image_id, annotations in annotation_by_image.items():
        gt_native = np.asarray([ann["bbox"] for ann in annotations], dtype=np.float64)
        need(np.isfinite(gt_native).all() and np.all(gt_native[:, 2:] > 0), "The exact author hull bounding rectangles")
        gt_native[:, 2:] += gt_native[:, :2]
        want = [iou_matrix(gt_native, native) for native in boxes[image_id]]
        labels.extend((ann["id"], groups[image_id]) for ann in annotations)
        maxima.extend(np.column_stack([v.max(axis=1) if v.shape[1] else np.zeros(len(annotations)) for v in want]))
        greedy.extend(np.column_stack([greedy50(value) for value in want]))
    names = [ann_id for ann_id, group in labels]
    values = np.asarray(maxima)
    decisions = np.asarray(greedy)
    need(len(names) == 139 and len(matches) == 92 and set(matches) <= set(names)
         and (values[:, 0] <= values[:, 1]).all() and (values[:, 1] <= values[:, 2]).all(),
         "Original all-hull fixed-matching denominator and nested geometric maxima")
    result = {}
    for number, view in enumerate(VIEWS):
        selected = np.asarray([i for i, (k, _) in enumerate(labels) if not number or k not in matches])
        identifiers = [labels[i] for i in selected]
        group_indices = defaultdict(list)
        for index, (_, group) in enumerate(identifiers):
            group_indices[group].append(index)
        result[view] = coverage(identifiers, values[selected], decisions[selected], groups=group_indices)
    need(result[VIEWS[0]]["hulls"] == 139 and result[VIEWS[0]]["source_groups"] == 23
         and result[VIEWS[1]]["hulls"] == 47 and len(set(groups.values())) == 31,
         "Exact author val and fixed native mask-unmatched subset")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("capture", "score", "code", "archive", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--protocol-sha", required=True)
    args = parser.parse_args()
    started = perf_counter()
    need(not args.output.exists() and sys.flags.isolated and sys.dont_write_bytecode
         and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "Fresh isolated CPU retrospective observation")
    pinned(args.protocol, args.protocol_sha)
    scorer = module(args.score / "score_sarssp.py", SCORE_WORKER_SHA)
    native = module(args.code / "INSTANCE_RSDD_FROZEN_HRSID_20261122/capture_instance_rsdd.py", NATIVE_SOURCE_SHA)
    records, names, boxes, previous, census = closed_inputs(args, native, scorer)
    print(json.dumps(dict(stage="closed_raw_and_retained_query_parity_before_labels", images=len(records),
                          verified_retained_boxes=census["retained_stock_boxes"])), flush=True)
    truth, groups, edges = targets(args.archive, scorer, names)
    # Native masks define the original fixed membership only. No AP or mask materialization for raw proposals.
    native_already_scored = previous["coco_mask_metric_points"]["native"]["AP"] / 100
    matches = scorer.native_matches(truth, records, native_already_scored)
    geometry = previous["fixed_native_match_all_truth"]
    need(len(matches) == geometry["matched_hulls"] == 92
         and geometry["unmatched_hulls"] == 47 and len(edges) == previous["edge_touch_hulls"],
         "Frozen original mask assignment is the only subgroup definition")
    result = dict(status=STATUS, phase="diagnostic", image_partition="val2017", images=232,
                  source_scene_groups=31, original_stock_mask_matched_hulls=92,
                  original_stock_mask_unmatched_hulls=47, box_candidates=census,
                  coverage=diagnose(truth, groups, boxes, matches), thresholds=list(THRESHOLDS),
                  ground_truth_opened_after_raw_parity=True, detector_forwards=0, optimizer_updates=0,
                  omitted_query_masks_materialized=False, segmentation_ap_for_raw_queries=False,
                  protocol_sha256=args.protocol_sha, worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  original_score_sha256=SCORE_SHA, capture_sha256=CAPTURE_SHA, capture_readback_sha256=READBACK_SHA,
                  historical_capture_audit_sha256=previous["native_audit_sha256"],
                  source_scope="retrospective analytic box coverage", seconds=perf_counter()-started)
    args.output.mkdir()
    data = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    with (args.output / RESULT).open("xb") as stream:
        stream.write(data)
    print(json.dumps(dict(status=STATUS, result_sha256=sha256(data).hexdigest())), flush=True)


if __name__ == "__main__":
    main()

