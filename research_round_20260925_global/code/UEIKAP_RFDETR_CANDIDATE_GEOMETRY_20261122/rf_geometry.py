"""Measure post-hoc geometry ceilings on the completed RF-DETR UEIKAP cohort."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from types import ModuleType

import numpy as np
from pycocotools import mask as codec
import shapely


TERMINALS = {
    "capture": "312e225a719cc68d1d64deaf65e0a5ef034c2a5258885416d846a4fafd485156",
    "verify-capture": "1ad8565802f1d83046fcf9ef8405ca9b0f5bc4e5bb10c28fb4c6b66e41934262",
    "score": "4057a19efa37d6d9bf03dd89450beeb178b234d4b5eb2f0283d3314acdf7e01c",
}
INTAKE = "d19bad46ea19ac4b01c6b462eb7470a45d3d0ed498015675cd7778a0e83752d8"
DIAG = "5713fbdd397a59f39632dcc0443aa14af7d8356b8ba760912f58eb8b3cb23dcf"
RF_SCORER = "9ecbac508e1506b6cee2706a31f497eb4a46dd9bb62fbd56340e9e64ec981fdc"
DFINE_BOUNDS = "8ff238513c37cf1c566a85d86025303fba51499be64a440236288d589bac20af"
LEVELS = (0.25, 0.50, 0.75)


def check(exact, message):
    if not exact:
        raise ValueError(message)


def pin(path, expected):
    data = path.read_bytes()
    check(sha256(data).hexdigest() == expected, "Digest mismatch: " + path.name)
    return data


def module(path, expected):
    result = ModuleType(path.stem)
    result.__file__ = str(path)
    exec(compile(pin(path, expected), str(path), "exec"), result.__dict__)
    return result


def match_iou(intersection, area, gt_area):
    return intersection / (area + gt_area - intersection)


def ceilings(area, gt_area, hit, capacity, gt_inside, area_inside, hit_inside):
    outside_hit = hit - hit_inside
    outside_area = area - area_inside
    ideal_hit = np.minimum(area, gt_area)
    guarded_steps = np.where((area >= 32) & (area < 1024), 8, 0)
    return {
        "mask": match_iou(hit, area, gt_area),
        "area_only": match_iou(ideal_hit, area, gt_area),
        "crop_only": gt_inside / gt_area,
        "area_and_crop": np.where(capacity >= area, match_iou(np.minimum(area, gt_inside), area, gt_area), 0.),
        "up_to_eight_anywhere": match_iou(np.minimum(ideal_hit, hit + 8), area, gt_area),
        "area_guard_and_up_to_eight_anywhere": match_iou(np.minimum(ideal_hit, hit + guarded_steps), area, gt_area),
        "within_box_equal_area_preserve_exterior": match_iou(outside_hit + np.minimum(area_inside, gt_inside), area, gt_area),
        "within_box_free_area_preserve_exterior": match_iou(outside_hit + gt_inside, outside_area + gt_inside, gt_area),
    }


def validate(args):
    frozen = {}
    for phase, digest in TERMINALS.items():
        event = json.loads(pin(args.run / ("observed_" + phase) / "TERMINAL.json", digest))
        check(event["phase"] == phase and event["actual_exit"] == 0 and event["output_footer_verified"] is True,
              "Original RF stage did not close")
        frozen[phase] = json.loads(pin(args.run / phase / "RESULT_SAFE.json", event["result_sha256"]))
    capture, verify, score = (frozen[x] for x in TERMINALS)
    intake = json.loads(pin(args.intake / "intake_terminal.json", INTAKE))
    check(intake["actual_exit"] == 0 and intake["status"] == "completed_verified"
          and intake["independently_verified_member_files"] == 216, "Incomplete author intake")
    check(capture["counts"] == verify["counts"] == {"eligible": 6199, "mapped": 10738, "overrides": 0, "rank_gaps": 62}
          and capture["GT_opened"] is verify["GT_opened"] is score["whole_scene_AP"] is False
          and score["previously_scored_acquisition_panel"] is True
          and capture["images"] == verify["images"] == score["images"] == 108
          and capture["model_forward_calls"] == 108 and verify["model_forward_calls"] == score["model_forwards"] == 0
          and capture["optimizer_updates"] == verify["optimizer_updates"] == score["optimizer_updates"] == 0,
          "Unexpected stage scope")
    check(capture["intake_sha256"] == verify["intake_sha256"] == intake["images_roster_sha256"]
          == score["inputs"]["image_roster_sha256"] and score["inputs"]["intake_terminal_sha256"] == INTAKE
          and capture["predictions_sha256"] == verify["predictions_sha256"] == score["inputs"]["predictions_sha256"]
          and capture["audit_sha256"] == verify["audit_sha256"] == score["inputs"]["candidate_audit_sha256"]
          and score["inputs"]["rf_scorer_sha256"] == RF_SCORER,
          "RF score and original capture inputs diverge")
    rf = module(args.rf_scorer, RF_SCORER)
    core = rf.load_geometry(args.frozen_scorer)
    diag = module(args.reference_diagnose, DIAG)
    pred, audit = [core.stream_pinned(args.run / "capture" / name, capture[key]) for name, key in (
        ("PREDICTIONS_PRIVATE.jsonl", "predictions_sha256"), ("CANDIDATE_AUDIT_PRIVATE.jsonl", "audit_sha256"))]
    labels = core.json_pinned(args.intake / "labels_roster.json", intake["labels_roster_sha256"])
    images = core.json_pinned(args.intake / "images_roster.json", intake["images_roster_sha256"])
    members = core.json_pinned(args.intake / "member_provenance.json", intake["member_provenance_sha256"])
    reserved = core.json_pinned(args.reservation, core.RESERVATION_SHA)
    core.verify_reservation(reserved, labels["items"], members)
    ids = list(range(1, 109))
    check(labels["expected_ids"] == images["expected_ids"] == ids and all(len(x) == 108 for x in (
        labels["items"], images["items"], pred, audit)), "Author roster differs")
    return rf, core, diag, zip(ids, images["items"], labels["items"], pred, audit), score, intake


def geometry_from_rle(masks, gt_rles, boxes, gt_areas):
    area = codec.area(masks).astype(np.int64)
    x0, y0, x1, y1 = boxes.T
    capacity = np.maximum(x1-x0, 0) * np.maximum(y1-y0, 0)
    crops = codec.frPyObjects(np.column_stack((x0, y0, np.maximum(x1-x0, 0), np.maximum(y1-y0, 0))).astype(float),
                              512, 512)
    clipped = [codec.merge([rle, crop], intersect=True) for rle, crop in zip(masks, crops)]
    inside = codec.area(clipped).astype(np.int64)
    sum_inside = np.stack([rectangle_sums(codec.decode(t), boxes, capacity) for t in gt_rles], axis=1)
    hit = np.asarray([[int(codec.area(codec.merge([rle, t], intersect=True))) for t in gt_rles] for rle in masks])
    hit_inside = np.asarray([[int(codec.area(codec.merge([rle, t], intersect=True))) for t in gt_rles] for rle in clipped])
    return inside, ceilings(area[:, None], gt_areas[None, :], hit, capacity[:, None], sum_inside,
                            inside[:, None], hit_inside)


def rectangle_sums(raster, boxes, capacity):
    table = np.pad(raster.cumsum(axis=0, dtype=np.int64).cumsum(axis=1), ((1, 0), (1, 0)))
    x0, y0, x1, y1 = boxes.T
    values = table[y1, x1] - table[y0, x1] - table[y1, x0] + table[y0, x0]
    return np.where(capacity > 0, values, 0)


def all_query_boxes(audit):
    raw = np.asarray(audit["raw_boxes_cxcywh"], dtype=np.float32)
    check(raw.shape == (164, 4) and np.isfinite(raw).all(), "Expected original raw Q164 box plane")
    centers, halves = raw[:, :2], .5 * raw[:, 2:]
    return (np.column_stack((centers-halves, centers+halves)) * 512).clip(0, 512).astype(np.float64)


def observations(args):
    rf, core, diag, stream, score, intake = validate(args)
    groups, counts = defaultdict(list), Counter()
    descriptions = defaultdict(list)
    for number, im, label, pred, audit in stream:
        check(number == im["image_id"] == label["image_id"] and im["original_hw"] == [512, 512], "Wrong chip")
        masks = rf.native_masks(core, pred, audit, number)
        check(masks, "RF capture unexpectedly supplies no mapped ship candidates")
        source = core.label_path(args.intake, label["label_ref"])
        truth, label_counts = core.parse_yolo(source, 512, 512, shapely)
        check(truth and label_counts == label["count_by_class"], "Unexpected class census")
        rles = [rle for _, rle in truth]
        gt_area = codec.area(rles).astype(np.int64)
        gt_bbox = np.asarray([polygon.bounds for polygon, _ in truth])
        candidates = audit["candidates"]
        area = codec.area(masks).astype(np.int64)
        box = np.asarray([x["box_xyxy"] for x in candidates], dtype=np.float64)
        qid = [x["query_id"] for x in candidates]
        raw_boxes = all_query_boxes(audit)
        check(len(set(qid)) == len(qid) and np.array_equal(box, raw_boxes[qid]) and np.all(gt_area > 0),
              "Original box pairing or raster area differs")
        integer_box = np.ceil(box).astype(np.int64)
        inside, values = geometry_from_rle(masks, rles, integer_box, gt_area)
        stock = codec.iou(masks, rles, [False] * len(rles))
        check(np.array_equal(values["mask"], stock), "RLE product versus standard COCO raster IoU differs")
        check(np.all(inside <= area), "Native intersection with own box exceeds mask area")
        values["box"] = diag.bbox_iou(box, gt_bbox)
        values["unfiltered_q164_box_only"] = diag.bbox_iou(raw_boxes, gt_bbox)
        masks_xywh = codec.toBbox(masks)
        truth_xywh = codec.toBbox(rles)
        values["tight_mask_extent_box"] = diag.bbox_iou(
            np.column_stack((masks_xywh[:, :2], masks_xywh[:, :2]+masks_xywh[:, 2:])),
            np.column_stack((truth_xywh[:, :2], truth_xywh[:, :2]+truth_xywh[:, 2:])))
        matches = core.match_native(masks, rles)
        check(diag.rank_matches(stock, .5) == matches, "Original greedy pairs differ")
        counts.update(images=1, hulls=len(truth), greedy_matches=len(matches), wakes=label_counts.get("1", 0),
                      mapped_candidates=len(masks), empty_candidates=int((area==0).sum()),
                      eligible_area_candidates=int(((area>=32) & (area<1024)).sum()),
                      candidates_with_fg_outside_box=int((area>inside).sum()),
                      candidates_with_forced_equal_area_infeasible=int((area>np.prod(
                          np.maximum(integer_box[:, 2:]-integer_box[:, :2], 0), axis=1)).sum()))
        descriptions["gt_raster_area_px"].extend(gt_area.tolist())
        descriptions["nonempty_mask_area_px"].extend(area[area > 0].tolist())
        descriptions["fraction_native_mask_outside_box"].extend(((area[area > 0]-inside[area > 0])/area[area > 0]).tolist())
        top_box = values["box"].argmax(axis=0)
        top_mask = values["mask"].argmax(axis=0)
        for j, (b, m) in enumerate(zip(top_box, top_mask)):
            descriptors = {"best_box_iou_per_hull": values["box"][b, j], "best_mask_iou_per_hull": values["mask"][m, j]}
            if values["box"][b, j] >= .5:
                descriptors.update(qualifying_best_box_corresponding_mask_iou=values["mask"][b, j],
                                   qualifying_best_box_mask_area_over_truth=area[b]/gt_area[j])
            if j not in matches:
                descriptors.update(greedy_unmatched_best_mask_area_over_truth=area[m]/gt_area[j],
                                   greedy_unmatched_best_mask_iou=values["mask"][m, j])
            for k, value in descriptors.items():
                descriptions[k].append(float(value))
            groups[label["acquisition_group"]].append({k: float(matrix[:, j].max()) for k, matrix in values.items()})
    check(counts["images"] == 108 and counts["hulls"] == 139 and len(groups) == 18
          and counts["wakes"] == 114 and counts["greedy_matches"] == score["labeled_hulls_and_coverage"]["native_matched"] == 17
          and counts["mapped_candidates"] == 10738 and counts["eligible_area_candidates"] == 6199, "Wrong closed census")
    return groups, dict(counts), {k: diag.summary(v) for k,v in descriptions.items()}, intake


def threshold_rows(groups):
    names = next(iter(groups.values()))[0].keys()
    thresholds = {}
    for t in LEVELS:
        out = {}
        for name in names:
            per_group = [[(row[name] >= t, row["mask"] >= t) for row in g] for g in groups.values()]
            hits = sum(h for rows in per_group for h, _ in rows)
            out[name] = {
                "hulls": hits,
                "groups": sum(any(h for h, _ in rows) for rows in per_group),
                "equal_acquisition_weight_fraction": float(np.mean([np.mean([h for h,_ in rows]) for rows in per_group])),
                "hulls_added_vs_any_native_mask": sum(h and not base for rows in per_group for h,base in rows),
                "hulls_missing_vs_any_native_mask": sum(base and not h for rows in per_group for h,base in rows),
            }
        for row in ("up_to_eight_anywhere", "area_guard_and_up_to_eight_anywhere", "area_only",
                    "within_box_equal_area_preserve_exterior", "within_box_free_area_preserve_exterior"):
            check(out[row]["hulls_missing_vs_any_native_mask"] == 0, "An edit upper bound lost native opportunity")
        check(out["area_guard_and_up_to_eight_anywhere"]["hulls"] <= out["up_to_eight_anywhere"]["hulls"] <=
              out["area_only"]["hulls"] and out["area_and_crop"]["hulls"] <= min(out["crop_only"]["hulls"],
              out["area_only"]["hulls"]) and out["box"]["hulls"] <= out["unfiltered_q164_box_only"]["hulls"],
              "Impossible geometry ceiling ordering")
        thresholds[str(t)] = out
    return thresholds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "intake", "reservation", "reference-diagnose", "frozen-scorer", "rf-scorer", "dfine-bounds", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    check(not args.output.exists(), "Use a fresh diagnosis directory")
    groups, counts, descriptions, intake = observations(args)
    prior = json.loads(pin(args.dfine_bounds, DFINE_BOUNDS))
    result = dict(status="UEIKAP_RFDETR_POSTHOC_GEOMETRY_COMPLETE", scope="Previously scored released hull cohort",
                  counts=counts, acquisitions=len(groups), thresholds=threshold_rows(groups), descriptions=descriptions,
                  dfine_closed_comparison=prior["scopes"], optimizer_updates=0, new_model_forwards=0,
                  frozen_predictions_modified=False, whole_scene_ap=False, original_intake_sha256=INTAKE,
                  label_roster_sha256=intake["labels_roster_sha256"], bound_rf_terminal_shas=TERMINALS,
                  external_dfine_bounds_sha256=DFINE_BOUNDS, diagnostic_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  definitions={
                      "box": "Reported detector box in pixel edges versus exact envelope of author polygon.",
                      "unfiltered_q164_box_only": "Every raw query box before class/score selection. No masks evaluated outside the mapped top100.",
                      "mask": "Observed mapped joint-top100 ship mask versus rasterized author hull.",
                      "tight_mask_extent_box": "Integer bounding boxes of both the RF raster and author hull raster.",
                      "area_only": "Exact stock foreground area, arbitrary position inside the chip.",
                      "crop_only": "Force raster inside reported integer box, arbitrary foreground area, ignoring original outside pixels.",
                      "area_and_crop": "Force raster inside reported integer box, exact stock area when feasible; infeasible candidates provide no reach.",
                      "up_to_eight_anywhere": "Keep stock area and replace up to eight pixels per sign anywhere on chip.",
                      "area_guard_and_up_to_eight_anywhere": "Same ideal swaps but spend zero outside original [32,1024) area guard.",
                      "within_box_equal_area_preserve_exterior": "Keep original pixels outside reported box and the original area inside, freely rearrange within.",
                      "within_box_free_area_preserve_exterior": "Keep original pixels outside reported box, freely rearrange or resize within.",
                      "limits": "Any-candidate, reusable-candidate per-hull oracle under GT; not a trained method or fixed matching result. RF stock may extend outside its own box."
                  })
    args.output.mkdir()
    data = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    (args.output / "RESULT_SAFE.json").write_bytes(data)
    print(json.dumps(dict(status=result["status"], result_sha256=sha256(data).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
