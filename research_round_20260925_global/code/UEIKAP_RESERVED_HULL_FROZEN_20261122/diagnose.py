"""Descriptive candidate opportunity in the completed UEIKAP reserved-hull evaluation."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import types

import numpy as np
from pycocotools import mask as codec
import shapely


TERMINALS = {
    "capture": "b71f8a67272116c5cca03c6384ee21f193a2fc1c8baad6126abd80df14bf5d6d",
    "verify-capture": "cd8cbced83856e8cf4cee29611c5382de09e4ec5a21d35d7ab550f2bda49117b",
    "prepare-edits": "271819fb8a9003aede1e88be242f798f9c6249466c6fba1218ceb0cf0febb344",
    "verify-edits": "8833e707c5d22174b4291af413a082b60afedc71e4f806d258c87038c0f01f9e",
    "score": "3a7d10aa52888673218322f142818fb48bc0e09e29e5a4c94c507a82e219eb6c",
}
INTAKE = "d19bad46ea19ac4b01c6b462eb7470a45d3d0ed498015675cd7778a0e83752d8"
LEVELS = (0.25, 0.50, 0.75)


def check(valid, message):
    if not valid:
        raise ValueError(message)


def pinned(path, expected):
    data = path.read_bytes()
    check(sha256(data).hexdigest() == expected, "Input digest mismatch: " + path.name)
    return data


def closed_results(run):
    results = {}
    for phase, digest in TERMINALS.items():
        terminal = json.loads(pinned(run / ("observed_" + phase) / "TERMINAL.json", digest))
        check(terminal["phase"] == phase and type(terminal["actual_exit"]) is int and
              terminal["actual_exit"] == 0 and terminal["output_footer_verified"], "Unclosed stage")
        results[phase] = json.loads(pinned(run / phase / "RESULT_SAFE.json", terminal["result_sha256"]))
    return results


def bbox_iou(boxes, targets):
    # Both arrays use continuous pixel edges. Point/line boxes carry zero area.
    wh = np.maximum(boxes[:, None, 2:] - boxes[:, None, :2], 0)
    truth_wh = targets[:, 2:] - targets[:, :2]
    intersection = np.maximum(np.minimum(boxes[:, None, 2:], targets[:, 2:]) -
                              np.maximum(boxes[:, None, :2], targets[:, :2]), 0).prod(axis=2)
    return intersection / (wh.prod(axis=2) + truth_wh.prod(axis=1) - intersection)


def rank_matches(matrix, threshold):
    # Match in the frozen rank order, keeping the higher truth index on ties.
    matches = {}
    for rank, row in enumerate(matrix, 1):
        offered = [(float(value), i) for i, value in enumerate(row)
                   if value >= threshold and i not in matches]
        if offered:
            _, i = max(offered)
            matches[i] = rank
    return matches


def summary(values):
    out = {"n": len(values)}
    if len(values):
        out.update(zip(("minimum", "q25", "median", "q75", "maximum"),
                       np.percentile(values, [0, 25, 50, 75, 100]).tolist()))
    return out


class Cohort:
    def __init__(self):
        self.candidates = Counter()
        self.joint = {str(t): Counter(both=0, box_only=0, mask_only=0, neither=0,
                                      same_candidate_meets_both=0, fixed_greedy_box=0, fixed_greedy_mask=0,
                                      any_box_raster_truth_bounds=0, any_mask_extent_box_raster_truth_bounds=0)
                      for t in LEVELS}
        self.describe = defaultdict(list)
        self.ranks = {str(t): defaultdict(list) for t in LEVELS}
        self.groups = {str(t): defaultdict(list) for t in LEVELS}

    def observe(self, boxes, mask_areas, mask_extents, scores, box_matrix, mask_matrix,
                box_raster_bounds, extent_iou, group, truth_areas):
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        positive = (widths > 0) & (heights > 0)
        self.candidates.update(rows=len(boxes), nonempty_masks=int((mask_areas > 0).sum()),
                               empty_masks=int((mask_areas == 0).sum()), positive_area_boxes=int(positive.sum()),
                               nonpositive_extent_boxes=int((~positive).sum()),
                               equal_endpoint_boxes=int(((widths == 0) | (heights == 0)).sum()),
                               nonpositive_extent_box_with_nonempty_mask=int(((~positive) & (mask_areas > 0)).sum()),
                               reversed_axis_boxes=int(((widths < 0) | (heights < 0)).sum()))
        self.describe["positive_box_width_px"].extend(widths[positive])
        self.describe["positive_box_height_px"].extend(heights[positive])
        self.describe["nonempty_mask_area_px"].extend(mask_areas[mask_areas > 0])
        self.describe["nonempty_mask_extent_width_px"].extend(mask_extents[mask_areas > 0, 2])
        self.describe["nonempty_mask_extent_height_px"].extend(mask_extents[mask_areas > 0, 3])
        bmax, mmax = box_matrix.max(axis=0), mask_matrix.max(axis=0)
        self.describe["best_box_iou_per_hull"].extend(bmax)
        self.describe["best_mask_iou_per_hull"].extend(mmax)
        for t in LEVELS:
            key = str(t)
            b, m = bmax >= t, mmax >= t
            self.groups[key][group].extend(zip(b.tolist(), m.tolist()))
            hits = self.joint[key]
            hits.update(both=int((b & m).sum()), box_only=int((b & ~m).sum()),
                        mask_only=int((~b & m).sum()), neither=int((~b & ~m).sum()),
                        same_candidate_meets_both=int(((box_matrix >= t) & (mask_matrix >= t)).any(axis=0).sum()),
                        fixed_greedy_box=len(rank_matches(box_matrix, t)),
                        fixed_greedy_mask=len(rank_matches(mask_matrix, t)),
                        any_box_raster_truth_bounds=int((box_raster_bounds.max(axis=0) >= t).sum()),
                        any_mask_extent_box_raster_truth_bounds=int((extent_iou.max(axis=0) >= t).sum()))
            for name, matrix, available in (("box", box_matrix, b), ("mask", mask_matrix, m)):
                ranks = self.ranks[key]
                for i in np.flatnonzero(available):
                    first = np.flatnonzero(matrix[:, i] >= t)[0]
                    best = int(np.argmax(matrix[:, i]))
                    ranks[name + "_first_qualifying_rank"].append(first + 1)
                    ranks[name + "_first_qualifying_score"].append(scores[first])
                    ranks[name + "_best_rank"].append(best + 1)
                    ranks[name + "_best_score"].append(scores[best])
                    if name == "box":
                        ranks["best_box_corresponding_mask_iou"].append(mask_matrix[best, i])
                        ranks["best_box_mask_area_over_truth_raster_area"].append(mask_areas[best] / truth_areas[i])

    def report(self):
        thresholds = {}
        for key, counter in self.joint.items():
            g = self.groups[key]
            check(len(g) == 18 and sum(len(x) for x in g.values()) == 139 and
                  sum(counter[k] for k in ("both", "box_only", "mask_only", "neither")) == 139, "Wrong denominator")
            group_means = np.asarray([np.mean(v, axis=0) for v in g.values()])
            ranks = self.ranks[key]
            thresholds[key] = dict(
                **counter, any_box=counter["both"] + counter["box_only"], any_mask=counter["both"] + counter["mask_only"],
                groups_with_any_box=int((group_means[:, 0] > 0).sum()),
                groups_with_any_mask=int((group_means[:, 1] > 0).sum()),
                equal_weight_family_coverage_fraction_box_mask=group_means.mean(axis=0).tolist(),
                qualifying_rank_and_score={k: summary(v) for k, v in sorted(ranks.items())},
            )
        return {"candidate_counts": dict(self.candidates), "thresholds": thresholds,
                "descriptive_summaries": {k: summary(v) for k, v in sorted(self.describe.items())}}


def execute(args):
    intake = json.loads(pinned(args.intake / "intake_terminal.json", INTAKE))
    check(intake["actual_exit"] == 0 and intake["status"] == "completed_verified" and
          intake["independently_verified_member_files"] == 216, "Author intake not closed")
    closed = closed_results(args.run)
    score = closed["score"]
    source_digest = score["inputs"]["scorer_sha256"]
    scorer = types.ModuleType("frozen_score")
    exec(compile(pinned(args.frozen_scorer, source_digest), "<verified frozen scorer>", "exec"), scorer.__dict__)
    labels = scorer.json_pinned(args.intake / "labels_roster.json", intake["labels_roster_sha256"])
    images = scorer.json_pinned(args.intake / "images_roster.json", intake["images_roster_sha256"])
    members = scorer.json_pinned(args.intake / "member_provenance.json", intake["member_provenance_sha256"])
    reservation = scorer.json_pinned(args.reservation, scorer.RESERVATION_SHA)
    scorer.verify_reservation(reservation, labels["items"], members)
    cap, verified = closed["capture"], closed["verify-capture"]
    check(cap["intake_terminal_sha256"] == verified["intake_terminal_sha256"] == INTAKE and
          cap["manifest_sha256"] == verified["manifest_sha256"] == intake["images_roster_sha256"] and
          cap["original_author_postprocess_parity"] and verified["independent_probability_and_candidate_census"] and
          cap["rows_at_100"] == 10800 and cap["post_nms_rows"] == 30910, "Native capture mismatch")
    hashes = ("predictions_sha256", "candidate_audit_sha256")
    check(all(verified[k] == cap[k] for k in hashes), "Readback hash mismatch")
    predictions, audits = [scorer.stream_pinned(args.run / "capture" / filename, verified[k]) for k, filename in
                           zip(hashes, ("DEV_MASK_PREDICTIONS_PRIVATE.jsonl", "DEV_CANDIDATE_AUDIT_PRIVATE.jsonl"))]
    ordered = list(range(1, 109))
    check(images["expected_ids"] == labels["expected_ids"] == ordered and
          all(len(seq) == len(ordered) for seq in (images["items"], labels["items"], predictions, audits)),
          "Unexpected chip inventory")
    scopes = {"top100": Cohort(), "all_post_nms": Cohort()}
    truth_stats = defaultdict(list)
    n_truth = n_wake = n_reproduced = 0
    for i, im, lbl, pred, audit in zip(ordered, images["items"], labels["items"], predictions, audits):
        check(im["image_id"] == lbl["image_id"] == i and im["original_hw"] == [512, 512], "Chip identity mismatch")
        scorer.label_path(args.intake, im["image_ref"])  # Verify member bytes without reopening pixel contents.
        truth, counts = scorer.parse_yolo(scorer.label_path(args.intake, lbl["label_ref"]), 512, 512, shapely)
        check(counts == lbl["count_by_class"] and truth, "Label census mismatch")
        n_truth += len(truth)
        n_wake += counts.get("1", 0)
        truth_rles = [rle for _, rle in truth]
        gt_boxes = np.array([polygon.bounds for polygon, _ in truth])
        gt_raster_xywh = np.asarray(codec.toBbox(truth_rles))
        gt_raster_boxes = gt_raster_xywh.copy()
        gt_raster_boxes[:, 2:] += gt_raster_boxes[:, :2]
        gt_areas = np.asarray(codec.area(truth_rles), dtype=float)
        size = gt_boxes[:, 2:] - gt_boxes[:, :2]
        check(np.all(size > 0), "Invalid author hull bounding geometry")
        truth_stats["polygon_envelope_width_px"].extend(size[:, 0])
        truth_stats["polygon_envelope_height_px"].extend(size[:, 1])
        truth_stats["polygon_envelope_short_side_px"].extend(size.min(axis=1))
        truth_stats["polygon_envelope_long_side_px"].extend(size.max(axis=1))
        truth_stats["polygon_area_px2"].extend([p.area for p, _ in truth])
        truth_stats["raster_area_px"].extend(gt_areas)
        top = scorer.verify_native(pred, audit, i, 512, 512)
        original_matches = scorer.match_native(top, truth_rles)
        caps = audit["candidates"]
        check(audit["post_nms"] == len(caps) >= 100 and audit["raw_queries"] == audit["internal_top"] == 300,
              "Incomplete native candidate cohort")
        stock = [p["segmentation"] for p in pred["coco"]]
        boxes = np.asarray([a["box_xyxy"] for a in caps], dtype=float)
        areas = np.asarray([a["mask_area"] for a in caps], dtype=float)
        extents = np.asarray(codec.toBbox(stock))
        scores = np.asarray([a["score"] for a in caps])
        check(boxes.shape == (len(caps), 4) and np.all(np.isfinite(boxes)) and
              np.all((boxes >= 0) & (boxes <= 512)),
              "Unexpected native detector box convention")
        check(np.all((scores >= 0) & (scores <= 1)), "Unexpected detector score")
        box_matrix = bbox_iou(boxes, gt_boxes)
        box_raster = bbox_iou(boxes, gt_raster_boxes)
        mask_matrix = codec.iou(stock, truth_rles, [False] * len(truth))
        mask_boxes = extents.copy()
        mask_boxes[:, 2:] += mask_boxes[:, :2]
        nonempty = areas > 0
        check(np.all(mask_boxes[nonempty, :2] >= np.ceil(boxes[nonempty, :2])) and
              np.all(mask_boxes[nonempty, 2:] <= np.ceil(boxes[nonempty, 2:])),
              "A nonempty native RLE extends outside its original half-open detector crop")
        mask_extent_matrix = bbox_iou(mask_boxes, gt_raster_boxes)
        check(rank_matches(mask_matrix[:100], 0.50) == original_matches, "Frozen matching discrepancy")
        n_reproduced += len(original_matches)
        for name, n in (("top100", 100), ("all_post_nms", len(caps))):
            scopes[name].observe(boxes[:n], areas[:n], extents[:n], scores[:n], box_matrix[:n], mask_matrix[:n],
                                 box_raster[:n], mask_extent_matrix[:n], lbl["acquisition_group"], gt_areas)
    old = score["labeled_hulls_and_coverage"]
    check(n_truth == old["labeled_hulls"] == 139 and n_wake == old["wake_rows_excluded"] == 114 and
          n_reproduced == old["native_matched"] == 2 and old["sf_positive"] == 0, "Frozen anchor mismatch")
    for name, count, empty in (("top100", "rows_at_100", "empty_at_100"),
                               ("all_post_nms", "post_nms_rows", "empty_post_nms")):
        check(scopes[name].candidates["rows"] == cap[count] and
              scopes[name].candidates["empty_masks"] == cap[empty], "Population census mismatch")
    return {
        "status": "posthoc_completed", "images": len(ordered), "acquisition_families": 18,
        "labeled_hulls": n_truth, "excluded_wakes": n_wake, "frozen_top100_matches_at_050": n_reproduced,
        "original_sf_positive": old["sf_positive"], "additional_model_forwards": 0, "optimizer_updates": 0,
        "scope": "Opportunity among released class-0 hulls. Descriptive target-label analysis after the frozen evaluation.",
        "definitions": {
            "box": "Native detector audit float xyxy in original 512x512 pixel edges, versus exact envelope of the scaled author polygon. No rounding or +1. Equal or reversed endpoints have zero extent and IoU; no reordering. Every nonempty RLE is verified within its audited half-open crop.",
            "mask": "Original post-NMS binary RLE versus author class-0 polygon raster from the frozen COCO scorer.",
            "availability": "Any single candidate reaching the threshold for one hull, allowing candidate reuse. Both permits separate candidates. Same-candidate overlap is also reported.",
            "greedy": "Descending frozen score, one-to-one per chip, highest available IoU with later truth index on ties. Only original top100 mask at .50 was prespecified.",
            "sensitivity": "any_box_raster_truth_bounds replaces analytic GT box with tight COCO raster box. any_mask_extent_box_raster_truth_bounds uses tight COCO boxes of both masks. Empty masks contribute zero.",
            "ranks": "Per qualifying hull, first means highest scoring qualifying row. Best means highest IoU, first rank on ties. Ranks start at 1.",
            "groups": "Equal-weight family coverage averages within-family hull coverage across all 18 families. Other counts are pooled.",
            "limits": "Post-hoc association cannot identify acquisition, scale, representation, calibration or annotation as the cause. No whole-scene AP, new holdout evidence or target tuning.",
        },
        "input_sha256": {**{k: cap[k] for k in hashes}, "frozen_scorer": source_digest,
                         "intake_terminal": INTAKE, "score_terminal": TERMINALS["score"],
                         "labels_roster": intake["labels_roster_sha256"], "images_roster": intake["images_roster_sha256"]},
        "diagnostic_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "truth_summaries": {k: summary(v) for k, v in sorted(truth_stats.items())},
        "scopes": {k: v.report() for k, v in scopes.items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("intake", "run", "reservation", "frozen-scorer", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    opts = parser.parse_args()
    result = execute(opts)
    with opts.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, sort_keys=True, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({"status": result["status"], "result_sha256": sha256(opts.output.read_bytes()).hexdigest()}))
