"""Score fixed native-query geometric interventions on the closed HRSID FIT export."""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import re
import sys
from time import perf_counter

import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO

from fit_query_export import INDEX_SHA, WIDTH, need, pinned, write
from geometry import analyze_pair

METHODS = ("baseline", "translation", "thickness", "joint", "joint_free", "local_band", "local_any")


def rank_pairs(overlap, threshold):
    """Fixed native order, greedily choosing the available highest-overlap GT."""
    claimed = np.zeros(overlap.shape[1], dtype=bool)
    pairs = []
    for query, values in enumerate(overlap):
        available = np.flatnonzero((values >= threshold) & ~claimed)[::-1]
        if available.size:
            gt = int(available[np.argmax(values[available])])
            claimed[gt] = True
            pairs.append((query, gt))
    return pairs


def iou(value):
    return value["tp"] / (value["tp"] + value["fp"] + value["fn"])


def strict(value):
    return value["tp"] >= 9 * (value["fp"] + value["fn"])


def summarize(groups, *, main):
    rows = [row for group in groups for row in group]
    total = len(rows)
    need(total > 0, "Fixed near-strict pair denominator")
    recoveries = {
        name: np.asarray([sum(strict(row["methods"][name]) for row in g) for g in groups])
        for name in METHODS
    }
    metrics = {}
    for name in METHODS:
        items = [row["methods"][name] for row in rows]
        success = [v for v in items if strict(v)]
        metrics[name] = dict(recovered=len(success), groups_recovered=int((recoveries[name] > 0).sum()),
                             mean_iou=sum(map(iou, items)) / total,
                             tp=sum(v["tp"] for v in items), fp=sum(v["fp"] for v in items),
                             fn=sum(v["fn"] for v in items), edits=sum(v["edits"] for v in items),
                             recovered_pair_edits=sum(v["edits"] for v in success))
    need(metrics["baseline"]["recovered"] == 0 and all(
        iou(row["methods"]["local_any"]) + 1e-15 >= iou(row["methods"]["local_band"]) >= iou(row["methods"]["joint"]) - 1e-15
        and iou(row["methods"]["joint_free"]) + 1e-15 >= iou(row["methods"]["joint"])
        for row in rows), "Identity and nested intervention dominance")
    def subgroup(subset):
        return dict(pairs=len(subset),
                    recovered={name: sum(strict(row["methods"][name]) for row in subset) for name in METHODS})
    by_iou = {}
    for lo, hi in ((.75, .80), (.80, .85), (.85, .90)):
        by_iou[f"{lo:.2f}-{hi:.2f}"] = subgroup(
            [row for row in rows if lo <= iou(row["methods"]["baseline"]) < hi])
    area = {k: subgroup([row for row in rows if row["small"] is small])
            for k, small in (("small", True), ("non_small", False))}
    need(sum(x["pairs"] for x in by_iou.values()) == sum(x["pairs"] for x in area.values()) == total,
         "Fixed IoU and annotation-area strata exhaust the pairs")
    result = dict(groups=len(groups), pairs=total, active_groups=sum(bool(group) for group in groups),
                  mean_budget=sum(row["budget"] for row in rows)/total,
                  pairs_with_budgeted_bulk_move=sum(row["bulk_admissible_count"] > 0 for row in rows),
                  pairs_with_band_errors=sum(row["local_repair_available"] > 0 for row in rows),
                  band_error_pixels=sum(row["local_repair_available"] for row in rows),
                  budget_impossible=total - metrics["local_any"]["recovered"],
                  band_only_impossible=metrics["local_any"]["recovered"] - metrics["local_band"]["recovered"],
                  methods=metrics, strata_by_iou=by_iou, strata_by_area=area)
    if main:
        indices = np.random.default_rng(5213).integers(0, len(groups), size=(4096, len(groups)))
        denominators = np.array([len(g) for g in groups])[indices].sum(axis=1)
        need(bool((denominators > 0).all()), "All fixed paired bootstrap denominators positive")
        contrasts = {}
        for control in ("joint", "joint_free"):
            difference = recoveries["local_band"] - recoveries[control]
            draws = difference[indices].sum(axis=1) / denominators
            contrasts[f"local_vs_{control}"] = dict(delta=float(difference.sum()/total),
                                                    interval=np.quantile(draws, [.025, .975], method="linear").tolist())
        local, bulk = metrics["local_band"], metrics["joint_free"]
        local_pass = (local["recovered"]/total >= .20 and local["groups_recovered"] >= 15
                      and contrasts["local_vs_joint"]["delta"] >= .10
                      and contrasts["local_vs_joint_free"]["delta"] >= .05
                      and contrasts["local_vs_joint_free"]["interval"][0] > 0)
        bulk_pass = bulk["recovered"]/total >= .20 and bulk["groups_recovered"] >= 15
        if local_pass:
            decision = "ADVANCE_LOCAL_PREDICTIVE_GEOMETRY"
        elif bulk_pass:
            decision = "PRIORITIZE_GLOBAL_DISPLACEMENT_AND_THICKNESS_CONTROL"
        else:
            decision = "STOP_LOCAL_AND_GLOBAL_ADVANCEMENT_REVIEW_BUDGET_OR_SUPPORT"
        result.update(contrasts=contrasts, local_gate=local_pass, bulk_gate=bulk_pass, decision=decision)
    return result


def run(args):
    started = perf_counter()
    previous = json.loads(pinned(args.previous, args.previous_sha256, 65536))
    export = previous["export"]
    need(previous["score"]["status"] == "ACTUAL_HRSID_FIT_Q164_RETENTION"
         and previous["proof"]["export_owner_closed"] is True
         and not previous["score"]["core"]["upper_bound_gate"]
         and previous["independent_recomputation"]["all_three_threshold_counts_identical"] is True,
         "Exact closed prior FIT inference and stop decision")
    cohort = json.loads(pinned(args.cohort, export["cohort_sha256"], 1024**2))
    blocks = [json.loads(line) for line in pinned(args.predictions, export["predictions_sha256"], 32 * 1024**2).splitlines()]
    gt = json.loads(pinned(args.index, INDEX_SHA, 32 * 1024**2))
    coco = COCO()
    coco.dataset = gt
    with redirect_stdout(io.StringIO()):
        coco.createIndex()
    need(len(coco.imgs) == 2937 and len(coco.anns) == 9260 and cohort["fit_only"] is True,
         "Frozen qualified FIT raster index")
    cohorts = {"core": [], "stress": []}
    native_counts = {k: dict.fromkeys(("images", "targets", "at50", "at75", "at90"), 0) for k in cohorts}
    for item, block in zip(cohort["images"], blocks, strict=True):
        image = item["image_id"]
        need(block["image_id"] == image and coco.imgs[image]["file_name"] == item["native_basename"],
             "Exact original candidate order and FIT image identity")
        query = block["native_ids"]
        all_rles = block["all_rles"]
        need(len(all_rles) == WIDTH and len(query) == len(set(query))
             and all(type(k) is int and 0 <= k < WIDTH for k in query), "Exact closed native query IDs")
        predictions = [all_rles[i] for i in query]
        targets, small = [], []
        for ann in coco.imgToAnns[image]:
            need(ann["category_id"] == 1, "FIT ship GT class")
            rle = coco.annToRLE(ann)
            if mask.area(rle) and not ann.get("iscrowd", 0) and not ann.get("ignore", 0):
                targets.append(rle)
                small.append(ann["area"] < 1024)
        matrix = (mask.iou(predictions, targets, [0] * len(targets)) if predictions and targets
                  else np.zeros((len(predictions), len(targets))))
        best = matrix.max(axis=0, initial=0)
        p50, p75, p90 = (rank_pairs(matrix, t) for t in (.50, .75, .90))
        strict_queries = {q for q, _ in p90}
        chosen = [(q, g) for q, g in p75 if best[g] < .90]
        need(not any(q in strict_queries for q, _ in chosen), "All fixed original strict queries remain protected")
        pair_rows = []
        for q, g in chosen:
            pred = mask.decode(predictions[q]).astype(bool)
            truth = mask.decode(targets[g]).astype(bool)
            need(pred.shape == truth.shape == (800, 800), "Exact original raster dimensions")
            stats = analyze_pair(pred, truth)
            need(abs(iou(stats["methods"]["baseline"]) - matrix[q, g]) < 1e-12,
                 "Fixed paired mask equals the original COCO IoU")
            stats["small"] = bool(small[g])
            pair_rows.append(stats)
        for name, region in cohorts.items():
            if item[name]:
                region.append(pair_rows)
                for key, value in (("images", 1), ("targets", len(targets)), ("at50", len(p50)),
                                   ("at75", len(p75)), ("at90", len(p90))):
                    native_counts[name][key] += value
    expected = dict(core=dict(images=109, targets=496, at50=453, at75=271, at90=37),
                    stress=dict(images=5, targets=613, at50=374, at75=59, at90=0))
    need(native_counts == expected and len(blocks) == 114, "Exact reproduction of original three-threshold native coverage")
    core, stress = summarize(cohorts["core"], main=True), summarize(cohorts["stress"], main=False)
    need(core["pairs"] == 234 and stress["pairs"] == 59, "Exact prelocked near-strict pair counts")
    result = dict(status="ACTUAL_HRSID_FIT_FIXED_CONTOUR_GEOMETRY", protocol=args.protocol,
                  previous_public_result_sha256=args.previous_sha256, predictions_sha256=export["predictions_sha256"],
                  cohort_sha256=export["cohort_sha256"], fit_index_sha256=INDEX_SHA, native=native_counts,
                  core=core, stress=stress, elapsed_seconds=perf_counter()-started,
                  FIT=True, DEV=False, TEST=False, AP=False, forward_calls=0, optimizer_updates=0)
    write(args.output, result)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("previous", "cohort", "predictions", "index", "output"):
        parser.add_argument("--" + field, required=True, type=Path)
    parser.add_argument("--previous-sha256", required=True)
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    need(re.fullmatch(r"[0-9a-f]{40}", args.protocol) and args.output.is_absolute()
         and args.output.parent.resolve(strict=True) == args.output.parent and not args.output.exists(),
         "Prelocked protocol and new isolated CPU output")
    def offline(event, _):
        if event in ("socket.connect", "socket.getaddrinfo"):
            raise RuntimeError("FIT geometric diagnostic is offline")
    sys.addaudithook(offline)
    run(args)


if __name__ == "__main__":
    main()
