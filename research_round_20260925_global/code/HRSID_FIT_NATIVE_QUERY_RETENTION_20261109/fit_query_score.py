"""Compare fixed FIT retention policies from the independently closed export."""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO

from fit_query_export import INDEX_SHA, WIDTH, need, pinned, write
from query_retention import analyze_image

POLICIES = ("native", "ship_same_count", "nonempty_same_count", "all_queries")
THRESHOLDS = ("0.50", "0.75", "0.90")
METRICS = ("independent", "maximum_one_to_one", "native_greedy")


def summarize(rows, *, main):
    valid = np.array([row["eligible_gt"] for row in rows])
    need(bool(len(valid)) and valid.sum() > 0, "Nonempty fixed diagnostic denominator")
    eligible = int(valid.sum())
    baseline = np.array([row["policies"]["native"]["thresholds"]["0.90"]["maximum_one_to_one"] for row in rows])
    summary = dict(images=len(rows), eligible_gt=eligible,
                   native_near_strict=sum(sum(row["native_near_strict"]) for row in rows), policies={})
    if main:
        rng = np.random.default_rng(5209)
        draws = rng.integers(0, len(rows), size=(4096, len(rows)))
        denominators = valid[draws].sum(axis=1)
        need(bool((denominators > 0).all()), "Valid fixed paired bootstrap denominators")
    for name in POLICIES:
        selected = [row["policies"][name] for row in rows]
        values = dict(candidates=sum(row["candidates"] for row in selected),
                      empty_candidates=sum(row["empty_candidates"] for row in selected))
        values["thresholds"] = {
            t: {metric: sum(r["thresholds"][t][metric] for r in selected) for metric in METRICS}
            for t in THRESHOLDS
        }
        difference = np.array([row["thresholds"]["0.90"]["maximum_one_to_one"] for row in selected]) - baseline
        values["strict_max_delta"] = int(difference.sum())
        values["strict_gain_groups"] = int((difference > 0).sum()) if main else None
        values["strict_max_delta_per_gt"] = float(difference.sum() / eligible)
        gain = loss = near_gain = 0
        for row, sel in zip(rows, selected, strict=True):
            native = np.asarray(row["policies"]["native"]["strict_reach"], dtype=bool)
            reach = np.asarray(sel["strict_reach"], dtype=bool)
            gained = reach & ~native
            gain += int(gained.sum())
            loss += int((native & ~reach).sum())
            near_gain += int((gained & np.asarray(row["native_near_strict"], dtype=bool)).sum())
        values.update(strict_independent_gains=gain, strict_independent_losses=loss, native_near_strict_gains=near_gain)
        if main:
            fractions = difference[draws].sum(axis=1) / denominators
            values["strict_paired_interval"] = np.quantile(fractions, [.025, .975], method="linear").tolist()
        summary["policies"][name] = values
    p = summary["policies"]
    need(p["native"] == p["ship_same_count"], "The stable slot-0 prefix is exactly the same-count ship-score control")
    need(p["all_queries"]["strict_independent_losses"] == 0, "The full query pool cannot remove native coverage")
    if main:
        upper = p["all_queries"]
        replacement = p["nonempty_same_count"]
        missing = eligible - int(baseline.sum())
        upper_pass = (missing > 0 and upper["strict_max_delta"] / missing >= .05 and upper["strict_gain_groups"] >= 10)
        pass_filter = (replacement["strict_max_delta_per_gt"] >= .02 and replacement["strict_paired_interval"][0] > 0
                       and replacement["thresholds"]["0.75"]["maximum_one_to_one"] >=
                       p["native"]["thresholds"]["0.75"]["maximum_one_to_one"])
        if not upper_pass:
            decision = "STOP_SELECTION_SHORTFALL_HYPOTHESIS_FOR_THIS_COHORT"
        elif pass_filter:
            decision = "ADVANCE_EQUAL_COUNT_FILTER_TO_PROSPECTIVE_CONTROL"
        else:
            decision = "RETAIN_SELECTION_HYPOTHESIS_STOP_FIXED_FILTER"
        summary.update(native_strict_missing=missing, full_pool_missing_fraction_recovered=upper["strict_max_delta"]/missing if missing else 0,
                       upper_bound_gate=upper_pass, equal_count_filter_gate=pass_filter, decision=decision)
    return summary


def run(args):
    started = perf_counter()
    report = json.loads(pinned(args.report, args.report_sha256, 65536))
    need(report["status"] == "HRSID_FIT_Q164_EXPORT_COMPLETE" and report["native_rows_bitwise_verified"]
         and report["images"] == report["forward_calls"] == 114
         and report["all_ship_queries"] == 114 * WIDTH
         and report["GT_opened"] is report["DEV"] is report["TEST"] is report["AP"] is False,
         "Original FIT-only and GT-free export")
    cohort = json.loads(pinned(args.cohort, report["cohort_sha256"], 1024**2))
    need(cohort["protocol"] == report["protocol"] and cohort["fit_only"] is True, "Exact locked FIT cohort")
    blocks = [json.loads(line) for line in pinned(args.predictions, report["predictions_sha256"], 32 * 1024**2).splitlines()]
    gt = json.loads(pinned(args.index, INDEX_SHA, 32 * 1024**2))
    coco = COCO()
    coco.dataset = gt
    with redirect_stdout(io.StringIO()):
        coco.createIndex()
    need(len(coco.imgs) == 2937 and len(coco.anns) == 9260, "Exact FIT index population")
    core, stress = [], []
    exclusions = dict(empty=0, crowd=0, ignore=0)
    mapped = all_empty = 0
    for item, block in zip(cohort["images"], blocks, strict=True):
        image_id = item["image_id"]
        need(block["image_id"] == image_id and coco.imgs[image_id]["file_name"] == item["native_basename"],
             "Exact frozen image and query order")
        query, ranks = block["native_ids"], block["native_ranks"]
        rles, score = block["all_rles"], np.array(block["ship_scores"])
        need(len(rles) == WIDTH and score.shape == (WIDTH,) and all(rle["size"] == [800, 800] for rle in rles)
             and len(query) == len(ranks) and ranks == sorted(set(ranks))
             and all(type(k) is int and 1 <= k <= 100 for k in ranks)
             and query == np.argsort(-score, kind="stable")[:len(query)].tolist(),
             "All raw query masks and exact native slot-0 prefix")
        mapped += len(query)
        empty = mask.area(rles) == 0
        all_empty += int(empty.sum())
        targets = []
        for ann in coco.imgToAnns[image_id]:
            need(ann["category_id"] == 1, "Original FIT ship category")
            target = coco.annToRLE(ann)
            if mask.area(target) == 0:
                exclusions["empty"] += 1
            elif ann.get("iscrowd", 0):
                exclusions["crowd"] += 1
            elif ann.get("ignore", 0):
                exclusions["ignore"] += 1
            else:
                targets.append(target)
        overlap = mask.iou(rles, targets, [0] * len(targets)) if targets else np.empty((WIDTH, 0))
        result = analyze_image(overlap, score, query, empty)
        if item["core"]:
            core.append(result)
        if item["stress"]:
            stress.append(result)
    need(len(blocks) == len(cohort["images"]) == 114 and len(core) == 109 and len(stress) == 5
         and len({r["group"] for r in cohort["images"] if r["core"]}) == 109
         and mapped == report["native_mapped_rows"] and all_empty == report["all_empty"],
         "Closed export image, candidate and group reconciliation")
    result = dict(status="ACTUAL_HRSID_FIT_Q164_RETENTION", protocol=report["protocol"],
                  cohort_sha256=report["cohort_sha256"], predictions_sha256=report["predictions_sha256"],
                  export_report_sha256=args.report_sha256, fit_index_sha256=INDEX_SHA,
                  core=summarize(core, main=True), stress=summarize(stress, main=False), exclusions=exclusions,
                  elapsed_seconds=perf_counter()-started, FIT=True, DEV=False, TEST=False, AP=False, forward_calls=0)
    write(args.output, result)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("report", "cohort", "predictions", "index", "output"):
        parser.add_argument("--" + field, required=True, type=Path)
    parser.add_argument("--report-sha256", required=True)
    args = parser.parse_args()
    need(args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent
         and not args.output.exists(), "Fresh CPU scoring result path")
    def offline(event, _):
        if event in ("socket.connect", "socket.getaddrinfo"):
            raise RuntimeError("FIT query scoring is offline")
    sys.addaudithook(offline)
    run(args)


if __name__ == "__main__":
    main()
