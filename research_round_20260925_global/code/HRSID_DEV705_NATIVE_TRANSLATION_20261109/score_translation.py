"""Summarize frozen historical HRSID DEV705 mask translation bounds on CPU."""
import argparse
from collections import defaultdict
from contextlib import redirect_stdout
from hashlib import sha256
from importlib.metadata import version
import io
import json
from pathlib import Path
import re
import sys
from time import perf_counter

import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO

PROTOCOL = "7aa4c821f3d6ff71dd3f57c4be9093d078bede6a"
GT_SHA = "393b3da233991dc5fafc98b4ddeaa341b042d9571434ef8ce53c3f2fdada5628"
PRED_SHA = "0f2221de4883c71e0349eb4dc4d8f4001d123fc339630e0664ccb9872e7b0aa7"
COVERAGE_SHA = "de18b2d993af7549e437add80c72b20633a3e596170f6b4789188539c7cc25fa"
CORE_SHA = "c713975658bb3d96261bc6f3d4eb84ee723e375cfde4cbc2f429b66bd14fbbec"
ROOT = Path(__file__).resolve().parent
IMAGES, GT, VALID, NEAR, GROUPS = 705, 1782, 1778, 1049, 28
MIB = 1024**2


def need(condition, message):
    if not condition:
        raise ValueError(message)


def load_code(path, digest):
    raw = path.read_bytes()
    need(0 < len(raw) <= 64 * 1024 and sha256(raw).hexdigest() == digest, "Exact diagnostic dependency")
    namespace = dict(__file__=str(path), __name__="_historical_translation_dependency")
    exec(compile(raw, str(path), "exec"), namespace)
    return namespace


def original_pairs(overlap):
    taken = np.zeros(overlap.shape[1], dtype=bool)
    for pred, overlaps in enumerate(overlap):
        possible = np.flatnonzero((overlaps >= .75) & ~taken)[::-1]
        if possible.size:
            gt = int(possible[np.argmax(overlaps[possible])])
            taken[gt] = True
            yield pred, gt


def strict(counts):
    return counts["tp"] >= 9 * (counts["fp"] + counts["fn"])


def fixed_summary(rows, direction):
    gain_groups, loss_groups = set(), set()
    initial = final = gains = losses = infeasible = 0
    for row in rows:
        before = strict(row["base"])
        moved = row[direction]
        after = strict(moved) if moved is not None else before
        initial += before
        final += after
        infeasible += moved is None
        if after and not before:
            gains += 1
            gain_groups.add(row["group"])
        elif before and not after:
            losses += 1
            loss_groups.add(row["group"])
    need(final - initial == gains - losses, "Fixed shift accounting")
    return dict(pairs=len(rows), initial_strict=initial, final_strict=final, gains=gains, losses=losses,
                net=gains - losses, infeasible=infeasible, gain_groups=len(gain_groups),
                loss_groups=len(loss_groups))


def shape_and_pixel_summary(rows):
    total = len(rows)
    def mean(values):
        return sum(values) / total if total else None
    area_impossible = sum(row["area_impossible"] for row in rows)
    one = sum(strict(row["oracle1"]) for row in rows)
    two = sum(strict(row["oracle2"]) for row in rows)
    need(0 <= one <= two <= total - area_impossible, "Nested translation bounds and area limits")
    result = dict(
        pairs=total, groups=len({row["group"] for row in rows}),
        radius1_recovered=one, radius2_recovered=two, area_impossible=area_impossible,
        area_feasible_not_recovered=total - two - area_impossible,
        fixed_up_recovered=sum(row["fixed_up"] is not None and strict(row["fixed_up"]) for row in rows),
        fixed_down_recovered=sum(row["fixed_down"] is not None and strict(row["fixed_down"]) for row in rows),
        mean_dx_pixels=mean(row["dx_pixels"] for row in rows),
        mean_dy_pixels=mean(row["dy_pixels"] for row in rows),
        mean_abs_dx_pixels=mean(abs(row["dx_pixels"]) for row in rows),
        mean_abs_dy_pixels=mean(abs(row["dy_pixels"]) for row in rows),
    )
    for method in ("base", "oracle1", "oracle2"):
        result[method] = {key: sum(row[method][key] for row in rows) for key in ("tp", "fp", "fn", "edits")}
    return result


def summary(coco_gt, blocks, measure, coverage):
    coco = COCO()
    coco.dataset = coco_gt
    with redirect_stdout(io.StringIO()):
        coco.createIndex()
    seen_groups = set()
    fixed, near = [], []
    for image, block in zip(coco_gt["images"], blocks, strict=True):
        match = re.match(r"^P[0-9]+(?=[_.-])", Path(image["file_name"]).name)
        need(match is not None, "Expected canonical DEV scene prefix")
        group = match.group()
        seen_groups.add(group)
        targets, areas = [], []
        for row in coco.imgToAnns[image["id"]]:
            rle = coco.annToRLE(row)
            if mask.area(rle) and not row.get("iscrowd", 0) and not row.get("ignore", 0):
                targets.append(rle)
                areas.append(row["area"])
        preds = [row["segmentation"] for row in block["coco"]]
        if not targets or not preds:
            continue
        overlaps = mask.iou(preds, targets, [0] * len(targets))
        best = overlaps.max(axis=0)
        for pred_index, target_index in original_pairs(overlaps):
            values = measure(mask.decode(preds[pred_index]).astype(bool, copy=False),
                             mask.decode(targets[target_index]).astype(bool, copy=False))
            base = values["base"]
            union = base["tp"] + base["fp"] + base["fn"]
            need(abs(base["tp"] / union - overlaps[pred_index, target_index]) <= 1e-15,
                 "Exact fixed original raster intersection and union")
            row = dict(values, group=group, small=areas[target_index] < 1024,
                       elongated=values["elongation"] >= 3)
            fixed.append(row)
            if best[target_index] < .90:
                need(not strict(base), "A near-strict fixed pair has no strict exported candidate")
                near.append(row)
    need(len(seen_groups) == GROUPS and len(fixed) == coverage["thresholds"]["0.75"]["native_greedy"] == 1153
         and len(near) == coverage["best_iou_bins"][3] == NEAR, "Fixed frozen DEV cohort")
    all_near = shape_and_pixel_summary(near)
    by_area = {label: shape_and_pixel_summary([row for row in near if row["small"] is value])
               for label, value in (("small", True), ("non_small", False))}
    by_shape = {label: shape_and_pixel_summary([row for row in near if row["elongated"] is value])
                for label, value in (("elongated", True), ("non_elongated", False))}
    by_iou_rows = {"0.75-0.80": [], "0.80-0.85": [], "0.85-0.90": []}
    for row in near:
        base = row["base"]
        union = base["tp"] + base["fp"] + base["fn"]
        iou_bin = ("0.75-0.80" if 100 * base["tp"] < 80 * union else
                   "0.80-0.85" if 100 * base["tp"] < 85 * union else "0.85-0.90")
        by_iou_rows[iou_bin].append(row)
    by_iou = {name: shape_and_pixel_summary(rows) for name, rows in by_iou_rows.items()}
    need(by_area["small"]["pairs"] == 903 and by_area["non_small"]["pairs"] == 146
         and sum(value["pairs"] for value in by_shape.values()) == NEAR
         and sum(value["pairs"] for value in by_iou.values()) == NEAR,
         "All fixed area, shape and headroom strata reconcile")

    grouped = defaultdict(list)
    for row in near:
        grouped[row["group"]].append(row)
    groups = sorted(seen_groups)
    samples = np.random.default_rng(5213).integers(0, GROUPS, size=(4096, GROUPS))
    den = np.array([len(grouped[key]) for key in groups])[samples].sum(axis=1)
    num = np.array([sum(strict(row["oracle2"]) for row in grouped[key]) for key in groups])[samples].sum(axis=1)
    need((den > 0).all(), "Positive frozen group bootstrap denominators")
    interval = np.quantile(num / den, [.025, .975], method="linear").tolist()
    fixed_shifts = {name: fixed_summary(fixed, name) for name in ("fixed_up", "fixed_down")}
    initial = fixed_shifts["fixed_up"]["initial_strict"]
    need(initial == fixed_shifts["fixed_down"]["initial_strict"] <= 104,
         "Fixed .75 matching strict subset scope")
    return dict(fixed_pairs=len(fixed), strict_on_fixed_pairs=initial, scene_prefix_groups=GROUPS,
                near=all_near, strata_by_area=by_area, strata_by_shape=by_shape, strata_by_initial_iou=by_iou,
                radius2_recovery_group_interval=interval, bootstrap_replicates=4096, bootstrap_seed=5213,
                fixed_one_pixel=fixed_shifts)


def no_network(event, _):
    if event.startswith("socket."):
        raise PermissionError("Offline historical diagnostic")


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    started = perf_counter()
    need(args.protocol == PROTOCOL and version("numpy") == "1.26.4" and version("pycocotools") == "2.0.11",
         "Exact locked protocol and CPU mask runtime")
    need(all(path.is_absolute() and path.resolve(strict=True) == path for path in (args.gt, args.predictions))
         and args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent
         and not args.output.exists() and not args.output.is_symlink(), "Canonical inputs and fresh output")
    sys.addaudithook(no_network)
    prior = load_code(ROOT.parent / "HRSID_DEV705_NATIVE_CANDIDATE_COVERAGE_20261109/candidate_coverage.py",
                      COVERAGE_SHA)
    probe = load_code(ROOT / "translation_probe.py", CORE_SHA)
    read = prior["read_pinned"]
    gt = json.loads(read(args.gt, GT_SHA, 4 * MIB))
    blocks = [json.loads(line) for line in read(args.predictions, PRED_SHA, 32 * MIB).splitlines()]
    need(len(gt["images"]) == len(blocks) == IMAGES and len(gt["annotations"]) == GT
         and all(image["height"] == image["width"] == 800 for image in gt["images"])
         and all(row["raw_query_count"] == 164 and row["native_count"] == 100
                 and row["unmapped_slot1_count"] + len(row["coco"]) == 100 for row in blocks),
         "Exact fixed historical native population")
    coverage = prior["census"](gt, blocks)
    need(coverage["eligible_gt"] == VALID and coverage["exclusions"] == dict(empty=4, crowd=0, ignore=0)
         and coverage["predictions"] == 62536 and coverage["empty_predictions"] == 23662
         and all(coverage["thresholds"][t][kind] == n
                 for t, n in (("0.50", 1634), ("0.75", 1153), ("0.90", 104))
                 for kind in ("independent", "maximum_one_to_one", "native_greedy")),
         "Exact reproduced historical candidate coverage")
    result = summary(gt, blocks, probe["measure"], coverage)
    result.update(status="ACTUAL_HISTORICAL_HRSID_DEV705_NATIVE_TRANSLATION", protocol=PROTOCOL,
                  gt_sha256=GT_SHA, predictions_sha256=PRED_SHA, coverage_source_sha256=COVERAGE_SHA,
                  probe_source_sha256=CORE_SHA, images=IMAGES, original_annotations=GT, eligible_gt=VALID,
                  empty_gt=4, elapsed_seconds=perf_counter()-started, forward_calls=0, AP=False,
                  official_TEST_opened=False)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
    print(json.dumps(dict(status=result["status"], pairs=result["near"]["pairs"],
                          radius2_recovered=result["near"]["radius2_recovered"],
                          elapsed_seconds=result["elapsed_seconds"])), flush=True)


if __name__ == "__main__":
    main()
