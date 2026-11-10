"""Score frozen paired center-ship masks against the author's native 101-pixel references."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from math import fsum, isfinite
import os
from pathlib import Path
import re
import stat

import numpy as np


SEEDS = (5201, 5202, 5203)
MODES = ("native", "center")
ROSTER = tuple((seed, mode) for seed in SEEDS for mode in MODES)
SIZE = 101
ROI = [40, 40, 61, 61]
RECOVERY = {"0.50": (1, 2), "0.75": (3, 4), "0.90": (9, 10)}
MIN_ACQUISITIONS = 20
BOOTSTRAP_DRAWS = 4096
BOOTSTRAP_SEED = 254101
MATERIAL_IOU = .01
MIB = 1024**2
PLAN_STATUS = "P254_SCHWEGMANN_CLOSED_SCORE_PLAN_V1"
EXPORT_STATUS = "P254_SCHWEGMANN_FROZEN_EXPORT_COMPLETE_PENDING_INDEPENDENT_ACCEPTANCE_V1"
SCORE_STATUS = "P254_SCHWEGMANN_SIX_ARM_CENTER_SCORE_PENDING_INDEPENDENT_ACCEPTANCE_V1"
SELECTION_RULE = "native_top100_then_slot0_highest_rank_box_center_in_fixed_roi"
AUTHOR_JSON_SHA256 = "78ff85f49eb5102716ae6d39bdbafaf27fc9c63a630be0d86a2c434ab09f053b"
EXPORT_WORKER_SHA256 = "79fe4da8fca69589421bd726b0ad0417bdfb52aef3957747c59c51548ab1ecef"
HRSID_SOURCE = {
    5201: ("d66baf3c57fafdba4242ad1b55e08cf76bfd77f7", "3e8720e8fad83d6a4b313ee8b38e261cabb4f1a2a3eefe3a5e33e83d574ac1a1",
           {"native": "198bc09cb93019e719342914965dffb31160bb284003a17b334f998c1f540c56",
            "center": "b95f1a41ab599b7eb27ed3f25029300838cf1b9da74d7bd0c3ccaaef449485db"}),
    5202: ("a49361b92977a605267025b1e7798420936cbe53", "8c689acf918ba5b14fcda730f459292bdc722b02f24a62efb4b791b54e4feda5",
           {"native": "1ef686c8449621ff3179150f12ba21c81340a8da4e553b23105746c403709909",
            "center": "84402a6cf29a6ae428853d3049a7759f97e342defa31ab719b74667ce932ef3a"}),
    5203: ("a49361b92977a605267025b1e7798420936cbe53", "4c5324ca061f310ef4dcaeff1a54974c972337201b7efcea57048d6682f46076",
           {"native": "8f47c6a8fa8f885ef552210dedea803f1372272ce0e85968a585412686f5cd0c",
            "center": "e6bdece9f2b8b613eb0963d923aa6bd27d119fe739dc4d44ed85c1934652bb5a"}),
}
FILES = {"terminal": "TERMINAL_PRIVATE.json", "report": "EXPORT_PRIVATE.json", "predictions": "PREDICTIONS_PRIVATE.jsonl",
         "stdout": "WORKER.stdout", "stderr": "WORKER.stderr"}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def exact(value, fixed, extra=()):
    return (type(value) is dict and set(value) == set(fixed).union(extra)
            and all(type(value[k]) is type(v) and value[k] == v for k, v in fixed.items()))


def finite(value):
    return type(value) is float and isfinite(value) or type(value) is int and abs(value) < 2**63


def bounded(value, lo, hi):
    return type(value) is int and lo <= value <= hi


def decode_rle(value):
    """Decode bounded native COCO RLE without passing malformed runs to C."""
    area = SIZE * SIZE
    require(exact(value, {"size": [SIZE, SIZE]}, ("counts",))
            and all(type(n) is int for n in value["size"])
            and type(value["counts"]) is str and 0 < len(value["counts"]) <= area * 4,
            "Expected original-grid 101×101 compressed COCO RLE")
    payload = value["counts"]
    counts, i, total = [], 0, 0
    while i < len(payload):
        number = shift = 0
        while True:
            require(i < len(payload) and shift <= 20, "Truncated or oversized COCO RLE run")
            code = ord(payload[i]) - 48
            i += 1
            require(0 <= code <= 63, "Invalid character in COCO RLE")
            number |= (code & 31) << shift
            shift += 5
            if not code & 32:
                if code & 16:
                    number -= 1 << shift
                break
        if len(counts) > 2:
            number += counts[-2]
        require(0 <= number <= area - total and len(counts) <= area, "COCO RLE run escapes native canvas")
        counts.append(number)
        total += number
    require(total == area, "Incomplete native COCO RLE coverage")
    flat = np.repeat(np.arange(len(counts), dtype=np.int64) % 2, counts)
    return flat.reshape(SIZE, SIZE, order="F").astype(np.bool_)


def _truth(image_ids, ground_truth):
    require(type(image_ids) in (tuple, list) and len(image_ids) > 0
            and all(type(i) is int and i > 0 for i in image_ids) and len(set(image_ids)) == len(image_ids),
            "Exact unique ordered image IDs required")
    require(exact(ground_truth, {"status": "P254_SCHWEGMANN_SCORER_ONLY_GT_V1"}, ("images",))
            and type(ground_truth["images"]) is list and len(ground_truth["images"]) == len(image_ids),
            "Exact independent positive center-ship GT population required")
    masks, groups = [], defaultdict(list)
    for index, (i, row) in enumerate(zip(image_ids, ground_truth["images"], strict=True)):
        require(exact(row, {"image_id": i}, ("acquisition", "segmentation"))
                and type(row["acquisition"]) is str and re.fullmatch(r"acq_[0-9]{3}", row["acquisition"]),
                "Ground truth must preserve image order and acquisition pseudonym")
        mask = decode_rle(row["segmentation"])
        require(mask.any(), "Positive center-ship ground truth must be nonempty")
        masks.append(mask)
        groups[row["acquisition"]].append(index)
    require(all(1 <= len(indices) <= 2 for indices in groups.values()), "At most two chips per physical acquisition required")
    return masks, tuple(groups[name] for name in sorted(groups))


def _candidate(value):
    require(exact(value, {}, ("native_rank", "score", "bbox_xyxy", "segmentation"))
            and bounded(value["native_rank"], 1, 100) and finite(value["score"]) and 0 <= value["score"] <= 1
            and type(value["bbox_xyxy"]) is list and len(value["bbox_xyxy"]) == 4
            and all(finite(x) and 0 <= x <= SIZE for x in value["bbox_xyxy"]), "Exact originally selected native candidate required")
    x0, y0, x1, y1 = value["bbox_xyxy"]
    require(x0 <= x1 and y0 <= y1 and ROI[0] <= (x0+x1)/2 < ROI[2] and ROI[1] <= (y0+y1)/2 < ROI[3],
            "Original predicted box center must lie in the fixed half-open ROI")
    return decode_rle(value["segmentation"])


def score_arm(image_ids, gt_masks, rows):
    require(type(rows) is list and len(rows) == len(image_ids), "Exact complete ordered prediction population required")
    result = []
    for i, target, row in zip(image_ids, gt_masks, rows, strict=True):
        fixed = dict(image_id=i, raw_query_count=164, native_count=100)
        require(exact(row, fixed, ("mapped_rows", "eligible_count", "candidate")) and bounded(row["mapped_rows"], 0, 100)
                and bounded(row["eligible_count"], 0, row["mapped_rows"])
                and (row["candidate"] is None) == (row["eligible_count"] == 0),
                "Prediction row lost original top100 census or candidate identity")
        selected = row["candidate"] is not None
        prediction = _candidate(row["candidate"]) if selected else None
        prediction_area = int(np.count_nonzero(prediction)) if selected else 0
        gt_area = int(np.count_nonzero(target))
        intersection = int(np.count_nonzero(target & prediction)) if prediction_area else 0
        union = gt_area + prediction_area - intersection
        result.append(dict(iou=intersection / union, dice=2 * intersection / (gt_area + prediction_area), selected=selected,
                           nonempty=prediction_area > 0, mapped_rows=row["mapped_rows"], eligible_rows=row["eligible_count"],
                           recovery={key: intersection * denominator >= union * numerator
                                     for key, (numerator, denominator) in RECOVERY.items()}))
    return result


def _group_means(values, groups):
    return [fsum(values[i] for i in group) / len(group) for group in groups]


def _average(values):
    return fsum(values) / len(values) if len(values) else None


def _arm_summary(rows, groups):
    totals = Counter()
    for row in rows:
        totals.update({key: row[key] for key in ("selected", "nonempty", "mapped_rows", "eligible_rows")})
    return dict(acquisition_equal_iou=_average(_group_means([x["iou"] for x in rows], groups)),
                acquisition_equal_dice=_average(_group_means([x["dice"] for x in rows], groups)),
                chip_mean_iou=_average([x["iou"] for x in rows]), chip_mean_dice=_average([x["dice"] for x in rows]),
                mapped_rows=totals["mapped_rows"], eligible_rows=totals["eligible_rows"],
                candidate_images=totals["selected"], missing_candidates=len(rows)-totals["selected"],
                nonempty_candidates=totals["nonempty"], selected_empty_masks=totals["selected"]-totals["nonempty"],
                recovery={key: dict(count=sum(x["recovery"][key] for x in rows),
                                    acquisition_equal_rate=_average(_group_means([x["recovery"][key] for x in rows], groups)))
                          for key in RECOVERY})


def _interval(values, draws):
    if len(values) < 2:
        return None
    sampled = np.asarray(values, dtype=np.float64)[draws].mean(axis=1)
    return np.quantile(sampled, [.025, .975]).tolist()


def _paired_boolean(a, b):
    counts = Counter((x, y) for x, y in zip(a, b, strict=True))
    return dict(native_only=counts[True, False], center_only=counts[False, True], both=counts[True, True],
                neither=counts[False, False])


def _paired(native, center, groups, draws):
    native_selected = [x["selected"] for x in native]
    center_selected = [x["selected"] for x in center]
    native_live = [x["nonempty"] for x in native]
    center_live = [x["nonempty"] for x in center]
    support = {i for i, (a, b) in enumerate(zip(native_live, center_live, strict=True)) if a and b}
    common_groups = [chosen for group in groups if (chosen := [i for i in group if i in support])]
    common = dict(images=len(support), acquisitions=len(common_groups))
    effects, group_effects = {}, {}
    for metric in ("iou", "dice"):
        a, b = ([x[metric] for x in rows] for rows in (native, center))
        values = [y - x for x, y in zip(a, b, strict=True)]
        group_effect = _group_means(values, groups)
        group_effects[metric] = group_effect
        effects[metric] = dict(mean_acquisition_equal_delta=_average(group_effect),
                               acquisition_cluster_percentile95=_interval(group_effect, draws),
                               positive_acquisitions=sum(x > 0 for x in group_effect),
                               negative_acquisitions=sum(x < 0 for x in group_effect))
        common[metric] = dict(native=_average(_group_means(a, common_groups)),
                              center=_average(_group_means(b, common_groups)),
                              center_minus_native=_average(_group_means(values, common_groups)))
    transitions = {key: _paired_boolean([row["recovery"][key] for row in native],
                                       [row["recovery"][key] for row in center]) for key in RECOVERY}
    report = dict(native=_arm_summary(native, groups), center=_arm_summary(center, groups), effects=effects,
                  candidate_presence_transitions=_paired_boolean(native_selected, center_selected),
                  nonempty_candidate_transitions=_paired_boolean(native_live, center_live),
                  recovery_transitions=transitions, common_nonempty=common)
    return report, group_effects


def _decision(seed_iou, pooled_iou, interval, groups):
    if groups < MIN_ACQUISITIONS:
        return "INSUFFICIENT_PHYSICAL_ACQUISITIONS"
    if all(x > 0 for x in seed_iou) and pooled_iou >= MATERIAL_IOU and interval[0] > 0:
        return "MATERIAL_POSITIVE_DIRECTION_SUPPORTED"
    if all(x <= 0 for x in seed_iou) and interval[1] <= 0:
        return "NONPOSITIVE_DIRECTION_SUPPORTED"
    if interval[1] < MATERIAL_IOU:
        return "PRACTICAL_ONE_POINT_BENEFIT_EXCLUDED"
    return "DIRECTION_NOT_ESTABLISHED"


def score_population(image_ids, ground_truth, arms):
    """Independently score all six closed arms on the same positive chip roster."""
    require(type(arms) is dict and set(arms) == set(ROSTER), "Exactly six HRSID seed/mode arms required")
    gt, groups = _truth(image_ids, ground_truth)
    count = len(groups)
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    draws = rng.integers(0, count, size=(BOOTSTRAP_DRAWS, count))
    reports, group_effects = {}, []
    for seed in SEEDS:
        a, b = (score_arm(image_ids, gt, arms[seed, mode]) for mode in MODES)
        report, effects = _paired(a, b, groups, draws)
        reports[str(seed)] = report
        group_effects.append(effects)
    pooled = {}
    for metric in ("iou", "dice"):
        values = [_average(xs) for xs in zip(*(effect[metric] for effect in group_effects), strict=True)]
        pooled["mean_acquisition_equal_" + metric + "_delta"] = _average(values)
        pooled[metric + "_acquisition_cluster_percentile95"] = _interval(values, draws)
    per_seed_iou = [reports[str(seed)]["effects"]["iou"]["mean_acquisition_equal_delta"] for seed in SEEDS]
    status = _decision(per_seed_iou, pooled["mean_acquisition_equal_iou_delta"],
                       pooled["iou_acquisition_cluster_percentile95"], count)
    return dict(images=len(image_ids), acquisitions=count, seeds=reports, pooled=pooled, decision=status,
                primary="acquisition_equal_center_ship_iou_with_misses_zero", material_margin_iou=MATERIAL_IOU,
                minimum_interpretable_acquisitions=MIN_ACQUISITIONS,
                bootstrap=dict(cluster="physical_acquisition", seed=BOOTSTRAP_SEED, draws=BOOTSTRAP_DRAWS,
                               shared_draws_across_arms_and_seeds=True), AP=False, official_TEST_opened=False)


def _hex(value, count=64):
    return type(value) is str and re.fullmatch(f"[0-9a-f]{{{count}}}", value) is not None


def _unique(pairs):
    result = dict(pairs)
    require(len(result) == len(pairs), "Duplicate scientific admission key")
    return result


def _nonfinite(_):
    raise ValueError("Nonfinite scientific JSON constant")


def _json(raw):
    return json.loads(raw, object_pairs_hook=_unique, parse_constant=_nonfinite)


def _file_token(meta):
    return meta.st_dev, meta.st_ino, meta.st_size, meta.st_mtime_ns, meta.st_ctime_ns


def _stable(path, sha, limit, length=None, *, empty=False):
    require(path.is_absolute() and path.resolve(strict=True) == path and _hex(sha), "Exact pinned canonical closed file required")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and int(not empty) <= before.st_size <= limit
                and (length is None or before.st_size == length), "Closed file length differs")
        raw = stream.read(before.st_size+1)
        require(len(raw) == before.st_size and sha256(raw).hexdigest() == sha
                and _file_token(before) == _file_token(os.fstat(stream.fileno())) == _file_token(path.stat()),
                "Exact stable closed file digest required")
    return raw


def _path(ref, limit, *, basename=None, empty=False):
    require(exact(ref, {}, ("path", "bytes", "sha256")) and type(ref["path"]) is str
            and bounded(ref["bytes"], int(not empty), limit) and _hex(ref["sha256"]), "Exact bounded file reference required")
    path = Path(ref["path"])
    require(path.is_absolute() and ref["path"] in (str(path), path.as_posix()) and ".." not in path.parts
            and (basename is None or path.name == basename), "Unexpected canonical closed artifact path")
    return path


def _read(ref, limit, *, empty=False):
    return _stable(_path(ref, limit, empty=empty), ref["sha256"], limit, ref["bytes"], empty=empty)


def _image_ids(ref):
    value = _json(_read(ref, MIB))
    require(exact(value, {"status": "P254_SCHWEGMANN_IMAGE_ONLY_V1"}, ("images",))
            and type(value["images"]) is list and 0 < len(value["images"]) <= 50, "Exact image-only manifest required")
    ids, hashes = [], set()
    for item in value["images"]:
        require(exact(item, {}, ("image_id", "basename", "bytes", "sha256")) and bounded(item["image_id"], 1, 9999)
                and item["basename"] == f'chip_{item["image_id"]:04d}.png' and bounded(item["bytes"], 1, 4*MIB)
                and _hex(item["sha256"]), "Exact original image-only manifest row required")
        ids.append(item["image_id"])
        hashes.add(item["sha256"])
    require(len(ids) == len(hashes) == len(set(ids)), "Duplicate image ID or exact inference chip bytes")
    return ids


def _closed_export(entry, plan, population, key):
    seed, mode = key
    training, pair, checkpoint = HRSID_SOURCE[seed]
    fixed = dict(seed=seed, mode=mode, checkpoint_sha256=checkpoint[mode], source_pair_receipt_sha256=pair)
    require(exact(entry, fixed, ("source_arm_receipt_sha256", "export_lock_sha256", *FILES))
            and all(_hex(entry[k]) for k in ("source_arm_receipt_sha256", "export_lock_sha256")),
            "Exact accepted HRSID source arm, pair and endpoint required")
    limits = dict(terminal=MIB, report=MIB, predictions=64*MIB, stdout=4*MIB, stderr=4*MIB)
    paths = {k: _path(entry[k], limits[k], basename=name, empty=k == "stderr") for k, name in FILES.items()}
    require(paths["predictions"].parent == paths["report"].parent, "Exact single export output directory required")
    require(paths["terminal"].parent == paths["stdout"].parent == paths["stderr"].parent,
            "Original process logs and terminal must have one control directory")
    terminal = _json(_read(entry["terminal"], MIB))
    expected = dict(seed=seed, mode=mode, original_os_wait=True, original_os_exit=0, resource_stop=False,
                    worker_closed=True, export_lock_sha256=entry["export_lock_sha256"],
                    stdout_sha256=entry["stdout"]["sha256"], stderr_sha256=entry["stderr"]["sha256"])
    require(exact(terminal, expected, ("elapsed_seconds", "peak_rss_bytes")) and finite(terminal["elapsed_seconds"])
            and terminal["elapsed_seconds"] > 0 and bounded(terminal["peak_rss_bytes"], 1, 2**63-1),
            "Original successful export closure required before scoring")
    report = _json(_read(entry["report"], MIB))
    expected = dict(status=EXPORT_STATUS, evaluation_protocol=plan["evaluation_protocol"], source_commit=plan["source_commit"],
                    source_training_protocol=training, source_seed=seed, source_mode=mode, checkpoint_sha256=checkpoint[mode],
                    source_pair_receipt_sha256=pair, source_arm_receipt_sha256=entry["source_arm_receipt_sha256"],
                    target_image_manifest_sha256=plan["image_manifest"]["sha256"], target_image_zip_sha256=plan["image_zip_sha256"],
                    lock_sha256=entry["export_lock_sha256"], worker_sha256=plan["export_worker_sha256"],
                    predictions_sha256=entry["predictions"]["sha256"], images=population, raw_queries=population*164,
                    native_top_rows=population*100, model_forward_calls=population, precision="FP32_TF32_disabled",
                    device="cpu", query_width=164, num_select=100, checkpoint_ema_updates=17640,
                    roi_xyxy=ROI, selection_rule=SELECTION_RULE, training=False, target_adaptation=False,
                    GT_opened=False, official_TEST_opened=False, AP=False)
    counters = ("mapped_rows", "eligible_rows", "candidate_images", "missing_candidate_images", "selected_empty_masks")
    require(exact(report, expected, (*counters, "elapsed_seconds")) and finite(report["elapsed_seconds"])
            and 0 < report["elapsed_seconds"] <= terminal["elapsed_seconds"]
            and bounded(report["mapped_rows"], 0, population*100)
            and bounded(report["eligible_rows"], 0, report["mapped_rows"])
            and all(bounded(report[k], 0, population) for k in counters[2:])
            and report["candidate_images"] + report["missing_candidate_images"] == population
            and report["selected_empty_masks"] <= report["candidate_images"] <= report["eligible_rows"],
            "Exact complete original image-only export, source and ROI required")
    stdout = _read(entry["stdout"], 4*MIB)
    _read(entry["stderr"], 4*MIB, empty=True)
    require(stdout.endswith(b"\n") and _json(stdout.splitlines()[-1]) == report, "Original final stdout must bind exact export report")
    return report, tuple(paths.values())


def score_closed_cohort(plan_path, plan_sha256, protocol):
    """Read truth only after the exact six original exports are closed and bound."""
    require(_hex(protocol, 40), "Exact evaluation protocol required")
    plan = _json(_stable(plan_path, plan_sha256, MIB))
    fields = ("source_commit", "export_worker_sha256", "image_zip_sha256", "image_manifest", "ground_truth", "arms")
    require(exact(plan, dict(status=PLAN_STATUS, evaluation_protocol=protocol), fields)
            and _hex(plan["source_commit"], 40) and plan["export_worker_sha256"] == EXPORT_WORKER_SHA256
            and _hex(plan["image_zip_sha256"])
            and type(plan["arms"]) is list and len(plan["arms"]) == len(ROSTER), "Exact six-arm evaluation plan required")
    image_manifest, ground_truth = (plan[k] for k in ("image_manifest", "ground_truth"))
    paths = [plan_path, _path(image_manifest, MIB), _path(ground_truth, MIB)]
    ids = _image_ids(image_manifest)
    admitted = {}
    for key, entry in zip(ROSTER, plan["arms"], strict=True):
        report, closed_paths = _closed_export(entry, plan, len(ids), key)
        admitted[key] = (entry, report)
        paths.extend(closed_paths)
    require(len(set(paths)) == len(paths), "Every closed role needs its own exact file path")
    for field in ("source_arm_receipt_sha256", "export_lock_sha256"):
        require(len({entry[field] for entry, _ in admitted.values()}) == len(ROSTER), "Distinct source and original export locks required")
    require(len({entry["terminal"]["sha256"] for entry, _ in admitted.values()}) == len(ROSTER),
            "Distinct original process closures required")
    rows = {}
    for key, (entry, _) in admitted.items():
        raw = _read(entry["predictions"], 64*MIB)
        lines = raw.splitlines()
        require(raw.endswith(b"\n") and len(lines) == len(ids), "Exact complete prediction JSONL required")
        rows[key] = [_json(line) for line in lines]
    truth = _json(_read(ground_truth, MIB))
    result = score_population(ids, truth, rows)
    result.update(status=SCORE_STATUS, evaluation_protocol=protocol, source_commit=plan["source_commit"],
                  closed_plan_sha256=plan_sha256, image_manifest_sha256=image_manifest["sha256"],
                  ground_truth_sha256=ground_truth["sha256"], image_zip_sha256=plan["image_zip_sha256"],
                  export_worker_sha256=plan["export_worker_sha256"], author_positive_file_expected_sha256=AUTHOR_JSON_SHA256,
                  external_auxiliary=True, positive_center_only=True, training=False, target_adaptation=False)
    source_arms = []
    for (seed, mode), (entry, report) in admitted.items():
        arm = result["seeds"][str(seed)][mode]
        expected = {key: arm[key] for key in ("mapped_rows", "eligible_rows", "candidate_images", "selected_empty_masks")}
        expected["missing_candidate_images"] = arm["missing_candidates"]
        require(all(report[k] == v for k, v in expected.items()), "Original prediction and export accounting differ")
        pins = {k: entry[k] for k in ("seed", "mode", "checkpoint_sha256", "source_pair_receipt_sha256",
                                     "source_arm_receipt_sha256", "export_lock_sha256")}
        pins.update({k+"_sha256": entry[k]["sha256"] for k in FILES})
        source_arms.append(dict(**pins, export_seconds=report["elapsed_seconds"]))
    result["source_arms"] = source_arms
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        require(args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent
                and not args.output.exists(), "Fresh absolute scoring output required")
        result = score_closed_cohort(args.plan, args.plan_sha256, args.protocol)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
            stream.flush()
            os.fsync(stream.fileno())
    except (OSError, ValueError) as exc:
        parser.exit(3, f"Schwegmann center score refused: {exc}\n")


if __name__ == "__main__":
    main()
