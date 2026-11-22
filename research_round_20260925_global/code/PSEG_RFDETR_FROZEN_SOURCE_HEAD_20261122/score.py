"""Frozen PSeg-SSDD RF-DETR evaluation of five masked edit policies."""
import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version
from io import StringIO
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

PRIOR = "07bbf74198bec0e55f131a2aeeca2f7efa4c6008654dd3816e7b3ea57d8178f4"
PROTOCOL = "be9d182f29ef33415b11c2423b45dbf7d3966b85cb879c1d20aec7343729f043"
GEOMETRY = "7cad8b51a9230101e2233db8455f84c28163abfb2ee8ddd8cbe8718a2e719bd0"
IOU = "77159ea493a95973fe422dac1d303eaeba603e6d8af7ae179c72aecc2e4301b0"
ARMS = ("raw", "fixed", "free_matched_count", "attraction_dynamic",
        "attraction_static_at_dynamic_count", "learned_at_attraction_count")
METRICS = ("AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl")
PAIRS = ((ARMS[5], ARMS[3]), (ARMS[5], ARMS[4]), (ARMS[2], ARMS[1]), (ARMS[5], ARMS[0]), (ARMS[3], ARMS[0]))
FIELDS = ("continuous_iou_points", "fp_pixels_per_gt_area_pct", "fn_pixels_per_gt_area_pct")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def pinned(path, expected):
    raw = path.read_bytes()
    require(digest(raw) == expected, "Pinned bytes: " + path.name)
    return raw


def load(path, expected):
    obj = ModuleType(path.stem)
    obj.__file__ = str(path)
    exec(compile(pinned(path, expected), str(path), "exec"), obj.__dict__)
    return obj


def document(path, expected):
    return json.loads(pinned(path, expected))


def stream(path, expected):
    return [json.loads(line) for line in pinned(path, expected).splitlines()]


def cohort(predictions, captures, edits):
    require(len(captures) == len(edits), "Aligned image and edit streams")
    blocks, counts = {}, Counter()
    index = 0
    for capture, edit in zip(captures, edits, strict=True):
        image_id = capture["image_id"]
        c, n = capture["stock_prediction_start"], capture["stock_prediction_count"]
        ranks = capture["native_ranks"]
        require(image_id == edit["image_id"] and image_id not in blocks and c == index
                and 0 <= n <= 100 and ranks == sorted(set(ranks)) and len(ranks) == n,
                "Native image and rank identity")
        stock = predictions[c:c+n]
        require(len(stock) == n and all(row["image_id"] == image_id and row["category_id"] == 1
                and row["segmentation"]["size"] == capture["original_hw"] for row in stock), "Exact native prediction window")
        edits_by_rank = {int(k): v for k, v in edit["changes"].items()}
        require(set(edits_by_rank) <= set(ranks) and len(edits_by_rank) == len(edit["changes"]), "Exact sparse ranks")
        for rank, original in zip(ranks, stock, strict=True):
            options = edits_by_rank.get(rank, {})
            require(set(options) <= set(ARMS[1:]), "Known prespecified policies")
            swaps = {name: options.get(name, {}).get("swaps", 0) for name in ARMS[1:]}
            require(swaps[ARMS[1]] == swaps[ARMS[2]] and swaps[ARMS[3]] == swaps[ARMS[4]] == swaps[ARMS[5]],
                    "Count matched within both comparison families")
            area = int(mask.area(original["segmentation"]))
            for arm, data in options.items():
                nswap = swaps[arm]
                rle = data["rle"]
                require(type(nswap) is int and 0 < nswap <= 8 and 32 <= area < 1024
                        and rle["size"] == original["segmentation"]["size"] and int(mask.area(rle)) == area,
                        "Independently admitted equal-area edit")
                counts[arm + "_candidates"] += 1
                counts[arm + "_swaps"] += nswap
        rgb = edit["rgb_sha256"]
        require(isinstance(rgb, str) and len(rgb) == 64 and all(c in "abcdef0123456789" for c in rgb), "Decoded identity")
        blocks[image_id] = dict(rgb=rgb, stock=stock, ranks=ranks, changes=edits_by_rank,
                                hw=capture["original_hw"])
        index += len(stock)
    require(index == len(predictions), "Complete original candidate roster")
    return blocks, dict(counts)


def score_one(gt, blocks, arm, anchors=False):
    rows, ids = [], []
    for image_id in blocks:
        item = blocks[image_id]
        for rank, original in zip(item["ranks"], item["stock"], strict=True):
            options = item["changes"].get(rank, {})
            edit = options.get(arm)
            rows.append(original if edit is None else dict(original, segmentation=edit["rle"]))
            ids.append((image_id, rank, original))
    with redirect_stdout(StringIO()):
        ref = COCO()
        ref.dataset = deepcopy(gt)
        ref.createIndex()
        det = ref.loadRes(deepcopy(rows))
        ev = COCOeval(ref, det, "segm")
        ev.params.catIds = [1]
        ev.params.imgIds = [im["id"] for im in gt["images"]]
        ev.params.maxDets = [1, 10, 100]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    scores = dict(zip(METRICS, map(float, ev.stats)))
    i90, = np.flatnonzero(np.isclose(ev.params.iouThrs, .90))
    precision = ev.eval["precision"][i90, :, 0, 0, -1]
    scores["AP90"] = float(precision[precision >= 0].mean())
    assignments = {}
    if anchors:
        for row in ev.evalImgs:
            if row is None or row["maxDet"] != 100 or list(row["aRng"]) != list(ev.params.areaRng[0]):
                continue
            for gt_id, det_id, ignored in zip(row["gtIds"], row["gtMatches"][0], row["gtIgnore"], strict=True):
                if det_id and not ignored:
                    record = ids[int(det_id)-1]
                    require(gt_id not in assignments and record[0] == row["image_id"], "Unique .50 original match")
                    assignments[int(gt_id)] = record
    return scores, assignments


def report_geometry(assignments, gt, blocks, code, shapely_site):
    sys.path.append(str(shapely_site))
    import shapely
    require(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2", "Historical polygon runtime")
    suite = load(code / "MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py", GEOMETRY)
    old = load(code / "DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/dev705_free_endpoint/score_endpoint.py", IOU)
    modules = suite.geometry_modules(code)
    truth, neutral, _ = modules.geometry.polygon_support(gt, modules.vector)
    require(len(truth) == len(gt["annotations"]) and not neutral, "Original author-float polygons")
    by_group, rows = defaultdict(list), []
    edits = Counter()
    for gid, (image_id, rank, stock) in assignments.items():
        raw = stock["segmentation"]
        area = int(mask.area(raw))
        if not 32 <= area < 1024:
            continue
        item = blocks[image_id]
        options = item["changes"].get(rank, {})
        base = old.continuous(raw, area, truth[gid][0], modules)
        values = [old.continuous(options[arm]["rle"], area, truth[gid][0], modules) if arm in options else base
                  for arm in ARMS]
        rows.append(values)
        by_group[item["rgb"]].append(values)
        edits.update(options.keys())
    require(rows, "Positive native area eligible match count")
    group_values = np.array([np.mean(by_group[k], axis=0) for k in sorted(by_group)], dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(5217))
    selectors = rng.integers(len(group_values), size=(4096, len(group_values)))
    paired = {}
    arrays = np.array(rows)
    for left, right in PAIRS:
        i, j = ARMS.index(left), ARMS.index(right)
        diff = group_values[:, i] - group_values[:, j]
        intervals = np.percentile(diff[selectors].mean(axis=1), [2.5,97.5], axis=0)
        measures = {k: {"delta": float(diff[:, c].mean()), "interval_95": intervals[:, c].tolist()}
                    for c,k in enumerate(FIELDS)}
        for threshold in (75, 90):
            a, b = arrays[:,i,0] >= threshold, arrays[:,j,0] >= threshold
            measures[str(threshold)] = dict(gained=int((a&~b).sum()), lost=int((b&~a).sum()), retained=int((a&b).sum()))
        paired[left + " minus " + right] = measures
    return dict(fixed_matches=len(assignments), area_eligible=len(rows), groups=len(by_group),
                edited_matches=dict(edits), paired_contrasts_points=paired,
                group_equal_arms={name:dict(zip(FIELDS,map(float,group_values[:,i].mean(axis=0))))
                                  for i,name in enumerate(ARMS)})


def run(args):
    require(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
            and version("pycocotools") == "2.0.11" and np.__version__ == "1.26.4", "Accepted isolated CPU evaluator")
    pinned(args.protocol, PROTOCOL)
    prior = document(args.prior, PRIOR)
    receipt = document(args.verify, args.verify_sha)
    observed = json.loads(args.verify_terminal.read_bytes())
    require(observed["actual_exit"] == 0 and observed["output_footer_verified"] is True
            and observed["result_sha256"] == args.verify_sha and receipt["GT_opened"] is False
            and receipt["status"] == "PSEG_TARGET_RF5201_PRELABEL_EDITS_VERIFIED",
            "Originally closed label-free independent verified edits")
    stock_sha = prior["arms"]["center"]["prediction_sha256"]
    require(stock_sha == receipt["predictions_sha256"], "Approved original carrier predictions")
    predictions = document(args.stock, stock_sha)
    captures = stream(args.capture / "CAPTURE_PRIVATE.jsonl", receipt["capture_stream_sha256"])
    edits = stream(args.edits, receipt["edits_sha256"])
    blocks, counts = cohort(predictions, captures, edits)
    require(len(blocks) == 187 and len(predictions) == 16161, "Historic DEV187 scope")
    gt = document(args.gt, prior["floating_gt_sha256"])
    require(set(im["id"] for im in gt["images"]) == set(blocks) and len(gt["annotations"]) == 380
            and all([im["height"],im["width"]] == blocks[im["id"]]["hw"] for im in gt["images"]),
            "Original float references and image dimensions")
    dedup = {}
    for image_id in sorted(blocks):
        dedup.setdefault(blocks[image_id]["rgb"], image_id)
    require(len(dedup) == 182, "Historic decoded identity count")
    views = {}
    assignments = {}
    for label, keep in (("full", set(blocks)), ("equal_rgb_deduplicated", set(dedup.values()))):
        subset = dict(gt, images=[r for r in gt["images"] if r["id"] in keep],
                      annotations=[r for r in gt["annotations"] if r["image_id"] in keep])
        selection = {k:v for k,v in blocks.items() if k in keep}
        metrics = {}
        for arm in ARMS:
            value, primary = score_one(subset, selection, arm, anchors=label=="full" and arm=="raw")
            metrics[arm] = value
            if primary:
                assignments = primary
        if label == "full":
            expected = prior["primary"]["center"]
            require([metrics["raw"][k] for k in METRICS] == expected["all12"]
                    and metrics["raw"]["AP90"] == expected["AP90"], "Exact 13 historical raw COCO metrics")
        contrasts = {a+" minus "+b:{key:100*(v-metrics[b][key]) if min(v, metrics[b][key])>=0 else None
                                   for key,v in metrics[a].items()} for a,b in PAIRS}
        views[label] = dict(images=len(keep), annotations=len(subset["annotations"]),
                            candidates=sum(len(b["stock"]) for b in selection.values()), arms_fraction=metrics,
                            paired_contrasts_points=contrasts)
    geometry = report_geometry(assignments, gt, blocks, args.code, args.shapely_site)
    key = ARMS[5] + " minus " + ARMS[3]
    decisions = dict(full_ap_positive=views["full"]["paired_contrasts_points"][key]["AP"]>0,
                     sensitivity_ap_positive=views["equal_rgb_deduplicated"]["paired_contrasts_points"][key]["AP"]>0,
                     positive_lower_iou=geometry["paired_contrasts_points"][key]["continuous_iou_points"]["interval_95"][0]>0)
    return dict(status="PSEG_RFDETR_FROZEN_SOURCE_HEAD_SCORE_COMPLETE", DEV=True, FIT=False, TEST=False,
                detector_forwards=0, optimizer_updates=0, history_raw_exact=True, views=views, geometry=geometry,
                readiness=decisions, advance=all(decisions.values()), independent_edit_receipt_sha256=args.verify_sha,
                protocol_sha256=PROTOCOL, all_candidate_edit_counts=counts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "code", "capture", "stock", "edits", "verify", "verify-terminal", "prior", "gt", "shapely-site", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--verify-sha", required=True)
    args = p.parse_args()
    require(not args.output.exists(), "Fresh score output")
    start = perf_counter()
    result = run(args)
    result["elapsed_seconds"] = perf_counter() - start
    result["source_sha256"] = digest(Path(__file__).read_bytes())
    args.output.mkdir()
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with (args.output / "RESULT_SAFE.json").open("xb") as handle:
        handle.write(raw)
    print(json.dumps(dict(status=result["status"],result_sha256=digest(raw))), flush=True)


if __name__=="__main__":
    main()

