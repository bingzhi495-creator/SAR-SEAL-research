"""Measure D-FINE fixed top100 and full post-NMS candidate coverage without AP."""
import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from hashlib import sha256
from importlib.metadata import version
from io import StringIO
import json
import os
from pathlib import Path
from types import ModuleType

import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO
from scipy.optimize import linear_sum_assignment

PROTOCOL = "63f7f3aef39a9262100a6a4b0b5a711bf7acc2b298f5516509639aa7d713e908"
COVERAGE_HELPER = "de18b2d993af7549e437add80c72b20633a3e596170f6b4789188539c7cc25fa"
ACCEPT = "37e51c1a1360a69751aa0f56c24d1b450e232266f1e0ba847626b80df2d93af4"
SCORE = "1d76e8d55be1da022e0962917e1490769994caf3636aaa4a2c1bbe36ab424588"
INDEPENDENT_SCORE = "7e179942859dddbaaa0c3ed576964a6664423d6a90ce5debf1a3fa3bc192950d"
EMPTY = "f3f2d644b2e0ce475b4a46b382ec7ec51f2efa4609c8f85b4f8f552ff57d5889"
MODES = ("top100", "full_post_nms")
THRESHOLDS = (.50, .75, .90)


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def pin(path, digest, cap=4*1024**2):
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(),
            "Exact canonical frozen file")
    size = path.stat().st_size
    require(0 < size <= cap, "Bounded exact input")
    raw = path.read_bytes()
    require(len(raw) == size and sha256(raw).hexdigest() == digest, "Pinned input bytes " + path.name)
    return raw


def write(path, value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256(raw).hexdigest()


def best_assignment(overlap, threshold):
    edges = overlap >= threshold
    bonus = min(overlap.shape) + 1
    rows, cols = linear_sum_assignment(np.where(edges, bonus + overlap, 0), maximize=True)
    valid = edges[rows, cols]
    return rows[valid], cols[valid]


def volume(candidates, targets, pair_rows, pair_cols):
    fields = ("matches", "predicted_pixels", "gt_pixels", "intersection_pixels", "fp_pixels", "fn_pixels")
    totals = Counter(dict.fromkeys(fields, 0))
    for r, c in zip(pair_rows, pair_cols, strict=True):
        pred, gt = candidates[int(r)], targets[int(c)]
        p, g = int(mask.area(pred)), int(mask.area(gt))
        shared = int(mask.area(mask.merge([pred, gt], intersect=True)))
        require(0 <= shared <= min(p, g), "Exact matched raster areas")
        totals.update(dict(matches=1, predicted_pixels=p, gt_pixels=g, intersection_pixels=shared,
                           fp_pixels=p-shared, fn_pixels=g-shared))
    return totals


def census(gt, blocks, empty_witness, coverage):
    coco = COCO()
    coco.dataset = gt
    with redirect_stdout(StringIO()):
        coco.createIndex()
    images = gt["images"]
    require([r["id"] for r in images] == [b["image_id"] for b in blocks] and
            len(coco.imgs) == len(images) == 705 and len(coco.anns) == len(gt["annotations"]) == 1782
            and [c["id"] for c in gt["categories"]] == [1], "Exact same raster GT and image IDs")
    exclusions = Counter(empty=0, crowd=0, ignore=0)
    population = Counter()
    blank_ids = set()
    modes = {mode: dict(counts=defaultdict(Counter), candidate=Counter(), bins=np.zeros((2, 5), dtype=np.int64))
             for mode in MODES}
    recovered = {f"{t:.2f}": Counter() for t in THRESHOLDS}
    for image, block in zip(images, blocks, strict=True):
        image_id, hw = image["id"], [image["height"], image["width"]]
        candidates = block["coco"]
        require(len(candidates) <= 300 and
                all(c["image_id"] == image_id and c["category_id"] == 1 and "bbox" not in c and
                    c["segmentation"]["size"] == hw and 0 <= c["score"] <= 1 for c in candidates) and
                all(a["score"] >= b["score"] for a,b in zip(candidates,candidates[1:])),
                "Exact ranked native RLEs and no boxes in scoring rows")
        masks = [c["segmentation"] for c in candidates]
        annotations = coco.imgToAnns[image_id]
        targets, small = [], []
        for ann in annotations:
            rle = coco.annToRLE(ann)
            require(ann["category_id"] == 1 and rle["size"] == hw, "Exact original ship raster")
            if not mask.area(rle):
                blank_ids.add((ann["id"], image_id))
                exclusions["empty"] += 1
            elif ann.get("iscrowd", 0):
                exclusions["crowd"] += 1
            elif ann.get("ignore", 0):
                exclusions["ignore"] += 1
            else:
                targets.append(rle)
                small.append(ann["area"] < 1024)
        population.update(dict(eligible=len(targets), small=sum(small), multi_gt=len(targets) >= 2))
        small = np.asarray(small, dtype=bool)
        areas = mask.area(masks) if masks else np.empty(0, dtype=np.uint32)
        overlap = mask.iou(masks, targets, [0]*len(targets)) if masks and targets else np.zeros((len(masks), len(targets)))
        require(overlap.shape == (len(masks), len(targets)) and np.isfinite(overlap).all() and
                (overlap >= 0).all() and (overlap <= 1).all(), "Valid native binary IoU matrix")
        prefix = min(len(masks), 100)
        best_top = overlap[:prefix].max(axis=0, initial=0)
        best_full = overlap.max(axis=0, initial=0)
        for mode, n in (("top100", prefix), ("full_post_nms", len(masks))):
            report = modes[mode]
            report["candidate"].update(dict(predictions=n, empty_predictions=int((areas[:n] == 0).sum()),
                                            images_without_candidates=int(n == 0),
                                            candidate_mask_pixels=int(areas[:n].sum())))
            best = best_top if mode == "top100" else best_full
            report["bins"][0] += coverage.bins(best[small])
            report["bins"][1] += coverage.bins(best[~small])
            for threshold in THRESHOLDS:
                key = f"{threshold:.2f}"
                independent, maximum, greedy = coverage.graph_counts(overlap[:n], threshold)
                rows, cols = best_assignment(overlap[:n], threshold)
                require(len(rows) == maximum, "Independent maximum-cardinality assignment witness")
                base = report["counts"][key]
                base.update(dict(independent=independent, maximum_one_to_one=maximum, native_greedy=greedy))
                base.update({f"matched_{k}": v for k,v in volume(masks, targets, rows, cols).items()})
                tail = rows >= 100
                base.update({f"tail_matched_{k}": v for k,v in volume(masks, targets, rows[tail], cols[tail]).items()})
        for threshold in THRESHOLDS:
            newly_covered = (best_top < threshold) & (best_full >= threshold)
            recovered[f"{threshold:.2f}"].update(dict(independent_recovered=int(newly_covered.sum()),
                                                     independent_recovered_small=int((newly_covered & small).sum())))
    require(blank_ids == empty_witness and len(blank_ids) == 4 and population["eligible"] + sum(exclusions.values()) == 1782,
            "Exact original exclusion reconciliation")
    for mode, report in modes.items():
        both = report["bins"].sum(axis=0)
        report["best_iou_bins"] = both.tolist()
        report["best_iou_bins_by_coco_area"] = dict(zip(("small", "non_small"), report.pop("bins").tolist()))
        for i, (key, t) in enumerate(report["counts"].items()):
            require(int(both[i+2:].sum()) == t["independent"] and t["matched_matches"] == t["maximum_one_to_one"],
                    "Coverage, bin and matched volume closure")
            t.update(independent_misses=population["eligible"]-t["independent"],
                     maximum_misses=population["eligible"]-t["maximum_one_to_one"],
                     sharing_gap=t["independent"]-t["maximum_one_to_one"],
                     rank_gap=t["maximum_one_to_one"]-t["native_greedy"])
            require(t["matched_predicted_pixels"] == t["matched_intersection_pixels"]+t["matched_fp_pixels"] and
                    t["matched_gt_pixels"] == t["matched_intersection_pixels"]+t["matched_fn_pixels"],
                    "Exact matched mask volumes")
    for key, changes in recovered.items():
        a, b = (modes[m]["counts"][key] for m in MODES)
        require(b["independent"]-a["independent"] == changes["independent_recovered"], "Nested independent rescue")
        changes.update(maximum_matching_increment=b["maximum_one_to_one"]-a["maximum_one_to_one"],
                       greedy_increment=b["native_greedy"]-a["native_greedy"],
                       fraction_top100_independent_misses_rescued=changes["independent_recovered"]/a["independent_misses"],
                       remaining_full_independent_misses=b["independent_misses"],
                       remaining_full_maximum_misses=b["maximum_misses"])
    return dict(images=705, annotations=1782, population=population, exclusions=exclusions, candidate_pools=modes,
                nested_recovery=recovered, best_iou_bin_labels=coverage.BIN_LABELS)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("protocol", "coverage-helper", "acceptance", "score", "independent-score", "predictions",
                  "gt", "empty", "output"):
        p.add_argument("--"+field, type=Path, required=True)
    a = p.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and version("pycocotools") == "2.0.11",
            "Exact CPU-only raster backend")
    pin(a.protocol, PROTOCOL)
    helper = ModuleType("_existing_native_coverage")
    helper.__file__ = str(a.coverage_helper)
    exec(compile(pin(a.coverage_helper, COVERAGE_HELPER), str(a.coverage_helper), "exec"), helper.__dict__)
    accepted = json.loads(pin(a.acceptance, ACCEPT))
    scored = json.loads(pin(a.score, SCORE))
    independent = json.loads(pin(a.independent_score, INDEPENDENT_SCORE))
    require(accepted["GT_opened"] is accepted["AP"] is accepted["TEST"] is False
            and accepted["job_drained"] is True and accepted["original_os_exit"] == 0 and
            independent["exact_26_metrics"] is True and independent["score_sha256"] == SCORE and
            scored["predictions_sha256"] == accepted["predictions_sha256"] and scored["export_acceptance_sha256"] == ACCEPT,
            "Exact natural, independent, frozen score lineage")
    raw = pin(a.predictions, accepted["predictions_sha256"], 64*1024**2)
    require(raw.endswith(b"\n"), "Complete exact closed prediction stream")
    blocks = [json.loads(line) for line in raw.splitlines()]
    require(len(blocks) == 705 and sum(len(b["coco"]) for b in blocks) == 184687, "Original prediction population")
    a.output.mkdir()
    prelabel = write(a.output/"PRELABEL_SAFE.json",dict(status="DFINE_DEV705_CANDIDATE_COVERAGE_PRELABEL",
                     protocol_sha256=PROTOCOL,predictions_sha256=accepted["predictions_sha256"],images=705,
                     post_nms_rows=184687,GT_opened=False,FIT=False,DEV=True,TEST=False,AP=False))
    gt = json.loads(pin(a.gt, scored["gt_sha256"]))
    empty = {(r["annotation_id"],r["image_id"]) for r in json.loads(pin(a.empty, EMPTY))}
    result = census(gt, blocks, empty, helper)
    for mode, prefix in (("top100", "rows_at_100"), ("full_post_nms", "post_nms_rows")):
        pool = result["candidate_pools"][mode]["candidate"]
        require(pool["predictions"] == accepted[prefix], "Original exact post-NMS candidate counts")
        require(pool["empty_predictions"] == accepted["empty_at_100" if mode == "top100" else "empty_post_nms"],
                "Original exact empty prediction count")
    result.update(status="DFINE_DEV705_CANDIDATE_COVERAGE_ACTUAL",FIT=False,DEV=True,TEST=False,AP=False,
                  GT_opened=True,model_forwards=0,optimizer_updates=0,author_post_nms_scope=True,
                  prior_score_sha256=SCORE,protocol_sha256=PROTOCOL,prelabel_sha256=prelabel,
                  predictions_sha256=accepted["predictions_sha256"], gt_sha256=scored["gt_sha256"])
    digest = write(a.output/"RESULT_SAFE.json", result)
    print(json.dumps(dict(status=result["status"],result_sha256=digest,nested_recovery=result["nested_recovery"]),
                     sort_keys=True))


if __name__ == "__main__":
    main()