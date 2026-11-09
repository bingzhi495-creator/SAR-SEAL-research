"""Cross-fit a signed detector-box shift on fixed HRSID FIT query/GT pairs."""
import argparse
from collections import Counter
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

from box_shift_oof import predict_oof
from fit_query_export import INDEX_SHA, WIDTH, need, pinned, write
from score_geometry import rank_pairs

SHIFTS = tuple((dy, dx) for dy in range(-2, 3) for dx in range(-2, 3))
POLICIES = ("identity", "constant", "mask", "box")
ZERO = (0, 0)


def union(value):
    return sum(value[:3])


def strict(value):
    return 10 * value[0] >= 9 * union(value)


def make_row(pred, truth, box, *, protected):
    yy, xx = np.nonzero(pred)
    need(yy.size > 0 and truth.any() and pred.shape == truth.shape, "Nonempty fixed raster pair")
    h, w = pred.shape
    y0, x0, y1, x1 = int(yy.min()), int(xx.min()), int(yy.max()), int(xx.max())
    bbox_mid = np.array([(y0 + y1 + 1) / 2, (x0 + x1 + 1) / 2])
    bbox_head_mid = np.array([box[1] * h, box[0] * w])
    centroid = np.array([yy.mean() + .5, xx.mean() + .5])
    truth_area = int(truth.sum())
    area = int(len(yy))
    table = {}
    for dy, dx in SHIFTS:
        if 0 <= y0 + dy and y1 + dy < h and 0 <= x0 + dx and x1 + dx < w:
            sy, sx = yy + dy, xx + dx
            tp = int(np.count_nonzero(truth[sy, sx]))
            edits = 2 * (area - int(np.count_nonzero(pred[sy, sx])))
            table[dy, dx] = (tp, area - tp, truth_area - tp, edits)
    # Areas are fixed by no-clipping translation, so IoU is monotone in TP.
    target = ZERO if protected else min(
        table, key=lambda s: (-table[s][0], table[s][3], abs(s[0]) + abs(s[1]), *s))
    return dict(box_gap=bbox_head_mid - bbox_mid, mask_gap=centroid - bbox_mid, table=table, target=target,
                protected=protected)


def score_region(rows, shifts, groups, *, primary):
    near = sum(not row["protected"] for row in rows)
    old_strict = len(rows) - near
    by_method = {}
    flags = {}
    for name in (*POLICIES, "oracle"):
        proposed = shifts[name] if name != "oracle" else [row["target"] for row in rows]
        values = []
        invalid = 0
        actions = Counter()
        for row, shift in zip(rows, proposed, strict=True):
            pair = tuple(map(int, shift))
            if pair not in row["table"]:
                pair = ZERO
                invalid += 1
            values.append(row["table"][pair])
            actions[pair] += 1
        counts = {f"{t/100:.2f}": sum(100 * v[0] >= t * union(v) for v in values) for t in (75, 85, 90)}
        flag = np.asarray([strict(v) for v in values])
        flags[name] = flag
        gains = sum(bool(f) and not row["protected"] for row, f in zip(rows, flag, strict=True))
        losses = sum(not f and row["protected"] for row, f in zip(rows, flag, strict=True))
        by_method[name] = dict(thresholds=counts, strict_gains=int(gains), strict_losses=int(losses),
                               effective_nonzero=len(rows) - actions[ZERO], clipping_fallback=invalid,
                               mean_iou=sum(v[0]/union(v) for v in values)/len(rows),
                               tp=sum(v[0] for v in values), fp=sum(v[1] for v in values),
                               fn=sum(v[2] for v in values), edits=sum(v[3] for v in values),
                               shifts={f"{s[0]},{s[1]}": n for s, n in sorted(actions.items())})
        need(counts["0.90"] == old_strict + gains - losses, "Exact strict gain and loss reconciliation")
        for row, value in zip(rows, values, strict=True):
            base = row["table"][ZERO]
            need(value[1] - base[1] == value[2] - base[2] and value[3] % 2 == 0,
                 "No-clipping rigid shift preserves area and signed FP/FN balance")
    result = dict(pairs=len(rows), near_strict=near, protected=old_strict, groups=len(groups), methods=by_method)
    if primary:
        group_index = {group: i for i, group in enumerate(groups)}
        ids = np.array([group_index[row["group"]] for row in rows])
        gsize = np.bincount(ids, minlength=len(groups))
        draw = np.random.default_rng(5217).integers(0, len(groups), size=(4096, len(groups)))
        den = gsize[draw].sum(axis=1)
        need((den > 0).all(), "Nonempty group bootstrap draws")
        contrasts = {}
        for control in ("identity", "constant", "mask"):
            diff = flags["box"].astype(np.int64) - flags[control].astype(np.int64)
            by_group = np.zeros(len(groups), dtype=np.int64)
            np.add.at(by_group, ids, diff)
            bootstrap = by_group[draw].sum(axis=1) / den
            contrasts[control] = dict(net=int(diff.sum()), gain_groups=int((by_group > 0).sum()),
                                      loss_groups=int((by_group < 0).sum()),
                                      fraction_interval=np.quantile(bootstrap, [.025, .975], method="linear").tolist())
        passed = (all(v["net"] >= 5 and v["gain_groups"] >= 5 and v["fraction_interval"][0] > 0
                      for v in contrasts.values())
                  and by_method["box"]["strict_losses"] <= 1
                  and by_method["box"]["thresholds"]["0.75"] >= by_method["identity"]["thresholds"]["0.75"])
        result.update(contrasts=contrasts, gate=passed,
                      decision="ADVANCE_BOX_SHIFT_TO_FIXED_ALL_CANDIDATE_EVALUATION" if passed
                      else "STOP_FIXED_BOX_GAP_SHIFT_RECIPE")
    return result


def run(args):
    start = perf_counter()
    report = json.loads(pinned(args.export_report, args.export_report_sha256, 65536))
    need(report["status"] == "HRSID_FIT_Q164_EXPORT_COMPLETE" and report["images"] == report["forward_calls"] == 114
         and report["native_mapped_rows"] == 10572 and report["native_rows_bitwise_verified"] is True
         and report["GT_opened"] is report["DEV"] is report["TEST"] is report["AP"] is False,
         "Closed exact GT-free box export")
    cohort = json.loads(pinned(args.cohort, report["cohort_sha256"], 1024**2))
    blocks = [json.loads(line) for line in pinned(args.predictions, report["predictions_sha256"], 32 * 1024**2).splitlines()]
    coco = COCO()
    coco.dataset = json.loads(pinned(args.index, INDEX_SHA, 32 * 1024**2))
    with redirect_stdout(io.StringIO()):
        coco.createIndex()
    need(len(coco.imgs) == 2937 and len(coco.anns) == 9260 and cohort["fit_only"] is True,
         "Frozen FIT index and cohort")
    rows = []
    mapped = 0
    groups = sorted({row["group"] for row in cohort["images"] if row["core"]})
    pressure_groups = sorted({row["group"] for row in cohort["images"] if row["stress"]})
    need(len(groups) == 109 and len(pressure_groups) == 4 and set(pressure_groups) <= set(groups),
         "Frozen core and pressure prefix memberships")
    for item, block in zip(cohort["images"], blocks, strict=True):
        image_id = item["image_id"]
        need(block["image_id"] == image_id and coco.imgs[image_id]["file_name"] == item["native_basename"]
             and item["core"] != item["stress"], "Exact disjoint image roles and order")
        native, boxes = block["native_ids"], block["native_box_cxcywh"]
        need(len(native) == len(boxes) and len(block["all_rles"]) == WIDTH
             and len(set(native)) == len(native) and all(type(q) is int and 0 <= q < WIDTH for q in native)
             and all(len(box) == 4 and np.isfinite(box).all() and box[2] >= 0 and box[3] >= 0 for box in boxes),
             "Normalized boxes exactly follow native query IDs")
        mapped += len(native)
        pred = [block["all_rles"][q] for q in native]
        gt = []
        for ann in coco.imgToAnns[image_id]:
            need(ann["category_id"] == 1, "FIT ship GT class")
            target = coco.annToRLE(ann)
            if mask.area(target) and not ann.get("iscrowd", 0) and not ann.get("ignore", 0):
                gt.append(target)
        overlaps = mask.iou(pred, gt, [0] * len(gt)) if pred and gt else np.zeros((len(pred), len(gt)))
        best = overlaps.max(axis=0, initial=0)
        protected = rank_pairs(overlaps, .90)
        near = [(q, g) for q, g in rank_pairs(overlaps, .75) if best[g] < .90]
        pairs = protected + near
        need(len({q for q, _ in pairs}) == len({g for _, g in pairs}) == len(pairs),
             "Exact original near and strict pairs are disjoint")
        for ordinal, (q, g) in enumerate(pairs):
            px, label = (mask.decode(x).astype(bool) for x in (pred[q], gt[g]))
            need(px.shape == label.shape == (800, 800), "Exact original raster canvas")
            row = make_row(px, label, boxes[q], protected=ordinal < len(protected))
            identity = row["table"][ZERO]
            need(abs(identity[0] / union(identity) - overlaps[q, g]) < 1e-12, "Exact native overlap reproduction")
            row.update(group=item["group"], fit=bool(item["core"]))
            rows.append(row)
    need(len(blocks) == len(cohort["images"]) == 114 and mapped == 10572 and len(rows) == 330,
         "Frozen image, native box, and pairing denominators")
    fit = np.array([row["fit"] for row in rows])
    predictions = predict_oof([row["group"] for row in rows], np.stack([row["box_gap"] for row in rows]),
                              np.stack([row["mask_gap"] for row in rows]), np.array([row["target"] for row in rows]),
                              all_groups=groups, fit_mask=fit)
    folds = predictions["folds"]
    core = score_region([r for r in rows if r["fit"]], {name: predictions[name][fit] for name in POLICIES},
                        groups, primary=True)
    stress = score_region([r for r in rows if not r["fit"]], {name: predictions[name][~fit] for name in POLICIES},
                          pressure_groups, primary=False)
    need(core["near_strict"] == 234 and core["protected"] == 37 and stress["near_strict"] == 59
         and stress["protected"] == 0 and core["methods"]["oracle"]["strict_gains"] == 50
         and stress["methods"]["oracle"]["strict_gains"] == 0, "Exact original pairs and translation oracle")
    sizes = [dict(fold=k, core_test_rows=int((fit & (folds == k)).sum()),
                  core_fit_rows=int((fit & (folds != k)).sum()), stress_test_rows=int((~fit & (folds == k)).sum()))
             for k in range(5)]
    result = dict(status="ACTUAL_HRSID_FIT_BOX_SHIFT", protocol=args.protocol,
                  fit_index_sha256=INDEX_SHA, cohort_sha256=report["cohort_sha256"],
                  prediction_sha256=report["predictions_sha256"], export_report_sha256=args.export_report_sha256,
                  core=core, stress=stress, folds=sizes, elapsed_seconds=perf_counter() - start,
                  FIT=True, DEV=False, TEST=False, AP=False, forward_calls=0, optimizer_updates=0)
    write(args.output, result)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("export-report", "cohort", "predictions", "index", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--export-report-sha256", required=True)
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    need(re.fullmatch(r"[0-9a-f]{40}", args.protocol) and args.output.is_absolute()
         and args.output.parent.resolve(strict=True) == args.output.parent and not args.output.exists(),
         "Prelocked protocol and isolated CPU scoring result")
    def offline(event, _):
        if event in ("socket.connect", "socket.getaddrinfo"):
            raise RuntimeError("FIT box shift scoring is offline")
    sys.addaudithook(offline)
    run(args)


if __name__ == "__main__":
    main()
