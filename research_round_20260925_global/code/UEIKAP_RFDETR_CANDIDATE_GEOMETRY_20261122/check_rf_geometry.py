"""Recompute the post-hoc UEIKAP opportunity using direct dense bitmap counts."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as codec


PRIMARY_RESULT = "48a57db4e59fc4171e2a2e1462a16ad63370a53dc918e45cc64bd94981565e64"
PRIMARY_SOURCE = "73b8e8f0ef8769a0ad179453ecbebd59fee1b5e733b4c41bd7576bfe42ebdabe"


def ratio(positive, area, truth):
    return positive / (truth + area - positive)


def box_extent_iou(candidate, bounds):
    x0, y0, x1, y1 = candidate
    l, t, r, b = bounds
    overlap = max(0., min(x1,r) - max(x0,l)) * max(0., min(y1,b) - max(y0,t))
    area = max(0., x1 - x0) * max(0., y1 - y0)
    return ratio(overlap, area, (r-l)*(b-t))


def box_edges(box):
    cx, cy, w, h = (np.float32(x) for x in box)
    two = np.float32(2.)
    return np.clip(np.float32(512.) * np.array((cx-w/two, cy-h/two, cx+w/two, cy+h/two)),0,512).tolist()


def native_extent(dense):
    ys, xs = np.nonzero(dense)
    return (xs.min(), ys.min(), xs.max()+1, ys.max()+1) if len(xs) else (0,0,0,0)


def additive_capacity(bind, args):
    rf, core, _, stream, _, _ = bind.validate(args)
    count, groups, categories = Counter(), defaultdict(list), Counter()
    for image_id, _, label, pred, audit in stream:
        masks = rf.native_masks(core, pred, audit, image_id)
        dense = np.moveaxis(codec.decode(masks), 2, 0).astype(bool)
        raw = [box_edges(row) for row in audit["raw_boxes_cxcywh"]]
        candidates = audit["candidates"]
        crop_admission = []
        for det, mask in zip(candidates, dense):
            area = int(mask.sum())
            l, t, r, b = (int(np.ceil(x)) for x in det["box_xyxy"])
            crop_stock = mask[t:b,l:r]
            n_inside = int(crop_stock.sum())
            crop_admission.append((area, l,t,r,b, crop_stock, n_inside, native_extent(mask)))
            count["candidates_with_fg_outside_box"] += n_inside < area
            count["candidates_with_forced_equal_area_infeasible"] += area > crop_stock.size

        for line in core.label_path(args.intake,label["label_ref"]).decode("ascii").splitlines():
            words = line.split()
            if not words or words[0] != "0":
                continue
            polygon = np.array(words[1:], dtype=float).reshape(-1,2) * 512
            rle, = codec.frPyObjects([polygon.reshape(-1).tolist()],512,512)
            truth = codec.decode(rle).astype(bool)
            gt_area = int(truth.sum())
            edges_polygon = tuple(polygon.min(axis=0)) + tuple(polygon.max(axis=0))
            edges_raster = native_extent(truth)
            best = dict(unfiltered_q164_box_only=max(box_extent_iou(box, edges_polygon) for box in raw))
            best_box = -1.
            corresponding_area = None
            for det, mask, (area, l,t,r,b, crop_stock, n_inside, stock_extent) in zip(candidates, dense, crop_admission):
                gt_crop = truth[t:b,l:r]
                hit = int(np.count_nonzero(mask[truth]))
                hit_in = int(np.count_nonzero(crop_stock[gt_crop]))
                gt_in = int(gt_crop.sum())
                inside_equal_overlap = hit - hit_in + min(n_inside, gt_in)
                inside_free_overlap = hit-hit_in+gt_in
                area_bound_intersection = min(area, gt_area)
                max_swap = min(8, area-hit, gt_area-hit)
                if not 32 <= area < 1024:
                    guarded = 0
                else:
                    guarded = max_swap
                box = box_extent_iou(det["box_xyxy"], edges_polygon)
                if box > best_box:
                    best_box, corresponding_area = box, area / gt_area
                vals = {
                    "box": box,
                    "mask": ratio(hit, area, gt_area),
                    "tight_mask_extent_box": box_extent_iou(stock_extent,edges_raster),
                    "area_only": ratio(area_bound_intersection,area,gt_area),
                    "crop_only": gt_in / gt_area,
                    "area_and_crop": ratio(min(area,gt_in),area,gt_area) if area <= gt_crop.size else 0.,
                    "up_to_eight_anywhere": ratio(hit+max_swap,area,gt_area),
                    "area_guard_and_up_to_eight_anywhere": ratio(hit+guarded,area,gt_area),
                    "within_box_equal_area_preserve_exterior": ratio(inside_equal_overlap,area,gt_area),
                    "within_box_free_area_preserve_exterior": ratio(inside_free_overlap,area-n_inside+gt_in,gt_area),
                }
                for k,v in vals.items():
                    best[k] = max(v, best.get(k, 0.))
            if best_box >= .5:
                categories["best_reported_box_passes_050"] += 1
                categories["associated_mask_area_less_than_half_truth"] += corresponding_area < .5
                categories["associated_mask_area_more_than_15_times_truth"] += corresponding_area > 1.5
            groups[label["acquisition_group"]].append(best)
    return count, groups, categories


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "intake", "reservation", "reference-diagnose", "frozen-scorer", "rf-scorer",
                 "primary", "primary-source", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    args = p.parse_args()
    source, previous = args.primary_source.read_bytes(), args.primary.read_bytes()
    if (sha256(source).hexdigest()!=PRIMARY_SOURCE or sha256(previous).hexdigest()!=PRIMARY_RESULT
            or args.output.exists()):
        raise ValueError("Primary authority differs or output is not fresh")
    spec = spec_from_file_location("bound_inputs", args.primary_source)
    bind = module_from_spec(spec)
    spec.loader.exec_module(bind)
    reference = json.loads(previous)
    outside, groups, underfill = additive_capacity(bind, args)
    if len(groups) != 18 or sum(len(x) for x in groups.values()) != 139:
        raise ValueError("Independent check did not retain the full cohort")
    for k,v in outside.items():
        if reference["counts"][k] != v:
            raise ValueError("Original mask extent census differs: "+k)
    checks = 0
    for key, metrics in reference["thresholds"].items():
        t = float(key)
        for metric, result in metrics.items():
            positives = [[(row[metric]>=t, row["mask"]>=t) for row in group] for group in groups.values()]
            flat = [pair for group in positives for pair in group]
            recomputed = (sum(x for x,y in flat), sum(any(x for x,y in g) for g in positives),
                          sum(x and not y for x,y in flat), sum(y and not x for x,y in flat))
            expected = tuple(result[name] for name in ("hulls", "groups", "hulls_added_vs_any_native_mask",
                                                        "hulls_missing_vs_any_native_mask"))
            if recomputed != expected:
                raise ValueError("Independent direct overlap disagrees at "+key+" / "+metric)
            group_mean = sum(sum(x for x,y in group) / len(group) for group in positives) / len(positives)
            if abs(group_mean - result["equal_acquisition_weight_fraction"]) > 1e-14:
                raise ValueError("Independent family-equal comparison differs")
            checks += 5
    result = dict(status="UEIKAP_RFDETR_INDEPENDENT_DENSE_GEOMETRY_COMPLETE", labeled_hulls=139,
                  acquisitions=18, verified_aggregate_cells=checks, primary_sha256=PRIMARY_RESULT,
                  independently_verified_marked_mask_extent=dict(outside), associated_mask_underfill=dict(underfill),
                  optimizer_updates=0, new_model_forwards=0, source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.mkdir()
    raw = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()
    (args.output / "RESULT_SAFE.json").write_bytes(raw)
    print(json.dumps(dict(status=result["status"], result_sha256=sha256(raw).hexdigest())),flush=True)


if __name__ == "__main__":
    main()
