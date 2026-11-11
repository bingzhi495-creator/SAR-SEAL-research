"""Audit original FIT training mask geometry on the full HRSID large-bearing cohort."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from statistics import fmean
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np
from pycocotools import mask as codec


P288 = "HRSID_FIT_LARGE_FOOTPRINT_20261111/large_fit_footprint.py"
P288_SHA = "321da3d5dffd77e4da33e539272322c748b861271ada9ba72788a1b427bffd9f"
OLD_RESULT_SHA = "c846323c4559d1e9bac864eed650cb0e205ef5499c24e7aa1f2aeb3f53052051"
PROTOCOL_SHA = "ca02731155e5e2034b92506d8e4858f1fe8dfe82bf03a4e454743a7025253a72"
GEOMETRY = "HRSID_FIT114_VECTOR_LABEL_GEOMETRY_20261110"
VECTOR_SHA = "4877c48d38a03e58e97163381ba544df3c90a78dde85c9141085fa0b92da3b09"
MATH_SHA = "0821d2cc3b6ca793d33fdfe35b634f3678aad072fc55b86352fa2501e969fc7e"
STATUS = "HRSID_FIT_LARGE_BEARING_VECTOR_SHAPE_COMPLETE"
ARMS = ("native", "center", "composed")
SIZES = ("small", "medium", "large")
PATHS = ("all", "direct", "crop", "crop_clipped", "crop_intact")
METRICS = ("iou_points", "fp_per_visible_gt_points", "fn_per_visible_gt_points",
           "centroid_abs_x_source_px", "centroid_abs_y_source_px", "centroid_l2_source_px")
TOL = 1e-10
MIB = 1024**2


def need(ok, message):
    if not ok:
        raise ValueError(message)


def pinned(path, expected, limit=MIB):
    need(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(), "Canonical closed input required")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    need(0 < len(raw) <= limit and sha256(raw).hexdigest() == expected, "Exact closed input bytes differ: " + path.name)
    return raw


def source(path, expected, name):
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(pinned(path, expected), str(path), "exec"), module.__dict__)
    return module


def geometry_sources(code):
    root = code / GEOMETRY
    return (source(root / "vector_label_geometry.py", VECTOR_SHA, "_fit_source_vector"),
            source(root / "vector_population_math.py", MATH_SHA, "_fit_horizontal_runs"))


def area_group(area):
    return "small" if area < 1024 else "medium" if area < 9216 else "large"


def encode(bitmap):
    return codec.encode(np.asfortranarray(bitmap, dtype=np.uint8))


def replay_one(runtime, image, target, seed):
    """Recover original A/B masks only through the validated augmentation seam."""
    observed = []
    stock = runtime.source

    def capture(row, *args, **kwargs):
        stock.target_summary(row, *args, **kwargs)
        observed.append(row)

    context = SimpleNamespace(_torchvision=stock._torchvision, set_seed=stock.set_seed, target_summary=capture)
    replay = runtime.replay.replay_view(runtime.audit, context, image, target, seed)
    need(len(observed) == 2, "Two original augmentations must be witnessed")
    live = []
    for item in observed:
        ids = item[runtime.replay.ROW_ID].numpy()
        masks = item["masks"].numpy()
        live.append({int(row_id): row for row_id, row in zip(ids, masks, strict=True) if bool(row.any())})
    a, b = live
    common = sorted(a.keys() & b.keys())
    need(len(common) == replay["rows"]["common_nonempty"], "Exact original common augmentation membership")
    composed = runtime.replay.center_reference(target["masks"].numpy()[common], replay["steps"])
    if replay["path"] == "direct":
        need(all(np.array_equal(b[row_id], item) for row_id,item in zip(common, composed, strict=True)),
             "Original composed and stepwise-center masks must match on a single resize")
    rows = [(key, {name:encode(row) for name,row in zip(ARMS, (a[key], b[key], c), strict=True)})
            for key,c in zip(common,composed,strict=True)]
    return replay["path"], replay["steps"], (set(a), set(b)), rows


def support_counts(count, a, b):
    common = a & b
    return dict(potential_rows=count, native_nonempty=len(a), center_nonempty=len(b),
                common_nonempty=len(common), native_only=len(a - b), center_only=len(b - a),
                neither=count - len(a | b))


def source_census(old, population, p288, views, support):
    planned = [[item["image"]["id"], item["ordinal"], item["prefix"], list(p288.view_seeds(item["ordinal"])),
                [item["annotations"][i]["id"] for i in item["large_indices"]]] for item in population]
    digest = sha256(p288.encoded(planned)).hexdigest()
    need(digest == old["design_sha256"] and dict(views) == old["views_by_path"],
         "Exact P288 large-bearing population, seed order and realized path counts required")
    for path in ("all", "direct", "crop"):
        historical = dict(old["support"][path])
        historical["potential_rows"] = historical.pop("potential_large_rows")
        need({key:value for key,value in support[path]["large"].items() if key != "composed_common_nonempty"}
             == historical, "Exact P288 original large A/B support required before geometry")
    return digest


def freeze(p288, runtime, train, layout_path, old):
    """Freeze all candidate bytes before any geometry measurements."""
    index_bytes = pinned(train / "_annotations.coco.json", p288.INDEX_SHA, 32*MIB)
    index = json.loads(index_bytes)
    layout = json.loads(pinned(layout_path, p288.LAYOUT_SHA, 2*MIB))
    need(len(index["images"]) == 2937 and len(index["annotations"]) == 9260
         and [row["id"] for row in index["categories"]] == [1]
         and layout["native_index_sha256"] == p288.INDEX_SHA and layout["native_index_bytes"] == len(index_bytes),
         "Exact closed FIT2937 source labels and layout required")
    population = p288.build_population(index, layout["images"])
    need(len(population) == 28 and sum(len(item["large_indices"]) for item in population) == 50
         and len({item["prefix"] for item in population}) == 4, "Complete 28-image P288 large FIT roster")
    views, original_counts = Counter(), Counter()
    views_with_common = {path:Counter() for path in ("direct","crop")}
    support = {path:{size:Counter() for size in SIZES} for path in ("all","direct","crop")}
    entries, reference = [], {}
    for item in population:
        image_info = item["image"]
        descriptor = item["layout"]
        data = pinned(train / image_info["file_name"], descriptor["encoded_sha256"], 4*MIB)
        image, scales = runtime.source.io_utils.decode_pil_image_bytes(data, draft_size=None)
        need(data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) == descriptor["encoded_bytes"]
             and image.mode == "RGB" and image.size == (800,800) and scales == (1.,1.)
             and sha256(np.asarray(image).tobytes(order="C")).hexdigest() == descriptor["decoded_rgb_sha256"],
             "Exact original FIT PNG and decoded RGB bytes required")
        image, target = p288.prepare_image(runtime, image, item)
        labels = item["annotations"]
        sizes = {key:{i for i,ann in enumerate(labels) if area_group(ann["area"]) == key} for key in SIZES}
        for i, ann in enumerate(labels):
            original_counts[area_group(ann["area"])] += 1
            reference[ann["id"]] = (ann.get("segmentation"), encode(target["masks"][i].numpy()))
        for seed in p288.view_seeds(item["ordinal"]):
            path, steps, (a,b), rows = replay_one(runtime, image, target, seed)
            views[path] += 1
            composed_live = {key for key, rles in rows if codec.area(rles["composed"]) > 0}
            for size, keys in sizes.items():
                tally = support_counts(len(keys), a & keys, b & keys)
                tally["composed_common_nonempty"] = len(composed_live & keys)
                views_with_common[path][size] += int(tally["common_nonempty"] > 0)
                support[path][size].update(tally)
                support["all"][size].update(tally)
            entries.extend(dict(annotation=labels[key]["id"], image=image_info["id"], prefix=item["prefix"],
                                size=area_group(labels[key]["area"]), path=path, steps=steps, rles=rles)
                           for key,rles in rows)
    need(len(entries) == sum(row["common_nonempty"] for row in support["all"].values()), "Complete frozen row accounting")
    digest = source_census(old, population, p288, views, support)
    return entries, reference, views, views_with_common, support, original_counts, digest


def polygon_union(rings, hw):
    import shapely
    from shapely.geometry import Polygon, box
    height, width = hw
    return shapely.union_all([Polygon(ring) for ring in rings]).intersection(box(0,0,width,height))


def measure(rings, hw, masks, inverse_yx, vector, math_api, *, dense):
    geometry = polygon_union(rings, hw)
    area = float(geometry.area)
    need(geometry.is_valid and area > 0., "Positive valid continuous reference required")
    geometry_center = (geometry.centroid.x, geometry.centroid.y)
    measured = math_api.run_footprint_metrics(rings, hw, masks)
    need(abs(measured["visible_polygon_area"] - area) <= 1e-8 * max(1.,area),
         "Exact vector reference and run geometry must agree")
    audit = None
    if dense:
        expected = vector.vector_footprint_metrics(rings, hw, masks)
        audit = max(abs(expected["by_arm"][arm][metric]-observed)
                    for arm, stats in measured["by_arm"].items() for metric, observed in stats.items())
        need(audit <= 1e-8, "Horizontal runs must match independent unit-cell intersections")
    result = {}
    sy, sx = inverse_yx
    for arm, bitmap in masks.items():
        yy, xx = np.nonzero(bitmap)
        need(len(yy)>0, "Common nonempty centroid must be defined")
        dx = abs((float(xx.mean())+.5-geometry_center[0]) * sx)
        dy = abs((float(yy.mean())+.5-geometry_center[1]) * sy)
        row = measured["by_arm"][arm]
        result[arm] = dict(iou_points=100*row["iou"], fp_per_visible_gt_points=100*row["fp_area"]/area,
                           fn_per_visible_gt_points=100*row["fn_area"]/area, centroid_abs_x_source_px=dx,
                           centroid_abs_y_source_px=dy, centroid_l2_source_px=math.hypot(dx,dy))
    return result, audit


def geometry_rows(entries, references, runtime, vector, math_api):
    eligible, skip = [], defaultdict(Counter)
    source, audits = {}, {}
    composed_empty = Counter()
    for entry in entries:
        aid, size, path = (entry[key] for key in ("annotation","size","path"))
        if aid not in source:
            segmentation, rle = references[aid]
            try:
                rings = vector.polygon_rings(segmentation)
                need(np.array_equal(vector.polygon_raster(rings, (800,800)), codec.decode(rle)),
                     "Original vector polygon must reproduce the original prepared training mask")
                source[aid] = (polygon_union(rings,(800,800)).area, None)
            except vector.UnsupportedPolygon as error:
                source[aid] = (None, str(error))
        source_area, reason = source[aid]
        if reason is None:
            try:
                segmentation, _ = references[aid]
                rings, hw = vector.final_polygons(segmentation, (800,800), entry["steps"])
                visible = polygon_union(rings, hw)
                if visible.area <= 0 or not visible.is_valid:
                    reason = "no_visible_valid_vector_area"
            except vector.UnsupportedPolygon as error:
                reason = str(error)
        if reason is not None:
            skip[path, size][reason] += 1
            continue
        masks = {arm:codec.decode(rle).astype(bool) for arm,rle in entry["rles"].items() if codec.area(rle) > 0}
        need(set(ARMS[:2]) <= masks.keys(), "Original A/B common masks must be nonempty")
        if "composed" not in masks:
            composed_empty[path,size] += 1
        inverse = runtime.replay.inverse_scales(entry["steps"])
        source_visible = visible.area * abs(inverse[0]*inverse[1])
        need(source_visible <= source_area+1e-7*max(1.,source_area), "Visible vector area cannot increase under crop")
        value, checked = measure(rings, hw, masks, inverse, vector, math_api, dense=(path,size) not in audits)
        if checked is not None:
            audits[path,size] = checked
        eligible.append(dict(annotation=aid, image=entry["image"], prefix=entry["prefix"], size=size, path=path,
                             crop_clipped=path == "crop" and source_visible < source_area-1e-8*max(1.,source_area),
                             values=value))
    need(len(eligible)+sum(sum(count.values()) for count in skip.values()) == len(entries),
         "All A/B common retained rows must be measured or explicitly excluded")
    return eligible, skip, composed_empty, audits


def aggregate(rows, a, b, metric):
    pairs = [row for row in rows if a in row["values"] and b in row["values"]]
    groups, annotations = defaultdict(list), defaultdict(list)
    effects = []
    for row in pairs:
        effect = row["values"][b][metric]-row["values"][a][metric]
        groups[row["prefix"]].append((row,effect))
        annotations[row["annotation"]].append(effect)
        effects.append(effect)
    means = {arm:fmean(row["values"][arm][metric] for row in pairs) if pairs else None for arm in (a,b)}
    group_means = [fmean(effect for _,effect in group) for group in groups.values()]
    leave = [fmean(effect for other, group in groups.items() if other != omitted for _,effect in group)
             for omitted in groups if len(groups) > 1]
    row_effect = fmean(effects) if pairs else None
    ann_effect = fmean(fmean(group) for group in annotations.values()) if pairs else None
    prefix_effect = fmean(group_means) if pairs else None
    signs = [row_effect, ann_effect, prefix_effect, *leave] if leave else []
    direction = ("positive" if signs and all(x>TOL for x in signs) else "negative"
                 if signs and all(x<-TOL for x in signs) else "mixed_or_indeterminate")
    return dict(rows=len(pairs), unique_annotations=len(annotations), unique_images=len({row["image"] for row in pairs}),
                prefix_groups=len(groups), arm_row_means=means, row_equal_delta=row_effect,
                annotation_equal_delta=ann_effect, prefix_equal_delta=prefix_effect,
                leave_one_prefix_out_min_delta=min(leave) if leave else None,
                leave_one_prefix_out_max_delta=max(leave) if leave else None, consistent_direction=direction,
                positive_rows=sum(int(x>TOL) for x in effects), negative_rows=sum(int(x<-TOL) for x in effects),
                tied_rows=sum(int(abs(x)<=TOL) for x in effects),
                prefix_proxy_summaries=[dict(rows=len(group), delta=fmean(effect for _,effect in group),
                                              images=len({row["image"] for row,_ in group}),
                                              annotations=len({row["annotation"] for row,_ in group}))
                                        for group in groups.values()])


def groups(eligible):
    result = {}
    for path in PATHS:
        result[path] = {}
        for size in SIZES:
            selected = [row for row in eligible if row["size"] == size and (
                path == "all" or row["path"] == path or (row["path"] == "crop"
                and path == ("crop_clipped" if row["crop_clipped"] else "crop_intact")))]
            complete = [row for row in selected if "composed" in row["values"]]
            result[path][size] = dict(
                native_center_full={key:aggregate(selected,"native","center",key) for key in METRICS},
                same_three_arm_support={b+"_minus_"+a:{key:aggregate(complete,a,b,key) for key in METRICS}
                                        for a,b in (("native","center"),("native","composed"),("center","composed"))})
    return result


def main(argv=None):
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code", "train", "layout", "old-result", "protocol", "shapely-site", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args(argv)
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
         and os.environ.get("OMP_NUM_THREADS") == os.environ.get("OPENBLAS_NUM_THREADS") == "1",
         "Exact isolated one-thread CPU required")
    need(args.train.is_absolute() and args.train.resolve(strict=True) == args.train and args.train.name == "train"
         and args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent
         and not args.output.is_relative_to(args.train.parent) and not args.output.exists() and not args.output.is_symlink(),
         "Exact closed FIT training directory and fresh separate result required")
    pinned(args.protocol, PROTOCOL_SHA)
    old = json.loads(pinned(args.old_result, OLD_RESULT_SHA))
    need(old["status"] == "HRSID_FIT_LARGE_FOOTPRINT_PAIRED_ACTUAL_V1"
         and old["FIT"] is True and old["DEV"] is old["TEST"] is old["AP"] is False,
         "Exact closed preceding FIT-only P288 support ledger required")
    started = perf_counter()
    sys.path.append(str(args.shapely_site.resolve(strict=True)))
    import shapely
    need(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2",
         "Exact preceding continuous geometry runtime required")
    p288 = source(args.code/P288, P288_SHA, "_closed_p288_population")
    runtime = p288.load_runtime(args.code)
    vector, math_api = geometry_sources(args.code)
    entries, references, views, views_with_common, support, census, digest = freeze(
        p288, runtime, args.train, args.layout, old)
    print(json.dumps(dict(event="HRSID_FIT_LARGE_AB_CENSUS_REPRODUCED_BEFORE_VECTOR",
                          views=sum(views.values()), common_large=support["all"]["large"]["common_nonempty"])),
          flush=True)
    eligible, skip, composed_empty, audits = geometry_rows(entries, references, runtime, vector, math_api)
    torch = runtime.source.torch
    need(not torch.cuda.is_initialized() and torch.get_num_threads() == torch.get_num_interop_threads() == 1
         and len(audits) > 0, "Exact no-CUDA geometry and independent witness required")
    results = groups(eligible)
    large = results["all"]["large"]["native_center_full"]["iou_points"]
    payload = dict(status=STATUS, protocol_sha256=PROTOCOL_SHA, original_p288_result_sha256=OLD_RESULT_SHA,
                   original_p288_source_sha256=P288_SHA, original_design_sha256=digest,
                   fit_index_sha256=p288.INDEX_SHA, fit_layout_sha256=p288.LAYOUT_SHA,
                   vector_source_sha256=VECTOR_SHA, horizontal_run_source_sha256=MATH_SHA,
                   FIT=True, DEV=False, TEST=False, AP=False, model_forward_calls=0, optimizer_updates=0,
                   images=28, total_views=sum(views.values()), views_by_realized_path=dict(views),
                   realized_views_with_common_support=views_with_common,
                   crop_row_definition="continuous visible area below original canvas-clipped vector area",
                   original_labels_by_size=census, original_prefix_proxies=4, source_area_edges=[1024,9216],
                   source_area_units="original COCO annotation pixels squared",
                   raw_common_support=support, primary_large_paired_iou_direction=large["consistent_direction"],
                   vector_exclusions=[dict(path=path,size=size,counts=counts) for (path,size),counts in sorted(skip.items())],
                   vector_eligible_composed_empty=[dict(path=path,size=size,count=count)
                                                   for (path,size),count in sorted(composed_empty.items())],
                   dense_witness=[dict(path=path,size=size,max_abs_native_geometry_difference=error)
                                  for (path,size),error in sorted(audits.items())],
                   results=results, elapsed_seconds=perf_counter()-started)
    raw = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=STATUS, result_sha256=sha256(raw).hexdigest())), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
