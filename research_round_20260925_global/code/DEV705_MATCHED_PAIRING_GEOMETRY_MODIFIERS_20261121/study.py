"""Prespecified area-adjusted width and neighboring-instance analysis of closed DEV705 masks."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
from pycocotools import mask as codec

D = Path(os.environ["SEAL_DEV705_SOURCE_ROOT"]).resolve(strict=True)
R = D / "github_release_20261011/research_round_20260925_global"
SRC = R / "code"
GT = "HRSID_OFFICIAL_PNG_DEV705_AUDIT_ACTUAL_PRIVATE_20261109_01/DEV_ORIGINAL_GT_PRIVATE.json"
DF_EDITS = "HRSID_DFINE_E24_DEV705_FREE_ENDPOINT_DRAFT_20261112_01/PRELABEL_ACTUAL/EDITS_PRIVATE.jsonl"
DF_NATIVE = "HRSID_DFINE_E24_DEV705_OPERATION_PRIVATE_20261112_01/CONTROL/EXPORT/DEV_MASK_PREDICTIONS_PRIVATE.jsonl"
DF_SCORE = "DFINE_DEV705_FREE_ENDPOINT_INDEPENDENT_SCORE_20261120_01/RESULT_SAFE.json"
DF_WIT = "DFINE_DEV705_FREE_ENDPOINT_INDEPENDENT_SCORE_20261120_01/STRICT_TRANSITIONS_PRIVATE.jsonl"
RF_WIT = "RFDETR_CENTER5202_DEV705_MATCHED_SCORE_ACTUAL_20261120_01/SHAPE_MATCH_WITNESS_PRIVATE.json"
RF_EDITS = "RFDETR_CENTER5202_DEV705_MATCHED_READOUT_ACTUAL_20261120_01/SPARSE_EDITS_PRIVATE.jsonl"
RF_NATIVE = "HRSID_MASK_COMMON_FIT_E24_DEV705_5202_PRIVATE_20261110_01/EXPORT_CENTER_ACTUAL_PRIVATE/DEV_NATIVE_PREDICTIONS_PRIVATE.jsonl"
RF_SCORE = "github_release_20261011/research_round_20260925_global/results/RFDETR_CENTER5202_DEV705_MATCHED_SCORE_PUBLIC_SAFE_20261120.json"
BIND = {
    GT: "393b3da233991dc5fafc98b4ddeaa341b042d9571434ef8ce53c3f2fdada5628",
    DF_WIT: "e1e62da5ee8dab72b70ab5d68ecf5dfd2e8f180d47a9643ab6f39557b5018432",
    DF_EDITS: "d81ef0f03a297ae983cb6a9c13a7e45cada611a2c4fcdb46956957c2c201ad15",
    DF_NATIVE: "80382f6f7318ac7a033e4d8169cf278322a9d51731f2d86e11b5f426d318d0ed",
    DF_SCORE: "e32f2e6f45fb2e1a777765a57b021877e0803588285c6868aa2a915c5c6542e8",
    RF_WIT: "cdd238a5bea193635a063b7ff8ce22c522097bf30f76e033648006fc7b26eded",
    RF_EDITS: "f8c20ce94715e651a1e5cd90e2213a3a2793c2b9454de45de212d07c69335a6b",
    RF_NATIVE: "da55f2d3f9bbf08b26d5fae76606b2496a5b482c5fb61f4fab53c23172e5c06e",
    RF_SCORE: "c0be05b24ca1bb23283ac19043b2ad96bbf2db39bc0ef525e1ea41a3dc9316e9",
}
PROTOCOL = "acdcf61fe49d8f3a0a4f03f6627f94bd3a1f6af86229279efc09646a487c6a6f"


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def source(path, expected):
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == expected, "Pinned source " + path.name)
    obj = ModuleType(path.stem)
    obj.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), obj.__dict__)
    return obj


def read_json(key):
    raw = (D/key).read_bytes()
    require(sha256(raw).hexdigest() == BIND[key], key)
    return json.loads(raw)


def stream(key):
    digest = sha256()
    with (D/key).open("rb") as f:
        for line in f:
            require(line.endswith(b"\n"), "Complete original row")
            digest.update(line)
            yield json.loads(line)
    require(digest.hexdigest() == BIND[key], key)


def geometry_metadata(gt, helpers, shp):
    shapes, neutral, _ = helpers.geometry.polygon_support(gt, helpers.vector)
    anns = {a["id"]: a for a in gt["annotations"]}
    groups = {r["id"]: Path(r["file_name"]).stem.split("_")[0] for r in gt["images"]}
    by_image = defaultdict(list)
    for aid in shapes:
        by_image[anns[aid]["image_id"]].append(aid)
    meta = {}
    for im, ids in by_image.items():
        for aid in ids:
            own = shapes[aid][0]
            perimeter = np.asarray(own.minimum_rotated_rectangle.exterior.coords)
            width = float(np.linalg.norm(np.diff(perimeter, axis=0), axis=1).min())
            others = [shapes[j][0] for j in ids if j != aid]
            distance = min((own.distance(g) for g in others), default=None)
            state = "singleton" if distance is None else ("near" if distance <= 4 else "far")
            exclusive = shp.union_all(others).difference(own) if state == "near" else None
            require(width > 0 and own.area > 0, "Valid continuous width and area")
            meta[aid] = dict(gt_id=aid, image=im, group=groups[im], width=width, area=own.area,
                             distance=distance, near=state, own=own, other=exclusive)
    return meta, len(neutral)


def cell_occupancy(flat, polygon, shp):
    if not len(flat) or polygon.is_empty:
        return 0.
    ys, xs = np.divmod(np.asarray(flat, dtype=np.int64), 800)
    boxes = shp.box(xs, ys, xs+1, ys+1)
    return float(shp.area(shp.intersection(boxes, polygon)).sum())


def incidence(data, adds, drops, shp):
    return np.array([cell_occupancy(adds, data["other"], shp), cell_occupancy(drops, data["other"], shp)])


def base_row(data, effects, overlaps):
    return dict(image=data["image"], group=data["group"], width=data["width"], area=data["area"],
                near=data["near"], delta=float(np.mean(effects)), spill=overlaps)


def rfd_rows(metadata, shp):
    grouped = defaultdict(lambda: defaultdict(dict))
    for wi in read_json(RF_WIT):
        store = grouped[wi["image_id"]][wi["rank"]]
        require(wi["seed"] not in store, "Unique original RF seed")
        store[wi["seed"]] = wi
    output = []
    for sparse, native in zip(stream(RF_EDITS), stream(RF_NATIVE), strict=True):
        im = native["image_id"]
        require(im == sparse["image_id"], "Aligned RF images")
        pts = {p["rank"]: p for p in sparse["points"]}
        indices = {rank: j for j, rank in enumerate(native["native_ranks"])}
        for rank, by_seed in grouped.pop(im, {}).items():
            require(set(by_seed) == {5217,5218,5219}, "Three head seeds per object")
            ref = by_seed[5217]
            data = metadata[ref["matched_gt_id"]]
            require(data["image"] == im and data["group"] == ref["group"] == sparse["group"], "Original RF object")
            effects, overlaps = [], []
            if data["near"] == "near":
                pred = native["coco"][indices[rank]]
                mask = codec.decode(pred["segmentation"]).ravel(order="C").astype(bool)
            for seed, witness in by_seed.items():
                require((witness["matched_gt_id"],witness["query_id"]) == (ref["matched_gt_id"], ref["query_id"]),
                        "Cross-seed original identity")
                arms = witness["arms"]
                require(abs(arms["N"]["T"]-data["area"]) < 1e-8, "Exact truth-area join")
                require(arms["SU=F"]["edits"] == arms["SF"]["edits"], "Original RF matched budget")
                effects.append(100*(arms["SU=F"]["iou"]-arms["SF"]["iou"]))
                if data["near"] == "near":
                    edits = sparse["seeds"][str(seed)].get(str(rank), {})
                    costs = []
                    for arm in ("SU=F", "SF"):
                        flipped = np.array(edits.get(arm, {}).get("flipped", []), dtype=np.int32)
                        require(len(flipped) == arms[arm]["edits"] and len(np.unique(flipped)) == len(flipped),
                                "Validated RF flips")
                        if len(flipped):
                            require(pts[rank]["query_id"] == ref["query_id"], "Actual RF query")
                        adds, drops = flipped[~mask[flipped]], flipped[mask[flipped]]
                        require(len(adds) == len(drops) == arms[arm]["adds"], "Balanced RF cell signs")
                        costs.append(incidence(data, adds, drops, shp))
                    overlaps.append(costs)
            output.append(base_row(data, effects, np.mean(overlaps, axis=0) if overlaps else None))
    require(not grouped and len(output) == 1461, "Complete RF fixed-match scope")
    return output


def dfine_rows(meta, helpers, scorer, shp):
    by_image = defaultdict(list)
    for rec in stream(DF_WIT):
        by_image[rec["image_id"]].append(rec)
    output = []
    for native, edited in zip(stream(DF_NATIVE), stream(DF_EDITS), strict=True):
        im = native["image_id"]
        require(im == edited["image_id"], "Aligned D-FINE images")
        for rec in by_image.pop(im, []):
            i = rec["native_rank"]
            data = meta[rec["original_gt_id"]]
            require(data["image"] == im, "Original D-FINE identity")
            stock_rle = native["coco"][i-1]["segmentation"]
            area = int(codec.area(stock_rle))
            changes = edited["changes"].get(str(i), {})
            effects, costs, flips_count = [], [], []
            if data["near"] == "near":
                stock = codec.decode(stock_rle).ravel(order="C").astype(bool)
            for arm in ("free_matched_count", "fixed"):
                edit = changes.get(arm, {})
                rle = edit.get("rle", stock_rle)
                k = edit.get("swaps", 0)
                require(k == rec["arms"][arm]["swaps"] and int(codec.area(rle)) == area, "D-FINE swaps and area")
                effects.append(scorer.continuous(rle, area, data["own"], helpers)[0])
                flips_count.append(k)
                if data["near"] == "near":
                    after = codec.decode(rle).ravel(order="C").astype(bool) if k else stock
                    add = np.flatnonzero(after & ~stock)
                    drop = np.flatnonzero(stock & ~after)
                    require(len(add) == len(drop) == k, "Balanced D-FINE cell signs")
                    costs.append(incidence(data, add, drop, shp))
            require(flips_count[0] == flips_count[1], "Same D-FINE per-candidate budget")
            output.append(base_row(data, [effects[0]-effects[1]], np.asarray(costs) if costs else None))
    require(not by_image and len(output) == 1447, "Complete D-FINE fixed-match scope")
    return output


def group_mean(rows, fn):
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(fn(r))
    return float(np.mean([np.mean(v) for v in groups.values()])) if groups else None


def describe(rows):
    return dict(matches=len(rows), images=len({r["image"] for r in rows}), prefixes=len({r["group"] for r in rows}),
                group_equal_delta_points=group_mean(rows, lambda r:r["delta"]))


def summarize(rows, keys, draws, expected, expected_ci=None):
    partitions = [[r for r in rows if r["group"] == key] for key in keys]
    require(len(partitions) == 27 and all(partitions), "27 paired source prefixes")
    mean = np.array([np.mean([r["delta"] for r in part]) for part in partitions])
    ci = np.percentile(mean[draws].mean(axis=1), [2.5, 97.5]).tolist()
    require(abs(mean.mean()-expected)<1e-9 and (expected_ci is None or np.allclose(ci,expected_ci,rtol=0,atol=1e-9)),
            "Original historical group-equal contrast replay")
    grams, cross = [], []
    for group in partitions:
        x = np.asarray([[1.,np.log2(r["width"]),np.log2(r["area"])] for r in group])
        y = np.asarray([r["delta"] for r in group])
        grams.append(x.T@x/len(group))
        cross.append(x.T@y/len(group))
    gram, rhs = np.array(grams), np.array(cross)
    coef = np.linalg.solve(gram.mean(axis=0), rhs.mean(axis=0))
    boots = np.linalg.solve(gram[draws].mean(axis=1), rhs[draws].mean(axis=1))
    require(np.isfinite(boots).all(), "All 4096 refits identifiable")
    filters = {"width_le_4":lambda x:x["width"]<=4, "width_gt_4":lambda x:x["width"]>4,
               "area_lt_1024":lambda x:x["area"]<1024, "area_1024_to_lt9216":lambda x:1024<=x["area"]<9216,
               "area_ge_9216":lambda x:x["area"]>=9216, **{k:lambda x,k=k:x["near"]==k for k in ("near","far","singleton")}}
    strata = {k:describe([r for r in rows if fn(r)]) for k,fn in filters.items()}
    near = [r for r in rows if r["near"]=="near"]
    require(all(r["spill"] is not None for r in near), "All near-object edited cells audited")
    incidence_stats = {}
    for j, arm in enumerate(("independent","fixed")):
        incidence_stats[arm] = dict(added=group_mean(near, lambda r:r["spill"][j,0]),
                                    removed=group_mean(near, lambda r:r["spill"][j,1]),
                                    net=group_mean(near, lambda r:r["spill"][j,0]-r["spill"][j,1]))
    def delta_incur(r):
        x = r["spill"]
        return float((x[0,0]-x[0,1])-(x[1,0]-x[1,1]))
    inc = [delta_incur(r) for r in near]
    incidence_stats.update(paired_net_change_sqpx=group_mean(near,delta_incur), increased=sum(x>1e-12 for x in inc),
                           decreased=sum(x< -1e-12 for x in inc), unchanged=sum(abs(x)<=1e-12 for x in inc))
    return dict(population=describe(rows), historical_contrast=expected, historical_parity=True,
                original_contrast_interval95_points=ci, slope_points_per_width_doubling=float(coef[1]),
                slope_interval95=np.percentile(boots[:,1],[2.5,97.5]).tolist(),
                log2_area_adjustment_coefficient=float(coef[2]), bootstrap_replicates=len(boots),
                strata=strata, nearby_other_instance_exclusive=incidence_stats)


def main():
    sys.path.append(str(D/"HRSID_FIT_FROZEN_VECTOR_OCCUPANCY_DRAFT_PRIVATE_20261111_01/shapely_cp310_runtime"))
    import shapely as shp
    require(shp.__version__ == "2.0.2" and shp.geos_version_string == "3.11.2", "Historical polygon engine")
    here = Path(__file__).resolve().parent
    require(sha256((here/"PROTOCOL.md").read_bytes()).hexdigest() == PROTOCOL, "Frozen study protocol")
    engine = source(SRC/"MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py",
                    "7cad8b51a9230101e2233db8455f84c28163abfb2ee8ddd8cbe8718a2e719bd0")
    scorer = source(SRC/"DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/dev705_free_endpoint/score_endpoint.py",
                    "77159ea493a95973fe422dac1d303eaeba603e6d8af7ae179c72aecc2e4301b0")
    helpers = engine.geometry_modules(SRC)
    gt = read_json(GT)
    meta, neutral = geometry_metadata(gt, helpers, shp)
    rf = rfd_rows(meta, shp)
    df = dfine_rows(meta, helpers, scorer, shp)
    rf_expected = read_json(RF_SCORE)["geometry_summary"]["means"]
    df_expected = read_json(DF_SCORE)["geometry_points"]["comparisons"]["free_matched_count_minus_fixed"]
    keys = sorted({r["group"] for r in rf})
    require(set(keys) == {r["group"] for r in df}, "Same 27 source-prefix bootstrap keys")
    draws = np.random.Generator(np.random.PCG64(5217)).integers(len(keys),size=(4096,len(keys)))
    result = dict(status="DEV705_EQUAL_COUNT_WIDTH_AND_NEIGHBOR_MODIFIERS_ACTUAL_V1", DEV=True, FIT=False, TEST=False,
                  protocol_sha256=PROTOCOL, source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  prefix_draws_sha256=sha256(draws.astype("<i8").tobytes()).hexdigest(), source_prefixes=len(keys),
                  annotation_count=len(gt["annotations"]), valid_reference_polygons=len(meta), neutral_annotation_count=neutral,
                  rf=summarize(rf, keys, draws, rf_expected["SU=F"]["group_equal_iou_points"]-
                               rf_expected["SF"]["group_equal_iou_points"]),
                  dfine=summarize(df, keys, draws, df_expected["delta_points"], df_expected["interval95_points"]))
    data=(json.dumps(result,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
    with (here/"RESULT_SAFE.json").open("xb") as writer:
        writer.write(data)
        writer.flush()
        os.fsync(writer.fileno())
    print(json.dumps(dict(status=result["status"],result_sha256=sha256(data).hexdigest())),flush=True)


if __name__ == "__main__":
    main()
