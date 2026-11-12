"""Read-only independent RLE membership audit of exact closed HRSID FIT36."""
import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from hashlib import sha256
from importlib.metadata import version
from io import StringIO
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

PINS = {
    "result": "17b32955ff8076ab71215fd4fc65c5bfc6fe510afbd6a46605506e3551d12346",
    "receipt": "d2beea23b7fd1436499962a96d4ce537df200c4099e753d29359feb8a71bd211",
    "variants": "e7470b57a15815c82ab9b252a18bd0772781f7ca0aaa5f8784207eb2d79d7c3e",
    "capture": "be37834f7ec657c938e15326ba04a9ac4a6dedaa4bf7a16c173490e93eb49cba",
    "anchor": "d089c9802a8d6009b3951a62c44f87e4c73cda4d1fbeef7f4544da2aa1b3698e",
    "label": "e95cf26cc60a2c6fb651e907541380a256f2df7b1e153244a6c0f803b0b19f8f",
    "geometry": "7cad8b51a9230101e2233db8455f84c28163abfb2ee8ddd8cbe8718a2e719bd0",
    "stock": "12fd1684b262a3c9b6732b90a0c1e624034c523a7d130919a079e17728f80075",
}
CATEGORIES = ("fixed_valid_small", "other_original_match", "unmatched")
EDIT_KEYS = ("total_edits", "distinct", "new_adds", "new_drops", "xor_pixels")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def exact(path, digest):
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file()
            and 0 < path.stat().st_size <= 64 * 2**20, "Bounded canonical file")
    value = path.read_bytes()
    require(sha256(value).hexdigest() == digest, "Exact pin: " + path.name)
    return value


def write(path, value):
    data = (json.dumps(value, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(data)
    return sha256(data).hexdigest()


def module(path, name, pin):
    obj = ModuleType(name)
    obj.__file__ = str(path)
    sys.modules[name] = obj
    exec(compile(exact(path, PINS[pin]), str(path), "exec"), obj.__dict__)
    return obj


def stock_assignment(image, annotations, candidates):
    gt, dt = COCO(), COCO()
    category = [{"id": 1, "name": "ship"}]
    gt.dataset = dict(images=[image], categories=category, annotations=[dict(a) for a in annotations])
    dt.dataset = dict(images=[image], categories=category, annotations=[
        dict(id=r["rank"], image_id=image["id"], category_id=1, segmentation=r["rle"],
             score=r["score"], area=r["area"], iscrowd=0) for r in candidates])
    gt.createIndex()
    dt.createIndex()
    ev = COCOeval(gt, dt, "segm")
    ev.params.iouThrs = np.array([.50])
    ev._prepare()
    key = (image["id"], 1)
    ev.ious[key] = ev.computeIoU(*key)
    scored = ev.evaluateImg(*key, ev.params.areaRng[0], 100)
    require(scored["dtIds"] == [r["rank"] for r in candidates], "Stable original candidate order")
    by_rank = {int(rank): int(identity) for rank, identity, ignored in
               zip(scored["dtIds"], scored["dtMatches"][0], scored["dtIgnore"][0], strict=True)
               if identity and not ignored}
    return by_rank, {identity: gt.anns[identity]["segmentation"] for identity in by_rank.values()}


def prepare(args):
    result = json.loads(exact(args.closed / "RESULT_SAFE.json", PINS["result"]))
    pre = json.loads(exact(args.closed / "PRELABEL_SAFE.json", PINS["receipt"]))
    capture = json.loads(exact(args.capture / "RESULT_SAFE.json", PINS["capture"]))
    require(result["status"] == "HRSID_FIT_SIGNED_DISTANCE_FUSION_ACTUAL"
            and result["FIT"] and result["GT_opened"] and not pre["GT_opened"]
            and not any(result[k] for k in ("DEV", "TEST", "logit_planes_opened"))
            and all(result[k] == v for k, v in pre.items() if k not in ("status", "GT_opened"))
            and pre["variants_sha256"] == PINS["variants"], "Closed FIT variant admission")
    anchor, = [x for x in capture["arms"] if x["seed"] == 5202]
    require(capture["FIT"] and not any(capture[k] for k in ("DEV", "TEST", "GT_opened"))
            and anchor["prediction_sha256"] == PINS["anchor"], "Closed FIT anchor admission")
    original = [json.loads(s) for s in exact(args.capture / "5202/CANDIDATES_PRIVATE.jsonl", PINS["anchor"]).splitlines()]
    edited = [json.loads(s) for s in exact(args.closed / "VARIANTS_PRIVATE.jsonl", PINS["variants"]).splitlines()]
    require(len(original) == len(edited) == 36 and len({r["group"] for r in original}) == 36
            and sum(len(r["candidates"]) for r in original) == 2620, "Exact 36-image cohort and candidates")
    observations, tally = [], Counter()
    for source, variant in zip(original, edited, strict=True):
        require(source["basename"] == variant["basename"], "Exact image join")
        roster = {r["rank"]: r for r in source["candidates"]}
        entries = {}
        for rank_text, row in variant["variants"].items():
            rank = int(rank_text)
            r0, r1, r2 = roster[rank]["rle"], row["sdf_both"], row["sdf_any"]
            require(r0["size"] == r1["size"] == r2["size"] == [800, 800], "Native mask dimensions")
            a, b, c = [mask.decode(r).astype(bool) for r in (r0, r1, r2)]
            old, new = a ^ b, a ^ c
            adds, drops = int((old & ~a).sum()), int((old & a).sum())
            require(adds == int((new & ~a).sum()) and drops == int((new & a).sum())
                    and 0 < adds + drops <= 16, "Exact same per-mask sign counts")
            incoming = new & ~old
            new_adds, new_drops = int((incoming & ~a).sum()), int((incoming & a).sum())
            xor_pixels = int((b ^ c).sum())
            require(xor_pixels == 2 * (new_adds + new_drops), "Paired edit replacements")
            row_counts = dict(total_edits=1, distinct=int(xor_pixels > 0), new_adds=new_adds,
                              new_drops=new_drops, xor_pixels=xor_pixels)
            tally.update(row_counts)
            tally.update(adds=adds, drops=drops)
            entries[rank] = (row_counts, r1, r2)
        observations.append((source, entries))
    require(dict(tally) == dict(total_edits=464, adds=2932, drops=3677, distinct=59,
                               new_adds=24, new_drops=103, xor_pixels=254), "Exact aggregate RLE replay")
    return result, observations, dict(tally)


def score(args, result, observations, geo, stock):
    fit = json.loads(exact(args.fit_index, PINS["label"]))
    require(len(fit["images"]) == 2937 and len(fit["annotations"]) == 9260, "Exact FIT label census")
    names = {source["basename"] for source, _ in observations}
    images = {r["file_name"]: r for r in fit["images"] if r["file_name"] in names}
    ids = {r["id"] for r in images.values()}
    require(set(images) == names and len(ids) == 36, "Exact FIT label join")
    annotations = defaultdict(list)
    for row in fit["annotations"]:
        if row["image_id"] in ids:
            annotations[row["image_id"]].append(row)
    subset = dict(images=list(images.values()), categories=fit["categories"],
                  annotations=[r for rows in annotations.values() for r in rows])
    supported, neutral, _ = geo.geometry.polygon_support(subset, geo.vector)
    counts = {k: Counter() for k in CATEGORIES}
    groups = {k: set() for k in CATEGORIES}
    contrast = {"raster": Counter(), "continuous": Counter()}
    all_stock, fixed_count = 0, 0
    fixed_groups = set()
    for source, entries in observations:
        image = images[source["basename"]]
        pool = source["candidates"]
        truth = annotations[image["id"]]
        with redirect_stdout(StringIO()):
            by_rank, rasters = stock_assignment(image, truth, pool)
            prior, _, _ = stock.freeze_coco(image, truth, pool)
        require(by_rank == {rank: identity for identity, rank in prior.items()}, "Independent exact fixed assignment")
        all_stock += len(by_rank)
        fixed = {r["rank"] for r in pool if 32 <= r["area"] < 1024 and r["rank"] in by_rank
                 and by_rank[r["rank"]] in supported and by_rank[r["rank"]] not in neutral}
        fixed_count += len(fixed)
        if fixed:
            fixed_groups.add(source["group"])
        for rank, (edits, both, any_) in entries.items():
            part = "fixed_valid_small" if rank in fixed else "other_original_match" if rank in by_rank else "unmatched"
            counts[part].update(edits)
            if not edits["distinct"]:
                continue
            groups[part].add(source["group"])
            if part != "fixed_valid_small":
                continue
            identity = by_rank[rank]
            polygon = supported[identity][0]
            raster_i, continuous_i = [], []
            for rle in (both, any_):
                raster_i.append(int(mask.area(mask.merge([rle, rasters[identity]], intersect=True))))
                boxes = geo.geometry.boxes_from_counts(geo.shift.runs(rle["counts"]), *rle["size"])
                continuous_i.append(geo.geometry.overlap_area(boxes, polygon))
            for key, intersections, tol in (("raster", raster_i, 0), ("continuous", continuous_i, 1e-10)):
                diff = intersections[1] - intersections[0]
                contrast[key]["equal" if abs(diff) <= tol else "higher_any" if diff > 0 else "lower_any"] += 1
    require(all_stock == result["edited_candidate_scope"]["stock_matches"] == 86
            and fixed_count == result["fixed_geometry_pairs"] == 69 and len(fixed_groups) == 31,
            "Exact original and fixed geometry denominators")
    require(tuple(counts[k]["total_edits"] for k in CATEGORIES) == (68, 0, 396), "Exact total edit membership")
    return dict(stock_matches=all_stock, fixed_valid_small_matches=fixed_count, fixed_valid_small_groups=len(fixed_groups),
                membership={k: {n: counts[k][n] for n in EDIT_KEYS} | {"distinct_groups": len(groups[k])} for k in CATEGORIES},
                changed_fixed_intersections={k: {n: c[n] for n in ("equal", "higher_any", "lower_any")}
                                             for k, c in contrast.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("protocol", "closed", "capture", "fit-index", "code", "shapely-site", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--protocol-sha", required=True)
    args = parser.parse_args()
    lock = json.loads(exact(args.protocol, args.protocol_sha))
    require(lock["pins"] == PINS and lock["source_sha256"] == sha256(Path(__file__).read_bytes()).hexdigest(),
            "Prelocked source and input digests")
    require(version("pycocotools") == "2.0.11", "Exact stock runtime")
    sys.path.append(str(args.shapely_site.resolve(strict=True)))
    import shapely
    require((shapely.__version__, shapely.geos_version_string) == ("2.0.2", "3.11.2"), "Exact polygon geometry runtime")
    geo = module(args.code / "MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py",
                 "_verify_sdf_geometry", "geometry").geometry_modules(args.code)
    stock = module(args.code / "HRSID_FIT_HULL_CDF_20261112/stock_coco_match.py", "_verify_sdf_stock", "stock")
    result, observations, tally = prepare(args)
    args.output.mkdir()
    receipt = dict(status="HRSID_FIT_SDF_MEMBERSHIP_PRELABEL", FIT=True, DEV=False, TEST=False, AP=False,
                   GT_opened=False, model_forwards=0, optimizer_updates=0, logit_planes_opened=False,
                   protocol_sha256=args.protocol_sha, pins=PINS, all_edits=tally)
    pre_sha = write(args.output / "PRELABEL_SAFE.json", receipt)
    analysis = score(args, result, observations, geo, stock)
    outcome = dict(receipt, **analysis, status="HRSID_FIT_SDF_MEMBERSHIP_ACTUAL", GT_opened=True,
                   prelabel_sha256=pre_sha, source_sha256=lock["source_sha256"])
    print(json.dumps(dict(result_sha256=write(args.output / "RESULT_SAFE.json", outcome), **analysis),
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    def offline(event, _):
        if event.startswith(("socket.", "subprocess.")):
            raise RuntimeError("Offline fixed membership audit")
    sys.addaudithook(offline)
    main()
