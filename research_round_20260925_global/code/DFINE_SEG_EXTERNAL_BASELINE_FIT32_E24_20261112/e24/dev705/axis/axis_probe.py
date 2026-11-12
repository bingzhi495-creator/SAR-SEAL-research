"""Fixed D-FINE DEV705 continuous shape versus one-axis translation headroom."""
import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from hashlib import sha256
from importlib.metadata import version
from io import StringIO
import json
import os
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
from pycocotools import mask as codec

PROTOCOL = "2db662d29a75da4ed997157449723b772225e072634aeaf17636509b7cf98a36"
EXPORT_ACCEPTED = "37e51c1a1360a69751aa0f56c24d1b450e232266f1e0ba847626b80df2d93af4"
COVERAGE = "ec99af81cb7be87a332e0787ac08cfa36e6cbed30f8d5628a8cbf0217828ac31"
SCORE = "1d76e8d55be1da022e0962917e1490769994caf3636aaa4a2c1bbe36ab424588"
SCORE_INDEPENDENT = "7e179942859dddbaaa0c3ed576964a6664423d6a90ce5debf1a3fa3bc192950d"
GEOM = "7cad8b51a9230101e2233db8455f84c28163abfb2ee8ddd8cbe8718a2e719bd0"
MATCH = "12fd1684b262a3c9b6732b90a0c1e624034c523a7d130919a079e17728f80075"
HULL = "dedd47d5453b2442ff02585c9b4d216917e5ed175e01eae07a354a22710f3a94"
MENU = (-1.5, -1.0, -.5, 0.0, .5, 1.0, 1.5)
MODES = ("raw", "short", "long", "best_axis")
THRESHOLDS = (.75, .90)


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def pin(path, digest, limit=4*1024**2):
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file()
            and 0 < path.stat().st_size <= limit, "Exact bounded input: "+path.name)
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == digest, "Exact closed digest: "+path.name)
    return raw


def write(path, value):
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    with path.open("xb") as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())
    return sha256(raw).hexdigest()


def load_module(path, digest, name):
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(pin(path, digest), str(path), "exec"), module.__dict__)
    return module


def inside(bounds, axes, width, height):
    x0, y0, x1, y1 = bounds
    for ux, uy in axes:
        dx, dy = abs(ux)*1.5, abs(uy)*1.5
        if x0-dx < -1e-9 or y0-dy < -1e-9 or x1+dx > width+1e-9 or y1+dy > height+1e-9:
            return False
    return True


def preferred(values):
    best = max(values.values())
    shift = min((x for x,y in values.items() if abs(y-best) <= 1e-12), key=lambda x: (abs(x),x))
    return shift, values[shift]


def continuous(rectangles, pred_area, truth, engine, delta=None):
    if delta is not None:
        rectangles = rectangles + np.tile(delta, 2)
    overlap = engine.overlap_area(rectangles, truth)
    cap = min(pred_area, truth.area)
    require(-1e-8 <= overlap <= cap+1e-8, "Exact valid area-preserving continuous intersection")
    intersection = max(0., min(cap, overlap))
    return float(intersection / (pred_area + truth.area - intersection))


def summary(rows):
    pairs = np.asarray([row[1] for row in rows])
    by_chip = defaultdict(list)
    for image, values, _, _, _ in rows:
        by_chip[image].append(values)
    groups = np.asarray([np.mean(by_chip[k], axis=0) for k in sorted(by_chip)])
    draws = np.random.Generator(np.random.PCG64(5217)).integers(len(groups), size=(4096,len(groups)))
    paired = {}
    for a,b in ((1,0),(2,0),(3,0),(1,2)):
        delta = 100*(groups[:,a]-groups[:,b])
        paired[MODES[a]+"_minus_"+MODES[b]] = dict(
            chip_equal_delta_points=float(delta.mean()),
            interval95_points=np.percentile(delta[draws].mean(axis=1),[2.5,97.5]).tolist(),
            pair_equal_delta_points=float(100*(pairs[:,a]-pairs[:,b]).mean()))
    return dict(pair_equal_iou_points=dict(zip(MODES,(100*pairs.mean(axis=0)).tolist())),
                chip_equal_iou_points=dict(zip(MODES,(100*groups.mean(axis=0)).tolist())),
                median_raw=float(np.median(pairs[:,0])),
                per_axis_nonzero=dict(short=sum(row[2] != 0 for row in rows), long=sum(row[3] != 0 for row in rows)),
                per_axis_at_menu_boundary=dict(short=sum(abs(row[2]) == 1.5 for row in rows),
                                               long=sum(abs(row[3]) == 1.5 for row in rows)),
                supported_images=len(groups), bootstrap_draws=4096, bootstrap_seed=5217, paired_effects=paired)


def strict_counts(rows):
    data = np.asarray([r[1] for r in rows])
    return {f"{t:.2f}": {m: int((data[:,i] >= t).sum()) for i,m in enumerate(MODES)} for t in THRESHOLDS}


def census(gt, blocks, modules, stock, hull, expected_stock):
    truth_by_id, neutral, support = modules.geometry.polygon_support(gt, modules.vector)
    annotations = defaultdict(list)
    eligible = {}
    for ann in gt["annotations"]:
        annotations[ann["image_id"]].append(ann)
        if ann["id"] in truth_by_id and ann["area"] < 1024:
            truth = truth_by_id[ann["id"]][0]
            aspect, _, tangent, normal = hull.hull(truth)
            if aspect >= 2:
                eligible[ann["id"]] = (truth, normal, tangent)
    require([i["id"] for i in gt["images"]] == [b["image_id"] for b in blocks] and
            len(gt["images"]) == 705 and len(gt["annotations"]) == 1782, "Exact original image and label roster")
    rows, boundary, all_raw = [], [], []
    counts = Counter(stock_matched=0, stock_matched_valid=0, stock_matched_ineligible=0,
                     stock_matched_invalid=0, eligible_matched=0, eligible_boundary=0,
                     eligible_common=0, images_without_gt=0)
    for im, block in zip(gt["images"], blocks, strict=True):
        image = im["id"]
        pool = [dict(rank=rank, score=row["score"], rle=row["segmentation"], area=int(codec.area(row["segmentation"])))
                for rank,row in enumerate(block["coco"][:100],1)]
        if not annotations[image]:
            counts["images_without_gt"] += 1
            continue
        with redirect_stdout(StringIO()):
            matches, _, _ = stock.freeze_coco(im, annotations[image], pool)
        counts["stock_matched"] += len(matches)
        for identity, rank in matches.items():
            if identity in neutral:
                counts["stock_matched_invalid"] += 1
                continue
            counts["stock_matched_valid"] += 1
            truth = truth_by_id[identity][0]
            cand = pool[rank-1]
            rle = cand["rle"]
            require(cand["rank"] == rank and rle["size"] == [im["height"],im["width"]],
                    "Exact top100 match identity and native canvas")
            rectangles = modules.geometry.boxes_from_counts(modules.shift.runs(rle["counts"]), *rle["size"])
            area = float(cand["area"])
            require(area > 0 and np.prod(rectangles[:,2:]-rectangles[:,:2],axis=1).sum() == area,
                    "Native unit-cell prediction area preservation")
            raw = continuous(rectangles, area, truth, modules.geometry)
            all_raw.append(raw)
            if identity not in eligible:
                counts["stock_matched_ineligible"] += 1
                continue
            counts["eligible_matched"] += 1
            _, short, long = eligible[identity]
            ceiling = min(area,truth.area)/max(area,truth.area)
            bounds = tuple(np.concatenate((rectangles[:,:2].min(axis=0),rectangles[:,2:].max(axis=0))))
            if not inside(bounds,(short,long),im["width"],im["height"]):
                counts["eligible_boundary"] += 1
                boundary.append((image,[raw]*4,0.,0.,ceiling))
                continue
            counts["eligible_common"] += 1
            ss, sv = preferred({s: raw if s == 0 else continuous(rectangles,area,truth,modules.geometry,s*short)
                                for s in MENU})
            ls, lv = preferred({s: raw if s == 0 else continuous(rectangles,area,truth,modules.geometry,s*long)
                                for s in MENU})
            require(min(sv,lv) >= raw-1e-12 and max(sv,lv) <= ceiling+1e-12, "Raw-in-menu and exact area ceiling")
            rows.append((image,[raw,sv,lv,max(sv,lv)],ss,ls,ceiling))
    require(counts["stock_matched"] == expected_stock == 1652 and
            counts["stock_matched"] == counts["stock_matched_valid"]+counts["stock_matched_invalid"] and
            counts["eligible_matched"] == len(rows)+len(boundary) and rows and
            counts["stock_matched_valid"] == len(all_raw) == counts["eligible_matched"]+counts["stock_matched_ineligible"],
            "Exact original stock and common-support cohort partitions")
    counts.update(all_gt=1782,source_polygon_valid=len(truth_by_id),source_polygon_neutral=len(neutral),
                  full_small_slender_gt=len(eligible),eligible_unmatched=len(eligible)-counts["eligible_matched"])
    return dict(cohort=counts, polygon_support=support, strict_common=strict_counts(rows),
                strict_full_eligible_gt=dict(denominator=len(eligible),hits=strict_counts(rows+boundary)),
                strict_raw_all_valid_matches={f"{t:.2f}": sum(x>=t for x in all_raw) for t in THRESHOLDS},
                area_ceiling_common={f"{t:.2f}": sum(r[4]>=t for r in rows) for t in THRESHOLDS},
                area_ceiling_eligible_matched={f"{t:.2f}": sum(r[4]>=t for r in rows+boundary) for t in THRESHOLDS},
                **summary(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("protocol", "code", "shapely-site", "acceptance", "coverage", "score", "score-independent",
                  "predictions", "gt", "output"):
        parser.add_argument("--"+field, type=Path, required=True)
    a = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and version("pycocotools") == "2.0.11"
            and np.__version__ == "1.26.4", "Exact CPU runtime")
    sys.path.append(str(a.shapely_site.resolve(strict=True)))
    import shapely
    require(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2", "Exact continuous geometry")
    pin(a.protocol, PROTOCOL)
    stock = load_module(a.code/"HRSID_FIT_HULL_CDF_20261112/stock_coco_match.py", MATCH, "_axis_stock")
    hull = load_module(a.code/"MASK_RTDETR_DEV705_FIXED_MATCH_GEOMETRY_20261112/fixed_match_geometry.py", HULL, "_axis_hull")
    source = load_module(a.code/"MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py", GEOM, "_axis_source")
    modules = source.geometry_modules(a.code)
    accepted = json.loads(pin(a.acceptance, EXPORT_ACCEPTED))
    coverage = json.loads(pin(a.coverage, COVERAGE))
    score = json.loads(pin(a.score, SCORE))
    verified = json.loads(pin(a.score_independent, SCORE_INDEPENDENT))
    require(accepted["original_os_exit"] == 0 and accepted["job_drained"] and
            not accepted["GT_opened"] and not accepted["TEST"] and not coverage["AP"] and not coverage["TEST"] and
            verified["exact_26_metrics"] is True and verified["score_sha256"] == SCORE and
            score["predictions_sha256"] == coverage["predictions_sha256"] == accepted["predictions_sha256"] and
            coverage["gt_sha256"] == score["gt_sha256"], "Exact closed export, coverage and scoring lineage")
    raw = pin(a.predictions, accepted["predictions_sha256"], 64*1024**2)
    blocks = [json.loads(line) for line in raw.splitlines()]
    require(raw.endswith(b"\n") and len(blocks) == 705 and sum(len(b["coco"]) for b in blocks) == 184687 and
            sum(min(len(b["coco"]),100) for b in blocks) == 70500, "Exact author post-NMS and native top100 rows")
    a.output.mkdir()
    prelabel = write(a.output/"PRELABEL_SAFE.json",dict(status="DFINE_DEV705_AXIS_PRELABEL",AP=False,
                     GT_opened=False,FIT=False,DEV=True,TEST=False,images=705,top100_rows=70500,
                     protocol_sha256=PROTOCOL,coverage_sha256=COVERAGE,predictions_sha256=accepted["predictions_sha256"]))
    gt = json.loads(pin(a.gt, score["gt_sha256"]))
    result = census(gt,blocks,modules,stock,hull,coverage["candidate_pools"]["top100"]["counts"]["0.50"]["native_greedy"])
    result.update(status="DFINE_DEV705_FIXED_MATCH_AXIS_ORACLE_ACTUAL",GT_opened=True,AP=False,FIT=False,DEV=True,
                  TEST=False,model_forwards=0,optimizer_updates=0,truth_assisted=True,protocol_sha256=PROTOCOL,
                  prelabel_sha256=prelabel,source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  shift_menu_px=MENU,score_sha256=SCORE,coverage_sha256=COVERAGE)
    digest = write(a.output/"RESULT_SAFE.json",result)
    print(json.dumps(dict(status=result["status"],result_sha256=digest,cohort=result["cohort"],
                          pair_equal_iou_points=result["pair_equal_iou_points"],strict_common=result["strict_common"]),
                     sort_keys=True))


if __name__ == "__main__":
    main()