"""Replay the exact historical DEV cohort for a mathematical lattice bound."""

import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
import json
from math import ceil
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter
from types import ModuleType

import numpy as np

PROTOCOL = "8eeef2e3436e9d8f068afe2ea2c06d65bfcaca7ffc1e8565ce9cb2072ddea2bc"
GRID = "3901cea51d96b9d79dc7efaa941e5e4c9011237b5efc60c69441a8deaa714200"
AXIS = "9cb46181f59f513f97d93e700a78a5d8d2d8004e2eea2661380a363ba3539386"
THRESHOLDS = (.75, .90)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(path, expected, name):
    data = path.read_bytes()
    require(sha256(data).hexdigest() == expected, "Exact source " + path.name)
    m = ModuleType(name)
    m.__file__ = str(path)
    sys.modules[name] = m
    exec(compile(data, str(path), "exec"), m.__dict__)
    return m


def assess(im, rle, meta, truth, width, modules, codec, shapely, grid):
    h, w = rle["size"]
    require(h == im["height"] and w == im["width"], "Native mask grid")
    xmin, ymin, xmax, ymax = meta["box_xyxy"]
    x0, y0 = max(0, ceil(xmin)), max(0, ceil(ymin))
    x1, y1 = min(w, ceil(xmax)), min(h, ceil(ymax))
    require(x0 < x1 and y0 < y1, "Nonempty author box")
    stock = codec.decode(rle)[y0:y1, x0:x1].astype(bool)
    pred_area = int(meta["mask_area"])
    require(int(stock.sum()) == pred_area > 0, "Author box retains stock mask")
    y, x = np.mgrid[y0:y1, x0:x1]
    cells = shapely.box(x, y, x + 1, y + 1)
    q = shapely.area(shapely.intersection(cells, truth))
    require(np.isfinite(q).all() and (q >= -1e-10).all() and (q <= 1 + 1e-10).all(),
            "Continuous unit-cell areas")
    q = np.clip(q, 0, 1)
    T = float(truth.area)
    epsilon = 1e-7 * max(1., T, q.size)
    I = float(q[stock].sum(dtype=float))
    boxes = modules.geometry.boxes_from_counts(modules.shift.runs(rle["counts"]), h, w)
    I_check = float(modules.geometry.overlap_area(boxes, truth))
    require(abs(I-I_check) <= epsilon and 0 <= I <= min(T, pred_area)+epsilon, "Stock geometry agreement")
    totals = {}
    for t in THRESHOLDS:
        a = grid.margin_envelope(q, T, t, (y0,x0), (h,w), (160,160))
        b = grid.margin_envelope(q, T, t, (y0,x0), (h,w), (480,480))
        raw = (1+t)*I - t*(pred_area+T)
        require(abs(a["free"]-b["free"]) <= epsilon and raw <= a["free"]+epsilon, "Box bound")
        eligible = raw < -epsilon and a["free"] > epsilon
        violation = raw > a["upper"]+epsilon
        extra = eligible and not violation and a["upper"] < -epsilon
        totals[str(t)] = dict(raw_reaches=raw >= 0, raw_definite_miss=raw < -epsilon,
                              raw_near=abs(raw)<=epsilon, box_excludes=a["free"] < -epsilon,
                              box_near=abs(a["free"])<=epsilon,
                              eligible_box_positive_raw_miss=eligible,
                              lattice160_excludes=extra, stock_160_violation=violation,
                              lattice160_near=abs(a["upper"])<=epsilon,
                              fine480_unresolved=extra and b["upper"] > epsilon,
                              fine480_excludes=extra and b["upper"] < -epsilon,
                              fine480_near=extra and abs(b["upper"])<=epsilon)
    group = Path(im["file_name"]).stem.split("_",1)[0]
    width_band = "under4" if width < 4 else "4to8" if width < 8 else "8plus"
    return (group, width_band, totals, abs(I-I_check))


def summarize(rows, key):
    groups = defaultdict(Counter)
    bands = defaultdict(Counter)
    counts = Counter()
    for group, band, values, _ in rows:
        groups[group].update(values[key])
        bands[band].update(values[key])
        counts.update(values[key])
    eligible = "eligible_box_positive_raw_miss"
    excluded = "lattice160_excludes"
    relevant = [c for c in groups.values() if c[eligible]]
    affected = sum(c[excluded] > 0 for c in relevant)
    f = counts[excluded]/counts[eligible] if counts[eligible] else 0.
    return dict(counts=counts, width_bands=bands, groups=len(groups), eligible_groups=len(relevant),
                affected_groups=affected, conditional_pair_rate=f,
                conditional_source_equal_rate=mean(c[excluded]/c[eligible] for c in relevant) if relevant else 0.,
                gate=(f >= .1 and bool(relevant) and affected >= .5*len(relevant)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol","code","shapely-site","acceptance","score","score-independent",
                 "predictions","audit","gt","output"):
        parser.add_argument("--"+name, type=Path, required=True)
    a = parser.parse_args()
    started = perf_counter()
    require(sha256(a.protocol.read_bytes()).hexdigest() == PROTOCOL, "Exact prelocked protocol")
    grid = load(Path(__file__).with_name("grid_envelope.py"), GRID, "_dfine_grid")
    sys.path.append(str(a.shapely_site.resolve(strict=True)))
    import shapely
    from pycocotools import mask as codec
    require(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2", "Exact geometry")
    root = a.code
    axis = load(root/"DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/dev705/axis/axis_probe.py", AXIS, "_axis")
    stock = axis.load_module(root/"HRSID_FIT_HULL_CDF_20261112/stock_coco_match.py", axis.MATCH, "_stock")
    hull = axis.load_module(root/"MASK_RTDETR_DEV705_FIXED_MATCH_GEOMETRY_20261112/fixed_match_geometry.py",
                            axis.HULL, "_hull")
    geom = axis.load_module(root/"MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py",
                            axis.GEOM, "_geom").geometry_modules(root)
    accepted = json.loads(axis.pin(a.acceptance, axis.EXPORT_ACCEPTED))
    score = json.loads(axis.pin(a.score, axis.SCORE))
    independent = json.loads(axis.pin(a.score_independent, axis.SCORE_INDEPENDENT))
    require(accepted["original_os_exit"] == 0 and accepted["job_drained"] and not accepted["GT_opened"]
            and independent["exact_26_metrics"] and independent["score_sha256"] == axis.SCORE
            and all(accepted[k] == score[k] for k in ("predictions_sha256", "candidate_audit_sha256")),
            "Exact closed candidate/audit/score lineage")
    blocks = [json.loads(x) for x in axis.pin(a.predictions, score["predictions_sha256"], 64*2**20).splitlines()]
    audit = [json.loads(x) for x in axis.pin(a.audit, score["candidate_audit_sha256"], 64*2**20).splitlines()]
    require(len(blocks) == len(audit) == 705 and sum(len(b["coco"]) for b in blocks) == 184687,
            "Exact prediction population")
    a.output.mkdir()
    pre = axis.write(a.output/"PRELABEL_SAFE.json", dict(protocol_sha256=PROTOCOL, grid_source_sha256=GRID,
                     predictions_sha256=score["predictions_sha256"], candidate_audit_sha256=score["candidate_audit_sha256"],
                     status="DFINE_IDEAL_LATTICE_DEV_PRELABEL", GT_opened=False))
    gt = json.loads(axis.pin(a.gt, score["gt_sha256"]))
    require(len(gt["images"]) == 705 and len(gt["annotations"]) == 1782, "Exact historical DEV truth")
    target, neutral, _ = geom.geometry.polygon_support(gt, geom.vector)
    anns, eligible = defaultdict(list), {}
    for ann in gt["annotations"]:
        anns[ann["image_id"]].append(ann)
        if ann["id"] in target and ann["area"] < 1024:
            aspect, width, _, _ = hull.hull(target[ann["id"]][0])
            if aspect >= 2:
                eligible[ann["id"]] = width
    rows = []
    count = 0
    for im, block, audit_image in zip(gt["images"], blocks, audit, strict=True):
        require(im["id"] == block["image_id"] == audit_image["image_id"], "Exact image order")
        pool = []
        for rank, (c, meta) in enumerate(zip(block["coco"][:100], audit_image["candidates"][:100], strict=True), 1):
            rle = c["segmentation"]
            require(meta["native_rank"] == rank and c["score"] == meta["score"]
                    and codec.area(rle) == meta["mask_area"], "Exact candidate identity")
            pool.append(dict(rank=rank, score=c["score"], area=meta["mask_area"], rle=rle))
        with redirect_stdout(StringIO()):
            matches, _, _ = stock.freeze_coco(im, anns[im["id"]], pool)
        count += len(matches)
        for ident, rank in matches.items():
            if ident in eligible and ident not in neutral:
                rows.append(assess(im, pool[rank-1]["rle"], audit_image["candidates"][rank-1], target[ident][0],
                                   eligible[ident], geom, codec, shapely, grid))
    require(len(eligible) == 1425 and count == 1652 and len(rows) == 1329, "Exact fixed matches")
    stats = {str(t): summarize(rows, str(t)) for t in THRESHOLDS}
    result = dict(status="DFINE_IDEAL_LATTICE_DEV_ACTUAL", DEV=True, FIT=False, TEST=False, AP=False,
                  model_forwards=0, detector_updates=0, matched=len(rows), eligible=len(eligible), stock_matches=count,
                  protocol_sha256=PROTOCOL, grid_source_sha256=GRID,
                  worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), prelabel_sha256=pre,
                  stats=stats, advance_ideal_resolution=stats["0.9"]["gate"],
                  maximum_stock_overlap_check_error=max(r[3] for r in rows), worker_seconds=perf_counter()-started)
    digest = axis.write(a.output/"RESULT_SAFE.json", result)
    print(json.dumps(dict(status=result["status"], result_sha256=digest, stats=stats), sort_keys=True), flush=True)


if __name__ == "__main__":
    def offline(event, _):
        if event.startswith(("socket.", "subprocess.")):
            raise RuntimeError("Offline CPU diagnostic")
    sys.addaudithook(offline)
    main()
