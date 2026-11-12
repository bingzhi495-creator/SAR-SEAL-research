"""D-FINE fixed-stock box collar reachability on historical DEV705."""

import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
import json
from math import ceil, floor
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

import numpy as np

PROTOCOL = "ca918f76425572db40a2df5a7b73cd0aada6e0c373ffa5ee89b37127dba8352a"
BOX = "2fad059c2c233e9a3a7dc915039248a7ec0db303c7fbde1a0950d5453babf6ea"
AXIS = "9cb46181f59f513f97d93e700a78a5d8d2d8004e2eea2661380a363ba3539386"
THRESHOLDS = (.75, .90)
COLLAR = "8fcf0176540c116aff5c1fdda1b12db2004353be0fca40dbab0e564b083e480c"


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


def assess(im, rle, candidate, truth, geom, shapely, codec, collar, reach):
    h, w = rle["size"]
    require(h == im["height"] and w == im["width"], "Exact native raster")
    xmin,ymin,xmax,ymax = candidate["box_xyxy"]
    x0,y0,x1,y1 = max(0,ceil(xmin)),max(0,ceil(ymin)),min(w,ceil(xmax)),min(h,ceil(ymax))
    tx0,ty0,tx1,ty1 = truth.bounds
    tx0,ty0,tx1,ty1 = max(0,floor(tx0)),max(0,floor(ty0)),min(w,ceil(tx1)),min(h,ceil(ty1))
    y,x = np.mgrid[ty0:ty1,tx0:tx1]
    cells = shapely.box(x,y,x+1,y+1)
    q = shapely.area(shapely.intersection(cells,truth))
    require(np.isfinite(q).all() and (q>=-1e-10).all() and (q<=1+1e-10).all(), "Continuous native occupancy")
    q = np.clip(q,0,1)
    stock = codec.decode(rle)[ty0:ty1,tx0:tx1].astype(bool)
    P = int(candidate["mask_area"])
    T = float(truth.area)
    eps = 1e-7*max(1.,T,(x1-x0)*(y1-y0))
    require(abs(float(q.sum(dtype=float))-T)<=eps, "Native cells cover entire truth")
    box = (x0,y0,x1,y1)
    inside = (x>=x0)&(x<x1)&(y>=y0)&(y<y1)
    expanded = (x>=x0-2)&(x<x1+2)&(y>=y0-2)&(y<y1+2)
    rectangles = geom.geometry.boxes_from_counts(geom.shift.runs(rle["counts"]),h,w)
    require((rectangles[:,0]>=x0).all() and (rectangles[:,1]>=y0).all() and
            (rectangles[:,2]<=x1).all() and (rectangles[:,3]<=y1).all(), "Original RLE remains inside box")
    I = float(q[stock].sum(dtype=float))
    footprint_error = abs(I-float(geom.geometry.overlap_area(rectangles,truth)))
    require(footprint_error<=eps, "Independent stock geometry")
    rows = {}
    for t in THRESHOLDS:
        raw = (1+t)*I-t*(P+T)
        A,D,N = collar.action_bonuses(q,stock,inside,expanded,P,t)
        free = reach.positive_margins(q,T,(ty0,tx0),box,t,(0,2))
        require(abs(raw+D+N-free["0"])<=eps and abs(raw+A+D+N-free["2"])<=eps, "Expanded bound agreement")
        margins = dict(raw=raw, collar_only=raw+A, collar_and_remove=raw+A+D,
                       collar_and_inner_add=raw+A+N, expanded_free=raw+A+D+N)
        states = {name:("positive" if m>eps else "negative" if m< -eps else "near")
                  for name,m in margins.items()}
        rows[str(t)] = (states, free["0"]< -eps and free["2"]>eps)
    return Path(im["file_name"]).stem.split("_",1)[0],rows,footprint_error


def summarize(rows, key):
    scopes = {name: defaultdict(Counter) for name in ("all_matched", "definite_raw_miss", "box2_rescues")}
    groups = {name:set() for name in scopes}
    for group, metrics, _ in rows:
        states,rescued = metrics[key]
        included = ("all_matched",) + (("definite_raw_miss",) if states["raw"]=="negative" else ()) + (
            ("box2_rescues",) if rescued else ())
        for name in included:
            groups[name].add(group)
            for arm,state in states.items():
                scopes[name][arm][state] += 1
    return {name:dict(cases=sum(v["raw"].values()),source_proxies=len(groups[name]),margins=v)
            for name,v in scopes.items()}



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol","code","shapely-site","acceptance","score","score-independent",
                 "predictions","audit","gt","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    a = parser.parse_args()
    started = perf_counter()
    require(sha256(a.protocol.read_bytes()).hexdigest() == PROTOCOL, "Exact prelocked protocol")
    collar = load(Path(__file__).with_name("stock_collar.py"),COLLAR,"_dfine_stock_collar")
    reach = load(a.code/"DFINE_DEV705_NATIVE_BOX_CEILING_20261112/box_ceiling.py",BOX,"_dfine_native_box")
    sys.path.append(str(a.shapely_site.resolve(strict=True)))
    import shapely
    from pycocotools import mask as codec
    require(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2", "Exact geometry")
    root = a.code
    axis = load(root/"DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/dev705/axis/axis_probe.py",AXIS,"_axis")
    stock = axis.load_module(root/"HRSID_FIT_HULL_CDF_20261112/stock_coco_match.py",axis.MATCH,"_stock")
    hull = axis.load_module(root/"MASK_RTDETR_DEV705_FIXED_MATCH_GEOMETRY_20261112/fixed_match_geometry.py",axis.HULL,"_hull")
    geom = axis.load_module(root/"MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py",
                            axis.GEOM,"_geom").geometry_modules(root)
    accepted = json.loads(axis.pin(a.acceptance,axis.EXPORT_ACCEPTED))
    score = json.loads(axis.pin(a.score,axis.SCORE))
    independent = json.loads(axis.pin(a.score_independent,axis.SCORE_INDEPENDENT))
    require(accepted["original_os_exit"] == 0 and accepted["job_drained"] and not accepted["GT_opened"]
            and independent["exact_26_metrics"] and independent["score_sha256"] == axis.SCORE
            and all(accepted[k] == score[k] for k in ("predictions_sha256","candidate_audit_sha256")),
            "Exact closed candidate, audit and score")
    blocks = [json.loads(x) for x in axis.pin(a.predictions,score["predictions_sha256"],64*2**20).splitlines()]
    audit = [json.loads(x) for x in axis.pin(a.audit,score["candidate_audit_sha256"],64*2**20).splitlines()]
    require(len(blocks)==len(audit)==705 and sum(len(b["coco"]) for b in blocks)==184687,"Exact candidate population")
    a.output.mkdir()
    pre = axis.write(a.output/"PRELABEL_SAFE.json",dict(status="DFINE_STOCK_COLLAR_DEV_PRELABEL",GT_opened=False,
                     protocol_sha256=PROTOCOL,box_source_sha256=BOX,collar_source_sha256=COLLAR,predictions_sha256=score["predictions_sha256"],
                     candidate_audit_sha256=score["candidate_audit_sha256"]))
    gt = json.loads(axis.pin(a.gt,score["gt_sha256"]))
    require(len(gt["images"])==705 and len(gt["annotations"])==1782,"Exact historical DEV GT")
    target,neutral,_ = geom.geometry.polygon_support(gt,geom.vector)
    anns,eligible = defaultdict(list),{}
    for ann in gt["annotations"]:
        anns[ann["image_id"]].append(ann)
        if ann["id"] in target and ann["area"]<1024:
            aspect,width,_,_ = hull.hull(target[ann["id"]][0])
            if aspect>=2:
                eligible[ann["id"]] = width
    rows = []
    matched = 0
    for im,block,trace in zip(gt["images"],blocks,audit,strict=True):
        require(im["id"]==block["image_id"]==trace["image_id"],"Exact image order")
        pool = []
        for rank,(c,meta) in enumerate(zip(block["coco"][:100],trace["candidates"][:100],strict=True),1):
            rle = c["segmentation"]
            require(rank==meta["native_rank"] and c["score"]==meta["score"] and codec.area(rle)==meta["mask_area"],
                    "Exact stock candidate identity")
            pool.append(dict(rank=rank,score=c["score"],area=meta["mask_area"],rle=rle))
        with redirect_stdout(StringIO()):
            matches,_,_ = stock.freeze_coco(im,anns[im["id"]],pool)
        matched += len(matches)
        for ident,rank in matches.items():
            if ident in eligible and ident not in neutral:
                rows.append(assess(im,pool[rank-1]["rle"],trace["candidates"][rank-1],target[ident][0],
                                   geom,shapely,codec,collar,reach))
    require(len(rows)==1329 and matched==1652 and len(eligible)==1425,"Exact fixed cohort")
    stats = {str(t):summarize(rows,str(t)) for t in THRESHOLDS}
    require(stats["0.9"]["box2_rescues"]["cases"]==134,"Reproduce the known expanded-box geometry count")
    result = dict(status="DFINE_STOCK_COLLAR_DEV_ACTUAL",DEV=True,FIT=False,TEST=False,AP=False,GT_opened=True,
                  model_forwards=0,detector_updates=0,matched=len(rows),eligible=len(eligible),stock_matches=matched,
                  protocol_sha256=PROTOCOL,box_source_sha256=BOX,collar_source_sha256=COLLAR,prelabel_sha256=pre,
                  worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  stats=stats,
                  maximum_stock_footprint_error=max(x[2] for x in rows),worker_seconds=perf_counter()-started)
    digest = axis.write(a.output/"RESULT_SAFE.json",result)
    print(json.dumps(dict(status=result["status"],result_sha256=digest,stats=stats),sort_keys=True),flush=True)


if __name__=="__main__":
    def offline(event,_):
        if event.startswith(("socket.","subprocess.")):
            raise RuntimeError("Offline CPU diagnostic")
    sys.addaudithook(offline)
    main()
