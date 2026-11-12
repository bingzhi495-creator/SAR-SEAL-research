"""D-FINE historical DEV705 truth-only inside-box edit menus."""

import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
import json
from math import ceil
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

import numpy as np

PROTOCOL = "905dd63b47ff3261884bcb3c08534c5ac698a886d407107077cc34df0ec86410"
BRIDGE = "2f3adfeabef3a4f3b39a5086f5223c5d709c52e6c3ec32de48326eb702830819"
AXIS = "9cb46181f59f513f97d93e700a78a5d8d2d8004e2eea2661380a363ba3539386"

MENU = "4b84e51855683acae44c09c2d0859cf249da1daa31edabfb76ea45f09da64555"


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


def assess(im, rle, candidate, truth, geom, shapely, codec, menu, bridge):
    h, w = rle["size"]
    require(h == im["height"] and w == im["width"], "Exact native raster")
    P = int(candidate["mask_area"])
    T = float(truth.area)
    boxes = geom.geometry.boxes_from_counts(geom.shift.runs(rle["counts"]),h,w)
    I = float(geom.geometry.overlap_area(boxes,truth))
    require(P>0 and T>0 and 0<=I<=min(P,T)+1e-8,"Exact original overlap")
    xmin,ymin,xmax,ymax = candidate["box_xyxy"]
    x0,y0,x1,y1 = max(0,ceil(xmin)),max(0,ceil(ymin)),min(w,ceil(xmax)),min(h,ceil(ymax))
    require((boxes[:,0]>=x0).all() and (boxes[:,1]>=y0).all() and
            (boxes[:,2]<=x1).all() and (boxes[:,3]<=y1).all(), "Original RLE is inside author box")
    cap = 2*min(8,P//10) if 32<=P<1024 else 0
    if cap:
        mask = codec.decode(rle)[y0:y1,x0:x1].astype(bool)
        require(int(mask.sum())==P,"Exact stock support")
        band,_,_ = bridge.two_layers(mask)

        def cover(selection):
            y,x = np.nonzero(selection)
            cells = shapely.box(x+x0,y+y0,x+x0+1,y+y0+1)
            q = shapely.area(shapely.intersection(cells,truth))
            require(np.isfinite(q).all() and (q>=-1e-10).all() and (q<=1+1e-10).all(),"Native occupancy")
            return np.clip(q,0,1)

        qa,qd = cover(band & ~mask),cover(band & mask)
    else:
        qa,qd = np.empty(0),np.empty(0)
    rows = menu.menu(P,T,I,qa,qd,cap)
    return Path(im["file_name"]).stem.split("_",1)[0],rows,cap,min(P,T)/max(P,T)


def summarize(rows):
    groups = defaultdict(list)
    for source,values,_,_ in rows:
        groups[source].append(values)
    modes = tuple(rows[0][1])
    metrics = {}
    for mode in modes:
        entries = [r[1][mode] for r in rows]
        group_means = [float(np.mean([r[mode][0] for r in group])) for group in groups.values()]
        metrics[mode] = dict(pair_equal_iou_points=100*float(np.mean([r[0] for r in entries])),
                             source_equal_iou_points=100*float(np.mean(group_means)),
                             strict75=sum(r[0]>=.75 for r in entries),strict90=sum(r[0]>=.90 for r in entries),
                             added=sum(r[1] for r in entries),dropped=sum(r[2] for r in entries),
                             changed=sum((r[1]+r[2])>0 for r in entries))
    strict_new = [r[1] for r in rows if r[1]["raw"][0]<.90 and r[1]["unrestricted"][0]>=.90]
    transitions = dict(new_90=len(strict_new),
                       new_90_not_balanced=sum(r["balanced"][0]<.90 for r in strict_new),
                       new_90_neither_single=sum(max(r["add_only"][0],r["drop_only"][0])<.90 for r in strict_new),
                       raw_strict_miss_fixed_area_bound=sum(r[1]["raw"][0]<.90 and r[3]<.90-1e-12 for r in rows),
                       eligible_raw_strict_miss_fixed_area_bound=sum(r[2]>0 and r[1]["raw"][0]<.90 and
                           r[3]<.90-1e-12 for r in rows))
    return dict(groups=len(groups),eligible=sum(r[2]>0 for r in rows),metrics=metrics,transitions=transitions)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol","code","shapely-site","acceptance","score","score-independent",
                 "predictions","audit","gt","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    a = parser.parse_args()
    started = perf_counter()
    require(sha256(a.protocol.read_bytes()).hexdigest() == PROTOCOL, "Exact prelocked protocol")
    menu = load(Path(__file__).with_name("inside_menus.py"),MENU,"_dfine_inside_menus")
    bridge = load(a.code/"HRSID_FIT_RLE_VOTE_SOURCE_BRIDGE_20261112/score_binary_bridge.py",BRIDGE,"_dfine_bridge")
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
    pre = axis.write(a.output/"PRELABEL_SAFE.json",dict(status="DFINE_INSIDE_MENUS_DEV_PRELABEL",GT_opened=False,
                     protocol_sha256=PROTOCOL,menu_source_sha256=MENU,band_source_sha256=BRIDGE,predictions_sha256=score["predictions_sha256"],
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
                                   geom,shapely,codec,menu,bridge))
    require(len(rows)==1329 and matched==1652 and len(eligible)==1425,"Exact fixed cohort")
    stats = summarize(rows)
    result = dict(status="DFINE_INSIDE_MENUS_DEV_ACTUAL",DEV=True,FIT=False,TEST=False,AP=False,GT_opened=True,
                  model_forwards=0,detector_updates=0,matched=len(rows),eligible=len(eligible),stock_matches=matched,
                  protocol_sha256=PROTOCOL,menu_source_sha256=MENU,band_source_sha256=BRIDGE,prelabel_sha256=pre,
                  worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  stats=stats,
                  worker_seconds=perf_counter()-started)
    digest = axis.write(a.output/"RESULT_SAFE.json",result)
    print(json.dumps(dict(status=result["status"],result_sha256=digest,stats=stats),sort_keys=True),flush=True)


if __name__=="__main__":
    def offline(event,_):
        if event.startswith(("socket.","subprocess.")):
            raise RuntimeError("Offline CPU diagnostic")
    sys.addaudithook(offline)
    main()
