"""Grouped fixed-count D-FINE area calibration on exact closed DEV outputs."""
import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter
from types import ModuleType

import numpy as np

PROTOCOL = "7ed59d5bac16bcc34abc6daa4a6cac2ad14c8e78ba3daeba6143211850583b92"
AXIS = "9cb46181f59f513f97d93e700a78a5d8d2d8004e2eea2661380a363ba3539386"
BRIDGE = "2f3adfeabef3a4f3b39a5086f5223c5d709c52e6c3ec32de48326eb702830819"
MODES = ("raw", "balanced", "size", "box", "oracle")
FEATURES = ("size", "box")


def require(ok, why):
    if not ok:
        raise ValueError(why)


def load(path, digest, name):
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == digest, "Exact helper " + path.name)
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def feasible_splits(count, available_adds, available_drops):
    return [(a, count-a) for a in range(max(0, count-available_drops), min(count, available_adds)+1)]


def select_delta(splits, predicted):
    return min(splits, key=lambda ad: (abs(ad[0]-ad[1]-predicted), abs(ad[0]-ad[1]), ad[0]-ad[1]))


def fit_line(x, target, weights):
    center = float(np.average(x, weights=weights))
    spread = float(np.sqrt(np.average((x-center)**2, weights=weights)))
    scale = spread if spread > 1e-12 else 1.0
    z = (x-center)/scale
    intercept = float(np.average(target, weights=weights))
    slope = float(np.average(z*(target-intercept), weights=weights) /
                  (np.average(z*z, weights=weights)+.01))
    return center, scale, intercept, slope


def author_box_contains(xs, ys, box):
    x1, y1, x2, y2 = box
    return (xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2)


def group_weighted(rows, fn):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(fn(row))
    return mean(mean(values) for values in groups.values())


def action_values(P, T, I, adds=0, drops=0):
    require(-1e-8 <= I <= min(P,T)+1e-8 and P > 0 and T > 0, "Valid mask geometry")
    I = max(0., min(I, P, T))
    return dict(iou=100*I/(P+T-I), area_error=100*abs(P-T)/T, fp=100*(P-I)/T, fn=100*(T-I)/T,
                strict75=I/(P+T-I)>=.75, strict90=I/(P+T-I)>=.90, adds=adds, drops=drops)


def ordered_indices(support, near, take):
    ids = np.flatnonzero(support)
    order = np.lexsort((ids, ~near.ravel()[ids]))
    return ids[order[:take]]


def proposal_rows(im, candidate, detail, truth, geom, bridge, codec, shapely):
    rle = candidate["rle"]
    P = int(candidate["area"])
    T = float(truth.area)
    boxes = geom.geometry.boxes_from_counts(geom.shift.runs(rle["counts"]), *rle["size"])
    I = float(geom.geometry.overlap_area(boxes, truth))
    values = action_values(P,T,I)
    group = Path(im["file_name"]).stem.split("_",1)[0]
    row = dict(group=group, raw=values, area=P, truth_area=T, editable=False, reason="predicted_area_under32")
    if P < 32:
        return row

    h,w = rle["size"]
    x1,y1,x2,y2 = detail["box_xyxy"]
    clipped = (max(0.,x1),max(0.,y1),min(float(w),x2),min(float(h),y2))
    require(clipped[0]<clipped[2] and clipped[1]<clipped[3], "Exact nonempty detector box")
    mask = codec.decode(rle).astype(bool)
    ys,xs = np.nonzero(mask)
    require(len(ys) == P and np.all(author_box_contains(xs,ys,clipped)), "Mask retained author native box clip")
    ylo, xlo = max(0,int(ys.min())-2), max(0,int(xs.min())-2)
    yhi, xhi = min(h,int(ys.max())+3), min(w,int(xs.max())+3)
    anchor = mask[ylo:yhi,xlo:xhi]
    band, near, _ = bridge.two_layers(anchor)
    yy,xx = np.indices(anchor.shape)
    permitted = author_box_contains(xx+xlo, yy+ylo, clipped)
    support_add = band & ~anchor & permitted
    support_drop = band & anchor
    m = 2*min(8,P//10)
    nadd, ndrop = int(support_add.sum()), int(support_drop.sum())
    row["reason"] = "balanced_infeasible"
    if min(nadd,ndrop) < m//2:
        return row

    aidx = ordered_indices(support_add, near, m)
    didx = ordered_indices(support_drop, near, m)
    def fractional_prefix(indices):
        y,x = np.divmod(indices, anchor.shape[1])
        x,y = x+xlo,y+ylo
        pixels = shapely.box(x, y, x+1, y+1)
        occupancy = shapely.area(shapely.intersection(pixels, truth))
        require(np.isfinite(occupancy).all() and (occupancy>=0.).all() and (occupancy<=1.+1e-9).all(),
                "Continuous native cell areas")
        return np.concatenate(([0.], np.cumsum(occupancy,dtype=float)))

    pa,pd = fractional_prefix(aidx),fractional_prefix(didx)
    splits = feasible_splits(m,nadd,ndrop)
    options = {(a,d): action_values(P+a-d,T,I+float(pa[a])-float(pd[d]),a,d) for a,d in splits}
    best = min(splits, key=lambda ad: (-options[ad]["iou"], abs(ad[0]-ad[1]), ad[0]-ad[1]))
    box_area = (clipped[2]-clipped[0])*(clipped[3]-clipped[1])
    tight_area = (int(xs.max())-int(xs.min())+1)*(int(ys.max())-int(ys.min())+1)
    row.update(editable=True, reason="editable", budget=m, options=options, oracle=options[best],
               balanced=options[m//2,m//2], size=float(np.log2(P)),
               box=float(np.log2(box_area/tight_area)), target=float(np.clip((T-P)/m,-1.,1.)))
    return row


def fit_and_score(rows):
    groups = sorted({r["group"] for r in rows}, key=lambda s: (sha256(("DFINE-area-v1|"+s).encode()).digest(),s))
    assignment = {g:i%5 for i,g in enumerate(groups)}
    fits = []
    for fold in range(5):
        fit = [r for r in rows if r["editable"] and assignment[r["group"]] != fold]
        held = [r for r in rows if assignment[r["group"]] == fold]
        group_counts = Counter(r["group"] for r in fit)
        require(fit and held, "All five source-held folds represented")
        w = np.asarray([1/group_counts[r["group"]] for r in fit])
        y = np.asarray([r["target"] for r in fit])
        controls = {}
        for mode in FEATURES:
            coeff = fit_line(np.asarray([r[mode] for r in fit]), y,w)
            controls[mode] = list(coeff)
            center, scale, offset, slope = coeff
            for row in held:
                if row["editable"]:
                    yhat = np.clip(offset+slope*(row[mode]-center)/scale,-1.,1.)
                    key = select_delta(row["options"], float(yhat)*row["budget"])
                    row[mode+"_metrics"] = row["options"][key]
        fold_difference = group_weighted(held,lambda r:r.get("box_metrics",r["raw"])["iou"]-
                                                     r.get("size_metrics",r["raw"])["iou"])
        fits.append(dict(fold=fold, train_editable=len(fit), train_groups=len(group_counts), held=len(held),
                         held_editable=sum(r["editable"] for r in held), held_groups=len({r["group"] for r in held}),
                         fits=controls, box_minus_size_iou_points=fold_difference))

    def select(row, mode):
        key = mode+"_metrics" if mode in FEATURES else mode
        return row.get(key,row["raw"])

    summary = {}
    for mode in MODES:
        vals = [select(r,mode) for r in rows]
        summary[mode] = dict(
            **{metric: group_weighted(rows,lambda r:select(r,mode)[metric]) for metric in ("iou","area_error","fp","fn")},
            pair_iou=mean(v["iou"] for v in vals), pair_area_error=mean(v["area_error"] for v in vals),
            strict75=sum(v["strict75"] for v in vals), strict90=sum(v["strict90"] for v in vals),
            adds=sum(v["adds"] for v in vals), drops=sum(v["drops"] for v in vals))
    budgets = sum(r["budget"] for r in rows if r["editable"])
    require(all(summary[m]["adds"]+summary[m]["drops"]==budgets for m in MODES if m!="raw"),
            "Each nonraw mode uses the exact fixed flip budget")
    gates = dict(area_margin=all(summary[m]["area_error"]-summary["box"]["area_error"] >= .25 for m in ("raw","size")),
                 iou_margin=all(summary["box"]["iou"]-summary[m]["iou"] >= .20 for m in ("raw","size")),
                 strict_retained=summary["box"]["strict90"] >= summary["raw"]["strict90"],
                 fold_directions=sum(f["box_minus_size_iou_points"]>0 for f in fits)>=4)
    return dict(groups=len(groups), matched=len(rows), editability=dict(Counter(r["reason"] for r in rows)),
                total_edit_budget=budgets, group_equal_metrics=summary, folds=fits, gates=gates,
                advance=all(gates.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for k in ("protocol","code","shapely-site","acceptance","score","score-independent","predictions",
              "audit","gt","output"):
        parser.add_argument("--"+k,required=True,type=Path)
    args=parser.parse_args()
    started=perf_counter()
    require(sha256(args.protocol.read_bytes()).hexdigest()==PROTOCOL,"Exact prospectively locked probe")
    sys.path.append(str(args.shapely_site.resolve(strict=True)))
    import shapely
    from pycocotools import mask as codec
    require(shapely.__version__=="2.0.2" and shapely.geos_version_string=="3.11.2","Exact geometry")
    code=args.code
    axis=load(code/"DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/dev705/axis/axis_probe.py",AXIS,"_area_axis")
    bridge=load(code/"HRSID_FIT_RLE_VOTE_SOURCE_BRIDGE_20261112/score_binary_bridge.py",BRIDGE,"_area_bridge")
    stock=axis.load_module(code/"HRSID_FIT_HULL_CDF_20261112/stock_coco_match.py",axis.MATCH,"_area_stock")
    hull=axis.load_module(code/"MASK_RTDETR_DEV705_FIXED_MATCH_GEOMETRY_20261112/fixed_match_geometry.py",
                          axis.HULL,"_area_hull")
    geom=axis.load_module(code/"MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py",
                          axis.GEOM,"_area_geometry").geometry_modules(code)
    accepted=json.loads(axis.pin(args.acceptance,axis.EXPORT_ACCEPTED))
    score=json.loads(axis.pin(args.score,axis.SCORE))
    independent=json.loads(axis.pin(args.score_independent,axis.SCORE_INDEPENDENT))
    require(accepted["original_os_exit"]==0 and accepted["job_drained"] and not accepted["GT_opened"]
            and independent["exact_26_metrics"] and independent["score_sha256"]==axis.SCORE
            and all(accepted[k]==score[k] for k in ("predictions_sha256","candidate_audit_sha256")),
            "Exact closed prediction/audit and score lineage")
    raw=axis.pin(args.predictions,score["predictions_sha256"],64*2**20)
    audit=axis.pin(args.audit,score["candidate_audit_sha256"],64*2**20)
    blocks=[json.loads(line) for line in raw.splitlines()]
    details=[json.loads(line) for line in audit.splitlines()]
    require(raw.endswith(b"\n") and audit.endswith(b"\n") and len(blocks)==len(details)==705
            and sum(len(b["coco"]) for b in blocks)==184687,"Exact closed DEV candidate rows")
    args.output.mkdir()
    prelabel=axis.write(args.output/"PRELABEL_SAFE.json",dict(status="DFINE_BOX_MASK_AREA_DEV_PRELABEL",
                          DEV=True,FIT=False,TEST=False,AP=False,GT_opened=False,protocol_sha256=PROTOCOL,
                          predictions_sha256=score["predictions_sha256"],candidate_audit_sha256=score["candidate_audit_sha256"]))
    gt=json.loads(axis.pin(args.gt,score["gt_sha256"]))
    require(len(gt["images"])==705 and len(gt["annotations"])==1782,"Exact DEV705 truth")
    target,neutral,_=geom.geometry.polygon_support(gt,geom.vector)
    anns=defaultdict(list)
    eligible=set()
    for ann in gt["annotations"]:
        anns[ann["image_id"]].append(ann)
        if ann["id"] in target and ann["area"]<1024 and hull.hull(target[ann["id"]][0])[0]>=2:
            eligible.add(ann["id"])
    rows=[]
    stock_matches=0
    for im,block,aud in zip(gt["images"],blocks,details,strict=True):
        require(im["id"]==block["image_id"]==aud["image_id"] and len(block["coco"])==len(aud["candidates"]),
                "Exact original image/candidate order")
        pool=[]
        for idx,(c,meta) in enumerate(zip(block["coco"][:100],aud["candidates"][:100],strict=True),1):
            rle=c["segmentation"]
            area=int(codec.area(rle))
            require(meta["native_rank"]==idx and meta["score"]==c["score"] and meta["mask_area"]==area,
                    "Exact native box/score/RLE row")
            pool.append(dict(rank=idx,score=c["score"],rle=rle,area=area))
        if not anns[im["id"]]:
            continue
        with redirect_stdout(StringIO()):
            match,_,_=stock.freeze_coco(im,anns[im["id"]],pool)
        stock_matches+=len(match)
        for identity,rank in match.items():
            if identity in eligible and identity not in neutral:
                rows.append(proposal_rows(im,pool[rank-1],aud["candidates"][rank-1],target[identity][0],
                                          geom,bridge,codec,shapely))
    require(len(eligible)==1425 and stock_matches==1652 and len(rows)==1329,"Exact previously closed cohort")
    result=dict(status="DFINE_BOX_MASK_AREA_DEV_ACTUAL",DEV=True,FIT=False,TEST=False,AP=False,GT_opened=True,
                protocol_sha256=PROTOCOL,source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                prelabel_sha256=prelabel,model_forwards=0,detector_updates=0,eligible_gt=len(eligible),
                stock_matches=stock_matches,**fit_and_score(rows),worker_seconds=perf_counter()-started)
    axis.write(args.output/"RESULT_SAFE.json",result)
    print(json.dumps({k:result[k] for k in ("status","matched","groups","editability","gates","advance")},
                     sort_keys=True),flush=True)


if __name__=="__main__":
    def offline(event,_args):
        if event.startswith(("socket.","subprocess.")):
            raise RuntimeError("Offline CPU diagnostic")
    sys.addaudithook(offline)
    main()
