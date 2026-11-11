"""One locked FIT-only score of consensus transverse shifts versus same-budget controls."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from hashlib import sha256
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
from PIL import Image
from pycocotools import mask

PROTOCOL_SHA = "54788552bf9fa0998446545c4288d2eb5ceca54b72d14737dc315a89e9904261"
CORE_SHA = "0338556c8e688648e6f4cb3659fbe68384083dc17f0c5a3d9c28d8b8b8f89907"
MATCH_SHA = "8741bebb448120e37e0c10b58467a83d724f2294d20fddbc93720301285084aa"
GEOMETRY_SHA = "7cad8b51a9230101e2233db8455f84c28163abfb2ee8ddd8cbe8718a2e719bd0"
HULL_SHA = "dedd47d5453b2442ff02585c9b4d216917e5ed175e01eae07a354a22710f3a94"
LABEL_SHA = "e95cf26cc60a2c6fb651e907541380a256f2df7b1e153244a6c0f803b0b19f8f"
ARMS = ("zero","paired","shuffle","sum","one_side","logit")
FIELDS = ("iou","fp","fn","short")


def need(ok, why):
    if not ok:
        raise ValueError(why)


def exact(path, expected, cap=2**20):
    need(path == path.resolve(strict=True) and 0 < path.stat().st_size <= cap,"Canonical bounded closed input")
    raw = path.read_bytes()
    need(sha256(raw).hexdigest()==expected,"Exact input digest "+path.name)
    return raw


def load(path, expected, name):
    obj = ModuleType(name)
    obj.__file__=str(path)
    exec(compile(exact(path,expected),str(path),"exec"),obj.__dict__)
    return obj


def route(image, block, logits, queries, match, flank):
    pool,mapped,_ = match.ranked_candidates(block)
    keys = {p["query"] for p in pool if 32 <= p["area"] < 1024}
    need(set(queries)==keys and len(queries)==len(keys) and logits.dtype==np.float32
         and logits.shape==(len(queries),144,144) and np.isfinite(logits).all(),"Exact frozen low-grid area subset")
    arrays=dict(zip(queries,logits,strict=True))
    decisions={}
    for i,p in enumerate(pool):
        query=p["query"]
        if query not in keys:
            continue
        current=mask.decode(p["rle"]).astype(bool)
        profile=flank.from_mask(current)
        if profile is None:
            continue
        previous=[q["rle"] for q in pool[:i]]
        overlap=bool(previous and np.max(mask.iou([p["rle"]],previous,[0]*len(previous)))>=.20)
        if overlap:
            proposed=flank.ZERO
            moves={arm:0. for arm in ARMS}
        else:
            values=flank.contrasts(image,profile)
            proposed=flank.selected(values)
            moves=dict(zero=0.,**flank.controls(values,flank.logit_contrasts(arrays[query],profile,image.shape),proposed))
        decisions[p["rank"]]=(p,profile[0],profile[1],moves,overlap)
    return pool, decisions, mapped, len(keys)


def measure(row, truth, modules, hull):
    candidate, centroid, normal, moves,_=row
    rle=candidate["rle"]
    rectangles=modules.geometry.boxes_from_counts(modules.shift.runs(rle["counts"]),*rle["size"])
    area=float(candidate["area"])
    need(abs(area-np.prod(rectangles[:,2:]-rectangles[:,:2],axis=1).sum())<1e-10,"Exact disjoint native footprint")
    _,_,_,truth_normal=hull.hull(truth)
    ref_center=np.array((truth.centroid.x,truth.centroid.y))
    target=float(truth.area)
    metrics={}
    for arm,value in moves.items():
        vector=normal*value
        offset=np.tile(vector,2)
        boxes=rectangles+offset
        need((boxes[:,:2]>=-1e-10).all() and (boxes[:,2:]<=800+1e-10).all(),"Fixed movements preserve frame support")
        intersect=modules.geometry.overlap_area(boxes,truth)
        need(-1e-9<=intersect<=min(area,target)+1e-8,"Exact translated intersection bounds")
        overlap=min(intersect,area,target)
        metrics[arm]=dict(iou=100*overlap/(area+target-overlap),fp=100*(area-overlap)/target,
                          fn=100*(target-overlap)/target,short=float(abs(truth_normal@(centroid+vector-ref_center))))
    return metrics


def summarize(rows, draws, footprint):
    if not rows:
        return dict(pairs=0,values={})
    values={}
    for arm in ARMS:
        arm_rows=[record[1][arm] for record in rows]
        d={field:mean(v[field] for v in arm_rows) for field in FIELDS}
        d["iou_delta_vs_zero"] = d["iou"]-mean(record[1]["zero"]["iou"] for record in rows)
        d["strict_added"] = sum(record[1]["zero"]["iou"]<90<=record[1][arm]["iou"] for record in rows)
        d["strict_lost"] = sum(record[1][arm]["iou"]<90<=record[1]["zero"]["iou"] for record in rows)
        d["harmful"] = sum(record[1][arm]["iou"]<record[1]["zero"]["iou"]-1e-10 for record in rows)
        values[arm]=d
    paired = {}
    for arm in ("zero","shuffle","sum","one_side","logit"):
        paired[arm] = {field:footprint.paired_cluster_stats([
            (idx,v["paired"][field]-v[arm][field]) for idx,v in rows],draws) for field in FIELDS}
    return dict(pairs=len(rows),values=values,paired_minus_control=paired)


def run(args):
    start=perf_counter()
    source=Path(__file__).resolve().parent
    exact(source/"PROTOCOL.md",PROTOCOL_SHA)
    flank=load(source/"two_flank.py",CORE_SHA,"_flank")
    anchor=load(args.match_source,MATCH_SHA,"_fit_anchor")
    geometry=load(args.code_root/"MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py",GEOMETRY_SHA,"_fit_geometry")
    hull=load(args.code_root/"MASK_RTDETR_DEV705_FIXED_MATCH_GEOMETRY_20261112/fixed_match_geometry.py",HULL_SHA,"_fit_hull")
    sys.path.append(str(args.shapely_site.resolve(strict=True)))
    import shapely
    need(shapely.__version__=="2.0.2" and shapely.geos_version_string=="3.11.2","Exact original geometry")
    modules=geometry.geometry_modules(args.code_root)
    request=json.loads(exact(args.request,args.request_sha))
    need(request["FIT"] is True and request["DEV"] is request["TEST"] is False
         and request["producer_terminal_verified"] and len(request["images"])==36,"Closed FIT-only original capture")
    refs=request["images"]
    ids=[row["image_id"] for row in refs]
    groups=[row["group"] for row in refs]
    blocks=[json.loads(line) for line in exact(Path(request["prediction_path"]),request["prediction_sha"],32*2**20).splitlines()]
    need(len(blocks)==len(set(ids))==len(set(groups))==36 and [block["image_id"] for block in blocks]==ids,
         "Exact fixed distinct-source FIT roster and prediction order")
    routed={}
    counts=Counter()
    for row,block in zip(refs,blocks,strict=True):
        name=row["native_basename"]
        need(Path(name).name==name and name.split("_",1)[0]==row["group"],"Exact image group identity")
        img_raw=exact(args.fit_images/name,row["encoded_sha256"],4*2**20)
        with Image.open(BytesIO(img_raw)) as decoded:
            rgb=np.asarray(decoded.convert("RGB"))
        need(rgb.shape==(800,800,3) and sha256(rgb.tobytes()).hexdigest()==row["decoded_rgb_sha256"],"Exact native rendered image")
        logits=np.load(BytesIO(exact(Path(row["logit_path"]),row["logit_sha256"],16*2**20)),allow_pickle=False)
        pool,decision,mapped,geometric_area=route(rgb.mean(axis=2),block,logits,row["query_ids"],anchor,flank)
        counts.update(mapped=mapped,area_eligible=geometric_area,profile_eligible=len(decision),
                      neighbor_abstain=sum(v[4] for v in decision.values()),
                      acted=sum(v[3]["paired"]!=0 for v in decision.values()))
        routed[row["image_id"]]=(pool,decision)
    # Prediction-only routing for all 36 views has completed before annotations open.
    truth=json.loads(exact(args.fit_index,LABEL_SHA,32*2**20))
    need(len(truth["images"])==2937 and len(truth["annotations"])==9260,"Exact original HRSID FIT")
    images={item["id"]:item for item in truth["images"] if item["id"] in routed}
    ann_by=defaultdict(list)
    for item in truth["annotations"]:
        if item["image_id"] in images:
            ann_by[item["image_id"]].append(item)
    selected_gt=dict(images=list(images.values()),annotations=[a for rows in ann_by.values() for a in rows],categories=truth["categories"])
    polygons,neutral,_=modules.geometry.polygon_support(selected_gt,modules.vector)
    rows,small,acted_rows=[],[],[]
    active_prefixes=set()
    with redirect_stdout(StringIO()):
        for ix,iid in enumerate(ids):
            pool,decision=routed[iid]
            linked,_,_=anchor.freeze_coco(images[iid],ann_by[iid],pool)
            counts["original_matches"]+=len(linked)
            for target in ann_by[iid]:
                aid=target["id"]
                row=decision.get(linked.get(aid))
                if row is None or aid in neutral:
                    continue
                metrics=measure(row,polygons[aid][0],modules,hull)
                pair=(ix,metrics)
                rows.append(pair)
                if float(target["area"])<1024:
                    small.append(pair)
                if row[3]["paired"]!=0:
                    active_prefixes.add(groups[ix])
                    acted_rows.append(pair)
                    for mode in ("shuffle","sum","one_side","logit"):
                        counts["same_sign_"+mode]+=int(row[3][mode]==row[3]["paired"])
    counts.update(matched_eligible=len(rows),matched_small=len(small),matched_acted=len(acted_rows),
                  matched_acted_prefixes=len(active_prefixes),unmatched_or_unsupported_acted=counts["acted"]-len(acted_rows))
    draws=modules.footprint.bootstrap_draws(len(ids))
    results={name:summarize(a,draws,modules.footprint) for name,a in (("all",rows),("small",small),("acted",acted_rows))}
    values=results["all"]["values"]
    adequate=len(acted_rows)>=8 and len(active_prefixes)>=4
    lead=(adequate and values["paired"]["iou"]-values["zero"]["iou"]>=.25
          and all(values["paired"]["iou"]-values[v]["iou"]>=.10 for v in ("shuffle","logit"))
          and values["paired"]["fn"]<=values["zero"]["fn"])
    outcome="DIRECTIONAL_LEAD" if lead else "NO_SCOUT_LEAD" if adequate else "SPARSE_ACTION_SUPPORT"
    result=dict(status="HRSID_FIT_CENTERED_TWO_FLANK_SCOUT_ACTUAL",FIT=True,DEV=False,TEST=False,
                optimizer_updates=0,model_forward_calls=0,original_capture_forwards=36,images=36,AP=False,
                protocol_sha256=PROTOCOL_SHA,request_sha256=args.request_sha,counts=dict(counts),groups=results,
                gate=outcome,wall_seconds=perf_counter()-start)
    data=(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    with args.output.open("xb") as stream:
        stream.write(data)
    print(json.dumps(dict(status=result["status"],result_sha256=sha256(data).hexdigest(),gate=outcome)),flush=True)


def main():
    parser=ArgumentParser()
    for name in ("request","fit-images","fit-index","code-root","match-source","shapely-site","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--request-sha",required=True)
    args=parser.parse_args()
    need(sys.flags.isolated and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1","Isolated FIT-only CPU diagnostic")
    def offline(event,_):
        if event.startswith("socket."):
            raise PermissionError("Offline FIT-only scoring")
    sys.addaudithook(offline)
    run(args)


if __name__=="__main__":
    main()

