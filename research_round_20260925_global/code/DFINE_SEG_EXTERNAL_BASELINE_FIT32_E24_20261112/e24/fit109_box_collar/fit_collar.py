"""Original-box collar additions on the exact closed D-FINE FIT109 predictions."""
import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from hashlib import sha256
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
from pycocotools import mask as codec

PROTOCOL="a948c9052acc9b8df53869a80a4076364c167a9d73d5bc23a5a6b82086e749fa"
MATH="694adb27f9be1f99924d91dcb49dd61b95c437480b081752c493eb555273f648"
BASE="9ce987e68f996c3ef4146727867de8b56c08bc56b008665128c36854366d6d23"
RANK="cb438f8937f8e29dbdcbc441abdf5a788ac9c442fd63f1b3ad504b8ae0fcca0b"
OFFICIAL="aaa08f598877b4043de3b1d09bc107994566506eee6f5a8c9a43002aaa3d0ff0"
PRIOR="f81e686a1a008929cdc525a666fc83ad4378618ab218fbbf128a033fecbf0535"
PRELABEL="DFINE_FIT109_BOX_COLLAR_PRELABEL"
MODES=("raw","ordinary","connected","equal_count")
MIB=2**20


def load(path,digest,name):
    raw=path.read_bytes()
    if sha256(raw).hexdigest()!=digest:
        raise ValueError("Exact fixed ordinary source")
    obj=ModuleType(name)
    obj.__file__=str(path)
    exec(compile(raw,str(path),"exec"),obj.__dict__)
    return obj


def prepare(a,base,rank,math,core,blocks):
    a.output.mkdir()
    h=sha256()
    totals=Counter()
    modes={mode:Counter() for mode in MODES[1:]}
    with (a.output/"ACTIONS_PRIVATE.jsonl").open("xb") as out:
        for block in blocks:
            ref=block["probabilities"]
            raw=base.exact(core/ref["filename"],ref["sha256"],11*MIB)
            planes=np.load(BytesIO(raw),allow_pickle=False)
            base.need(planes.shape==(100,160,160) and planes.dtype==np.float32,"Exact native top100 planes")
            actions={}
            for index,(cand,row) in enumerate(zip(block["candidates"][:100],block["coco"][:100],strict=True)):
                area=cand["mask_area"]
                totals.update(candidates=1,empty=int(area==0))
                base.need(cand["native_rank"]==index+1 and cand["probability_row"]==index and cand["score"]==row["score"],
                          "Exact unchanged stock rank and probability lineage")
                if not 1<=area<1024:
                    continue
                totals["eligible"]+=1
                stock=codec.decode(row["segmentation"]).astype(bool)
                base.need(int(stock.sum())==area,"Exact raw binary area")
                region,coords=math.collar_region(cand["box_xyxy"],stock.shape)
                p=rank.sample_probability(planes[index],coords,800)
                positive=np.flatnonzero(stock)
                original_p=rank.sample_probability(planes[index],positive,800)
                totals.update(collar_pixels=len(p),collar_exact_half=int((p==.50).sum()),
                              stock_positive_pixels=len(original_p),stock_positive_below_half=int((original_p<.50).sum()),
                              stock_positive_exact_half=int((original_p==.50).sum()))
                edits=math.choose(stock,region,coords,p)
                base.need(len(edits["connected"])==len(edits["equal_count"])<=len(edits["ordinary"]),
                          "Exact control candidate addition counts")
                if edits["ordinary"]:
                    actions[str(index+1)]=edits
                for mode,ids in edits.items():
                    modes[mode].update(edited=int(bool(ids)),added_pixels=len(ids))
            line=base.encoded(dict(image_id=block["image_id"],group=block["group"],actions=actions))
            out.write(line)
            h.update(line)
            if block["ordinal"]%25==0 or block["ordinal"]==109:
                print(json.dumps(dict(event="DFINE_FIT109_COLLAR_PREPARE_PROGRESS",images=block["ordinal"])),flush=True)
        out.flush()
        os.fsync(out.fileno())
    base.need(totals["candidates"]==10900 and dict(modes["connected"])==dict(modes["equal_count"]),
              "Exact all-candidate and count control closure")
    result=dict(status=PRELABEL,protocol_sha256=PROTOCOL,math_sha256=MATH,
                worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),actions_sha256=h.hexdigest(),
                FIT=True,DEV=False,TEST=False,GT_opened=False,AP=False,model_forwards=0,optimizer_updates=0,
                images=109,counts=dict(totals),predicted_edit_totals={m:dict(t) for m,t in modes.items()})
    digest=base.seal(a.output/"PRELABEL_SAFE.json",base.encoded(result))
    print(json.dumps(dict(status=PRELABEL,result_sha256=digest),sort_keys=True),flush=True)


def metric(rle,area,truth,truth_raster,neighbors,box,act,geom,math,base):
    import shapely
    cells=geom.geometry.boxes_from_counts(geom.shift.runs(rle["counts"]),*rle["size"])
    base.need(np.prod(cells[:,2:]-cells[:,:2],axis=1).sum()==area,"Exact original disjoint pixel cells")
    raw_i=geom.geometry.overlap_area(cells,truth)
    _,collar=math.collar_region(box,(800,800))
    y,x=np.divmod(collar,800)
    fractions=np.asarray(shapely.area(shapely.intersection(shapely.box(x,y,x+1,y+1),truth)))
    base.need(np.isfinite(fractions).all() and (fractions>=0).all() and (fractions<=1+1e-12).all(),
              "Exact continuous truth cell fractions")

    def indices(ids):
        a=np.asarray(ids,dtype=np.int64)
        idx=np.searchsorted(collar,a)
        base.need(len(a)==len(set(ids)) and np.all(idx<len(collar)) and np.array_equal(collar[idx],a),
                  "Exact unique outer collar positions")
        return a,idx

    metrics=[]
    for mode in MODES:
        ids,idx=indices(act[mode] if act and mode!="raw" else [])
        contribution=float(fractions[idx].sum())
        geometry=math.geometry(area,truth.area,raw_i,len(ids),contribution)
        near=int(neighbors[ids].sum())
        metrics.append(np.concatenate((geometry,[len(ids),int(truth_raster[ids].sum()),contribution,near,int(near>0)])))
    _,qualified=indices(act["ordinary"] if act else [])
    oracle_all=math.max_iou(area,truth.area,raw_i,fractions)
    oracle_p=math.max_iou(area,truth.area,raw_i,fractions[qualified])
    base.need(100*oracle_all+1e-9>=100*oracle_p and 100*oracle_p+1e-9>=max(m[0] for m in metrics),
              "Exact nested additive-only truth bounds")
    return np.asarray(metrics),oracle_all,oracle_p


def summarize(rows):
    group=defaultdict(list)
    for name,value,_,_ in rows:
        group[name].append(value)
    keys=sorted(group)
    groups=np.asarray([np.mean(group[key],axis=0) for key in keys])
    pairs=np.asarray([r[1] for r in rows])
    rng=np.random.Generator(np.random.PCG64(5217))
    draws=rng.integers(len(groups),size=(4096,len(groups)))
    out={}
    for i,m in enumerate(MODES):
        totals=pairs[:,i].sum(axis=0)
        out[m]=dict(group_equal_iou_points=float(groups[:,i,0].mean()),
                    pair_equal_iou_points=float(pairs[:,i,0].mean()),
                    fp_per_truth_points=float(groups[:,i,1].mean()),fn_per_truth_points=float(groups[:,i,2].mean()),
                    strict_75=int((pairs[:,i,0]>=75).sum()),strict_90=int((pairs[:,i,0]>=90).sum()),
                    added_pixels=int(totals[3]),binary_true_added=int(totals[4]),continuous_true_added=float(totals[5]),
                    binary_add_purity=float(totals[4]/totals[3]) if totals[3] else None,
                    continuous_add_purity=float(totals[5]/totals[3]) if totals[3] else None,
                    neighbor_added_pixels=int(totals[6]),neighbor_affected_matches=int(totals[7]),
                    neighbor_added_rate=float(totals[6]/totals[3]) if totals[3] else None)
    effects={}
    for i,j in ((1,0),(2,0),(3,0),(2,1),(2,3)):
        delta=groups[:,i,0]-groups[:,j,0]
        effects[MODES[i]+"_minus_"+MODES[j]]=dict(group_equal_delta_points=float(delta.mean()),
              interval95_points=np.percentile(delta[draws].mean(axis=1),[2.5,97.5]).tolist())
    raw=out["raw"]["strict_90"]
    bound_all=sum(row[2]>=.90 for row in rows)
    bound_p=sum(row[3]>=.90 for row in rows)
    return dict(matches=len(rows),source_groups=len(keys),metrics=out,effects=effects,
                oracle_strict90=dict(raw=raw,all_collar=bound_all,probability_eligible=bound_p,
                                     new_all_collar=bound_all-raw,new_probability_eligible=bound_p-raw,
                                     denominator=len(rows)))


def read_labels(a,base,blocks):
    source=json.loads(base.exact(a.fit_gt,base.GT,64*MIB))
    base.need(len(source["images"])==2937 and len(source["annotations"])==9260,"Exact original author FIT population")
    lookup={r["id"]:r for r in source["images"]}
    images=[lookup[b["image_id"]] for b in blocks]
    ids={r["id"] for r in images}
    base.need(len(ids)==109 and all(im["file_name"]==b["basename"] and (im["height"],im["width"])==(800,800)
                                 for im,b in zip(images,blocks,strict=True)),"Exact accepted native FIT image join")
    anns=defaultdict(list)
    for a in source["annotations"]:
        if a["image_id"] in ids:
            anns[a["image_id"]].append(a)
    return dict(images=images,categories=source["categories"],annotations=[r for rr in anns.values() for r in rr]),anns


def read_predictions(a,base,math,blocks,planned,gt,anns,receipt):
    sys.path.append(str(a.shapely_site.resolve(strict=True)))
    import shapely
    base.need(shapely.__version__=="2.0.2" and shapely.geos_version_string=="3.11.2","Exact geometry")
    engine=base.load(a.code/"MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py",base.GEOM,"_collar_geom")
    geom=engine.geometry_modules(a.code)
    matcher=base.load(a.code/"HRSID_FIT_HULL_CDF_20261112/stock_coco_match.py",base.MATCH,"_collar_match")
    truth,neutral,_=geom.geometry.polygon_support(gt,geom.vector)
    modes={m:[] for m in MODES}
    budgets={m:Counter() for m in MODES}
    scope=Counter()
    rows=[]
    for im,b,p in zip(gt["images"],blocks,planned,strict=True):
        image=b["image_id"]
        base.need(image==im["id"]==p["image_id"] and b["group"]==p["group"],"Exact prelabel and GT identity")
        items=list(zip(b["candidates"][:100],b["coco"][:100],strict=True))
        pool=[dict(rank=i,score=r["score"],rle=r["segmentation"],area=c["mask_area"]) for i,(c,r) in enumerate(items,1)]
        with redirect_stdout(StringIO()):
            matched,rasters,_=matcher.freeze_coco(im,anns[image],pool)
        scope["stock_matches"]+=len(matched)
        by_rank={rank:identity for identity,rank in matched.items()}
        binary={identity:codec.decode(r).astype(bool).ravel(order="C") for identity,r in rasters.items()
                if identity in truth and identity not in neutral}
        for i,(c,r) in enumerate(items,1):
            act=p["actions"].get(str(i))
            identity=by_rank.get(i)
            role="valid" if identity in binary else "neutral" if identity is not None else "unmatched"
            for mode in MODES:
                pixels=act[mode] if act and mode!="raw" else []
                if pixels:
                    added=base.apply(r["segmentation"],(pixels,[]))
                    base.need(int(codec.area(added))==c["mask_area"]+len(pixels),"Exact no-delete raw stock preservation")
                    modes[mode].append(dict(r,segmentation=added))
                    budgets[mode].update({"edited_"+role:1,"added_"+role:len(pixels),
                                          "edited":1,"added":len(pixels)})
                else:
                    modes[mode].append(r)
                    budgets[mode]["no_op"]+=1
            if role!="valid" or not 1<=c["mask_area"]<1024:
                continue
            other=np.zeros(800*800,dtype=bool)
            for key,reference in binary.items():
                if key!=identity:
                    other|=reference
            scope["eligible_valid"]+=1
            values,all_bound,p_bound=metric(r["segmentation"],c["mask_area"],truth[identity][0],
                                                binary[identity],other,c["box_xyxy"],act,geom,math,base)
            rows.append((b["group"],values,all_bound,p_bound))
    base.need(all(len(v)==10900 for v in modes.values()) and scope["stock_matches"]==467 and len(rows)==scope["eligible_valid"],
              "Exact original top100 rows and stock match denominator")
    for mode in MODES:
        budget=budgets[mode]
        base.need(budget["edited"]+budget["no_op"]==10900 and budget["edited"]==
                  budget["edited_valid"]+budget["edited_neutral"]+budget["edited_unmatched"],
                  "Exact candidate edit scope partition")
        if mode!="raw":
            fixed=receipt["predicted_edit_totals"][mode]
            base.need((budget["edited"],budget["added"])==(fixed["edited"],fixed["added_pixels"]),
                      "Exact sealed all-candidate budget")
    return modes,{m:dict(t) for m,t in budgets.items()},rows,dict(scope)


def score(a,base,math,blocks):
    receipt=json.loads(base.exact(a.prelabel/"PRELABEL_SAFE.json",a.prelabel_sha))
    wait=json.loads(base.exact(a.wait,a.wait_sha))
    t=wait["terminal"]
    tail=json.loads(t["output"].strip().splitlines()[-1])
    base.need(t["exit_code"]==0 and tail==dict(status=PRELABEL,result_sha256=a.prelabel_sha)
              and receipt["protocol_sha256"]==PROTOCOL and receipt["math_sha256"]==MATH
              and receipt["worker_sha256"]==sha256(Path(__file__).read_bytes()).hexdigest()
              and receipt["GT_opened"] is False and receipt["AP"] is False,"Exact naturally closed label-free scope")
    planned=[json.loads(l) for l in base.exact(a.prelabel/"ACTIONS_PRIVATE.jsonl",receipt["actions_sha256"],64*MIB).splitlines()]
    base.need(len(planned)==len(blocks)==109,"Exact label-free artifact scope")
    previous=json.loads(base.exact(a.prior_result,PRIOR))
    scorer=load(a.official,OFFICIAL,"_exact_fixed_coco")
    a.output.mkdir()
    pre=base.seal(a.output/"PRELABEL_SCORE_SAFE.json",base.encoded(dict(status="DFINE_FIT109_COLLAR_SCORE_PRELABEL",
                  FIT=True,DEV=False,TEST=False,GT_opened=False,AP=False,prelabel_sha256=a.prelabel_sha,
                  original_tool_wait_sha256=a.wait_sha,protocol_sha256=PROTOCOL)))
    gt,anns=read_labels(a,base,blocks)
    modes,budgets,rows,scope=read_predictions(a,base,math,blocks,planned,gt,anns,receipt)
    coco={m:scorer.official(gt,v) for m,v in modes.items()}
    base.need(all(abs(coco["raw"][k]-v)<=1e-12 for k,v in previous["pooled_oof_coco"]["raw"].items()),
              "Exact independently closed original raw FIT109 metrics")
    summary=summarize(rows)
    metrics=summary["metrics"]
    effects=summary["effects"]
    strong_ap=max(coco[m]["AP"] for m in ("ordinary","equal_count"))
    strong_iou=max(metrics[m]["group_equal_iou_points"] for m in ("ordinary","equal_count"))
    def raw_gates(mode):
        d=effects[mode+"_minus_raw"]
        return dict(AP_gain=coco[mode]["AP"]-coco["raw"]["AP"]>=.50,
                    geometry_gain=d["group_equal_delta_points"]>=.20 and d["interval95_points"][0]>0,
                    AP75_retained=coco[mode]["AP75"]>=coco["raw"]["AP75"]-1e-10,
                    strict90_retained=metrics[mode]["strict_90"]>=metrics["raw"]["strict_90"])
    gates=raw_gates("connected")
    gates.update(beats_strong_control=(coco["connected"]["AP"]-strong_ap>=.10 or
                  metrics["connected"]["group_equal_iou_points"]-strong_iou>=.20),
                 neighbor_retained=metrics["connected"]["neighbor_added_pixels"]<=
                 min(metrics[m]["neighbor_added_pixels"] for m in ("ordinary","equal_count")))
    result=dict(status="DFINE_FIT109_BOX_COLLAR_ACTUAL",FIT=True,DEV=False,TEST=False,GT_opened=True,AP=True,
                model_forwards=0,optimizer_updates=0,protocol_sha256=PROTOCOL,math_sha256=MATH,
                worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),prelabel_sha256=a.prelabel_sha,
                prelabel_score_sha256=pre,counts=receipt["counts"],images=109,annotations=len(gt["annotations"]),
                budgets=budgets,match_scope=scope,pooled_coco=coco,fixed_geometry=summary,gates=gates,
                advance=all(gates.values()),ordinary_raw_gates=raw_gates("ordinary"))
    digest=base.seal(a.output/"RESULT_SAFE.json",base.encoded(result))
    print(json.dumps(dict(status=result["status"],result_sha256=digest,advance=result["advance"],gates=gates),
                     sort_keys=True),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    stages=parser.add_subparsers(dest="stage",required=True)
    common=("protocol","capture","accepted","base","rank-math","collar-math","output")
    extra=("prelabel","wait","fit-gt","code","shapely-site","prior-result","official")
    for mode,fields in (("prepare",()),("score",extra)):
        stage=stages.add_parser(mode)
        for f in common+fields:
            stage.add_argument("--"+f,type=Path,required=True)
        if mode=="score":
            stage.add_argument("--prelabel-sha",required=True)
            stage.add_argument("--wait-sha",required=True)
    a=parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="-1":
        raise ValueError("Isolated CPU only")
    base=load(a.base,BASE,"_exact_balanced_carrier")
    base.exact(a.protocol,PROTOCOL)
    rank=load(a.rank_math,RANK,"_exact_cpu_probability")
    math=load(a.collar_math,MATH,"_exact_collar_math")
    core,blocks=base.read_core(a)
    if a.stage=="prepare":
        prepare(a,base,rank,math,core,blocks)
    else:
        score(a,base,math,blocks)


if __name__=="__main__":
    main()