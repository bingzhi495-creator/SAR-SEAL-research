"""Frozen D-FINE FIT109 label-free pairing and equal-budget CPU head training."""
import argparse
from collections import Counter,defaultdict
from contextlib import redirect_stdout
from hashlib import sha256
from io import BytesIO,StringIO
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from zipfile import ZipFile

import numpy as np
from pycocotools import mask as codec

PROTOCOL="f353ceb224e6f4551d900a6b0d682c5647621141d01c62436c1a24bdc47085bb"
MATH="8fe7a2c146c08c89f377b1f20b321276a60356de4bdbb27be2a00544dfc77962"
TRAIN="3111c34afd791cb15987ed94c5e898de4c3ff2cad61677b616420651335d812c"
BASE="9ce987e68f996c3ef4146727867de8b56c08bc56b008665128c36854366d6d23"
RANK="cb438f8937f8e29dbdcbc441abdf5a788ac9c442fd63f1b3ad504b8ae0fcca0b"
OFFICIAL="aaa08f598877b4043de3b1d09bc107994566506eee6f5a8c9a43002aaa3d0ff0"
PRIOR="f81e686a1a008929cdc525a666fc83ad4378618ab218fbbf128a033fecbf0535"
COHORT="70e9a0e8fb30f5ba70b48b9067c1b95d07bfec23d6b248d92c5983727acceb6e"
PRELABEL="DFINE_FIT109_PAIRED_POINT_PRELABEL"
MODES=("raw","independent","shuffled","paired","equal_count_independent")
GEOMETRY=(*MODES,"oracle")
MIB=2**20


def load(path,digest,name):
    raw=path.read_bytes()
    if sha256(raw).hexdigest()!=digest:
        raise ValueError("Exact fixed ordinary source")
    module=ModuleType(name)
    module.__file__=str(path)
    exec(compile(raw,str(path),"exec"),module.__dict__)
    return module


def folds_for(blocks,math):
    groups=sorted({b["group"] for b in blocks},key=lambda g:(sha256((math.SALT+g).encode()).hexdigest(),g))
    return {g:i%5 for i,g in enumerate(groups)}


def image_roster(archive,blocks,base):
    raw=archive.read("COHORT_PRIVATE.json")
    doc=json.loads(raw)
    selected={r["image_id"]:r for r in doc["images"] if r["core"]}
    base.need(sha256(raw).hexdigest()==COHORT and doc["fit_only"] is True and len(selected)==109,
              "Exact closed FIT109 core roster")
    for b in blocks:
        r=selected[b["image_id"]]
        base.need((r["native_basename"],r["group"],r["decoded_rgb_sha256"])==
                  (b["basename"],b["group"],b["rgb_sha256"]),"Exact image-only candidate/source join")
    return selected


def read_png(archive,row,base):
    from PIL import Image

    info=archive.getinfo("images/"+row["native_basename"])
    base.need(0<info.file_size==row["encoded_bytes"]<=4*MIB,"Exact source PNG member")
    raw=archive.read(info)
    base.need(sha256(raw).hexdigest()==row["encoded_sha256"],"Exact source PNG bytes")
    with Image.open(BytesIO(raw)) as decoded:
        base.need(decoded.format=="PNG" and decoded.size==(800,800),"Exact native chip shape")
        rgb=np.asarray(decoded.convert("RGB"),dtype=np.uint8)
    base.need(sha256(rgb.tobytes(order="C")).hexdigest()==row["decoded_rgb_sha256"],"Exact source FIT RGB")
    return rgb


def prepare(a,base,rank,math,core,blocks):
    folds=folds_for(blocks,math)
    a.output.mkdir()
    totals=Counter()
    meta=[]
    base.need(a.bundle.stat().st_size==151320868,"Exact closed historical source archive")
    with ZipFile(a.bundle) as archive:
        roster=image_roster(archive,blocks,base)
        for b in blocks:
            ref=b["probabilities"]
            raw=base.exact(core/ref["filename"],ref["sha256"],11*MIB)
            probs=np.load(BytesIO(raw),allow_pickle=False)
            base.need(probs.shape==(100,160,160) and probs.dtype==np.float32,"Exact captured author probability carrier")
            image=math.image_features(read_png(archive,roster[b["image_id"]],base))
            coords,features,depth,ranks=[],[],[],[]
            for i,(c,r) in enumerate(zip(b["candidates"][:100],b["coco"][:100],strict=True),1):
                area=c["mask_area"]
                totals.update(candidates=1,empty=int(area==0))
                base.need(c["native_rank"]==i and c["probability_row"]==i-1 and c["score"]==r["score"],
                          "Exact original native candidate order")
                if not 32<=area<1024:
                    continue
                totals["area_eligible"]+=1
                stock=codec.decode(r["segmentation"]).astype(bool)
                paired=math.paired_points(stock,c["box_xyxy"],area,r["score"],probs[i-1],image,rank)
                if paired is None:
                    totals["zero_pairs"]+=1
                    continue
                points,feat,layer,discord,half=paired
                totals.update(supported_candidates=1,pairs=len(points),threshold_discord=discord,exact_half=half)
                coords.append(points)
                features.append(feat)
                depth.append(layer)
                ranks.extend([i]*len(points))
            n=len(ranks)
            array=dict(ranks=np.asarray(ranks,dtype=np.int16),
                       coordinates=np.concatenate(coords) if coords else np.empty((0,2),dtype=np.int32),
                       features=np.concatenate(features) if features else np.empty((0,2,math.FEATURES),dtype=np.float32),
                       layers=np.concatenate(depth) if depth else np.empty((0,2),dtype=np.int8))
            buf=BytesIO()
            np.savez(buf,**array)
            filename=f"{b['ordinal']:03d}.npz"
            digest=base.seal(a.output/filename,buf.getvalue())
            meta.append(dict(ordinal=b["ordinal"],image_id=b["image_id"],group=b["group"],fold=folds[b["group"]],
                             file=filename,sha256=digest,pairs=n))
            if b["ordinal"]%25==0 or b["ordinal"]==109:
                print(json.dumps(dict(event="DFINE_FIT109_PAIRED_POINT_PREPARE_PROGRESS",images=b["ordinal"])),
                      flush=True)
    base.need(totals["candidates"]==10900 and totals["area_eligible"]==totals["zero_pairs"]+totals["supported_candidates"],
              "Exact unchanged label-free support partition")
    meta_sha=base.seal(a.output/"POINTS_MANIFEST_PRIVATE.json",base.encoded(meta))
    receipt=dict(status=PRELABEL,FIT=True,DEV=False,TEST=False,GT_opened=False,AP=False,images=109,
                 protocol_sha256=PROTOCOL,worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                 math_sha256=MATH,train_sha256=TRAIN,points_manifest_sha256=meta_sha,source_cohort_sha256=COHORT,
                 counts=dict(totals),fold_counts={str(i):sum(f==i for f in folds.values()) for i in range(5)})
    digest=base.seal(a.output/"PRELABEL_SAFE.json",base.encoded(receipt))
    print(json.dumps(dict(status=PRELABEL,result_sha256=digest),sort_keys=True),flush=True)


def sealed_points(a,base,math,blocks):
    receipt=json.loads(base.exact(a.prelabel/"PRELABEL_SAFE.json",a.prelabel_sha))
    wait=json.loads(base.exact(a.wait,a.wait_sha))
    term=wait["terminal"]
    tail=json.loads(term["output"].strip().splitlines()[-1])
    base.need(term["exit_code"]==0 and tail==dict(status=PRELABEL,result_sha256=a.prelabel_sha)
              and receipt["GT_opened"] is False and receipt["AP"] is False and
              (receipt["protocol_sha256"],receipt["math_sha256"],receipt["train_sha256"])==(PROTOCOL,MATH,TRAIN)
              and receipt["worker_sha256"]==sha256(Path(__file__).read_bytes()).hexdigest(),
              "Exact natural prediction-only preparation")
    meta=json.loads(base.exact(a.prelabel/"POINTS_MANIFEST_PRIVATE.json",receipt["points_manifest_sha256"]))
    folds=folds_for(blocks,math)
    results=[]
    for b,m in zip(blocks,meta,strict=True):
        base.need((b["ordinal"],b["image_id"],b["group"],folds[b["group"]])==
                  (m["ordinal"],m["image_id"],m["group"],m["fold"]) and m["file"]==f"{b['ordinal']:03d}.npz",
                  "Exact fixed native point roster")
        raw=base.exact(a.prelabel/m["file"],m["sha256"],MIB)
        with np.load(BytesIO(raw),allow_pickle=False) as package:
            r,c,f,l=(package[k] for k in ("ranks","coordinates","features","layers"))
        n=m["pairs"]
        base.need(r.shape==(n,) and c.shape==l.shape==(n,2) and f.shape==(n,2,math.FEATURES)
                  and r.dtype==np.int16 and c.dtype==np.int32 and f.dtype==np.float32 and l.dtype==np.int8
                  and np.isfinite(f).all() and np.all((c>=0)&(c<640000)) and np.all((r>=1)&(r<=100)),
                  "Exact finite native point feature array")
        by={}
        for rank in np.unique(r):
            select=r==rank
            coords,feat,depth=c[select],f[select],l[select]
            base.need(1<=len(coords)<=32 and all(len(np.unique(coords[:,i]))==len(coords) for i in (0,1))
                      and np.all(feat[:,0,1]==0) and np.all(feat[:,1,1]==1),"Exact disjoint opposite-sign point pairs")
            by[int(rank)]=dict(coordinates=coords,features=feat,contexts=math.contexts(feat,depth,b["group"],int(rank)))
        results.append(dict(fold=m["fold"],group=m["group"],data=by))
    base.need(sum(len(v["coordinates"]) for r in results for v in r["data"].values())==receipt["counts"]["pairs"],
              "Exact label-free point partition")
    return receipt,results


def read_labels(a,base,blocks):
    gt=json.loads(base.exact(a.fit_gt,base.GT,64*MIB))
    all_images={im["id"]:im for im in gt["images"]}
    images=[all_images[b["image_id"]] for b in blocks]
    base.need(len(gt["images"])==2937 and len(gt["annotations"])==9260 and
              all(i["file_name"]==b["basename"] for i,b in zip(images,blocks,strict=True)),"Exact author FIT identity")
    ids={im["id"] for im in images}
    anns=defaultdict(list)
    for r in gt["annotations"]:
        if r["image_id"] in ids:
            anns[r["image_id"]].append(r)
    return dict(images=images,categories=gt["categories"],annotations=[r for rows in anns.values() for r in rows]),anns


def fixed_slots(a,base,blocks,planned,gt,anns):
    sys.path.append(str(a.shapely_site.resolve(strict=True)))
    import shapely
    base.need(shapely.__version__=="2.0.2" and shapely.geos_version_string=="3.11.2","Exact continuous geometry")
    engine=base.load(a.code/"MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py",base.GEOM,"_pair_geom")
    geom=engine.geometry_modules(a.code)
    matcher=base.load(a.code/"HRSID_FIT_HULL_CDF_20261112/stock_coco_match.py",base.MATCH,"_pair_stock")
    truth,neutral,_=geom.geometry.polygon_support(gt,geom.vector)
    slots=[]
    scope=Counter()
    for b,p,im in zip(blocks,planned,gt["images"],strict=True):
        items=list(zip(b["candidates"][:100],b["coco"][:100],strict=True))
        pool=[dict(rank=i,score=r["score"],area=c["mask_area"],rle=r["segmentation"])
              for i,(c,r) in enumerate(items,1)]
        with redirect_stdout(StringIO()):
            matches,_,_=matcher.freeze_coco(im,anns[im["id"]],pool)
        rank_to_gt={rank:k for k,rank in matches.items()}
        scope["stock_matches"]+=len(matches)
        for rank,(c,r) in enumerate(items,1):
            identity=rank_to_gt.get(rank)
            role="valid" if identity in truth and identity not in neutral else "neutral" if identity is not None else "unmatched"
            area=c["mask_area"]
            data=p["data"].get(rank)
            slot=dict(group=b["group"],fold=p["fold"],area=area,row=r,role=role,data=data,truth=None,raw=None)
            if role=="valid" and 32<=area<1024:
                polygon=truth[identity][0]
                rle=r["segmentation"]
                boxes=geom.geometry.boxes_from_counts(geom.shift.runs(rle["counts"]),*rle["size"])
                base.need(np.prod(boxes[:,2:]-boxes[:,:2],axis=1).sum()==area,"Exact native disjoint cell area")
                i=geom.geometry.overlap_area(boxes,polygon)
                slot["raw"]=(polygon.area,i)
                scope["fixed_valid"]+=1
                if data is not None:
                    y,x=np.divmod(data["coordinates"],800)
                    target=np.asarray(shapely.area(shapely.intersection(shapely.box(x,y,x+1,y+1),polygon)))
                    base.need(target.shape==data["coordinates"].shape and np.all((target>=0)&(target<=1+1e-12)),
                              "Exact fractional source point supervision")
                    slot["truth"]=target
            slots.append(slot)
    base.need(scope["stock_matches"]==467 and scope["fixed_valid"]==412 and len(slots)==10900,
              "Exact prior original matching and candidate cohort")
    return slots,dict(scope)


def train_and_select(slots,math,learner):
    chosen={m:[[] for _ in slots] for m in MODES}
    audits=[]
    for fold in range(5):
        models,center,scale,audit=learner.train(slots,fold)
        edits=Counter()
        for i,s in enumerate(slots):
            data=s["data"]
            if s["fold"]!=fold or data is None:
                continue
            cap=min(8,s["area"]//10,len(data["coordinates"]))
            f=data["features"]
            gains={m:learner.gains(model,f,data["contexts"][m],center,scale) for m,model in models.items()}
            for m in learner.HEADS:
                chosen[m][i]=math.choose_pairs(gains[m],cap)
                edits[m]+=len(chosen[m][i])
            chosen["equal_count_independent"][i]=math.choose_pairs(gains["independent"],cap,forced=len(chosen["paired"][i]))
        audits.append(dict(fold=fold,selected_pairs=dict(edits),**audit))
        print(json.dumps(dict(event="DFINE_FIT109_PAIRED_POINT_FOLD_TRAINED",fold=fold,steps=200,
                              paired_edits=edits["paired"]),sort_keys=True),flush=True)
    return chosen,audits


def fixed_geometry(slots,chosen,math):
    rows=[]
    for i,s in enumerate(slots):
        if s["raw"] is None:
            continue
        truth_area,raw_i=s["raw"]
        q=s["truth"]
        oracle=math.choose_pairs(q[:,0]-q[:,1],min(8,s["area"]//10,len(q))) if q is not None else []
        metrics=[]
        for action in [chosen[m][i] for m in MODES]+[oracle]:
            change=float((q[action,0]-q[action,1]).sum()) if action else 0.
            intersection=raw_i+change
            if not -1e-8<=intersection<=min(s["area"],truth_area)+1e-8:
                raise ValueError("Exact balanced continuous area witness")
            intersection=float(np.clip(intersection,0,min(s["area"],truth_area)))
            metrics.append((100*intersection/(s["area"]+truth_area-intersection),
                            100*(s["area"]-intersection)/truth_area,100*(truth_area-intersection)/truth_area))
        rows.append((s["group"],s["fold"],np.asarray(metrics)))
    by=defaultdict(list)
    foldmap={}
    for name,fold,values in rows:
        by[name].append(values)
        foldmap[name]=fold
    keys=sorted(by)
    groups=np.asarray([np.mean(by[g],axis=0) for g in keys])
    pairs=np.asarray([v for _,_,v in rows])
    folds=np.asarray([foldmap[k] for k in keys])
    draws=np.random.Generator(np.random.PCG64(5217)).integers(len(keys),size=(4096,len(keys)))
    metrics={}
    for i,m in enumerate(GEOMETRY):
        metrics[m]=dict(group_equal_iou_points=float(groups[:,i,0].mean()),
                        pair_equal_iou_points=float(pairs[:,i,0].mean()),
                        fp_per_truth_points=float(groups[:,i,1].mean()),fn_per_truth_points=float(groups[:,i,2].mean()),
                        strict_75=int((pairs[:,i,0]>=75).sum()),strict_90=int((pairs[:,i,0]>=90).sum()))
    effects={}
    for i,m in enumerate(GEOMETRY):
        if m=="paired":
            continue
        delta=groups[:,3,0]-groups[:,i,0]
        effects[m]=dict(delta_points=float(delta.mean()),
                        interval95_points=np.percentile(delta[draws].mean(axis=1),[2.5,97.5]).tolist(),
                        folds=[float(delta[folds==f].mean()) for f in range(5)])
    return dict(pairs=len(rows),source_groups=len(keys),metrics=metrics,paired_minus=effects)


def score(a,base,math,blocks):
    receipt,planned=sealed_points(a,base,math,blocks)
    previous=json.loads(base.exact(a.prior_result,PRIOR))
    official=load(a.official,OFFICIAL,"_fixed_cohort_coco")
    learner=load(a.training,TRAIN,"_equal_cpu_point_heads")
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    a.output.mkdir()
    pre=base.seal(a.output/"PRELABEL_SCORE_SAFE.json",base.encoded(dict(status="DFINE_FIT109_PAIRED_POINT_SCORE_PRELABEL",
                  FIT=True,DEV=False,TEST=False,GT_opened=False,AP=False,prelabel_sha256=a.prelabel_sha,
                  original_prepare_wait_sha256=a.wait_sha,protocol_sha256=PROTOCOL)))
    gt,anns=read_labels(a,base,blocks)
    slots,scope=fixed_slots(a,base,blocks,planned,gt,anns)
    chosen,training=train_and_select(slots,math,learner)
    cohorts={}
    budgets={}
    for m in MODES:
        rows=[]
        counts=Counter()
        for i,s in enumerate(slots):
            row=s["row"]
            ids=chosen[m][i]
            if ids:
                coords=s["data"]["coordinates"][ids]
                edited=base.apply(row["segmentation"],(coords[:,0].tolist(),coords[:,1].tolist()))
                base.need(int(codec.area(edited))==s["area"],"Exact balanced area for all native candidates")
                rows.append(dict(row,segmentation=edited))
                counts.update(edited=1,swaps=len(ids))
                counts.update({"edited_"+s["role"]:1})
            else:
                rows.append(row)
                counts["no_op"]+=1
        base.need(len(rows)==10900 and counts["no_op"]+counts["edited"]==10900,"Full unchanged candidate scores and count")
        cohorts[m]=official.official(gt,rows)
        budgets[m]={k:counts[k] for k in ("edited","swaps","edited_valid","edited_unmatched","edited_neutral","no_op")}
    base.need(all(abs(cohorts["raw"][k]-v)<=1e-12 for k,v in previous["pooled_oof_coco"]["raw"].items()) and
              budgets["paired"]["swaps"]==budgets["equal_count_independent"]["swaps"],"Exact raw witness and same-count control")
    geometry=fixed_geometry(slots,chosen,math)
    comparisons=geometry["paired_minus"]
    strongest=max(("raw","independent","shuffled"),key=lambda m:cohorts[m]["AP"])
    gates=dict(AP_gain=cohorts["paired"]["AP"]-cohorts[strongest]["AP"]>=.50,
               AP75_retained=cohorts["paired"]["AP75"]>=cohorts["raw"]["AP75"]-1e-10,
               strict90_retained=geometry["metrics"]["paired"]["strict_90"]>=geometry["metrics"]["raw"]["strict_90"])
    for rival in ("raw","independent","shuffled"):
        t=comparisons[rival]
        gates["geometry_vs_"+rival]=t["delta_points"]>=.20 and t["interval95_points"][0]>0
    result=dict(status="DFINE_FIT109_PAIRED_POINT_ACTUAL",FIT=True,DEV=False,TEST=False,GT_opened=True,AP=True,
                detector_forwards=0,detector_optimizer_updates=0,point_optimizer_updates=3000,
                protocol_sha256=PROTOCOL,math_sha256=MATH,training_sha256=TRAIN,prelabel_sha256=a.prelabel_sha,
                score_prelabel_sha256=pre,images=109,counts=receipt["counts"],match_scope=scope,training=training,
                cohorts=cohorts,budgets=budgets,fixed_geometry=geometry,strongest_AP_control=strongest,
                positive_folds_vs_strongest=sum(x>0 for x in comparisons[strongest]["folds"]),gates=gates,
                advance=all(gates.values()))
    digest=base.seal(a.output/"RESULT_SAFE.json",base.encoded(result))
    print(json.dumps(dict(status=result["status"],result_sha256=digest,advance=result["advance"],gates=gates),
                     sort_keys=True),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    stages=p.add_subparsers(dest="mode",required=True)
    common=("protocol","capture","accepted","base","rank-math","pair-math","output")
    for mode,extra in (("prepare",("bundle",)),("score",("prelabel","wait","fit-gt","code","shapely-site","prior-result","official","training"))):
        stage=stages.add_parser(mode)
        for field in common+extra:
            stage.add_argument("--"+field,type=Path,required=True)
        if mode=="score":
            stage.add_argument("--prelabel-sha",required=True)
            stage.add_argument("--wait-sha",required=True)
    a=p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="-1":
        raise ValueError("CPU only")
    base=load(a.base,BASE,"_exact_frozen_fit109")
    base.exact(a.protocol,PROTOCOL)
    rank=load(a.rank_math,RANK,"_original_cpu_feature")
    math=load(a.pair_math,MATH,"_frozen_pair_support")
    core,blocks=base.read_core(a)
    if a.mode=="prepare":
        prepare(a,base,rank,math,core,blocks)
    else:
        score(a,base,math,blocks)


if __name__=="__main__":
    main()