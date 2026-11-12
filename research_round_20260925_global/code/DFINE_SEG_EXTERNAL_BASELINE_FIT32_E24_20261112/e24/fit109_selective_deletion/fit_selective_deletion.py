"""Source-group OOF reject-or-delete on the exact frozen D-FINE FIT109 top100."""
import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import sys

import numpy as np
from pycocotools import mask as codec

PROTOCOL="4793c2ca425c1c76907310c3c009f5c0bcb3c9c0c19f4a279c238621d85b13f7"
MATH="bf35b1b26c6e4cfbdd2199f5ea4ff57dfbf0e6fb73e3d74aa2e3298f286a1db2"
BASE="9ce987e68f996c3ef4146727867de8b56c08bc56b008665128c36854366d6d23"
RANK_MATH="cb438f8937f8e29dbdcbc441abdf5a788ac9c442fd63f1b3ad504b8ae0fcca0b"
PRELABEL="DFINE_FIT109_SELECTIVE_DELETE_PRELABEL"
FOLD_SALT="DFINE-FIT109-selective-deletion-v1|"
THRESHOLDS=(None,.5000,.5005,.501,.502,.504,.506,.508,.510,.515,.520,.530,.540,.550,.575,.600,.650)
ARMS=("raw","global","context","full")
GEOMETRY=(*ARMS,"oracle")
MIB=2**20


def source(path,digest,name):
    from types import ModuleType
    raw=path.read_bytes()
    if sha256(raw).hexdigest()!=digest:
        raise ValueError("Exact fixed source")
    mod=ModuleType(name)
    mod.__file__=str(path)
    exec(compile(raw,str(path),"exec"),mod.__dict__)
    return mod


def folds_for(blocks):
    groups=sorted({b["group"] for b in blocks},key=lambda g:(sha256((FOLD_SALT+g).encode()).hexdigest(),g))
    return {g:i%5 for i,g in enumerate(groups)}


def box_pixels(box):
    values=np.clip(np.ceil(box).astype(np.int64),0,800)
    return max(1,(int(values[2])-int(values[0]))*(int(values[3])-int(values[1])))


def proposal(c,row,plane,rank_math,deletion):
    hard=codec.decode(row["segmentation"]).astype(bool)
    area=c["mask_area"]
    pools,_=rank_math.editable_pools(hard,c["box_xyxy"])
    near,n_dist=pools["drop1"]
    far,f_dist=pools["drop2"]
    coords=np.concatenate((near,far))
    p=rank_math.sample_probability(plane,coords,800)
    k=min(8,area//10,len(coords))
    if not k:
        return None,0,0
    pixels=deletion.ordered_deletions(coords,p,k)
    lookup={int(x):i for i,x in enumerate(coords)}
    ii=np.asarray([lookup[x] for x in pixels])
    dist=np.concatenate((n_dist,f_dist))[ii]
    is_far=(ii>=len(near)).astype(np.float64)
    ps=p[ii].astype(np.float64)
    cardinality=np.arange(1,k+1)
    box_count=box_pixels(c["box_xyxy"])
    features=np.column_stack((cardinality/area,np.full(k,np.log2(area)),np.full(k,area/box_count),
                              np.full(k,row["score"]),np.cumsum(dist)/cardinality,
                              np.cumsum(is_far)/cardinality,np.cumsum(ps)/cardinality,ps))
    return dict(pixels=pixels,p=ps.tolist(),features=features.tolist(),area=area),len(coords),int((p<.5).sum())


def prepare(a,base,rank_math,deletion,core,blocks):
    folds=folds_for(blocks)
    a.output.mkdir()
    h=sha256()
    totals=Counter()
    with (a.output/"PREFIXES_PRIVATE.jsonl").open("xb") as f:
        for b in blocks:
            ref=b["probabilities"]
            raw=base.exact(core/ref["filename"],ref["sha256"],11*MIB)
            prob=np.load(BytesIO(raw),allow_pickle=False)
            base.need(prob.shape==(100,160,160) and prob.dtype==np.float32,"Exact raw native probability plane")
            rows={}
            for i,(c,r) in enumerate(zip(b["candidates"][:100],b["coco"][:100],strict=True)):
                base.need(c["native_rank"]==i+1 and c["probability_row"]==i and c["score"]==r["score"],
                          "Exact candidate rank and score linkage")
                area=c["mask_area"]
                totals.update(candidates=1,empty=int(area==0))
                if not 32<=area<1024:
                    continue
                totals["eligible_area"]+=1
                action,queried,discord=proposal(c,r,prob[i],rank_math,deletion)
                totals.update(queried=queried,discordant=discord)
                if action:
                    rows[str(i+1)]=action
                    totals.update(capable=1,prefixes=len(action["pixels"]))
                else:
                    totals["zero_capacity"]+=1
            line=base.encoded(dict(image_id=b["image_id"],group=b["group"],fold=folds[b["group"]],prefixes=rows))
            f.write(line)
            h.update(line)
            if b["ordinal"]%25==0 or b["ordinal"]==109:
                print(json.dumps(dict(event="DFINE_FIT109_SELECTIVE_DELETE_PREPARE",images=b["ordinal"])),flush=True)
        f.flush()
        os.fsync(f.fileno())
    base.need(totals["candidates"]==10900 and totals["capable"]+totals["zero_capacity"]==totals["eligible_area"],
              "Exact pred-only capacity partition")
    result=dict(status=PRELABEL,FIT=True,DEV=False,TEST=False,GT_opened=False,AP=False,images=109,
                protocol_sha256=PROTOCOL,worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),math_sha256=MATH,
                prefixes_sha256=h.hexdigest(),fold_counts={str(i):sum(f==i for f in folds.values()) for i in range(5)},
                closed_capture_sha256=base.CAPTURE,counts=dict(totals))
    digest=base.seal(a.output/"PRELABEL_SAFE.json",base.encoded(result))
    print(json.dumps(dict(status=PRELABEL,result_sha256=digest),sort_keys=True),flush=True)


def official(gt, rows):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with redirect_stdout(StringIO()):
        c=COCO()
        c.dataset=deepcopy(gt)
        c.createIndex()
        d=c.loadRes(deepcopy(rows))
        ev=COCOeval(c,d,"segm")
        ev.params.imgIds=list(c.imgs)
        ev.params.catIds=[1]
        ev.params.maxDets=[1,10,100]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    ix=int(np.flatnonzero(np.isclose(ev.params.iouThrs,.90)).item())
    strict=ev.eval["precision"][ix,:,0,0,-1]
    return dict(AP=float(ev.stats[0])*100,AP75=float(ev.stats[2])*100,
                AP90=float(strict[strict>=0].mean())*100,AR100=float(ev.stats[8])*100)


def geometry_for(p,area,truth,base_mask,geom,deletion):
    import shapely
    boxes=geom.geometry.boxes_from_counts(geom.shift.runs(base_mask["counts"]),*base_mask["size"])
    if np.prod(boxes[:,2:]-boxes[:,:2],axis=1).sum()!=area:
        raise ValueError("Exact disjoint stock unit-cell area")
    inter=geom.geometry.overlap_area(boxes,truth)
    if p:
        y,x=np.divmod(np.asarray(p["pixels"]),800)
        fractions=shapely.area(shapely.intersection(shapely.box(x,y,x+1,y+1),truth))
        y_train=inter/(area+truth.area)-np.cumsum(fractions)/np.arange(1,len(fractions)+1)
    else:
        fractions=[]
        y_train=[]
    metrics=deletion.continuous_after_drop(area,truth.area,inter,fractions)
    if p and np.any(np.sign(metrics[1:,0]-metrics[0,0])[np.abs(y_train)>1e-12]!=np.sign(y_train[np.abs(y_train)>1e-12])):
        raise ValueError("Exact break-even target sign")
    return metrics,y_train


def reference_population(a,base,blocks,planned):
    index=json.loads(base.exact(a.fit_gt,base.GT,64*MIB))
    base.need(len(index["images"])==2937 and len(index["annotations"])==9260,"Exact official FIT labels")
    ims={i["id"]:i for i in index["images"]}
    ids={b["image_id"] for b in blocks}
    image_rows=[ims[b["image_id"]] for b in blocks]
    anns=defaultdict(list)
    for ann in index["annotations"]:
        if ann["image_id"] in ids:
            anns[ann["image_id"]].append(ann)
    folds=folds_for(blocks)
    base.need(all(i["file_name"]==b["basename"] and i["id"]==p["image_id"]==b["image_id"]
                  and p["group"]==b["group"] and p["fold"]==folds[b["group"]]
                  for i,b,p in zip(image_rows,blocks,planned,strict=True)), "Exact original source order and locked fold")
    return dict(images=image_rows,annotations=[r for rr in anns.values() for r in rr],categories=index["categories"]),anns


def fixed_tables(a,base,deletion,blocks,planned,gt,anns):
    sys.path.append(str(a.shapely_site.resolve(strict=True)))
    import shapely
    base.need(shapely.__version__=="2.0.2" and shapely.geos_version_string=="3.11.2","Exact GEOS runtime")
    geometry=base.load(a.code/"MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py",
                       base.GEOM,"_deletion_geom")
    geom=geometry.geometry_modules(a.code)
    match=base.load(a.code/"HRSID_FIT_HULL_CDF_20261112/stock_coco_match.py",base.MATCH,"_deletion_match")
    truth,neutral,_=geom.geometry.polygon_support(gt,geom.vector)
    slots, scope = [], Counter()
    for b,p,image in zip(blocks,planned,gt["images"],strict=True):
        top=list(zip(b["candidates"][:100],b["coco"][:100],strict=True))
        pool=[dict(rank=i,score=r["score"],rle=r["segmentation"],area=c["mask_area"]) for i,(c,r) in enumerate(top,1)]
        with redirect_stdout(StringIO()):
            matched,_,_=match.freeze_coco(image,anns[image["id"]],pool)
        by_rank={rank:key for key,rank in matched.items()}
        scope["stock_matches"]+=len(matched)
        for i,(c,r) in enumerate(top,1):
            identity=by_rank.get(i)
            edit=p["prefixes"].get(str(i))
            role="valid" if identity in truth and identity not in neutral else "neutral" if identity is not None else "unmatched"
            area=c["mask_area"]
            if edit:
                base.need(edit["area"]==area and 1<=len(edit["pixels"])<=min(8,area//10) and
                          len(edit["pixels"])==len(set(edit["pixels"])) and edit["p"]==sorted(edit["p"]),
                          "Frozen exact candidate prefix")
            slot=dict(row=r,group=p["group"],fold=p["fold"],role=role,edit=edit,metrics=None)
            if role=="valid" and 32<=area<1024:
                slot["metrics"],slot["target"]=geometry_for(edit,area,truth[identity][0],r["segmentation"],geom,deletion)
                scope["fixed_valid"]+=1
            slots.append(slot)
    base.need(len(slots)==10900 and scope["stock_matches"]==467 and scope["fixed_valid"]==412,
              "Original fixed-candidate match reproduction")
    return slots,scope


def subset(gt, ids):
    return dict(images=[r for r in gt["images"] if r["id"] in ids],
                annotations=[r for r in gt["annotations"] if r["image_id"] in ids],categories=gt["categories"])


def fit_model(slots,train,column_count,deletion):
    selected=[s for i in train for s in [slots[i]] if s["metrics"] is not None and s["edit"]]
    n=Counter()
    for s in selected:
        n[s["group"]]+=len(s["target"])
    x,y,w=[],[],[]
    for s in selected:
        x.extend(row[:column_count] for row in s["edit"]["features"])
        y.extend(s["target"])
        w.extend([1/n[s["group"]]]*len(s["target"]))
    return deletion.ridge(np.asarray(x),np.asarray(y),np.asarray(w)),dict(prefixes=len(y),groups=len(n))


def global_model(gt, slots, train, materialize):
    ids={slots[i]["row"]["image_id"] for i in train}
    train_gt=subset(gt,ids)
    all_scores=[]
    for cutoff in THRESHOLDS:
        ks=[int(np.searchsorted(slots[i]["edit"]["p"],cutoff,side="left")) if cutoff is not None and slots[i]["edit"] else 0
            for i in train]
        scores=official(train_gt,[materialize(i,k) for i,k in zip(train,ks,strict=True)])
        all_scores.append(dict(threshold=cutoff,train_AP=scores["AP"],train_deleted_pixels=sum(ks)))
    best=max(r["train_AP"] for r in all_scores)
    winner=min((i for i,r in enumerate(all_scores) if best-r["train_AP"]<=1e-10),
               key=lambda i:(all_scores[i]["train_deleted_pixels"],i))
    return THRESHOLDS[winner],all_scores,winner


def summarize_geometry(slots,chosen):
    by_group=defaultdict(list)
    group_fold={}
    for i,s in enumerate(slots):
        if s["metrics"] is None:
            continue
        values=s["metrics"]
        picks=[chosen[m][i] for m in ARMS]+[int(np.argmax(values[:,0]))]
        by_group[s["group"]].append(values[picks])
        group_fold[s["group"]]=s["fold"]
    groups=sorted(by_group)
    data=np.asarray([np.mean(by_group[g],axis=0) for g in groups])
    pairs=np.concatenate([by_group[g] for g in groups],axis=0)
    fold_ids=np.asarray([group_fold[g] for g in groups])
    draws=np.random.Generator(np.random.PCG64(5217)).integers(len(groups),size=(4096,len(groups)))
    metrics={}
    for i,m in enumerate(GEOMETRY):
        metrics[m]=dict(pair_equal_iou_points=float(pairs[:,i,0].mean()),
                        group_equal_iou_points=float(data[:,i,0].mean()),fp_per_gt_points=float(data[:,i,1].mean()),
                        fn_per_gt_points=float(data[:,i,2].mean()),strict_75=int((pairs[:,i,0]>=75).sum()),
                        strict_90=int((pairs[:,i,0]>=90).sum()))
    contrasts={}
    for i,m in enumerate(ARMS[:-1]):
        delta=data[:,3,0]-data[:,i,0]
        contrasts[m]=dict(delta_points=float(delta.mean()),
                          interval95_points=np.percentile(delta[draws].mean(axis=1),[2.5,97.5]).tolist(),
                          folds=[float(delta[fold_ids==f].mean()) for f in range(5)])
    return dict(pairs=len(pairs),source_groups=len(groups),metrics=metrics,full_minus=contrasts)


def evaluate(a,base,deletion,gt,slots,scope,receipt,score_pregt):
    cache={}
    def materialize(i,k):
        row=slots[i]["row"]
        if not k:
            return row
        key=(i,k)
        if key not in cache:
            pixels=slots[i]["edit"]["pixels"][:k]
            rle=base.apply(row["segmentation"],([],pixels))
            base.need(int(codec.area(rle))==slots[i]["edit"]["area"]-k,"Exact native deletion and no empty mask")
            cache[key]=dict(row,segmentation=rle)
        return cache[key]

    chosen={m:[0]*len(slots) for m in ARMS}
    calibration=[]
    fold_scores={}
    for fold in range(5):
        train=[i for i,s in enumerate(slots) if s["fold"]!=fold]
        test=[i for i,s in enumerate(slots) if s["fold"]==fold]
        global_cut,grid,winner=global_model(gt,slots,train,materialize)
        models={m:fit_model(slots,train,n,deletion) for m,n in (("context",6),("full",8))}
        for i in test:
            edit=slots[i]["edit"]
            if not edit:
                continue
            if global_cut is not None:
                chosen["global"][i]=int(np.searchsorted(edit["p"],global_cut,side="left"))
            z=np.asarray(edit["features"])
            for m,n in (("context",6),("full",8)):
                model,_=models[m]
                chosen[m][i]=deletion.choose_prefix(model(z[:,:n]),edit["area"])
        ids={slots[i]["row"]["image_id"] for i in test}
        held=subset(gt,ids)
        fold_scores[str(fold)]={m:official(held,[materialize(i,chosen[m][i]) for i in test]) for m in ARMS}
        calibration.append(dict(fold=fold,train_images=len({slots[i]["row"]["image_id"] for i in train}),
                                test_images=len(ids),global_cutoff=global_cut,global_grid=grid,global_winner=winner,
                                ridge_training={m:v[1] for m,v in models.items()}))
        print(json.dumps(dict(event="DFINE_FIT109_SELECTIVE_DELETE_FOLD_COMPLETE",fold=fold,
                              full_AP=fold_scores[str(fold)]["full"]["AP"]),sort_keys=True),flush=True)
    coco={m:official(gt,[materialize(i,k) for i,k in enumerate(chosen[m])]) for m in ARMS}
    edited={}
    for m in ARMS:
        actions=Counter()
        for s,k in zip(slots,chosen[m],strict=True):
            actions["deleted_pixels"]+=k
            actions["no_op"]+=int(k==0)
            if k:
                actions["edited"]+=1
                actions["edited_"+s["role"]]+=1
        edited[m]={name:actions[name] for name in ("deleted_pixels","no_op","edited","edited_valid","edited_neutral","edited_unmatched")}
    geometry=summarize_geometry(slots,chosen)
    contrast=geometry["full_minus"]
    gates=dict(full_AP_margin=coco["full"]["AP"]-max(coco[m]["AP"] for m in ARMS[:-1])>=.50,
               full_AP75_retained=coco["full"]["AP75"]>=coco["raw"]["AP75"]-1e-10,
               strict90_retained=geometry["metrics"]["full"]["strict_90"]>=geometry["metrics"]["raw"]["strict_90"],
               raw_positive_four_folds=sum(x>0 for x in contrast["raw"]["folds"])>=4)
    for m,t in contrast.items():
        gates["geometry_vs_"+m]=t["delta_points"]>=.20 and t["interval95_points"][0]>0
    result=dict(status="DFINE_FIT109_SELECTIVE_DELETE_ACTUAL",FIT=True,DEV=False,TEST=False,GT_opened=True,AP=True,
                model_forwards=0,optimizer_updates=0,protocol_sha256=PROTOCOL,math_sha256=MATH,
                worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),prelabel_sha256=a.prelabel_sha,
                prelabel_score_sha256=score_pregt,counts=receipt["counts"],match_scope=dict(scope),images=109,
                annotations=len(gt["annotations"]),pooled_oof_coco=coco,fixed_geometry=geometry,
                fold_coco=fold_scores,fold_calibration=calibration,realized_budget=edited,gates=gates,advance=all(gates.values()))
    out=base.seal(a.output/"RESULT_SAFE.json",base.encoded(result))
    print(json.dumps(dict(status=result["status"],result_sha256=out,advance=result["advance"],gates=gates),
                     sort_keys=True),flush=True)


def score(a,base,deletion,blocks):
    receipt=json.loads(base.exact(a.prelabel/"PRELABEL_SAFE.json",a.prelabel_sha))
    wait=json.loads(base.exact(a.wait,a.wait_sha))
    term=wait["terminal"]
    tail=json.loads(term["output"].strip().splitlines()[-1])
    base.need(tail==dict(status=PRELABEL,result_sha256=a.prelabel_sha) and term["exit_code"]==0
              and receipt["status"]==PRELABEL and receipt["GT_opened"] is False and receipt["AP"] is False
              and receipt["protocol_sha256"]==PROTOCOL and receipt["math_sha256"]==MATH and
              receipt["worker_sha256"]==sha256(Path(__file__).read_bytes()).hexdigest(),
              "Exact natural prediction-only preparation closure")
    planned=[json.loads(line) for line in base.exact(a.prelabel/"PREFIXES_PRIVATE.jsonl",
                                                   receipt["prefixes_sha256"],64*MIB).splitlines()]
    base.need(len(planned)==109 and sum(len(p["prefixes"]) for p in planned)==receipt["counts"]["capable"],
              "Exact frozen prediction-only action roster")
    a.output.mkdir()
    pre=base.seal(a.output/"PRELABEL_SCORE_SAFE.json",base.encoded(dict(status="DFINE_FIT109_SELECTIVE_DELETE_SCORE_PRELABEL",
                  protocol_sha256=PROTOCOL,prelabel_sha256=a.prelabel_sha,prelabel_wait_sha256=a.wait_sha,
                  FIT=True,DEV=False,TEST=False,GT_opened=False,AP=False)))
    gt,anns=reference_population(a,base,blocks,planned)
    slots,scope=fixed_tables(a,base,deletion,blocks,planned,gt,anns)
    evaluate(a,base,deletion,gt,slots,scope,receipt,pre)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    stages=p.add_subparsers(dest="stage",required=True)
    common=("protocol","capture","accepted","base","rank-math","deletion-math","output")
    for name,extra in (("prepare",()),("score",("prelabel","wait","fit-gt","code","shapely-site"))):
        q=stages.add_parser(name)
        for arg in common+extra:
            q.add_argument("--"+arg,type=Path,required=True)
        if name=="score":
            q.add_argument("--prelabel-sha",required=True)
            q.add_argument("--wait-sha",required=True)
    a=p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="-1":
        raise ValueError("CPU only")
    base=source(a.base,BASE,"_exact_balanced_base")
    base.exact(a.protocol,PROTOCOL)
    rank_math=source(a.rank_math,RANK_MATH,"_exact_balanced_math")
    deletion=source(a.deletion_math,MATH,"_fixed_delete_math")
    core,blocks=base.read_core(a)
    if a.stage=="prepare":
        prepare(a,base,rank_math,deletion,core,blocks)
    else:
        score(a,base,deletion,blocks)


if __name__=="__main__":
    main()