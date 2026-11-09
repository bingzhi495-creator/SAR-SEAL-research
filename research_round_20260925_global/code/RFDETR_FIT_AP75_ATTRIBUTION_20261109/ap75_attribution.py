"""One historical COCO replay after the fixed-prefix AP75 gate failure."""
import argparse
from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from io import StringIO
import json
import os
from pathlib import Path
import sys
from types import ModuleType

import numpy as np

PREFIX_SHA="2d0744d446bddc1f2a703d71d26d12deb57a7943db10e43157a64f781648d293"
LOSS_SHA="e6bdf3c9df789aac7a2d5266d0cf73abee0dad59e4bc2f74e0d96d814060220d"
RESULT_SHA="621ba706a5466abd4d066ac21bc82eeee1a4fb0b5493d87ecce391c6ebb3d88e"
ARMS=("bce","prefix")
T=.75


def load_pinned(path,digest,name):
    raw=path.read_bytes()
    if sha256(raw).hexdigest()!=digest:
        raise ValueError("Exact frozen source or result required")
    if name is None:
        return json.loads(raw)
    m=ModuleType(name)
    m.__file__=str(path)
    sys.modules[name]=m
    exec(compile(raw,str(path),"exec"),m.__dict__)
    return m


def coco_eval(gt,pred,ids,matrices=None):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with redirect_stdout(StringIO()):
        g=COCO()
        g.dataset=deepcopy(gt)
        g.createIndex()
        d=g.loadRes(deepcopy(pred))
        ev=COCOeval(g,d,"segm")
        p=ev.params
        p.imgIds=sorted(ids)
        p.catIds=[1]
        p.iouThrs=np.array([T],dtype=float)
        p.areaRng=p.areaRng[:1]
        p.areaRngLbl=["all"]
        if matrices is not None:
            ev.computeIoU=lambda image,category:matrices[(image,category)]
        ev.evaluate()
        ev.accumulate()
    if tuple(p.maxDets)!=(1,10,100) or len(ev.evalImgs)!=len(ids):
        raise ValueError("Exact COCO mask AP75 scope")
    grid=ev.eval["precision"][0,:,0,0,-1]
    if grid.shape!=(101,) or (grid<0).any():
        raise ValueError("Complete 101-point precision grid")
    return ev,grid


def same_trace(a,b):
    keys=("dtIds","gtIds","dtScores","dtMatches","gtMatches","dtIgnore","gtIgnore")
    return all((x is None and y is None) or
               (x is not None and y is not None and all(np.array_equal(x[k],y[k]) for k in keys))
               for x,y in zip(a.evalImgs,b.evalImgs,strict=True))


def transitions(a,b):
    count=Counter()
    changed_images=0
    for x,y in zip(a.evalImgs,b.evalImgs,strict=True):
        if x is None and y is None:
            continue
        if x is None or y is None or any(not np.array_equal(x[k],y[k]) for k in ("dtIds","gtIds","dtScores","gtIgnore")):
            raise ValueError("Fixed canonical candidate and annotation identity")
        key=x["image_id"],x["category_id"]
        u,v=(np.asarray(ev.ious[key]) for ev in (a,b))
        cols={gt["id"]:i for i,gt in enumerate(a._gts[key])}
        if [g["id"] for g in b._gts[key]]!=list(cols):
            raise ValueError("Fixed IoU annotation columns")
        count.update(eligible_edges_lost=int(((u>=T)&(v<T)).sum()),eligible_edges_gained=int(((u<T)&(v>=T)).sum()))
        image_changed=False
        for i,(old,new,oi,ni) in enumerate(zip(x["dtMatches"][0],y["dtMatches"][0],x["dtIgnore"][0],y["dtIgnore"][0],strict=True)):
            if old==new and oi==ni:
                continue
            image_changed=True
            before="ignore" if oi else "TP" if old else "FP"
            after="ignore" if ni else "TP" if new else "FP"
            count[before+"_to_"+after]+=1
            if before=="TP" and after=="FP":
                count["TP_loss_direct_edge" if v[i,cols[old]]<T else "TP_loss_rematch"]+=1
            if before=="FP" and after=="TP":
                count["TP_gain_direct_edge" if u[i,cols[new]]<T else "TP_gain_rematch"]+=1
        changed_images+=int(image_changed)
    return dict(changed_images=changed_images,counts=dict(sorted(count.items())))


def role_predictions(env,items,ref,pr,teacher,base,local,policy,score):
    ids=[]
    pred={k:[] for k in ARMS}
    identity=[]
    changed=Counter()
    for item in items:
        c=env.image(item)
        rows,x=pr.fields(c,item["dataset_index"],teacher,base,local)
        masks,_=pr.choices(c,rows,x,score,policy)
        iid=item["image_id"]
        ids.append(iid)
        identity.extend((iid,r,q,s.hex()) for r,q,s in zip(c["rank"],c["query"],c["scores"],strict=True))
        original=env.s.coerce_rles(c["native"],env.codec)
        for arm in ARMS:
            rles=list(original)
            changed[arm]+=len(masks[arm])
            for j,value in masks[arm].items():
                if np.count_nonzero(value)!=np.count_nonzero(c["native"][j]):
                    raise ValueError("Balanced edit changed area")
                rles[j]=env.codec.encode(np.asfortranarray(value.astype(np.uint8)))
            pred[arm].extend(dict(image_id=iid,category_id=1,score=s,segmentation=m)
                             for s,m in zip(c["scores"],rles,strict=True))
    digest=sha256(json.dumps(identity,separators=(",",":")).encode()).hexdigest()
    if digest!=ref["identity_sha256"] or any(changed[k]*8!=ref["edited_per_non_native"] for k in ARMS):
        raise ValueError("Exact closed P170 candidates and edit support")
    idset=set(ids)
    gt=dict(info=env.gt["info"],images=[x for x in env.gt["images"] if x["id"] in idset],
            annotations=[x for x in env.gt["annotations"] if x["image_id"] in idset],categories=env.gt["categories"])
    return gt,pred,ids


def analyze(gt,pred,ids,ref,mix):
    actual={k:coco_eval(gt,pred[k],ids) for k in ARMS}
    if not all(float(grid.mean())==ref["metrics"][k]["AP75"] for k,(_,grid) in actual.items()):
        raise ValueError("Exact published P170 AP75 reproduction")
    scores={}
    for e in ARMS:
        left=actual[e][0].ious
        for r in ARMS:
            right=actual[r][0].ious
            if left.keys()!=right.keys():
                raise ValueError("Same canonical image-category keys")
            alt={k:mix(left[k],right[k]) for k in left}
            replay,grid=coco_eval(gt,pred[e],ids,alt)
            if e==r and not (same_trace(replay,actual[e][0]) and np.array_equal(grid,actual[e][1])):
                raise ValueError("Canonical endpoint matching and precision grid identity")
            scores[e+"_"+r]=float(grid.mean())
    u,v=(actual[k][1] for k in ARMS)
    a,b,c,d=(scores[k] for k in ("bce_bce","bce_prefix","prefix_bce","prefix_prefix"))
    return dict(AP75=scores,eligibility_effect=.5*((c-a)+(d-b)),priority_effect=.5*((b-a)+(d-c)),
                interaction=d-c-b+a,actual_delta=d-a,transitions=transitions(actual["bce"][0],actual["prefix"][0]),
                precision_grid=dict(lower=int((v<u).sum()),higher=int((v>u).sum()),equal=int((v==u).sum())),
                exact_own_endpoint_match_and_precision=True)


def run(args,pr,teacher,base):
    previous=load_pinned(args.reference,RESULT_SHA,None)
    if previous["gate"] or previous["DEV"] or previous["TEST"]:
        raise ValueError("Only closed historical failed-gate replay")
    prior=load_pinned(args.local_source,base.LOCAL_SHA,"_ap75_prior")
    local=load_pinned(args.local_math,base.MATH_SHA,"_ap75_local")
    policy=load_pinned(args.policy,pr.POLICY_SHA,"_ap75_policy")
    losses=load_pinned(args.losses,LOSS_SHA,"_ap75_losses")
    mix=load_pinned(args.factors,args.factors_sha,"_ap75_factors").mix_edges
    env=teacher.Environment(args)
    try:
        fit_items=base.parts(env.fit)["point_fit"]
        allfit=json.loads(env.cap.exact_bytes(args.fit_index,348469,env.cap.FIT_INDEX_SHA))["items"]
        cal=[x for x in allfit if x["partition"]=="calibration"]
        train=base.point_training(env,teacher,prior,local,fit_items)
        if len(train)!=623 or sum(len(r["y"]) for r in train)!=142170:
            raise ValueError("Exact same historical point fit")
        score,report,_=pr.fit(env.torch,train,prior,losses)
        for k in ARMS:
            for m in ("bce","objective","ranking","updates"):
                if report[k][m]!=previous["training"][k][m]:
                    raise ValueError("Exact closed point-head training replay")
        result=dict(status="HISTORICAL_FIT_AP75_MATCH_ATTRIBUTION_COMPLETE",protocol=args.protocol,
                    closed_prefix_protocol=previous["protocol"],gate=False,DEV=False,TEST=False,
                    synthetic_mixes_diagnostic_only=True)
        for role,items in (("primary114",cal),("secondary111",env.hold)):
            ref=previous[role]
            gt,pred,ids=role_predictions(env,items,ref,pr,teacher,base,local,policy,score)
            if len(ids)!=ref["images"]:
                raise ValueError("Exact historical role size")
            result[role]=analyze(gt,pred,ids,ref,mix)
            print(json.dumps(dict(event="AP75_ROLE_CLOSED",role=role,images=len(ids))),flush=True)
    finally:
        env.close()
    args.output.mkdir()
    with (args.output/"RESULT.json").open("x",encoding="utf8") as f:
        json.dump(result,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(result,sort_keys=True,allow_nan=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("teacher","selective-source","prefix-source"):
        p.add_argument("--"+k,type=Path,required=True)
    first,_=p.parse_known_args()
    pr=load_pinned(first.prefix_source,PREFIX_SHA,"_closed_prefix")
    base=load_pinned(first.selective_source,pr.SELECTIVE_SHA,"_ap75_selective")
    teacher=base.teacher_module(first.teacher)
    teacher.common_args(p)
    for k in ("local-source","local-math","policy","losses","reference","factors"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--factors-sha",required=True)
    run(p.parse_args(),pr,teacher,base)
