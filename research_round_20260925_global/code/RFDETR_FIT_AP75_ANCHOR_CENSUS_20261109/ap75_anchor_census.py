"""Trace AP75 loss anchors and audit training support without selecting a method."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from types import ModuleType

import numpy as np

PRIOR_SHA="34fbff972d8b8354b6a3d80fafb4b9db07a7311effa639b2774787a1856a0b7e"
FORENSIC_SHA="94a6db0f44df783683041b03c41ba2bb88732acb1713cff448659dc8dbbcd2e8"
ARMS=("native","bce","prefix")
FOLDS=5


def source(path,digest,name):
    raw=path.read_bytes()
    if sha256(raw).hexdigest()!=digest:
        raise ValueError("Exact frozen diagnostic source required")
    m=ModuleType(name)
    m.__file__=str(path)
    sys.modules[name]=m
    exec(compile(raw,str(path),"exec"),m.__dict__)
    return m


def slack(mask,truth):
    return 4*int(np.count_nonzero(mask&truth))-3*int(np.count_nonzero(mask|truth))


def fold_scores(torch,items,rows,prior,base):
    groups=sorted({r["image_sha256"] for r in items},
                  key=lambda s:sha256(("P175_AP75_GROUP_OOF_V1:"+s).encode()).digest())
    if len(items)!=290 or len(groups)!=288:
        raise ValueError("Original 290-image fit role")
    assignment={s:i%FOLDS for i,s in enumerate(groups)}
    predicted={}
    counts=[]
    for fold in range(FOLDS):
        train=[r for r in rows if assignment[r["group"]]!=fold]
        held=[(i,r) for i,r in enumerate(rows) if assignment[r["group"]]==fold]
        fit=lambda k:np.concatenate([r[k] for r in train])
        scorer,_=base.train_model(torch,fit("x"),fit("y"),prior.weights(train),True)
        x=np.concatenate([r["x"] for _,r in held])
        cuts=np.cumsum([len(r["x"]) for _,r in held])[:-1]
        for (i,_),z in zip(held,np.split(scorer(x),cuts),strict=True):
            predicted[i]=z
        counts.append(dict(fold=fold,train_rows=len(train),held_rows=len(held),
                           held_groups=sum(f==fold for f in assignment.values()),updates=96))
    if len(predicted)!=len(rows):
        raise ValueError("Each labelled row held out from exactly one BCE fit")
    return predicted,counts,assignment


def train_census(env,items,rows,z,assignment,local,policy,m):
    pos=0
    sums=Counter()
    role=[Counter() for _ in range(FOLDS)]
    vulnerable={k:set() for k in ("native","oof_bce","oof_bce_only")}
    min_hits={k:Counter() for k in vulnerable}
    bce_margin=Counter()
    statuses={k:Counter() for k in vulnerable}
    for i,item in enumerate(items,1):
        c=env.image(item)
        _,truth,matches,_=env.matches(item,c)
        group=item["image_sha256"]
        fold=assignment[group]
        for j,k in matches:
            q=c["query"][j]
            if not bool(c["ctx"]["active"][q]):
                continue
            mask=c["native"][j]
            gt=truth[k]
            xy,_,_,_,inside=local.boundary_samples(mask,128,5201+100*item["dataset_index"]+q)
            if not len(xy):
                continue
            row=rows[pos]
            y=gt[xy[:,1],xy[:,0]]!=inside
            if row["group"]!=group or not np.array_equal(row["y"],y) or not np.array_equal(row["inside"],inside):
                raise ValueError("Exact original group and paired point labels")
            scores=z[pos]
            pos+=1
            limits=m.bounds(y,inside)
            if limits is None:
                sums["ineligible_short"]+=1
                continue
            lo,hi=limits
            px=xy[:,1]*c["hw"][1]+xy[:,0]
            a,d=policy.choose_points(px,inside,scores,4)
            hits=int(y[a].sum()+y[d].sum())
            if not lo<=hits<=hi:
                raise ValueError("Cross-fit action respects attainable edit counts")
            s0=slack(mask,gt)
            r=m.needs_hits(s0)
            sb=m.slack_after(s0,hits)
            sums.update(eligible=1,native_pass=int(s0>=0),oof_bce_pass=int(sb>=0),oof_bce_only_pass=int(sb>=0>s0))
            role[fold].update(eligible=1)
            for name,anchor in (("native",s0>=0),("oof_bce",sb>=0),("oof_bce_only",sb>=0>s0)):
                if not anchor:
                    continue
                if r>hi:
                    if name!="native":
                        raise ValueError("Observed BCE pass must be feasible")
                    statuses[name]["all_actions_fail"]+=1
                elif r<=lo:
                    statuses[name]["all_actions_pass"]+=1
                else:
                    statuses[name]["choice_matters"]+=1
                    vulnerable[name].add(group)
                    min_hits[name][r]+=1
                    role[fold][name+"_vulnerable"]+=1
                    if name=="oof_bce":
                        bce_margin[hits-r]+=1
        if i%100==0 or i==len(items):
            print(json.dumps(dict(event="AP75_ANCHOR_CENSUS_PROGRESS",images=i)),flush=True)
    if pos!=len(rows):
        raise ValueError("All original point-fitting rows audited")
    return dict(totals=sums,role_counts=role,anchor_statuses=statuses,
                oof_bce_vulnerable_hit_margin=bce_margin,
                vulnerable={k:dict(groups=len(v),cases=sum(min_hits[k].values()),needed_hits=min_hits[k])
                            for k,v in vulnerable.items()})


def historical_predictions(env,items,ref,pr,teacher,base,local,policy,score):
    pred={k:[] for k in ARMS}
    ids=[]
    identity=[]
    edits=Counter()
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
            edits[arm]+=len(masks[arm])
            for j,value in masks[arm].items():
                rles[j]=env.codec.encode(np.asfortranarray(value.astype(np.uint8)))
            pred[arm].extend(dict(image_id=iid,category_id=1,score=s,segmentation=m)
                             for s,m in zip(c["scores"],rles,strict=True))
    if (sha256(json.dumps(identity,separators=(",",":")).encode()).hexdigest()!=ref["identity_sha256"]
            or len(ids)!=ref["images"] or any(edits[k]*8!=ref["edited_per_non_native"] for k in ARMS[1:])):
        raise ValueError("Exact closed historical candidates")
    idset=set(ids)
    gt=dict(info=env.gt["info"],images=[x for x in env.gt["images"] if x["id"] in idset],
            annotations=[x for x in env.gt["annotations"] if x["image_id"] in idset],categories=env.gt["categories"])
    return gt,pred,ids


def origins(gt,pred,ids,ref,forensic,diag,codec,m):
    ev={}
    for k in ARMS:
        ev[k],grid=diag.coco_eval(gt,pred[k],ids)
        if float(grid.mean())!=ref["metrics"][k]["AP75"]:
            raise ValueError("Exact native/BCE/prefix AP75 endpoints")
    if diag.transitions(ev["bce"],ev["prefix"])!=forensic["transitions"]:
        raise ValueError("Exact P174 historical match transitions")
    results=Counter()
    hist=Counter()
    for n,b,p in zip(*(ev[k].evalImgs for k in ARMS),strict=True):
        if b is None:
            continue
        if any(not np.array_equal(x[k],b[k]) for x in (n,p) for k in ("dtIds","dtScores","gtIds")):
            raise ValueError("Canonical historical candidate alignment")
        key=b["image_id"],b["category_id"]
        cols={g["id"]:i for i,g in enumerate(ev["prefix"]._gts[key])}
        for i,(did,gid,dst,ib,ip) in enumerate(zip(b["dtIds"],b["dtMatches"][0],p["dtMatches"][0],
                                                    b["dtIgnore"][0],p["dtIgnore"][0],strict=True)):
            if not gid or dst or ib or ip or ev["prefix"].ious[key][i,cols[gid]]>=.75:
                continue
            target=ev["bce"].cocoGt.anns[gid]
            if target["iscrowd"]:
                raise ValueError("Noncrowd true-positive loss only")
            y=codec.decode(target["segmentation"]).astype(bool)
            masks=[codec.decode(ev[k].cocoDt.anns[did]["segmentation"]).astype(bool) for k in ARMS]
            if len({int(x.sum()) for x in masks})!=1:
                raise ValueError("Fixed candidate area")
            s0,sb,sp=[slack(x,y) for x in masks]
            if not sb>=0>sp:
                raise ValueError("Exact integer 0.75 threshold crossing")
            label="native_pass" if s0>=0 else "bce_only_pass"
            results[label]+=1
            hist[m.needs_hits(s0)]+=1
    if sum(results.values())!=forensic["transitions"]["counts"].get("TP_loss_direct_edge",0):
        raise ValueError("All previously verified direct TP losses attributed")
    return dict(lost_edges=sum(results.values()),native_pass=results["native_pass"],
                bce_only_pass=results["bce_only_pass"],needed_hits_from_native=hist)


def run(args,pr,diag,teacher,base):
    previous=diag.load_pinned(args.reference,diag.RESULT_SHA,None)
    forensic=diag.load_pinned(args.forensic,FORENSIC_SHA,None)
    local=diag.load_pinned(args.local_math,base.MATH_SHA,"_anchor_local")
    prior=diag.load_pinned(args.local_source,base.LOCAL_SHA,"_anchor_prior")
    policy=diag.load_pinned(args.policy,pr.POLICY_SHA,"_anchor_policy")
    losses=diag.load_pinned(args.losses,diag.LOSS_SHA,"_anchor_losses")
    m=source(args.slack,args.slack_sha,"_anchor_slack")
    env=teacher.Environment(args)
    try:
        fit_items=base.parts(env.fit)["point_fit"]
        allfit=json.loads(env.cap.exact_bytes(args.fit_index,348469,env.cap.FIT_INDEX_SHA))["items"]
        cal=[x for x in allfit if x["partition"]=="calibration"]
        rows=base.point_training(env,teacher,prior,local,fit_items)
        if len(rows)!=623 or sum(len(r["y"]) for r in rows)!=142170:
            raise ValueError("Exact original training labels")
        score,report,_=pr.fit(env.torch,rows,prior,losses)
        for k in ARMS[1:]:
            for n in ("bce","objective","ranking","updates"):
                if report[k][n]!=previous["training"][k][n]:
                    raise ValueError("Exact paired original model replays")
        z,folds,assignment=fold_scores(env.torch,fit_items,rows,prior,base)
        census=train_census(env,fit_items,rows,z,assignment,local,policy,m)
        outcome={}
        for name,items in (("primary114",cal),("secondary111",env.hold)):
            gt,pred,ids=historical_predictions(env,items,previous[name],pr,teacher,base,local,policy,score)
            outcome[name]=origins(gt,pred,ids,previous[name],forensic[name],diag,env.codec,m)
            print(json.dumps(dict(event="AP75_ANCHOR_ORIGIN_COMPLETE",role=name,images=len(ids))),flush=True)
    finally:
        env.close()
    result=dict(status="HISTORICAL_FIT_AP75_ANCHOR_CENSUS_COMPLETE",protocol=args.protocol,
                p174_source=forensic["protocol"],origins=outcome,train_census=census,folds=folds,
                model_selection=False,DEV=False,TEST=False,p170_gate=False)
    args.output.mkdir()
    with (args.output/"RESULT.json").open("x",encoding="utf8") as f:
        json.dump(result,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(result,sort_keys=True,allow_nan=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("teacher","selective-source","prefix-source","diagnostic-source"):
        p.add_argument("--"+k,type=Path,required=True)
    first,_=p.parse_known_args()
    diag=source(first.diagnostic_source,PRIOR_SHA,"_closed_diag")
    pr=diag.load_pinned(first.prefix_source,diag.PREFIX_SHA,"_anchor_prefix")
    base=diag.load_pinned(first.selective_source,pr.SELECTIVE_SHA,"_anchor_selective")
    teacher=base.teacher_module(first.teacher)
    teacher.common_args(p)
    for k in ("local-source","local-math","policy","losses","reference","slack","forensic"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--slack-sha",required=True)
    run(p.parse_args(),pr,diag,teacher,base)
