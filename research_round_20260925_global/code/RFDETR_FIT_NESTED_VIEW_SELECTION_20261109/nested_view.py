"""Nested image-content-group OOF choice between two exact frozen actions."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np

FLIP_SHA="16da7ffa344dca41dfc5ecc7c6e82ade336723d4e88e0202cab4746c9bbca6f9"
REFERENCE_SHA="22d482fe4c49d238c7f23d20a8ebcf6e986805d03995f6ab0ec7b1816cdb6c54"
ACTIONS=("raw38","aligned")


def pinned(path,digest,name=None):
    from types import ModuleType
    import sys

    b=path.read_bytes()
    if sha256(b).hexdigest()!=digest:
        raise ValueError("Exact previously verified source or result")
    if name is None:
        return json.loads(b)
    m=ModuleType(name)
    m.__file__=str(path)
    sys.modules[name]=m
    exec(compile(b,str(path),"exec"),m.__dict__)
    return m


def decisions(predict,indices,rows,records,flip,policy,driver):
    out={}
    for old in (True,False):
        sub=[i for i in indices if records[i]["old"]==old]
        if not sub:
            continue
        cuts=np.cumsum([len(rows[i]["x"]) for i in sub])[:-1]
        scores={k:np.split(predict(k,np.concatenate([flip.feature(rows[i],records[i],k) for i in sub])),cuts)
                for k in ACTIONS}
        for j,i in enumerate(sub):
            r,c=rows[i],records[i]
            if not c["linked"]:
                raise ValueError("Exact fully linked fixed-pair role")
            choose={k:policy.choose_points(c["pixel"],r["inside"],scores[k][j],4) for k in ACTIONS}
            a,b=(choose[k] for k in ACTIONS)
            overlap=[len(np.intersect1d(x,y,assume_unique=True))/4 for x,y in zip(a,b,strict=True)]
            x=r["x"]
            common=[float(x[0,k]) for k in (3,4,5)]+overlap
            def margin(name):
                v=c["extra"][name]
                return [float(v[y].mean()-v[x].mean()) for x,y in zip(a,b,strict=True)]
            common+=margin("raw39")
            evidence=np.array(common+margin("aligned"),dtype=np.float64)
            control=np.array(common+margin("sham"),dtype=np.float64)
            # All predictor features and actions above are defined from prediction-only arrays.
            hits={k:tuple(int(r["y"][q].sum()) for q in v) for k,v in choose.items()}
            h={k:driver.case_thresholds(c["intersection"],c["area_sum"],sum(v)) for k,v in hits.items()}
            target=int(h["aligned"][7:].sum()-h["raw38"][7:].sum())
            out[i]=dict(target=target,group=c["group"],fold=c["fold"],raw=h["raw38"],flip=h["aligned"],hits=hits,
                        eligible=not all(v==1 for v in overlap),aligned_features=evidence,sham_features=control)
    return out


def describe(items):
    y=np.array([v["target"] for v in items])
    out={}
    for label,keep in (("gain",y>0),("loss",y<0)):
        chosen=[v for v,k in zip(items,keep,strict=True) if k]
        groups=Counter(v["group"] for v in chosen)
        out[label]=dict(rows=len(chosen),groups=len(groups),folds=len({v["fold"] for v in chosen}),
                        largest_group_rows=max(groups.values(),default=0))
    out["ties"]=int((y==0).sum())
    return out


def opportunity(records,picks,driver,math):
    items=[picks[i] for i in range(len(records))]
    raw,alt=(np.stack([d[k] for d in items]) for k in ("raw","flip"))
    native=np.stack([driver.case_thresholds(c["intersection"],c["area_sum"],4) for c in records])
    upper=math.geometric_choice(native,raw,alt)
    upper["high_total"]=sum(upper["maximum_passed"][7:])
    upper["high_gain_over_raw"]=sum(upper["gained_over_raw"][7:])
    upper["high_loss_by_flip"]=sum(upper["lost_by_flip"][7:])
    for d in items:
        x,y=(sum(d["hits"][k]) for k in ACTIONS)
        if (y-x)*d["target"]<0:
            raise ValueError("Equal area makes threshold passes monotone in correct hits")
    delta=np.array([sum(d["hits"]["aligned"])-sum(d["hits"]["raw38"]) for d in items])
    changed=np.array([d["target"]!=0 for d in items])
    upper["correct_flip_gain_loss"]={}
    for k,keep in (("unchanged_high_passes",~changed),("changed_high_passes",changed)):
        v=delta[keep]
        upper["correct_flip_gain_loss"][k]=dict(rows=int(keep.sum()),positive=int(v.clip(0).sum()),
                                               negative=int((-v).clip(0).sum()),net=int(v.sum()))
    upper["label_support"]=describe(items)
    upper["different_actions"]=sum(d["eligible"] for d in items)
    return native,raw,alt,upper


def nested(env,rows,records,outer,flip,prior,prefix,driver,policy,math):
    allfolds=tuple(range(5))
    cache={}
    for f in allfolds:
        for g in range(f+1,5):
            train=[i for i,c in enumerate(records) if c["fold"] not in (f,g)]
            if {records[i]["fold"] for i in train}&{f,g}:
                raise ValueError("Both outer and inner held groups excluded")
            score,report=flip.fit(env.torch,train,rows,records,prior,prefix,driver)
            cache[f,g]={}
            for h in (f,g):
                evaluate=[i for i,c in enumerate(records) if c["fold"]==h]
                cache[f,g][h]=decisions(score,evaluate,rows,records,flip,policy,driver)
            print(json.dumps(dict(event="NESTED_VIEW_INNER_FIT_COMPLETE",excluded=[f,g],updates=96,heads=len(report))),flush=True)
    routed={k:set() for k in ("selector","sham_selector","same_rate_hash")}
    reports=[]
    for f in allfolds:
        train={}
        for g in allfolds:
            if g!=f:
                train.update(cache[tuple(sorted((f,g)))][g])
        held=[i for i,c in enumerate(records) if c["fold"]==f]
        if len(train)+len(held)!=len(records) or set(train)&set(held):
            raise ValueError("Complete inner OOF without outer fold labels")
        ids=[i for i,v in train.items() if v["eligible"]]
        e=[i for i in held if outer[i]["eligible"]]
        gs=[train[i]["group"] for i in ids]
        if set(gs)&{outer[i]["group"] for i in held}:
            raise ValueError("Outer evaluation content groups excluded")
        y=np.array([train[i]["target"] for i in ids],dtype=np.float64)
        chosen={}
        for k,x in (("selector","aligned_features"),("sham_selector","sham_features")):
            model=math.fit_ridge([train[i][x] for i in ids],y,gs)
            z=model(np.stack([outer[i][x] for i in e]))
            if not np.isfinite(z).all():
                raise ValueError("Finite held gate predictions")
            chosen[k]=[i for i,v in zip(e,z,strict=True) if v>0]
        order=sorted(e,key=lambda i:sha256(f"P187_RATE_CONTROL:{outer[i]['group']}:{i}".encode()).digest())
        chosen["same_rate_hash"]=order[:len(chosen["selector"])]
        for k,v in chosen.items():
            routed[k].update(v)
        reports.append(dict(fold=f,training_rows=len(ids),training_groups=len(set(gs)),eligible_held=len(e),
                            switched={k:len(v) for k,v in chosen.items()},training_support=describe([train[i] for i in ids])))
    return routed,reports


def report_select(native,raw,alt,picks,routed,driver,ref):
    masks={"raw38":raw,"aligned":alt}
    for k,values in routed.items():
        use=np.array([i in values for i in range(len(raw))])
        masks[k]=np.where(use[:,None],alt,raw)
    metrics={}
    for k,a in masks.items():
        metrics[k]=dict(passed=a.sum(0).tolist(),added_vs_raw=(a&~raw).sum(0).tolist(),
                        lost_vs_raw=(raw&~a).sum(0).tolist(),native_90_lost=int((native[:,8]&~a[:,8]).sum()))
    p=masks["selector"][:,7:].sum(1)
    gs=[picks[i]["group"] for i in range(len(raw))]
    paired={k:driver.paired_difference(gs,p,a[:,7:].sum(1)) for k,a in masks.items() if k!="selector"}
    s=metrics["selector"]
    gate=dict(high_micro_ge_0_01_vs_all=all(v["micro"]>=.01 for v in paired.values()),
              group_interval_positive_vs_all=all(v["group_percentile95"][0]>0 for v in paired.values()),
              fixed_ap75_nondecrease=s["passed"][5]>=ref["metrics"]["raw38"]["passed"][5],
              native_ap90_loss_at_most_5_percent=20*s["native_90_lost"]<=int(native[:,8].sum()))
    return dict(metrics=metrics,paired=paired,gate_checks=gate,gate=all(gate.values()))


def run(args,flip,driver,broad,base,teacher):
    ref=pinned(args.reference,REFERENCE_SHA)
    geometry_ref=pinned(args.geometry_reference,flip.REFERENCE_SHA)
    factor=pinned(args.factors,driver.FACTORS_SHA,"_choice_factors")
    prefix=pinned(args.prefix,driver.PREFIX_SHA,"_choice_prefix")
    slack=pinned(args.slack,driver.SLACK_SHA,"_choice_slack")
    evidence=pinned(args.evidence,args.evidence_sha,"_choice_evidence")
    math=pinned(args.selection,args.selection_sha,"_choice_math")
    prior=pinned(args.local_source,base.LOCAL_SHA,"_choice_prior")
    local=pinned(args.local_math,base.MATH_SHA,"_choice_local")
    policy=pinned(args.policy,broad.POLICY_SHA,"_choice_policy")
    env=teacher.Environment(args)
    try:
        roles=base.parts(env.fit)
        gs={k:{r["image_sha256"] for r in v} for k,v in roles.items()}
        old=gs["point_fit"]
        added=gs["gate_fit"]|gs["gate_cal"]
        if (len(env.fit),len(old),len(added))!=(484,288,191):
            raise ValueError("Exact frozen 484 FIT role")
        folds=factor.fold_map(old,added)
        rows=base.point_training(env,teacher,prior,local,env.fit)
        if len(rows)!=1082 or sum(len(r["y"]) for r in rows)!=245685:
            raise ValueError("Exact original fixed-pair population")
        rec,coverage=flip.collect(env,env.fit,rows,folds,old,teacher,local,slack,evidence)
        if coverage["counts"]!=ref["coverage"]["counts"]:
            raise ValueError("Exact closed original auxiliary associations")
        outer={}
        for f,prev in enumerate(ref["fold_reports"]):
            ti=[i for i,v in enumerate(rec) if v["fold"]!=f]
            hi=[i for i,v in enumerate(rec) if v["fold"]==f]
            scorer,report=flip.fit(env.torch,ti,rows,rec,prior,prefix,driver)
            if (f,len(ti),len(hi))!=(prev["fold"],prev["train_rows"],prev["held_rows"]) or any(
                report[k][n]!=prev["arms"][k][n] for k in flip.LEARNED for n in ("updates","bce","ranking","objective")):
                raise ValueError("Exact original four outer fitted heads")
            flip.score_fold(scorer,hi,rows,rec,policy)
            pred=decisions(scorer,hi,rows,rec,flip,policy,driver)
            for i,d in pred.items():
                if d["hits"]!={k:rec[i]["hits"][k] for k in ACTIONS}:
                    raise ValueError("Frozen actions exactly reproduced")
            outer.update(pred)
            print(json.dumps(dict(event="NESTED_VIEW_OUTER_REPLAY_COMPLETE",fold=f)),flush=True)
        replay=flip.summarize(rec,driver,geometry_ref)
        if any(replay[k]!=ref[k] for k in ("metrics","correct_hits_by_add_delete","high_IoU_paired","gate_checks")):
            raise ValueError("Exact original full fixed geometry endpoint replay")
        n,r,a,bound=opportunity(rec,outer,driver,math)
        if (bound["high_total"],bound["high_gain_over_raw"],bound["high_loss_by_flip"])!=(1161,38,40):
            raise ValueError("Exact P186 oracle bound")
        feasible=20*bound["minimum_native_ap90_losses"]<=int(n[:,8].sum())
        if feasible:
            routed,fold_reports=nested(env,rows,rec,outer,flip,prior,prefix,driver,policy,math)
            selector=report_select(n,r,a,outer,routed,driver,ref)
        else:
            fold_reports=[]
            selector=None
    finally:
        env.close()
    result=dict(status="HISTORICAL_FIT_NESTED_VIEW_SELECTION_COMPLETE",protocol=args.protocol,
                exact_original_actions=True,fit_images=484,fit_matches=1082,fit_content_groups=479,
                bound=bound,analytic_safety_feasible=feasible,native_ap90_total=int(n[:,8].sum()),
                selector=selector,fold_reports=fold_reports,actual_nested_fit=feasible,
                gate=bool(selector and selector["gate"]),AP=False,DEV=False,TEST=False)
    args.output.mkdir()
    with (args.output/"RESULT.json").open("x",encoding="utf8") as f:
        json.dump(result,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(result,sort_keys=True,allow_nan=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("teacher","selective-source","old-driver","broad","flip"):
        p.add_argument("--"+k,type=Path,required=True)
    first,_=p.parse_known_args()
    flip=pinned(first.flip,FLIP_SHA,"_closed_flip")
    driver=pinned(first.old_driver,flip.DRIVER_SHA,"_closed_utility")
    broad=pinned(first.broad,driver.BROAD_SHA,"_closed_broad")
    base=pinned(first.selective_source,broad.SELECTIVE_SHA,"_choice_selective")
    teacher=base.teacher_module(first.teacher)
    teacher.common_args(p)
    for k in ("local-source","local-math","policy","slack","factors","prefix","evidence","selection","reference","geometry-reference"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--evidence-sha",required=True)
    p.add_argument("--selection-sha",required=True)
    run(p.parse_args(),flip,driver,broad,base,teacher)
