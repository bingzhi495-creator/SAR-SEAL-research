"""Matched four-arm training and fixed-geometry evaluation in historical FIT."""
import argparse
from collections import Counter,defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time
from types import ModuleType

import numpy as np

BROAD_SHA="72df216dab5be65b298a79327c5780dc2daa7ab87a846264b7b660de05b28b19"
FACTORS_SHA="093eb0059944e90716e1683e0051a9d2399e774fca2318b5bf80e9dab51d3ebb"
ACTION_SHA="d8dbe768e26c6827f2aaa420b2fcf0f8cdba54456d955eb86df28eed41aa2b46"
PREFIX_SHA="e6bdf3c9df789aac7a2d5266d0cf73abee0dad59e4bc2f74e0d96d814060220d"
SLACK_SHA="1286df02c04781f6b82d95c474f4f6ca0dd0db56d7f960a70bbe6754f99bdb06"
REFERENCE_SHA="e174b39a79d21438c50ae04380adcdccc01ee31fc637015ddd14745d99fb0754"
ARMS=("bce","prefix","linear","threshold")
M=np.arange(10,20,dtype=np.int64)
STEPS=96


def pinned(path,digest,name=None):
    raw=path.read_bytes()
    if sha256(raw).hexdigest()!=digest:
        raise ValueError("Exact closed inputs required")
    if name is None:
        return json.loads(raw)
    m=ModuleType(name)
    m.__file__=str(path)
    sys.modules[name]=m
    exec(compile(raw,str(path),"exec"),m.__dict__)
    return m


def case_thresholds(i,a,h):
    return (20+M)*(i+h-4)>=M*a


def paired_difference(groups,new,old):
    delta=new-old
    group=defaultdict(list)
    for k,n in zip(groups,delta,strict=True):
        group[k].append(n)
    values=np.array([np.mean(v) for _,v in sorted(group.items())])
    sample=np.random.default_rng(5201).integers(0,len(values),size=(2048,len(values)))
    ci=np.quantile(values[sample].mean(1),[.025,.975]).tolist()
    return dict(micro=float(delta.mean()),group_mean=float(values.mean()),groups=len(values),group_percentile95=ci)


def geometry(env,items,rows,folds,old,local,slack):
    records=[]
    for i,item in enumerate(items,1):
        c=env.image(item)
        _,truth,matches,_=env.matches(item,c)
        group=item["image_sha256"]
        for j,k in matches:
            q=c["query"][j]
            if not bool(c["ctx"]["active"][q]):
                continue
            mask=c["native"][j]
            gt=truth[k]
            xy,_,_,_,inside=local.boundary_samples(mask,128,5201+100*item["dataset_index"]+q)
            if not len(xy):
                continue
            row=rows[len(records)]
            y=gt[xy[:,1],xy[:,0]]!=inside
            if row["group"]!=group or not np.array_equal(row["inside"],inside) or not np.array_equal(row["y"],y):
                raise ValueError("Same content, side, and label row as frozen point fitting")
            intersection=int(np.count_nonzero(mask&gt))
            area_sum=int(mask.sum()+gt.sum())
            px=xy[:,1]*c["hw"][1]+xy[:,0]
            records.append(dict(group=group,old=group in old,fold=folds[group],pixel=px,
                                intersection=intersection,area_sum=area_sum,slack=7*intersection-3*area_sum,
                                limit=slack.bounds(y,inside)))
        if i%100==0 or i==len(items):
            print(json.dumps(dict(event="COCO_UTILITY_GEOMETRY_PROGRESS",images=i)),flush=True)
    if len(records)!=len(rows):
        raise ValueError("All matched fitting rows have exact integer geometry")
    return records


def packed(rows,w):
    index=[]
    start=0
    for r in rows:
        inside=r["inside"]
        index.append(tuple(start+np.flatnonzero(inside==flag) for flag in (False,True)))
        start+=len(inside)
    length=max(len(v) for pair in index for v in pair)
    gather=np.zeros((len(rows),2,length),dtype=np.int64)
    valid=np.zeros(gather.shape,dtype=bool)
    for j,pair in enumerate(index):
        for k,v in enumerate(pair):
            gather[j,k,:len(v)]=v
            valid[j,k,:len(v)]=True
    weight=(w[gather]*valid).sum(2)
    if int(valid.sum())!=start or start!=len(w) or length>128 or abs(float(weight.sum())-1)>1e-5:
        raise ValueError("Exact original per-point weights and common candidate mass")
    return gather,valid,weight


def train(torch,rows,geo,prior,prefix,action):
    nn=torch.nn
    x,y=(torch.as_tensor(np.concatenate([r[k] for r in rows]),device="cuda",dtype=torch.float32)
         for k in ("x","y"))
    wn=prior.weights(rows)
    w=torch.as_tensor(wn,device="cuda",dtype=torch.float32)
    mu=(x*w[:,None]).sum(0)
    scale=((x-mu).square()*w[:,None]).sum(0).sqrt().clamp_min(1e-4)
    x=(x-mu)/scale
    idx,valid,ws=packed(rows,wn)
    idx=torch.as_tensor(idx,device="cuda")
    valid=torch.as_tensor(valid,device="cuda")
    ws=torch.as_tensor(ws,device="cuda",dtype=torch.float32)
    labels=y[idx].bool()
    cweight=ws.sum(1)
    i,a=(torch.as_tensor([r[k] for r in geo],device="cuda",dtype=torch.int64) for k in ("intersection","area_sum"))
    u=action.iou_table(i,a)
    models={}
    report={}
    init=None
    for arm in ARMS:
        torch.manual_seed(5201)
        model=nn.Sequential(nn.Linear(38,32),nn.GELU(),nn.Linear(32,1)).cuda()
        if init is None:
            init={k:v.detach().clone() for k,v in model.state_dict().items()}
        if sum(t.numel() for t in model.parameters())!=1281 or not all(torch.equal(t,init[k]) for k,t in model.state_dict().items()):
            raise ValueError("Same point head capacity and paired exact initialization")
        opt=torch.optim.AdamW(model.parameters(),lr=.008,weight_decay=.02)
        torch.cuda.synchronize()
        started=time.perf_counter()
        for _ in range(STEPS):
            opt.zero_grad(set_to_none=True)
            z=model(x).flatten()
            bce=(nn.functional.binary_cross_entropy_with_logits(z,y,reduction="none")*w).sum()
            reg=bce.new_zeros(())
            if arm=="prefix":
                reg=(prefix.precision_ramp(z[idx].flatten(0,1),labels.flatten(0,1),valid.flatten(0,1))*ws.flatten()).sum()
            elif arm in ("linear","threshold"):
                t,l=action.regrets(z[idx],labels,valid,u)
                reg=((l if arm=="linear" else t)*cweight).sum()
            loss=bce if arm=="bce" else bce+.5*reg
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        if {int(v["step"].item()) for v in opt.state.values()}!={STEPS} or not torch.isfinite(loss):
            raise ValueError("Completed finite equal optimizer clocks")
        report[arm]=dict(seconds=time.perf_counter()-started,updates=STEPS,bce=float(bce.detach()),
                         ranking=float(reg.detach()),objective=float(loss.detach()))
        models[arm]=model
    def scores(x):
        with torch.no_grad():
            a=(torch.as_tensor(x,device="cuda",dtype=torch.float32)-mu)/scale
            out={k:m(a).flatten().cpu().numpy() for k,m in models.items()}
        if not all(np.isfinite(z).all() for z in out.values()):
            raise ValueError("Finite scores for all paired held candidates")
        return out
    return scores,report


def score_fold(scorer,indices,rows,records,policy):
    by={}
    for old in (True,False):
        sub=[i for i in indices if records[i]["old"]==old]
        if not sub:
            continue
        x=np.concatenate([rows[i]["x"] for i in sub])
        cuts=np.cumsum([len(rows[i]["x"]) for i in sub])[:-1]
        split={k:np.split(v,cuts) for k,v in scorer(x).items()}
        for j,i in enumerate(sub):
            row=rows[i]
            rec=records[i]
            h={}
            if rec["limit"] is not None:
                for arm in ARMS:
                    a,d=policy.choose_points(rec["pixel"],row["inside"],split[arm][j],4)
                    h[arm]=int(row["y"][a].sum()+row["y"][d].sum())
                    lo,hi=rec["limit"]
                    if not lo<=h[arm]<=hi:
                        raise ValueError("Chosen points respect candidate-side feasibility")
            by[i]=h
    return by


def totals(records):
    valid=[r for r in records if r["limit"] is not None]
    pa=defaultdict(list)
    groups=[]
    all_active=set()
    curved=set()
    strict_counts=Counter()
    for r in valid:
        i,a=r["intersection"],r["area_sum"]
        lo,hi=r["limit"]
        groups.append(r["group"])
        pa["native"].append(case_thresholds(i,a,4))
        pa["oracle"].append(case_thresholds(i,a,hi))
        for arm in ARMS:
            pa[arm].append(case_thresholds(i,a,r["hits"][arm]))
        u=np.array([case_thresholds(i,a,h).sum() for h in range(lo,hi+1)])
        if u[0]!=u[-1]:
            all_active.add(r["group"])
            strict_counts["threshold_active_rows"]+=1
            if ((u-u[0])*(hi-lo)!=np.arange(len(u))*(u[-1]-u[0])).any():
                curved.add(r["group"])
                strict_counts["nonaffine_rows"]+=1
    pa={k:np.stack(v) for k,v in pa.items()}
    n=pa["native"]
    b=pa["bce"]
    oracle=pa["oracle"]
    summary={}
    for arm,v in pa.items():
        summary[arm]=dict(mean_thresholds=float(v.sum(1).mean()),passed=v.sum(0).tolist(),
                          gained_vs_native=(v&~n).sum(0).tolist(),lost_vs_native=(n&~v).sum(0).tolist(),
                          gained_vs_BCE=(v&~b).sum(0).tolist(),lost_vs_BCE=(b&~v).sum(0).tolist(),
                          total_deficit_to_oracle=int((oracle&~v).sum()))
    p={k:paired_difference(groups,pa["threshold"].sum(1),pa[k].sum(1)) for k in ARMS if k!="threshold"}
    checks=dict(micro_gain_ge_0_01_over_every_control=all(v["micro"]>=.01 for v in p.values()),
                group_interval_lower_positive_vs_every_control=all(v["group_percentile95"][0]>0 for v in p.values()),
                ap75_proxy_pass_nondecrease=summary["threshold"]["passed"][5]>=summary["bce"]["passed"][5],
                native_ap90_loss_no_increase_and_le_5_percent=summary["threshold"]["lost_vs_native"][8]<=summary["bce"]["lost_vs_native"][8]
                and 20*summary["threshold"]["lost_vs_native"][8]<=summary["native"]["passed"][8])
    return dict(eligible=len(valid),thresholds=[f"{m/20:.2f}" for m in M],metrics=summary,paired=p,
                geometry=dict(groups=len(set(groups)),threshold_active_groups=len(all_active),nonaffine_groups=len(curved),
                              **strict_counts),gate_checks=checks,gate=all(checks.values()))


def run(args,broad,base,teacher):
    result_ref=pinned(args.reference,REFERENCE_SHA)
    fold_support=pinned(args.factors,FACTORS_SHA,"_utility_folds")
    action=pinned(args.action,ACTION_SHA,"_utility_action")
    prefix=pinned(args.prefix,PREFIX_SHA,"_utility_prefix")
    slack=pinned(args.slack,SLACK_SHA,"_utility_slack")
    prior=pinned(args.local_source,base.LOCAL_SHA,"_utility_prior")
    local=pinned(args.local_math,base.MATH_SHA,"_utility_local")
    policy=pinned(args.policy,broad.POLICY_SHA,"_utility_policy")
    env=teacher.Environment(args)
    try:
        roles=base.parts(env.fit)
        group={k:{r["image_sha256"] for r in v} for k,v in roles.items()}
        old=group["point_fit"]
        added=group["gate_fit"]|group["gate_cal"]
        if (len(env.fit),len(old),len(added))!=(484,288,191):
            raise ValueError("Exact old plus added 484 FIT groups")
        folds=fold_support.fold_map(old,added)
        rows=base.point_training(env,teacher,prior,local,env.fit)
        if len(rows)!=result_ref["training_matches"] or sum(len(r["y"]) for r in rows)!=result_ref["training_points"]:
            raise ValueError("Exact closed 484-image matched point population")
        records=geometry(env,env.fit,rows,folds,old,local,slack)
        reports=[]
        for f,prev in enumerate(result_ref["fold_reports"]):
            ti=[i for i,r in enumerate(records) if r["fold"]!=f]
            hi=[i for i,r in enumerate(records) if r["fold"]==f]
            score,report=train(env.torch,[rows[i] for i in ti],[records[i] for i in ti],prior,prefix,action)
            if (f,len(ti),sum(records[i]["old"] for i in hi),sum(not records[i]["old"] for i in hi),report["bce"]["bce"])!=(
                prev["fold"],prev["fits"]["large"]["train_rows"],prev["old_held_rows"],prev["added_held_rows"],prev["fits"]["large"]["loss"]):
                raise ValueError("Exact per-fold original BCE endpoint and role counts")
            scored=score_fold(score,hi,rows,records,policy)
            for i,h in scored.items():
                records[i]["hits"]=h
            reports.append(dict(fold=f,train_rows=len(ti),held_rows=len(hi),arms=report))
            print(json.dumps(dict(event="COCO_UTILITY_FOLD_COMPLETE",fold=f,held_rows=len(hi))),flush=True)
    finally:
        env.close()
    for name,pop in (("all",records),("old",[r for r in records if r["old"]]),
                     ("added",[r for r in records if not r["old"]])):
        c=broad.census(pop,"bce",slack)
        if json.loads(json.dumps(c,sort_keys=True))!=result_ref["census"][name]["large"]:
            raise ValueError("Exact closed expanded BCE AP75 fixed-match replay")
    out=totals(records)
    out.update(status="HISTORICAL_FIT_COCO_ACTION_UTILITY_OOF_COMPLETE",protocol=args.protocol,folds=reports,
               exact_P177_BCE_replay=True,fit_images=484,fit_content_groups=479,fit_matches=len(records),
               AP=False,DEV=False,TEST=False)
    args.output.mkdir()
    with (args.output/"RESULT.json").open("x",encoding="utf8") as f:
        json.dump(out,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(out,sort_keys=True,allow_nan=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("teacher","selective-source","broad"):
        p.add_argument("--"+k,type=Path,required=True)
    first,_=p.parse_known_args()
    broad=pinned(first.broad,BROAD_SHA,"_closed_broad")
    base=pinned(first.selective_source,broad.SELECTIVE_SHA,"_utility_selective")
    teacher=base.teacher_module(first.teacher)
    teacher.common_args(p)
    for k in ("local-source","local-math","policy","slack","factors","prefix","action","reference"):
        p.add_argument("--"+k,type=Path,required=True)
    run(p.parse_args(),broad,base,teacher)
