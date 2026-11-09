"""Paired OOF candidate-context readout at two fixed training budgets."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time
from types import ModuleType

import numpy as np

DRIVER_SHA="a463b538900c5092617883b413dfb85c33c78f562d65b80ef0da2cd7ecc15150"
REFERENCE_SHA="b6d5d0e6817d81a2d044b35ae04d265c012eccddc045e39d3932e737a0bf8f29"
ARMS=("bce","raw","pooled","side")
ENDPOINTS=(96,384)


def pinned(path,digest,name=None):
    raw=path.read_bytes()
    if sha256(raw).hexdigest()!=digest:
        raise ValueError("Exact closed source or result")
    if name is None:
        return json.loads(raw)
    m=ModuleType(name)
    m.__file__=str(path)
    sys.modules[name]=m
    exec(compile(raw,str(path),"exec"),m.__dict__)
    return m


def views(rows,rank):
    out={k:[r["x"] for r in rows] for k in ("bce","raw")}
    audit={}
    for arm in ("pooled","side"):
        x=[]
        flat=0
        ties=0
        start=time.perf_counter()
        for r in rows:
            v=rank.ranked_inputs(r["x"],r["inside"],arm=="side")
            if v.shape!=r["x"].shape or not np.array_equal(v[:,:6],r["x"][:,:6]):
                raise ValueError("Keep six original columns and exact dimensions")
            x.append(v)
        elapsed=time.perf_counter()-start
        for r in rows:
            parts=[r["x"][:,6:]] if arm=="pooled" else [r["x"][r["inside"]==f,6:] for f in (False,True)]
            for a in parts:
                if len(a):
                    flat+=int((np.ptp(a,axis=0)==0).sum())
                    b=np.sort(a,axis=0)
                    ties+=int((b[1:]==b[:-1]).sum())
        out[arm]=x
        audit[arm]=dict(seconds=elapsed,flat_channels=flat,tied_adjacent_pairs=ties)
    return out,audit


def fit(torch,indices,rows,features,prior,prefix,old):
    fit_rows=[rows[i] for i in indices]
    y=torch.as_tensor(np.concatenate([r["y"] for r in fit_rows]),device="cuda",dtype=torch.float32)
    w_np=prior.weights(fit_rows)
    w=torch.as_tensor(w_np,device="cuda",dtype=torch.float32)
    gather,valid,side_weights=old.packed(fit_rows,w_np)
    idx=torch.as_tensor(gather,device="cuda")
    valid=torch.as_tensor(valid,device="cuda").flatten(0,1)
    ws=torch.as_tensor(side_weights,device="cuda",dtype=torch.float32).flatten()
    y_side=y[idx].bool().flatten(0,1)
    nn=torch.nn
    initial=None
    fits={e:{} for e in ENDPOINTS}
    snapshots={e:{} for e in ENDPOINTS}
    normalization={}
    for arm in ARMS:
        a=np.concatenate([features[arm][i] for i in indices])
        x=torch.as_tensor(a,device="cuda",dtype=torch.float32)
        mu=(x*w[:,None]).sum(0)
        scale=((x-mu).square()*w[:,None]).sum(0).sqrt().clamp_min(1e-4)
        x=(x-mu)/scale
        normalization[arm]=(mu,scale)
        torch.manual_seed(5201)
        net=nn.Sequential(nn.Linear(38,32),nn.GELU(),nn.Linear(32,1)).cuda()
        if initial is None:
            initial={k:t.detach().clone() for k,t in net.state_dict().items()}
        if sum(t.numel() for t in net.parameters())!=1281 or not all(torch.equal(t,initial[k]) for k,t in net.state_dict().items()):
            raise ValueError("Equal point head and exact paired initialization")
        opt=torch.optim.AdamW(net.parameters(),lr=.008,weight_decay=.02)
        torch.cuda.synchronize()
        start=time.perf_counter()
        for step in range(1,ENDPOINTS[-1]+1):
            opt.zero_grad(set_to_none=True)
            z=net(x).flatten()
            bce=(nn.functional.binary_cross_entropy_with_logits(z,y,reduction="none")*w).sum()
            ranking=bce.new_zeros(()) if arm=="bce" else (prefix.precision_ramp(z[idx].flatten(0,1),y_side,valid)*ws).sum()
            obj=bce if arm=="bce" else bce+.5*ranking
            obj.backward()
            opt.step()
            if step in ENDPOINTS:
                torch.cuda.synchronize()
                if {int(v["step"].item()) for v in opt.state.values()}!={step} or not torch.isfinite(obj):
                    raise ValueError("Exact cumulative optimizer clocks")
                fits[step][arm]=dict(seconds=time.perf_counter()-start,updates=step,bce=float(bce.detach()),
                                      ranking=float(ranking.detach()),objective=float(obj.detach()))
                snapshots[step][arm]={k:t.detach().clone() for k,t in net.state_dict().items()}
    def predict(step,arm,rows):
        mu,scale=normalization[arm]
        a=(torch.as_tensor(rows,device="cuda",dtype=torch.float32)-mu)/scale
        p=snapshots[step][arm]
        with torch.no_grad():
            # Same Linear/GELU kernels and same held batch shapes as P180.
            z=nn.functional.linear(a,p["0.weight"],p["0.bias"])
            z=nn.functional.gelu(z)
            z=nn.functional.linear(z,p["2.weight"],p["2.bias"]).flatten()
        if not torch.isfinite(z).all():
            raise ValueError("Finite prediction scores")
        return z.cpu().numpy()
    return predict,fits


def pick(predict,held,rows,features,records,policy):
    out={e:{} for e in ENDPOINTS}
    for step in ENDPOINTS:
        for is_old in (True,False):
            sub=[i for i in held if records[i]["old"]==is_old]
            if not sub:
                continue
            cuts=np.cumsum([len(rows[i]["x"]) for i in sub])[:-1]
            split={}
            for arm in ARMS:
                x=np.concatenate([features[arm][i] for i in sub])
                split[arm]=np.split(predict(step,arm,x),cuts)
            for j,i in enumerate(sub):
                r=rows[i]
                rec=records[i]
                h={}
                choice={}
                for arm in ARMS:
                    a,d=policy.choose_points(rec["pixel"],r["inside"],split[arm][j],4)
                    h[arm]=(int(r["y"][a].sum()),int(r["y"][d].sum()))
                    lo,hi=rec["limit"]
                    if not lo<=sum(h[arm])<=hi:
                        raise ValueError("Exact four-plus-four attainable edits")
                    choice[arm]=(np.sort(a),np.sort(d))
                changes={k:[int(not np.array_equal(a,b)) for a,b in zip(choice["raw"],choice[k],strict=True)]
                         for k in ("pooled","side")}
                out[step][i]=dict(hits=h,changed=changes)
    return out


def summarize(records,choices,old):
    z={k:[] for k in ("native","oracle",*ARMS)}
    groups=[]
    hit_totals={k:np.zeros(2,dtype=np.int64) for k in ARMS}
    changes={k:np.zeros(2,dtype=np.int64) for k in ("pooled","side")}
    for i,r in enumerate(records):
        iou=lambda h:old.case_thresholds(r["intersection"],r["area_sum"],h)
        groups.append(r["group"])
        z["native"].append(iou(4))
        z["oracle"].append(iou(r["limit"][1]))
        row=choices[i]
        for k in ARMS:
            h=row["hits"][k]
            z[k].append(iou(sum(h)))
            hit_totals[k]+=h
        for k in changes:
            changes[k]+=row["changed"][k]
    z={k:np.stack(v) for k,v in z.items()}
    n,b,o=(z[k] for k in ("native","bce","oracle"))
    metrics={}
    for k,v in z.items():
        metrics[k]=dict(mean_thresholds=float(v.sum(1).mean()),passed=v.sum(0).tolist(),
                        gained_vs_native=(v&~n).sum(0).tolist(),lost_vs_native=(n&~v).sum(0).tolist(),
                        gained_vs_BCE=(v&~b).sum(0).tolist(),lost_vs_BCE=(b&~v).sum(0).tolist(),
                        total_deficit_to_oracle=int((o&~v).sum()))
    paired={k:old.paired_difference(groups,z["side"][:,7:].sum(1),z[k][:,7:].sum(1)) for k in ARMS if k!="side"}
    safety=dict(ap75_proxy_pass_nondecrease=metrics["side"]["passed"][5]>=metrics["bce"]["passed"][5],
                native_ap90_loss_no_increase_and_le_5_percent=metrics["side"]["lost_vs_native"][8]<=metrics["bce"]["lost_vs_native"][8]
                and 20*metrics["side"]["lost_vs_native"][8]<=metrics["native"]["passed"][8])
    return dict(metrics=metrics,high_IoU_paired=paired,safety=safety,
                correct_hits_by_add_delete={k:v.tolist() for k,v in hit_totals.items()},
                side_choice_changes_vs_raw={k:v.tolist() for k,v in changes.items()})


def run(args,old,broad,base,teacher):
    ref=pinned(args.reference,REFERENCE_SHA)
    factor=pinned(args.factors,old.FACTORS_SHA,"_rank_factors")
    prefix=pinned(args.prefix,old.PREFIX_SHA,"_rank_prefix")
    slack=pinned(args.slack,old.SLACK_SHA,"_rank_slack")
    rank=pinned(args.rank,args.rank_sha,"_candidate_rank")
    prior=pinned(args.local_source,base.LOCAL_SHA,"_rank_prior")
    local=pinned(args.local_math,base.MATH_SHA,"_rank_local")
    policy=pinned(args.policy,broad.POLICY_SHA,"_rank_policy")
    env=teacher.Environment(args)
    try:
        roles=base.parts(env.fit)
        by={k:{r["image_sha256"] for r in v} for k,v in roles.items()}
        small=by["point_fit"]
        extra=by["gate_fit"]|by["gate_cal"]
        if (len(env.fit),len(small),len(extra))!=(484,288,191):
            raise ValueError("Exact original broad point group roles")
        folds=factor.fold_map(small,extra)
        rows=base.point_training(env,teacher,prior,local,env.fit)
        if len(rows)!=1082 or sum(len(r["y"]) for r in rows)!=245685:
            raise ValueError("Exact original broad point features and labels")
        records=old.geometry(env,env.fit,rows,folds,small,local,slack)
        if any(r["limit"] is None for r in records):
            raise ValueError("Exact original full four-plus-four support")
        features,audit=views(rows,rank)
        prediction={e:{} for e in ENDPOINTS}
        reports=[]
        for f,prior_fold in enumerate(ref["folds"]):
            ti=[i for i,r in enumerate(records) if r["fold"]!=f]
            hi=[i for i,r in enumerate(records) if r["fold"]==f]
            predict,report=fit(env.torch,ti,rows,features,prior,prefix,old)
            if (f,len(ti),len(hi))!=(prior_fold["fold"],prior_fold["train_rows"],prior_fold["held_rows"]):
                raise ValueError("Exact original fold budget")
            for k,q in (("bce","bce"),("raw","prefix")):
                if any(report[96][k][n]!=prior_fold["arms"][q][n] for n in ("bce","ranking","objective","updates")):
                    raise ValueError("Exact original BCE and raw prefix 96-step endpoints")
            selected=pick(predict,hi,rows,features,records,policy)
            for e in ENDPOINTS:
                prediction[e].update(selected[e])
            reports.append(dict(fold=f,train_rows=len(ti),held_rows=len(hi),endpoints=report))
            print(json.dumps(dict(event="SIDE_CONTEXT_FOLD_COMPLETE",fold=f,held_rows=len(hi))),flush=True)
    finally:
        env.close()
    summary={e:summarize(records,prediction[e],old) for e in ENDPOINTS}
    for k,q in (("bce","bce"),("raw","prefix"),("native","native"),("oracle","oracle")):
        if summary[96]["metrics"][k]!=ref["metrics"][q]:
            raise ValueError("Exact closed original endpoint fixed geometry replay")
    p=summary[96]
    v=summary[384]
    checks=dict(high_micro_ge_0_01_vs_all=all(a["micro"]>=.01 for a in p["high_IoU_paired"].values()),
                high_group_interval_positive_vs_all=all(a["group_percentile95"][0]>0 for a in p["high_IoU_paired"].values()),
                primary_safety=all(p["safety"].values()),
                longer_same_direction_and_safety=all(v["high_IoU_paired"][k]["micro"]>=0 for k in ("raw","pooled"))
                and all(v["safety"].values()))
    out=dict(status="HISTORICAL_FIT_SIDE_CONTEXT_RANK_COMPLETE",protocol=args.protocol,
             exact_P181_raw_BCE_replay=True,fit_images=484,fit_matches=1082,fit_content_groups=479,
             fold_reports=reports,feature_audit=audit,endpoints=summary,gate_checks=checks,gate=all(checks.values()),
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
    for k in ("teacher","selective-source","old-driver","broad"):
        p.add_argument("--"+k,type=Path,required=True)
    first,_=p.parse_known_args()
    old=pinned(first.old_driver,DRIVER_SHA,"_closed_utility")
    broad=pinned(first.broad,old.BROAD_SHA,"_closed_broad")
    base=pinned(first.selective_source,broad.SELECTIVE_SHA,"_rank_selective")
    teacher=base.teacher_module(first.teacher)
    teacher.common_args(p)
    for k in ("local-source","local-math","policy","slack","factors","prefix","rank","reference"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--rank-sha",required=True)
    run(p.parse_args(),old,broad,base,teacher)
