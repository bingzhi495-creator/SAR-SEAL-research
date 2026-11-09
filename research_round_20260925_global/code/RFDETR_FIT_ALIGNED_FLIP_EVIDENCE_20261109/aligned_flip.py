"""Two-view aligned directional signal under fixed SSDD candidate edit support."""
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
LEARNED=("raw38","raw39","sham","aligned")
ARMS=(*LEARNED,"mean_tta")
STEPS=96


def pinned(path,digest,name=None):
    raw=path.read_bytes()
    if sha256(raw).hexdigest()!=digest:
        raise ValueError("Exact previously verified input")
    if name is None:
        return json.loads(raw)
    m=ModuleType(name)
    m.__file__=str(path)
    sys.modules[name]=m
    exec(compile(raw,str(path),"exec"),m.__dict__)
    return m


def flip_forward(env,item,c,check):
    from PIL import Image

    encoded=env.bundle.read(f'images/{item["image_sha256"]}.jpg')
    if len(encoded)!=item["image_bytes"] or sha256(encoded).hexdigest()!=item["image_sha256"]:
        raise ValueError("Exact same original image content")
    im,scale=env.decode(encoded,draft_size=None)
    hw=c["hw"]
    if im.mode!="RGB" or im.size!=(hw[1],hw[0]) or scale!=(1.,1.):
        raise ValueError("Exact same original RGB raster")
    f=Image.Transpose.FLIP_LEFT_RIGHT
    flipped=im.transpose(f)
    if check and flipped.transpose(f).tobytes()!=im.tobytes():
        raise ValueError("Exact discrete horizontal flip and inverse")
    torch=env.torch
    torch.cuda.synchronize()
    start=time.perf_counter()
    raw,_,_=env.infer(flipped)
    torch.cuda.synchronize()
    seconds=time.perf_counter()-start
    with torch.no_grad():
        masks,_,ids,_=env.core._read_stock(raw,hw,env.top,torch)
        keep=[j for j,(_,slot) in enumerate(ids) if slot==0]
        queries=[ids[j][0] for j in keep]
        m=masks[keep,0].flip(-1).cpu().numpy()
    if len(queries)!=len(set(queries)):
        raise ValueError("Distinct flipped stock queries")
    start=time.perf_counter()
    rr=env.s.coerce_rles(c["native"],env.codec)
    ar=env.s.coerce_rles(m,env.codec)
    iou=env.codec.iou(rr,ar,[0]*len(ar)) if rr and ar else np.zeros((len(rr),len(ar)))
    pairs=dict(env.s.greedy(iou))
    if len(set(pairs.values()))!=len(pairs) or any(iou[i,j]<.5 for i,j in pairs.items()):
        raise ValueError("One-to-one GT-free native-rank mask IoU association")
    matching=time.perf_counter()-start
    return raw["pred_masks"][0],queries,m,pairs,seconds,matching


def collect(env,items,rows,folds,old,teacher,local,slack,evidence):
    records=[]
    totals=Counter()
    forward_s=matching_s=render_s=0.
    first=True
    for i,item in enumerate(items,1):
        c=env.image(item)
        aux_z,aux_q,aux_m,links,f,m=flip_forward(env,item,c,first)
        first=False
        forward_s+=f
        matching_s+=m
        areas=c["native"].sum((1,2))
        small=(areas>0)&(areas<1024)
        totals.update(original_candidates=len(c["query"]),auxiliary_candidates=len(aux_q),linked_candidates=len(links),
                      original_small_candidates=int(small.sum()),linked_small_candidates=sum(bool(small[j]) for j in links))
        # Auxiliary association and all candidate coverage precede the GT-based fixed-pair readout.
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
            r=rows[len(records)]
            y=gt[xy[:,1],xy[:,0]]!=inside
            if r["group"]!=group or not np.array_equal(r["inside"],inside) or not np.array_equal(r["y"],y):
                raise ValueError("Exact original point labels and ordering")
            p0=1/(1+np.exp(-np.clip(r["x"][:,0],-60,60)))
            p1=p0
            linked=j in links
            if linked:
                b=links[j]
                zq=aux_q[b]
                t=time.perf_counter()
                plane=teacher.render(aux_z[zq:zq+1],c["hw"])[0].flip(-1).detach().cpu().numpy()
                render_s+=time.perf_counter()-t
                if not np.array_equal(plane>0,aux_m[b]):
                    raise ValueError("Exact original-raster auxiliary logit polarity after reflection")
                p1=1/(1+np.exp(-np.clip(plane[xy[:,1],xy[:,0]],-60,60)))
            base,d,avg=evidence.signed_scores(p0,p1,inside)
            sham=evidence.shuffled(d,p0,inside,5201+1000*item["dataset_index"]+q) if linked else base
            totals.update(linked_fixed_pairs=int(linked),sham_changed_points=int((sham!=d).sum()) if linked else 0,
                          linked_points=len(xy) if linked else 0)
            record=dict(group=group,old=group in old,fold=folds[group],
                        intersection=int(np.count_nonzero(mask&gt)),area_sum=int(mask.sum()+gt.sum()),
                        limit=slack.bounds(y,inside),pixel=xy[:,1]*c["hw"][1]+xy[:,0],
                        linked=linked,mean_tta=avg)
            if record["limit"] is None:
                raise ValueError("Exact original full 4+4 support")
            record["extra"]=dict(raw39=base,sham=sham,aligned=d if linked else base)
            records.append(record)
        if i%100==0 or i==len(items):
            print(json.dumps(dict(event="ALIGNED_FLIP_GEOMETRY_PROGRESS",images=i)),flush=True)
    if len(records)!=len(rows):
        raise ValueError("Same complete fixed-pair population")
    return records,dict(counts=totals,additional_forward_s=forward_s,prediction_matching_s=matching_s,
                        linked_point_render_s=render_s,exact_double_flip=True)


def feature(r,record,arm):
    x=r["x"]
    if arm=="raw38":
        return x
    return np.column_stack((x,record["extra"][arm])).astype(np.float32,copy=False)


def fit(torch,idx,rows,records,prior,prefix,driver):
    nn=torch.nn
    fit_rows=[rows[i] for i in idx]
    y=torch.as_tensor(np.concatenate([r["y"] for r in fit_rows]),device="cuda",dtype=torch.float32)
    wn=prior.weights(fit_rows)
    w=torch.as_tensor(wn,device="cuda",dtype=torch.float32)
    gather,v,ws=driver.packed(fit_rows,wn)
    g=torch.as_tensor(gather,device="cuda")
    v=torch.as_tensor(v,device="cuda").flatten(0,1)
    ws=torch.as_tensor(ws,device="cuda",dtype=torch.float32).flatten()
    yy=y[g].bool().flatten(0,1)
    fits={}
    models={}
    normal={}
    initial={}
    for arm in LEARNED:
        a=np.concatenate([feature(rows[i],records[i],arm) for i in idx])
        x=torch.as_tensor(a,device="cuda",dtype=torch.float32)
        mu=(x*w[:,None]).sum(0)
        scale=((x-mu).square()*w[:,None]).sum(0).sqrt().clamp_min(1e-4)
        x=(x-mu)/scale
        normal[arm]=(mu,scale)
        width=38 if arm=="raw38" else 39
        torch.manual_seed(5201)
        net=nn.Sequential(nn.Linear(width,32),nn.GELU(),nn.Linear(32,1)).cuda()
        if width not in initial:
            initial[width]={k:t.detach().clone() for k,t in net.state_dict().items()}
        if a.shape[1]!=width or sum(t.numel() for t in net.parameters())!=32*width+65 or not all(
            torch.equal(t,initial[width][k]) for k,t in net.state_dict().items()):
            raise ValueError("Identical initialization per controlled input dimension")
        opt=torch.optim.AdamW(net.parameters(),lr=.008,weight_decay=.02)
        torch.cuda.synchronize()
        start=time.perf_counter()
        for _ in range(STEPS):
            opt.zero_grad(set_to_none=True)
            z=net(x).flatten()
            bce=(nn.functional.binary_cross_entropy_with_logits(z,y,reduction="none")*w).sum()
            rank=(prefix.precision_ramp(z[g].flatten(0,1),yy,v)*ws).sum()
            obj=bce+.5*rank
            obj.backward()
            opt.step()
        torch.cuda.synchronize()
        if {int(v["step"].item()) for v in opt.state.values()}!={STEPS} or not torch.isfinite(obj):
            raise ValueError("Exact finite optimizer endpoints")
        fits[arm]=dict(seconds=time.perf_counter()-start,updates=STEPS,bce=float(bce.detach()),
                       ranking=float(rank.detach()),objective=float(obj.detach()))
        models[arm]=net
    def predict(arm,data):
        mu,scale=normal[arm]
        with torch.no_grad():
            x=(torch.as_tensor(data,device="cuda",dtype=torch.float32)-mu)/scale
            z=models[arm](x).flatten()
        if not torch.isfinite(z).all():
            raise ValueError("Finite all-head held predictions")
        return z.cpu().numpy()
    return predict,fits


def score_fold(predict,held,rows,records,policy):
    for is_old in (True,False):
        sub=[i for i in held if records[i]["old"]==is_old]
        if not sub:
            continue
        cuts=np.cumsum([len(rows[i]["x"]) for i in sub])[:-1]
        split={k:np.split(predict(k,np.concatenate([feature(rows[i],records[i],k) for i in sub])),cuts) for k in LEARNED}
        for j,i in enumerate(sub):
            r=rows[i]
            a=records[i]
            z={k:split[k][j] for k in LEARNED}
            if a["linked"]:
                z["mean_tta"]=a["mean_tta"]
            else:
                for k in ("sham","aligned","mean_tta"):
                    z[k]=z["raw39"]
            h={}
            for k in ARMS:
                aa,dd=policy.choose_points(a["pixel"],r["inside"],z[k],4)
                h[k]=(int(r["y"][aa].sum()),int(r["y"][dd].sum()))
                lo,hi=a["limit"]
                if not lo<=sum(h[k])<=hi:
                    raise ValueError("Exact 4+4 action within support")
            a["hits"]=h


def summarize(records,driver,ref):
    z={k:[] for k in ("native","oracle",*ARMS)}
    correct={k:np.zeros(2,dtype=np.int64) for k in ARMS}
    groups=[]
    for r in records:
        key=lambda h:driver.case_thresholds(r["intersection"],r["area_sum"],h)
        groups.append(r["group"])
        z["native"].append(key(4))
        z["oracle"].append(key(r["limit"][1]))
        for k,h in r["hits"].items():
            z[k].append(key(sum(h)))
            correct[k]+=h
        if not r["linked"] and len({r["hits"][k] for k in ("raw39","aligned","sham","mean_tta")})!=1:
            raise ValueError("Identical unlinked fallback")
    z={k:np.stack(v) for k,v in z.items()}
    n,b,o=(z[k] for k in ("native","raw38","oracle"))
    metrics={}
    for k,v in z.items():
        metrics[k]=dict(mean_thresholds=float(v.sum(1).mean()),passed=v.sum(0).tolist(),
                        gained_vs_native=(v&~n).sum(0).tolist(),lost_vs_native=(n&~v).sum(0).tolist(),
                        gained_vs_raw38=(v&~b).sum(0).tolist(),lost_vs_raw38=(b&~v).sum(0).tolist(),
                        total_deficit_to_oracle=int((o&~v).sum()))
    for k,q in (("native","native"),("raw38","prefix"),("oracle","oracle")):
        if any(metrics[k][f]!=ref["metrics"][q][f] for f in ("mean_thresholds","passed","gained_vs_native","lost_vs_native","total_deficit_to_oracle")):
            raise ValueError("Exact closed raw 38-D fixed geometry reference")
    paired={k:driver.paired_difference(groups,z["aligned"][:,7:].sum(1),z[k][:,7:].sum(1)) for k in ARMS if k!="aligned"}
    checks=dict(high_micro_ge_0_01_vs_all=all(a["micro"]>=.01 for a in paired.values()),
                high_group_interval_positive_vs_all=all(a["group_percentile95"][0]>0 for a in paired.values()),
                ap75_proxy_pass_nondecrease=metrics["aligned"]["passed"][5]>=ref["metrics"]["bce"]["passed"][5],
                native_ap90_loss_no_increase_and_le_5_percent=metrics["aligned"]["lost_vs_native"][8]<=ref["metrics"]["bce"]["lost_vs_native"][8]
                and 20*metrics["aligned"]["lost_vs_native"][8]<=metrics["native"]["passed"][8])
    return dict(metrics=metrics,high_IoU_paired=paired,gate_checks=checks,gate=all(checks.values()),
                correct_hits_by_add_delete={k:v.tolist() for k,v in correct.items()})


def run(args,driver,broad,base,teacher):
    ref=pinned(args.reference,REFERENCE_SHA)
    factors=pinned(args.factors,driver.FACTORS_SHA,"_flip_factors")
    prefix=pinned(args.prefix,driver.PREFIX_SHA,"_flip_prefix")
    slack=pinned(args.slack,driver.SLACK_SHA,"_flip_slack")
    evidence=pinned(args.evidence,args.evidence_sha,"_flip_evidence")
    prior=pinned(args.local_source,base.LOCAL_SHA,"_flip_prior")
    local=pinned(args.local_math,base.MATH_SHA,"_flip_local")
    policy=pinned(args.policy,broad.POLICY_SHA,"_flip_policy")
    env=teacher.Environment(args)
    try:
        roles=base.parts(env.fit)
        groups={k:{r["image_sha256"] for r in v} for k,v in roles.items()}
        old=groups["point_fit"]
        added=groups["gate_fit"]|groups["gate_cal"]
        if (len(env.fit),len(old),len(added))!=(484,288,191):
            raise ValueError("Original 484 FIT roles")
        folds=factors.fold_map(old,added)
        rows=base.point_training(env,teacher,prior,local,env.fit)
        if len(rows)!=1082 or sum(len(r["y"]) for r in rows)!=245685:
            raise ValueError("Original matched point population")
        records,coverage=collect(env,env.fit,rows,folds,old,teacher,local,slack,evidence)
        reports=[]
        for f,prev in enumerate(ref["folds"]):
            ti=[i for i,r in enumerate(records) if r["fold"]!=f]
            hi=[i for i,r in enumerate(records) if r["fold"]==f]
            predict,report=fit(env.torch,ti,rows,records,prior,prefix,driver)
            if (f,len(ti),len(hi))!=(prev["fold"],prev["train_rows"],prev["held_rows"]) or any(
                report["raw38"][k]!=prev["arms"]["prefix"][k] for k in ("bce","ranking","objective","updates")):
                raise ValueError("Exact original per-fold 38-D prefix fit")
            score_fold(predict,hi,rows,records,policy)
            reports.append(dict(fold=f,train_rows=len(ti),held_rows=len(hi),arms=report))
            print(json.dumps(dict(event="ALIGNED_FLIP_FOLD_COMPLETE",fold=f,held_rows=len(hi))),flush=True)
    finally:
        env.close()
    result=summarize(records,driver,ref)
    result.update(status="HISTORICAL_FIT_ALIGNED_FLIP_EVIDENCE_COMPLETE",protocol=args.protocol,
                  exact_raw38_reference_replay=True,fit_images=484,fit_matches=1082,fit_content_groups=479,
                  coverage=coverage,fold_reports=reports,AP=False,DEV=False,TEST=False)
    args.output.mkdir()
    with (args.output/"RESULT.json").open("x",encoding="utf8") as f:
        json.dump(result,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(result,sort_keys=True,allow_nan=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("teacher","selective-source","old-driver","broad"):
        p.add_argument("--"+k,type=Path,required=True)
    first,_=p.parse_known_args()
    driver=pinned(first.old_driver,DRIVER_SHA,"_closed_utility")
    broad=pinned(first.broad,driver.BROAD_SHA,"_closed_broad")
    base=pinned(first.selective_source,broad.SELECTIVE_SHA,"_flip_selective")
    teacher=base.teacher_module(first.teacher)
    teacher.common_args(p)
    for k in ("local-source","local-math","policy","slack","factors","prefix","evidence","reference"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--evidence-sha",required=True)
    run(p.parse_args(),driver,broad,base,teacher)
