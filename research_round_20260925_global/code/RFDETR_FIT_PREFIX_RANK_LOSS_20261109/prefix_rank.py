"""Train and compare exact-prefix point ranks on frozen SSDD candidates."""
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

SELECTIVE_SHA="4840f5511d4225ef1701e88f0b96da2669149ed529b7c071b4dd908d882b074b"
POLICY_SHA="44c79d7459a59d9dd601d5192c9743ecce815725dd24e1456ba14c70573e0031"
REFERENCE_SHA="63e1f7316247130bddd044aa582ace074b61b0b35be3c7f323300deebcc1e59f"
ARMS=("native","uncertainty","bce","pairwise","prefix")
STEPS=96


def need(ok,message):
    if not ok:
        raise ValueError(message)


def module(path,digest,name):
    raw=path.read_bytes()
    need(sha256(raw).hexdigest()==digest,"Exact published source digest")
    m=ModuleType(name)
    m.__file__=str(path)
    sys.modules[m.__name__]=m
    exec(compile(raw,str(path),"exec"),m.__dict__)
    return m


def list_index(rows,w):
    indices=[]
    pos=0
    for r in rows:
        side=r["inside"]
        for flag in (False,True):
            selected=np.flatnonzero(side==flag)
            if len(selected):
                indices.append(pos+selected)
        pos+=len(side)
    count=np.array([len(k) for k in indices])
    gather=np.zeros((len(indices),int(count.max())),dtype=np.int64)
    valid=np.zeros(gather.shape,dtype=bool)
    for i,k in enumerate(indices):
        gather[i,:len(k)]=k
        valid[i,:len(k)]=True
    weight=(w[gather]*valid).sum(1)
    need(int(valid.sum())==pos==len(w) and gather.shape[1]<=128 and abs(float(weight.sum())-1)<1e-5,
         "Exact original side weights and training points")
    return gather,valid,weight


def fit(torch,training,prior,losses):
    x=torch.as_tensor(np.concatenate([r["x"] for r in training]),device="cuda",dtype=torch.float32)
    y=torch.as_tensor(np.concatenate([r["y"] for r in training]),device="cuda",dtype=torch.float32)
    w_np=prior.weights(training)
    w=torch.as_tensor(w_np,device="cuda",dtype=torch.float32)
    mu=(x*w[:,None]).sum(0)
    scale=((x-mu).square()*w[:,None]).sum(0).sqrt().clamp_min(1e-4)
    x=(x-mu)/scale
    idx,v,wside=list_index(training,w_np)
    idx=torch.as_tensor(idx,device="cuda")
    v=torch.as_tensor(v,device="cuda")
    wside=torch.as_tensor(wside,device="cuda",dtype=torch.float32)
    labels=y[idx].bool()
    side_counts=(labels&v).sum(1)
    side_lengths=v.sum(1)
    k=side_lengths.clamp_max(4)
    lists=dict(total=len(v),short=int((side_lengths<4).sum()),zero_positive=int((side_counts==0).sum()),
               all_positive=int((side_counts==side_lengths).sum()),full_selection=int((side_lengths==k).sum()),
               partial_positive=int(((side_counts>0)&(side_counts<k)).sum()))
    init=None
    models={}
    reports={}
    nn=torch.nn
    for k in ARMS[2:]:
        torch.manual_seed(5201)
        m=nn.Sequential(nn.Linear(38,32),nn.GELU(),nn.Linear(32,1)).cuda()
        if init is None:
            init={n:t.detach().clone() for n,t in m.state_dict().items()}
        need(sum(t.numel() for t in m.parameters())==1281 and
             all(torch.equal(t,init[n]) for n,t in m.state_dict().items()),"Paired initial point head and capacity")
        opt=torch.optim.AdamW(m.parameters(),lr=.008,weight_decay=.02)
        torch.cuda.synchronize()
        start=time.perf_counter()
        for _ in range(STEPS):
            opt.zero_grad(set_to_none=True)
            z=m(x).flatten()
            bce=(nn.functional.binary_cross_entropy_with_logits(z,y,reduction="none")*w).sum()
            ranking=bce.new_zeros(())
            if k=="pairwise":
                ranking=(losses.uniform_pairwise(z[idx],labels,v)*wside).sum()
            elif k=="prefix":
                ranking=(losses.precision_ramp(z[idx],labels,v)*wside).sum()
            obj=bce+.5*ranking
            obj.backward()
            opt.step()
        need({int(t["step"].item()) for t in opt.state.values()}=={STEPS} and torch.isfinite(obj),
             "Exact completed optimizer clocks")
        torch.cuda.synchronize()
        reports[k]=dict(seconds=time.perf_counter()-start,updates=STEPS,bce=float(bce.detach()),
                        ranking=float(ranking.detach()),objective=float(obj.detach()))
        models[k]=m
    need(reports["bce"]["bce"]==0.3621605634689331,"Exact existing BCE fit endpoint")
    def score(x):
        with torch.no_grad():
            a=(torch.as_tensor(x,device="cuda",dtype=torch.float32)-mu)/scale
            out={k:m(a).flatten().cpu().numpy() for k,m in models.items()}
        out["uncertainty"]=-np.abs(x[:,0])
        need(all(np.isfinite(z).all() for z in out.values()),"Finite point ranks")
        return out
    return score,reports,lists


def fields(c,index,teacher,base,local):
    rows=[]
    for j,q in enumerate(c["query"]):
        if not bool(c["ctx"]["active"][q]):
            continue
        xy,_,_,d,inside=local.boundary_samples(c["native"][j],128,5201+100*index+q)
        if int(inside.sum())>=4 and int((~inside).sum())>=4:
            rows.append((j,q,xy,d,inside))
    if not rows:
        return [],None
    z=[]
    qs=[r[1] for r in rows]
    for k in range(0,len(qs),16):
        z.extend(teacher.render(c["z"][qs[k:k+16]],c["hw"]).detach().cpu().numpy())
    features=[]
    for (j,_,xy,d,side),plane in zip(rows,z,strict=True):
        area=float(np.log1p(np.count_nonzero(c["native"][j])))
        features.append(base.logit_inputs(plane,xy,d,side,area,c["scores"][j],c["rank"][j]))
    return rows,np.concatenate(features)


def choices(c,rows,x,score,policy):
    masks={k:{} for k in ARMS}
    selected={k:{} for k in ARMS}
    if rows:
        cuts=np.cumsum([len(r[2]) for r in rows])[:-1]
        for arm,scores in score(x).items():
            for (j,_,xy,_,inside),z in zip(rows,np.split(scores,cuts),strict=True):
                px=xy[:,1]*c["hw"][1]+xy[:,0]
                a,d=policy.choose_points(px,inside,z,4)
                pair=(px[a],px[d])
                selected[arm][j]=pair
                masks[arm][j]=policy.edited_mask(c["native"][j],*pair)
    return masks,selected


def score_role(env,items,role,reference,teacher,base,local,policy,score):
    pred={k:[] for k in ARMS}
    geom={k:Counter() for k in ARMS}
    groups=defaultdict(list)
    identity=[]
    ids=[]
    start=time.perf_counter()
    for i,item in enumerate(items,1):
        c=env.image(item)
        rows,x=fields(c,item["dataset_index"],teacher,base,local)
        masks,selected=choices(c,rows,x,score,policy)
        native=c["native"]
        iid=item["image_id"]
        ids.append(iid)
        identity.extend((iid,r,q,s.hex()) for r,q,s in zip(c["rank"],c["query"],c["scores"],strict=True))
        original=env.s.coerce_rles(native,env.codec)
        for arm,value in masks.items():
            rle=list(original)
            for j,p in value.items():
                rle[j]=env.codec.encode(np.asfortranarray(p.astype(np.uint8)))
            pred[arm].extend(dict(image_id=iid,category_id=1,score=s,segmentation=z)
                             for s,z in zip(c["scores"],rle,strict=True))
        # Choices and complete predictions precede fixed-match geometry.
        _,truth,pairs,_=env.matches(item,c)
        matched=dict(pairs)
        prior={j:env.geometry.signed_counts(native[j],truth[t]) for j,t in pairs}
        attained={}
        for arm,changes in masks.items():
            g=geom[arm]
            n=0
            g.update(edited_rows=len(changes),original_matches=len(matched),original_strict=sum(v["strict"] for v in prior.values()),
                     matched_edits=sum(j in matched for j in changes),unmatched_edits=sum(j not in matched for j in changes))
            for j,p in changes.items():
                if j not in matched:
                    continue
                gt=truth[matched[j]]
                after=env.geometry.signed_counts(p,gt)
                before=prior[j]
                aa,dd=selected[arm][j]
                ac=int(gt.ravel()[aa].sum())
                dc=int((~gt.ravel()[dd]).sum())
                n+=ac+dc
                g.update(correct_add=ac,correct_delete=dc,fp_delta=after["fp"]-before["fp"],fn_delta=after["fn"]-before["fn"],
                         strict_loss=int(before["strict"] and not after["strict"]),
                         strict_gain=int(after["strict"] and not before["strict"]))
            attained[arm]=n
        if any(j in matched for j in masks["bce"]):
            groups[item["image_sha256"]].append((attained["prefix"]-attained["bce"],sum(j in matched for j in masks["bce"])))
        if i%50==0 or i==len(items):
            print(json.dumps(dict(event="PREFIX_RANK_PROGRESS",role=role,images=i)),flush=True)
    digest=sha256(json.dumps(identity,separators=(",",":")).encode()).hexdigest()
    keys=[(i,1,s) for i,_,_,s in identity]
    need(digest==reference["identity_sha256"] and len(keys)==reference["predictions"] and
         all([(r["image_id"],r["category_id"],r["score"].hex()) for r in v]==keys for v in pred.values()) and
         all(geom[k]["edited_rows"]==reference["eligible"] for k in ARMS[1:]),
         "Exact original candidate identities and same eight edits on each eligible row")
    group_means=np.array([sum(x for x,_ in v)/sum(n for _,n in v) for _,v in sorted(groups.items())])
    sample=np.random.default_rng(5201).integers(0,len(group_means),size=(2048,len(group_means)))
    ci=np.quantile(group_means[sample].mean(1),[.025,.975]).tolist()
    idset=set(ids)
    gt=dict(info=env.gt["info"],images=[x for x in env.gt["images"] if x["id"] in idset],
            annotations=[x for x in env.gt["annotations"] if x["image_id"] in idset],categories=env.gt["categories"])
    m={k:env.s.fit_stats(gt,v,ids) for k,v in pred.items()}
    need(m["native"]==reference["metrics"]["native"] and m["bce"]==reference["metrics"]["balanced"],
         "Exact closed native and BCE full AP replay")
    return dict(metrics=m,geometry=geom,images=len(ids),edited_per_non_native=reference["eligible"]*8,
                identity_sha256=digest,grouped_correct_diff=dict(groups=len(group_means),
                mean=float(group_means.mean()),percentile95=ci),elapsed_s=time.perf_counter()-start)


def run(args,teacher,base,prior,local,policy,losses):
    start=time.perf_counter()
    raw=args.reference.read_bytes()
    need(sha256(raw).hexdigest()==REFERENCE_SHA,"Exact closed point-opportunity endpoints")
    ref=json.loads(raw)
    env=teacher.Environment(args)
    fit_items=base.parts(env.fit)["point_fit"]
    allfit=json.loads(env.cap.exact_bytes(args.fit_index,348469,env.cap.FIT_INDEX_SHA))["items"]
    cal=[x for x in allfit if x["partition"]=="calibration"]
    groups=[{x["image_sha256"] for x in role} for role in (fit_items,cal,env.hold)]
    need([len(role) for role in (fit_items,cal,env.hold)]==[290,114,111] and len(groups[0])==288
         and len(set.union(*groups))==sum(map(len,groups)),"Exact image roles with disjoint encoded groups")
    training=base.point_training(env,teacher,prior,local,fit_items)
    need(len(training)==623 and sum(len(r["y"]) for r in training)==142170,"Exact common fitting labels")
    score,reports,lists=fit(env.torch,training,prior,losses)
    a=score_role(env,cal,"historical_calibration114",ref["primary114"],teacher,base,local,policy,score)
    b=score_role(env,env.hold,"historical_holdout111",ref["secondary111"],teacher,base,local,policy,score)
    env.close()
    m=a["metrics"]
    g=a["geometry"]
    gain=m["prefix"]["AP"]-max(m[k]["AP"] for k in ARMS if k!="prefix")
    gate=(gain>=.003 and m["prefix"]["AP75"]>=m["bce"]["AP75"] and m["prefix"]["AP90"]-m["bce"]["AP90"]>=.01
          and g["prefix"]["strict_loss"]<=g["bce"]["strict_loss"] and g["prefix"]["strict_loss"]*20<=g["prefix"]["original_strict"]
          and g["prefix"]["correct_add"]>g["bce"]["correct_add"] and g["prefix"]["correct_delete"]>g["bce"]["correct_delete"]
          and a["grouped_correct_diff"]["percentile95"][0]>0 and b["metrics"]["prefix"]["AP"]>=b["metrics"]["bce"]["AP"])
    out=dict(status="SSDD_FIT_PREFIX_RANK_LOSS_COMPLETE",protocol=args.protocol,original_protocol=ref["protocol"],
             primary114=a,secondary111=b,training=reports,side_lists=lists,fit_matches=len(training),
             fit_points=142170,gate=bool(gate),primary_gain_fraction=gain,detector_updates=0,
             truth_used_for_holdout_decisions=False,AP=True,DEV=False,TEST=False,
             elapsed_s=time.perf_counter()-start,cuda_peak_bytes=env.torch.cuda.max_memory_allocated())
    args.output.mkdir()
    with (args.output/"RESULT.json").open("x",encoding="utf-8") as f:
        json.dump(out,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(out,sort_keys=True,allow_nan=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("selective-source","teacher"):
        p.add_argument("--"+n,type=Path,required=True)
    first,_=p.parse_known_args()
    base=module(first.selective_source,SELECTIVE_SHA,"_prefix_selective")
    teacher=base.teacher_module(first.teacher)
    teacher.common_args(p)
    for n in ("local-source","local-math","policy","losses","reference"):
        p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--losses-sha",required=True)
    args=p.parse_args()
    prior=module(args.local_source,base.LOCAL_SHA,"_prefix_prior")
    local=module(args.local_math,base.MATH_SHA,"_prefix_local")
    policy=module(args.policy,POLICY_SHA,"_prefix_policy")
    losses=module(args.losses,args.losses_sha,"_prefix_losses")
    run(args,teacher,base,prior,local,policy,losses)

