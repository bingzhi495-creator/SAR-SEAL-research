"""Full fixed-candidate SSDD readout of a paired logit edit with abstention."""
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

TEACHER_SHA="67cd6687d1882889bc671fda2207ed88150627772b0f137ff97b29faec82bf3d"
LOCAL_SHA="0e763ae445c86482620faf163b28352f00acd2e151ffc06f800d43a63d129cc0"
MATH_SHA="eeb3c12c1abf1edc61a98d9716dbe8a20d8d3f28cd6c537393bd5c451e104733"
REFERENCE_SHA="d598aab9e2cbb5260fc0cd1a14f8e16e938516245d3354a55880f613f7708de2"
SEED=5201
K=4
STEPS=96
OFFSETS=np.array([(dx*r,dy*r) for r in (1,2,3,4)
                  for dx,dy in ((1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1))])
ARMS=("native","forced_all","score_equal","raw_equal","selective")


def teacher_module(path):
    raw=path.read_bytes()
    if sha256(raw).hexdigest()!=TEACHER_SHA:
        raise ValueError("Exact original detector and FIT environment")
    mod=ModuleType("_selective_teacher")
    mod.__file__=str(path)
    sys.modules[mod.__name__]=mod
    exec(compile(raw,str(path),"exec"),mod.__dict__)
    return mod


def parts(fit):
    groups=sorted({x["image_sha256"] for x in fit},
                  key=lambda x:sha256(("P154_ACTION_SPLIT_V1:"+x).encode()).digest())
    assignment={group:("point_fit" if i%5<3 else "gate_fit" if i%5==3 else "gate_cal")
                for i,group in enumerate(groups)}
    return {k:[x for x in fit if assignment[x["image_sha256"]]==k]
            for k in ("point_fit","gate_fit","gate_cal")}


def logit_inputs(plane,xy,distance,inside,size,score,rank):
    h,w=plane.shape
    x,y=xy.T
    z=plane[y,x]
    if not np.array_equal(z>0,inside):
        raise ValueError("Exact original raster logit polarity")
    base=np.column_stack((z,distance,inside,np.full(len(x),size),np.full(len(x),score),
                          np.full(len(x),rank/100)))
    px=np.clip(x[:,None]+OFFSETS[None,:,0],0,w-1)
    py=np.clip(y[:,None]+OFFSETS[None,:,1],0,h-1)
    return np.column_stack((base,plane[py,px])).astype(np.float32)


def point_training(env,teacher,prior,local,items):
    torch=env.torch
    projection=torch.as_tensor(np.linalg.qr(np.random.default_rng(SEED).normal(size=(256,32)))[0]
                               .astype(np.float32),device="cuda")
    rows=[]
    exact=False
    for i,item in enumerate(items,1):
        c=env.image(item)
        _,truth,matches,_=env.matches(item,c)
        for j,k in matches:
            q=c["query"][j]
            if not bool(c["ctx"]["active"][q]):
                continue
            mask=c["native"][j]
            xy,anchor,normal,distance,inside=local.boundary_samples(mask,128,SEED+100*item["dataset_index"]+q)
            if not len(xy):
                continue
            size=float(np.log1p(np.count_nonzero(mask)))
            plane=teacher.render(c["z"][q:q+1],c["hw"])[0].detach().cpu().numpy()
            x=logit_inputs(plane,xy,distance,inside,size,c["scores"][j],c["rank"][j])
            if not exact:
                f=c["basis"].float()
                basis=(projection.T@f.flatten(1)).reshape(32,*f.shape[-2:])
                reference,_=prior.point_inputs(c,j,q,xy,anchor,normal,distance,inside,basis,projection,env,teacher)
                env.s.need(np.array_equal(x,reference["logit"]),"Byte-exact P153 point input on real FIT pair")
                exact=True
            y=truth[k][xy[:,1],xy[:,0]]!=inside
            rows.append(dict(group=item["image_sha256"],inside=inside,x=x,y=y))
        if i%100==0 or i==len(items):
            print(json.dumps(dict(event="SELECTIVE_ACTION_PROGRESS",role="point_fit",images=i)),flush=True)
    env.s.need(exact and len(rows)>0,"Exact logit point control qualification")
    return rows


def train_model(torch,x,y,weight,hidden):
    x=torch.as_tensor(x,device="cuda",dtype=torch.float32)
    y=torch.as_tensor(y,device="cuda",dtype=torch.float32)
    weight=torch.as_tensor(weight,device="cuda",dtype=torch.float32)
    mu=(x*weight[:,None]).sum(0)
    scale=((x-mu).square()*weight[:,None]).sum(0).sqrt().clamp_min(1e-4)
    x=(x-mu)/scale
    torch.manual_seed(SEED)
    nn=torch.nn
    model=(nn.Sequential(nn.Linear(x.shape[1],32),nn.GELU(),nn.Linear(32,1)) if hidden
           else nn.Linear(x.shape[1],1)).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=.008,weight_decay=.02)
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        logits=model(x).flatten()
        loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction="none")*weight).sum()
        loss.backward()
        opt.step()
    if {int(v["step"].item()) for v in opt.state.values()}!={STEPS} or not torch.isfinite(loss):
        raise ValueError("Exact finite optimizer clocks")
    def score(rows):
        with torch.no_grad():
            z=model((torch.as_tensor(rows,device="cuda",dtype=torch.float32)-mu)/scale).flatten()
            if not torch.isfinite(z).all():
                raise ValueError("Finite point/action predictions")
            return z.cpu().numpy()
    return score,float(loss.detach())


def proposals(c,dataset_index,point_score,teacher,local,policy):
    """Produces actions from frozen outputs and learned parameters, with no label argument."""
    chosen=[]
    for j,q in enumerate(c["query"]):
        if not bool(c["ctx"]["active"][q]):
            continue
        xy,_,_,distance,inside=local.boundary_samples(c["native"][j],128,SEED+100*dataset_index+q)
        counts=(int((~inside).sum()),int(inside.sum()))
        if min(counts)<K:
            continue
        chosen.append((j,q,xy,distance,inside,counts))
    if not chosen:
        return []
    planes=[]
    queries=[row[1] for row in chosen]
    for start in range(0,len(queries),16):
        planes.extend(teacher.render(c["z"][queries[start:start+16]],c["hw"]).detach().cpu().numpy())
    profile=[]
    for (j,_,xy,distance,inside,_),plane in zip(chosen,planes,strict=True):
        area=float(np.log1p(np.count_nonzero(c["native"][j])))
        profile.append(logit_inputs(plane,xy,distance,inside,area,c["scores"][j],c["rank"][j]))
    cut=np.cumsum([len(v) for v in profile])[:-1]
    point=np.split(point_score(np.concatenate(profile)),cut)
    out=[]
    for (j,q,xy,_,inside,counts),x,z in zip(chosen,profile,point,strict=True):
        pixel=xy[:,1]*c["hw"][1]+xy[:,0]
        add,delete=policy.choose_points(pixel,inside,z,K)
        common=[x[0,4],x[0,5],x[0,3],np.log1p(counts[0]),np.log1p(counts[1])]
        signed=common+[z[add].mean(),z[delete].mean()]
        raw=common+[np.abs(x[add,0]).mean(),np.abs(x[delete,0]).mean()]
        out.append(dict(position=j,query=q,rank=c["rank"][j],score=c["scores"][j],
                        add=pixel[add],delete=pixel[delete],signed=signed,raw=raw))
    return out


def action_outcome(mask,edited,gt,geometry):
    if gt is None:
        return dict(matched=False,useful=False,strict_loss=False,margin=0)
    before=geometry.signed_counts(mask,gt)
    after=geometry.signed_counts(edited,gt)
    margin=before["error"]-after["error"]
    lost=bool(before["strict"] and not after["strict"])
    return dict(matched=True,useful=bool(margin>0 and not lost),strict_loss=lost,margin=margin,
                fp_delta=after["fp"]-before["fp"],fn_delta=after["fn"]-before["fn"],
                strict_gain=bool(after["strict"] and not before["strict"]))


def labeled_actions(env,teacher,local,policy,items,role,point):
    out=[]
    totals=Counter()
    for i,item in enumerate(items,1):
        c=env.image(item)
        actions=proposals(c,item["dataset_index"],point,teacher,local,policy)
        _,truth,matches,_=env.matches(item,c)
        by_position=dict(matches)
        totals.update(images=1,candidates=len(c["query"]),eligible=len(actions),original_matches=len(matches))
        for a in actions:
            j=a["position"]
            mask=c["native"][j]
            edited=policy.edited_mask(mask,a["add"],a["delete"])
            gt=truth[by_position[j]] if j in by_position else None
            label=action_outcome(mask,edited,gt,env.geometry)
            out.append(dict(group=item["image_sha256"],signed=a["signed"],raw=a["raw"],**label))
        if i%50==0 or i==len(items):
            print(json.dumps(dict(event="SELECTIVE_ACTION_PROGRESS",role=role,images=i)),flush=True)
    totals.update(matched_eligible=sum(r["matched"] for r in out),useful=sum(r["useful"] for r in out))
    return out,dict(totals)


def action_features(rows,key):
    return np.asarray([r[key] for r in rows],dtype=np.float32)


def fit_gates(torch,rows):
    groups=Counter(x["group"] for x in rows)
    weight=[1/(len(groups)*groups[x["group"]]) for x in rows]
    y=[r["useful"] for r in rows]
    return {k:train_model(torch,action_features(rows,k),y,weight,False) for k in ("signed","raw")}


def add_eval_image(env,policy,item,c,actions,gates,threshold,pred,diag):
    # Build decisions and candidate edits before original matching or GT-based diagnostics.
    n=len(actions)
    if n:
        signed=gates["signed"][0](action_features(actions,"signed"))
        raw=gates["raw"][0](action_features(actions,"raw"))
        chosen=np.flatnonzero(signed>=threshold) if threshold is not None else np.empty(0,dtype=np.int64)
        ranks=[a["rank"] for a in actions]
        sets=dict(native=[],forced_all=np.arange(n),selective=chosen,
                  score_equal=policy.select_equal_count([a["score"] for a in actions],ranks,len(chosen)),
                  raw_equal=policy.select_equal_count(raw,ranks,len(chosen)))
    else:
        sets={k:[] for k in ARMS}
    if len({len(sets[k]) for k in ("score_equal","raw_equal","selective")})!=1:
        raise ValueError("Exact per-image action budget match")
    positions={k:{actions[i]["position"] for i in indices} for k,indices in sets.items()}
    edited={a["position"]:policy.edited_mask(c["native"][a["position"]],a["add"],a["delete"]) for a in actions}
    native_rles=env.s.coerce_rles(c["native"],env.codec)
    changed_rles={j:env.codec.encode(np.asfortranarray(p.astype(np.uint8))) for j,p in edited.items()}
    iid=item["image_id"]
    for k in ARMS:
        pred[k].extend(dict(image_id=iid,category_id=1,score=s,native_rank=c["rank"][j],native_query=c["query"][j],segmentation=changed_rles[j]
                            if j in positions[k] else native_rles[j]) for j,s in enumerate(c["scores"]))
    _,truth,matches,_=env.matches(item,c)
    matched=dict(matches)
    native_strict=sum(env.geometry.signed_counts(c["native"][j],truth[k])["strict"] for j,k in matches)
    overlap=env.codec.iou(native_rles,env.s.coerce_rles(truth,env.codec),[0]*len(truth)) if len(native_rles) and len(truth) else None
    for k in ARMS:
        d=diag[k]
        selected=positions[k]
        d.update(images=1,candidates=len(native_rles),original_matches=len(matches),original_strict=native_strict,eligible=n,
                 edited_rows=len(selected),edited_pixels=2*K*len(selected),images_edited=int(bool(selected)))
        for j in selected:
            gt=truth[matched[j]] if j in matched else None
            o=action_outcome(c["native"][j],edited[j],gt,env.geometry)
            d.update(useful=o["useful"],matched_edits=o["matched"],unmatched_edits=not o["matched"],
                     gt_overlap_unmatched=int(not o["matched"] and overlap is not None and overlap[j].max()>=.50),
                     lost_strict=o["strict_loss"],new_strict=o.get("strict_gain",False),margin=o["margin"],
                     fp_delta=o.get("fp_delta",0),fn_delta=o.get("fn_delta",0))
    return [(iid,rank,q,score.hex()) for rank,q,score in zip(c["rank"],c["query"],c["scores"],strict=True)]


def run(args,teacher,prior,local,policy):
    start=time.perf_counter()
    raw=args.reference.read_bytes()
    if sha256(raw).hexdigest()!=REFERENCE_SHA:
        raise ValueError("Exact closed reference report")
    reference=json.loads(raw)
    env=teacher.Environment(args)
    roles=parts(env.fit)
    groups={k:{r["image_sha256"] for r in v} for k,v in dict(roles,internal_holdout=env.hold).items()}
    env.s.need(sum(len(x) for x in groups.values())==len(set.union(*groups.values()))==589,
               "Exact mutual content-group disjointness")
    timing={}
    ts=time.perf_counter()
    pt=point_training(env,teacher,prior,local,roles["point_fit"])
    point,point_loss=train_model(env.torch,np.concatenate([r["x"] for r in pt]),np.concatenate([r["y"] for r in pt]),
                                 prior.weights(pt),True)
    timing["point_fit"]=time.perf_counter()-ts
    ts=time.perf_counter()
    gtfit,fit_counts=labeled_actions(env,teacher,local,policy,roles["gate_fit"],"gate_fit",point)
    gates=fit_gates(env.torch,gtfit)
    gtcal,cal_counts=labeled_actions(env,teacher,local,policy,roles["gate_cal"],"gate_cal",point)
    fitted=gates["signed"][0](action_features(gtcal,"signed"))
    policy_cut=policy.risk_threshold(fitted,*([r[k] for r in gtcal] for k in ("useful","strict_loss","matched","margin","group")))
    timing["gate_fit_and_calibration"]=time.perf_counter()-ts
    print(json.dumps(dict(event="SELECTIVE_ACTION_THRESHOLD_LOCKED",risk=policy_cut)),flush=True)
    ts=time.perf_counter()
    pred={k:[] for k in ARMS}
    diag={k:Counter() for k in ARMS}
    identity=[]
    for i,item in enumerate(env.hold,1):
        c=env.image(item)
        actions=proposals(c,item["dataset_index"],point,teacher,local,policy)
        identity.extend(add_eval_image(env,policy,item,c,actions,gates,policy_cut["threshold"],pred,diag))
        if i%25==0 or i==len(env.hold):
            print(json.dumps(dict(event="SELECTIVE_ACTION_PROGRESS",role="internal_holdout",images=i)),flush=True)
    env.close()
    ids=[r["image_id"] for r in env.hold]
    idset=set(ids)
    gt=dict(info=env.gt["info"],images=[x for x in env.gt["images"] if x["id"] in idset],
            annotations=[x for x in env.gt["annotations"] if x["image_id"] in idset],categories=env.gt["categories"])
    exact=lambda v:[(r["image_id"],r["native_rank"],r["native_query"],r["score"].hex()) for r in v]
    env.s.need(len(identity)==9082 and all(exact(v)==identity for v in pred.values()),
               "Exact original candidate identities, ranks and class score bits")
    metrics={k:env.s.fit_stats(gt,[{x:r[x] for x in ("image_id","category_id","score","segmentation")} for r in v],ids)
             for k,v in pred.items()}
    env.s.need(metrics["native"]==reference["metrics"]["native"] and diag["native"]["original_strict"]==60,
               "Exact native AP and fixed strict historical replay")
    env.s.need(len({diag[k]["edited_rows"] for k in ("score_equal","raw_equal","selective")})==1,
               "Exact total selected-action coverage")
    timing["holdout_five_readouts_and_COCO"]=time.perf_counter()-ts
    delta={k:{m:metrics[k][m]-metrics["native"][m] for m in ("AP","AP75","AP90","APsmall","AR100")}
           for k in ARMS if k!="native"}
    main=delta["selective"]
    excess={k:metrics["selective"]["AP"]-metrics[k]["AP"] for k in ("score_equal","raw_equal")}
    gate=(policy_cut["threshold"] is not None and main["AP"]>=.005 and main["AP75"]>=0 and main["AP90"]>=.01
          and all(v>=.002 for v in excess.values()) and diag["selective"]["lost_strict"]<=3
          and diag["selective"]["new_strict"]>diag["selective"]["lost_strict"])
    result=dict(status="SSDD_FIT_SELECTIVE_LOGIT_ACTION_COMPLETE",protocol=args.protocol,gate=bool(gate),
                roles={k:dict(images=len(v),encoded_groups=len(groups[k])) for k,v in dict(roles,internal_holdout=env.hold).items()},
                point_fit_pairs=len(pt),point_fit_points=sum(len(x["y"]) for x in pt),point_fit_loss=point_loss,
                optimizer_updates=dict(point=96,signed_gate=96,raw_gate=96,detector=0),
                gate_fit=fit_counts,gate_cal=cal_counts,calibration=policy_cut,
                metrics=metrics,diagnostics=diag,delta_vs_native=delta,selective_minus_equal_budget_AP=excess,
                all_candidate_identities_sha256=sha256(json.dumps(identity,separators=(",",":")).encode()).hexdigest(),
                original_candidate_replay=True,exact_point_input_replay=True,truth_used_for_holdout_decisions=False,
                fixed_edited_points_per_side=K,timing=timing,elapsed_s=time.perf_counter()-start,
                cuda_peak_bytes=env.torch.cuda.max_memory_allocated(),AP=True,DEV=False,TEST=False)
    args.output.mkdir()
    with (args.output/"RESULT.json").open("x",encoding="utf-8") as f:
        json.dump(result,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(result,sort_keys=True,allow_nan=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher",type=Path,required=True)
    first,_=p.parse_known_args()
    teacher=teacher_module(first.teacher)
    teacher.common_args(p)
    for name in ("local-source","local-math","policy","reference"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--policy-sha",required=True)
    args=p.parse_args()
    prior=teacher.module(args.local_source,LOCAL_SHA,"_p153_signed_source")
    local=teacher.module(args.local_math,MATH_SHA,"_p153_boundary_math")
    policy=teacher.module(args.policy,args.policy_sha,"_p154_action_policy")
    run(args,teacher,prior,local,policy)

