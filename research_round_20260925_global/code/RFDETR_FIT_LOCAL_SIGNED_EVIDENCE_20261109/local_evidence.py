"""Equal-budget direction evidence on native SSDD mask boundary supports."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time
from types import ModuleType

import numpy as np

TEACHER_SHA="67cd6687d1882889bc671fda2207ed88150627772b0f137ff97b29faec82bf3d"
SEED=5201
ARMS=("logit","point","global","tangent","normal")
BUDGET=4
CAP=128
RADIUS=2.0
STEPS=96


def teacher_module(path):
    data=path.read_bytes()
    if sha256(data).hexdigest()!=TEACHER_SHA:
        raise ValueError("Exact original frozen detector environment")
    mod=ModuleType("_signed_local_teacher")
    mod.__file__=str(path)
    sys.modules[mod.__name__]=mod
    exec(compile(data,str(path),"exec"),mod.__dict__)
    return mod


def point_inputs(c, position, q, xy, boundary, normal, distance, inside, basis, projection, env, teacher):
    torch=env.torch
    h,w=c["hw"]
    tangent=np.column_stack((-normal[:,1],normal[:,0]))
    coord=np.stack((xy,boundary+RADIUS*normal,boundary-RADIUS*normal,
                       boundary+RADIUS*tangent,boundary-RADIUS*tangent))
    norm=(2*(coord+.5)/np.array((w,h))-1).astype(np.float32)
    with torch.no_grad():
        grid=torch.as_tensor(norm,device=basis.device)
        feats=torch.nn.functional.grid_sample(basis[None],grid[None],mode="bilinear",
                    align_corners=False,padding_mode="border")[0].permute(1,2,0).cpu().numpy()
        global_feature=(projection.T@c["ctx"]["contrast"][q].float()).cpu().numpy()
        plane=teacher.render(c["z"][q:q+1],(h,w))[0].cpu().numpy()
    x,y=xy.T
    z=plane[y,x]
    env.s.need(np.array_equal(z>0,inside),"Point logit sign equals native original raster")
    size=float(np.log1p(np.count_nonzero(c["native"][position])))
    base=np.column_stack((z,distance,inside,np.full(len(x),size),np.full(len(x),c["scores"][position]),
                          np.full(len(x),c["rank"][position]/100)))
    offsets=np.array([(dx*r,dy*r) for r in (1,2,3,4)
                      for dx,dy in ((1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1))])
    px=np.clip(x[:,None]+offsets[None,:,0],0,w-1)
    py=np.clip(y[:,None]+offsets[None,:,1],0,h-1)
    center=feats[0]
    extra={
        "logit":plane[py,px],
        "point":center,
        "global":np.column_stack((center[:,:16],np.broadcast_to(global_feature[:16],(len(x),16)))),
        "tangent":np.column_stack((center[:,:16],feats[3,:,:16]-feats[4,:,:16])),
        "normal":np.column_stack((center[:,:16],feats[1,:,:16]-feats[2,:,:16])),
    }
    return {k:np.column_stack((base,value)).astype(np.float32) for k,value in extra.items()},z


def collect(env,teacher,local,items,role,projection):
    torch=env.torch
    rows=[]
    counts=Counter(images=0)
    for i,item in enumerate(items,1):
        c=env.image(item)
        _,truth,matches,_=env.matches(item,c)
        with torch.no_grad():
            f=c["basis"].float()
            basis=(projection.T@f.flatten(1)).reshape(32,*f.shape[-2:])
        counts.update(images=1,candidates=len(c["query"]),matched=len(matches),unmatched=len(c["query"])-len(matches))
        for j,k in matches:
            q=c["query"][j]
            if not bool(c["ctx"]["active"][q]):
                counts.update(inactive=1)
                continue
            mask=c["native"][j]
            xy,boundary,normal,distance,inside=local.boundary_samples(mask,CAP,SEED+100*item["dataset_index"]+q)
            if not len(xy):
                counts.update(empty_support=1)
                continue
            inputs,logit=point_inputs(c,j,q,xy,boundary,normal,distance,inside,basis,projection,env,teacher)
            x,y=xy.T
            target=truth[k][y,x]
            stock=env.geometry.signed_counts(mask,truth[k])
            counts.update(used_matches=1,points=len(xy),strict_before=stock["strict"],
                          small=int(stock["area"]<1024),available_add=int((~inside).sum()),
                          available_delete=int(inside.sum()),true_add=int((target&~inside).sum()),
                          true_delete=int((~target&inside).sum()))
            rows.append(dict(group=item["image_sha256"],inputs=inputs,logit=logit,
                             pixels=y*c["hw"][1]+x,inside=inside,truth=target,stock=stock))
        if i%100==0 or i==len(items):
            print(json.dumps(dict(event="SIGNED_LOCAL_COLLECTION",role=role,images=i)),flush=True)
    return rows,dict(counts)


def weights(rows):
    groups=Counter(r["group"] for r in rows)
    out=[]
    for row in rows:
        sign=row["inside"]
        sides=np.unique(sign)
        weight=np.empty(len(sign),dtype=np.float32)
        for side in sides:
            idx=sign==side
            weight[idx]=1/(len(groups)*groups[row["group"]]*len(sides)*int(idx.sum()))
        out.append(weight)
    return np.concatenate(out)


def train_and_score(torch,training,evaluation):
    targets=np.concatenate([x["truth"]!=x["inside"] for x in training]).astype(np.float32)
    y=torch.as_tensor(targets,device="cuda")
    weight=torch.as_tensor(weights(training),device="cuda")
    scored={}
    updates={}
    endpoint={}
    for arm in ARMS:
        x=torch.as_tensor(np.concatenate([r["inputs"][arm] for r in training]),device="cuda")
        mean=(x*weight[:,None]).sum(0)
        spread=((x-mean).square()*weight[:,None]).sum(0).sqrt().clamp_min(1e-4)
        x=(x-mean)/spread
        if x.shape[1]!=38 or not torch.isfinite(x).all():
            raise ValueError("Common finite predictor width")
        torch.manual_seed(SEED)
        m=torch.nn.Sequential(torch.nn.Linear(38,32),torch.nn.GELU(),torch.nn.Linear(32,1)).cuda()
        opt=torch.optim.AdamW(m.parameters(),lr=.008,weight_decay=.02)
        for _ in range(STEPS):
            opt.zero_grad(set_to_none=True)
            logits=m(x).flatten()
            loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction="none")*weight).sum()
            loss.backward()
            opt.step()
        counters={int(state["step"].item()) for state in opt.state.values()}
        if counters!={STEPS} or not torch.isfinite(loss):
            raise ValueError("Actual equal finite optimizer clocks")
        updates[arm]=STEPS
        endpoint[arm]=float(loss.detach())
        with torch.no_grad():
            ev=torch.as_tensor(np.concatenate([r["inputs"][arm] for r in evaluation]),device="cuda")
            logits=m((ev-mean)/spread).flatten()
            if not torch.isfinite(logits).all():
                raise ValueError("Finite held feature scores")
            scored[arm]=np.split(logits.cpu().numpy(),np.cumsum([len(r["truth"]) for r in evaluation])[:-1])
    return scored,updates,endpoint


def assess(local,evaluation,scores,budget):
    summaries={}
    per_group={}
    for arm,pred in scores.items():
        total=Counter()
        groups=defaultdict(list)
        for r,z in zip(evaluation,pred,strict=True):
            x=local.signed_prefix(r["pixels"],r["inside"],r["truth"],z,budget)
            stock=r["stock"]
            fp=stock["fp"]+x["fp_delta"]
            fn=stock["fn"]+x["fn_delta"]
            if min(fp,fn)<0:
                raise ValueError("Exact signed edits cannot yield negative error")
            after=stock["area"]>0 and 10*fn+9*fp<=stock["area"]
            total.update({k:v for k,v in x.items() if k!="selected"})
            total.update(before_strict=stock["strict"],after_strict=int(after),
                         new_strict=int(after and not stock["strict"]),lost_strict=int(stock["strict"] and not after),
                         both_error_sides_reduced=int(x["fp_delta"]<0 and x["fn_delta"]<0))
            if stock["strict"]:
                total["strict_margin_change"]+=x["add_margin"]+x["delete_margin"]
            groups[r["group"]].append((x["add_margin"],x["delete_margin"]))
        keys=sorted(groups)
        per_group[arm]=np.array([np.mean(groups[k],axis=0) for k in keys])
        summaries[arm]=dict(total,group_mean_add=float(per_group[arm][:,0].mean()),
                            group_mean_delete=float(per_group[arm][:,1].mean()))
    lengths={(x["add_count"],x["delete_count"]) for x in summaries.values()}
    if len(lengths)!=1:
        raise ValueError("Actual per-side equal edit budget")
    return summaries,per_group


def paired_controls(values):
    direction=values["normal"].sum(1)
    boot=np.random.default_rng(SEED).integers(0,len(direction),size=(2048,len(direction)))
    pairs={}
    for k,v in values.items():
        if k in ("normal","perfect_support"):
            continue
        d=direction-v.sum(1)
        pairs[k]=dict(mean=float(d.mean()),percentile95=np.quantile(d[boot].mean(1),[.025,.975]).tolist())
    return pairs


def run(args,teacher,local):
    start=time.perf_counter()
    env=teacher.Environment(args)
    index=json.loads(env.cap.exact_bytes(args.fit_index,348469,env.cap.FIT_INDEX_SHA))["items"]
    eval_items=[x for x in index if x["partition"]=="calibration"]
    train_groups={x["image_sha256"] for x in env.fit}
    eval_groups={x["image_sha256"] for x in eval_items}
    hold_groups={x["image_sha256"] for x in env.hold}
    env.s.need((len(eval_items),len(eval_groups))==(114,109) and
               not (train_groups&eval_groups or hold_groups&eval_groups),"Exact role/content separation")
    rng=np.random.default_rng(SEED)
    orthogonal=np.linalg.qr(rng.normal(size=(256,32)))[0].astype(np.float32)
    projection=env.torch.as_tensor(orthogonal,device="cuda")
    training,train_counts=collect(env,teacher,local,env.fit,"fit",projection)
    evaluation,eval_counts=collect(env,teacher,local,eval_items,"historical_calibration",projection)
    env.close()
    scores,updates,loss=train_and_score(env.torch,training,evaluation)
    scores["uncertainty"]=[-np.abs(r["logit"]) for r in evaluation]
    scores["perfect_support"]=[(r["truth"]!=r["inside"]).astype(np.float32) for r in evaluation]
    summaries,values=assess(local,evaluation,scores,BUDGET)
    zero=np.zeros_like(values["normal"])
    paired=paired_controls(dict(values,no_edit=zero))
    sensitivity,_=assess(local,evaluation,scores,1)
    n=summaries["normal"]
    eval_used_groups=len({r["group"] for r in evaluation})
    pass_gate=(eval_used_groups>=60 and n["group_mean_add"]>0 and n["group_mean_delete"]>0
               and paired["point"]["mean"]>=1.0 and all(x["percentile95"][0]>0 for x in paired.values())
               and n["lost_strict"]<=summaries["point"]["lost_strict"])
    result=dict(status="SSDD_FIT_LOCAL_SIGNED_EVIDENCE_COMPLETE",protocol=args.protocol,
                training=dict(train_counts,encoded_groups=len(train_groups),groups_with_rows=len({r["group"] for r in training})),
                evaluation=dict(eval_counts,encoded_groups=len(eval_groups),groups_with_rows=eval_used_groups),
                edits_per_side=BUDGET,primary=summaries,normal_minus_controls=paired,sensitivity_k1=sensitivity,
                probe_optimizer_updates=updates,final_train_bce=loss,gate=bool(pass_gate),
                width=38,parameters_each=1281,optimizer_updates_detector=0,AP=False,DEV=False,TEST=False,
                historical_calibration_role=True,elapsed_s=time.perf_counter()-start,
                cuda_peak_bytes=env.torch.cuda.max_memory_allocated())
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
    p.add_argument("--local-math",type=Path,required=True)
    p.add_argument("--local-math-sha",required=True)
    args=p.parse_args()
    local=teacher.module(args.local_math,args.local_math_sha,"_signed_local_math")
    run(args,teacher,local)

