"""Cross-fitted finite gradient directions on the closed SSDD teacher student."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time
from types import ModuleType

import numpy as np

FIDELITY_SHA="58935b02379669d52f431dd291a6255cc60c9bc45ce43278affc16928b55db56"
FIDELITY_RESULT_SHA="5e396f5f157c2be94c2b20f06d50cbf7068d647571323ee6d29dcdd36fd32c6d"
MATH_SHA="39e7cffee486aced350414a96b1da1d967618efa0e48986af022bbf6df3b3bc0"
SEED=5201
SALTED="SSDD-FIT-TEACHER-DIRECTION-20261109-v1"
REL_STEP=.002
STRATA=("accepted","already_strict","rejected_miss")
ARMS=("accepted","projected","scalarized","mixed")


def module(path,digest,name):
    raw=path.read_bytes()
    if sha256(raw).hexdigest()!=digest:
        raise ValueError("Exact pinned source")
    mod=ModuleType(name)
    mod.__file__=str(path)
    sys.modules[name]=mod
    exec(compile(raw,str(path),"exec"),mod.__dict__)
    return mod


def masks_for(t,rows,device):
    keep=t.tensor([r["keep"] for r in rows],dtype=t.bool,device=device)
    strict=t.tensor([r["strict_before"] for r in rows],dtype=t.bool,device=device)
    return dict(accepted=keep,already_strict=strict,rejected_miss=~(keep|strict))


def per_row_loss(logits,target,weights):
    from torch.nn.functional import binary_cross_entropy_with_logits
    return (binary_cross_entropy_with_logits(logits,target,reduction="none")*weights).sum(1)


def flatten_gradient(t,loss,params,retain):
    grad=t.autograd.grad(loss,params,retain_graph=retain)
    return t.cat([x.flatten() for x in grad]).detach().double().cpu().numpy()


def compare_ci(values):
    a=np.asarray(values,dtype=np.float64)
    boot=np.random.default_rng(SEED).integers(0,len(a),size=(2048,len(a)))
    return dict(groups=len(a),mean=float(a.mean()),percentile95=np.quantile(a[boot].mean(1),[.025,.975]).tolist())


def build_step(t,anchor,delta):
    head=deepcopy(anchor)
    offset=0
    realized=[]
    with t.no_grad():
        for old,new in zip(anchor.parameters(),head.parameters(),strict=True):
            size=new.numel()
            change=t.from_numpy(delta[offset:offset+size]).reshape_as(new).to(device=new.device,dtype=new.dtype)
            new.add_(change)
            realized.append((new-old).flatten().double().cpu().numpy())
            offset+=size
    if offset!=len(delta):
        raise ValueError("Exact step layout")
    return head.requires_grad_(False).eval(),float(np.linalg.norm(np.concatenate(realized)))


def run(args,fi,student,source):
    began=time.perf_counter()
    math=module(Path(__file__).with_name("direction_math.py"),MATH_SHA,"_direction_math")
    env=source.Environment(args)
    t=env.torch
    need=env.s.need
    result_raw=args.fidelity_result.read_bytes()
    teacher_raw=args.teacher_result.read_bytes()
    student_raw=args.student_result.read_bytes()
    need(sha256(result_raw).hexdigest()==FIDELITY_RESULT_SHA
         and sha256(teacher_raw).hexdigest()==fi.TEACHER_RESULT_SHA
         and sha256(student_raw).hexdigest()==fi.STUDENT_RESULT_SHA,"Exact closed previous evidence")
    fr,tr,sr=map(json.loads,(result_raw,teacher_raw,student_raw))
    for path,digest in ((args.students,sr["student_archive_sha256"]),(args.teachers,tr["teacher_archive_sha256"])):
        with path.open("rb") as f:
            need(env.cap.file_digest(f)==digest,"Exact private checkpoint bytes")
    archive=t.load(args.students,map_location="cpu",weights_only=True)
    stored=t.load(args.teachers,map_location="cpu",weights_only=True)
    need(archive["protocol"]==sr["protocol"] and archive["steps"]==sr["updates"]
         and stored["protocol"]==tr["protocol"] and len(stored["records"])==len(env.fit)==484,
         "Exact closed student and teacher")
    head=env.head.MaskCoefficientHead()
    head.load_state_dict(archive["students"]["teacher"],strict=True)
    head=head.cuda().eval()
    params=tuple(head.parameters())
    param_l2=float(t.linalg.vector_norm(t.cat([x.detach().flatten().double() for x in params])))
    budget=REL_STEP*param_l2
    keys=sorted({x["image_sha256"] for x in env.fit},key=lambda k:(sha256(f"{SALTED}:{k}".encode()).digest(),k))
    fold={key:i%2 for i,key in enumerate(keys)}
    need(len(fold)==479,"Exact encoded grouping for both diagnostic halves")
    grad=defaultdict(lambda:defaultdict(list))
    grad_rows=Counter()
    for i,(row,packet) in enumerate(zip(env.fit,stored["records"],strict=True),1):
        c=env.image(row)
        item=student.locked_supervision(env,source,row,c,packet)
        if item is not None:
            qi,grid,weights,targets=item
            f,q,z,ctx=(c[k] for k in ("basis","q","z","ctx"))
            flags=masks_for(t,packet["rows"],z.device)
            edit=env.head.bounded_readout(f,q,z,ctx,head(ctx["contrast"]))
            losses=per_row_loss(student.sampled(edit,qi,grid),targets[2],weights)
            group=grad[row["image_sha256"]]
            for k in STRATA[:2]:
                if bool(flags[k].any()):
                    group[k].append(flatten_gradient(t,losses[flags[k]].mean(),params,True))
                    grad_rows[k]+=int(flags[k].sum())
            mixed=losses.mean()+.05*(edit-z).square().mean()
            group["mixed"].append(flatten_gradient(t,mixed,params,False))
            grad_rows["mixed"]+=len(packet["rows"])
        if i%100==0 or i==len(env.fit):
            print(json.dumps(dict(event="DIRECTION_GRADIENT_PROGRESS",images=i)),flush=True)
    need(grad_rows==dict(accepted=752,already_strict=277,mixed=1082),"All original support strata")
    trials={}
    plans=[]
    for f in range(2):
        group_keys=[k for k in keys if fold[k]==f]
        vectors={}
        group_counts={}
        for k in ("accepted","already_strict","mixed"):
            arrays=[np.mean(grad[key][k],axis=0) for key in group_keys if grad[key][k]]
            vectors[k]=np.mean(arrays,axis=0)
            group_counts[k]=len(arrays)
        steps,info=math.make_directions(vectors["accepted"],vectors["already_strict"],vectors["mixed"],budget)
        info["derivation_group_counts"]=group_counts
        info["derivation_fold"]=f
        info["evaluation_fold"]=1-f
        realized={}
        for k,v in steps.items():
            model,n=build_step(t,head,v)
            need(abs(n-(0. if info["degenerate"][k] else budget))<=max(1e-7,budget*1e-4),
                 "Equal realized finite parameter lengths")
            trials[(1-f,k)]=model
            realized[k]=n
        info["realized_step_lengths"]=realized
        plans.append(info)
    grad.clear()

    grouped=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    original=defaultdict(lambda:defaultdict(list))
    rms=defaultdict(list)
    geometry=defaultdict(Counter)
    for i,(row,packet) in enumerate(zip(env.fit,stored["records"],strict=True),1):
        c=env.image(row)
        item=student.locked_supervision(env,source,row,c,packet)
        if item is not None:
            qi,grid,weights,targets=item
            f,q,z,ctx,hw=(c[k] for k in ("basis","q","z","ctx","hw"))
            flags=masks_for(t,packet["rows"],z.device)
            key=row["image_sha256"]
            anchors=[r for r in packet["rows"] if r["strict_before"]]
            anns,truth,_,_=env.matches(row,c) if anchors else (None,None,None,None)
            gt_by_ann={a["id"]:truth[j] for j,a in enumerate(anns)} if anchors else {}
            with t.no_grad():
                base=env.head.bounded_readout(f,q,z,ctx,head(ctx["contrast"]))
                base_kl=fi.regret(student.sampled(base,qi,grid),targets[2],weights)
                anchor_queries=[a["query"] for a in anchors]
                baseline_masks=source.render(base[anchor_queries],hw).gt(0).cpu().numpy() if anchors else None
                for j,a in enumerate(anchors):
                    strict=env.geometry.signed_counts(baseline_masks[j],gt_by_ann[a["annotation_id"]])["strict"]
                    geometry["baseline"].update(retained=strict,rows=1)
                for k in STRATA:
                    original[k][key].extend(base_kl[flags[k]].cpu().tolist())
                for name in ARMS:
                    trial=trials[(fold[key],name)]
                    z1=env.head.bounded_readout(f,q,z,ctx,trial(ctx["contrast"]))
                    kl=fi.regret(student.sampled(z1,qi,grid),targets[2],weights)
                    change=kl-base_kl
                    rms[name].append(float((z1-base).square().mean().sqrt()))
                    for k in STRATA:
                        grouped[k][key][name].extend(change[flags[k]].cpu().tolist())
                    if anchors:
                        after=source.render(z1[anchor_queries],hw).gt(0).cpu().numpy()
                        for j,a in enumerate(anchors):
                            gt=gt_by_ann[a["annotation_id"]]
                            old=env.geometry.signed_counts(baseline_masks[j],gt)["strict"]
                            new=env.geometry.signed_counts(after[j],gt)["strict"]
                            geometry[name].update(new_losses=int(old and not new),regained=int(new and not old),
                                                  retained=new,rows=1)
        if i%100==0 or i==len(env.fit):
            print(json.dumps(dict(event="DIRECTION_FINITE_STEP_PROGRESS",images=i)),flush=True)
    env.close()

    outcomes={}
    for stratum in STRATA:
        groups=[k for k in keys if original[stratum][k]]
        base=float(np.mean([np.mean(original[stratum][k]) for k in groups]))
        need(abs(base-fr["strata"][stratum]["mean"]["teacher"])<1e-10,"Exact previous no-step fidelity endpoint")
        arr={k:np.array([np.mean(grouped[stratum][key][k]) for key in groups]) for k in ARMS}
        outcomes[stratum]=dict(baseline=base,deltas={k:compare_ci(x) for k,x in arr.items()},
           projected_minus_accepted=compare_ci(arr["projected"]-arr["accepted"]),
           scalarized_minus_accepted=compare_ci(arr["scalarized"]-arr["accepted"]),
           mixed_minus_accepted=compare_ci(arr["mixed"]-arr["accepted"]))
    a=outcomes["accepted"]["deltas"]
    s=outcomes["already_strict"]["deltas"]
    conflict=all(x["cosine"] is not None and x["cosine"]<=-.10 for x in plans)
    conflict=conflict and a["accepted"]["percentile95"][1]<0 and s["accepted"]["percentile95"][0]>0
    gates={}
    for name in ("projected","scalarized","mixed"):
        gates[name]=bool(conflict and a[name]["mean"]<=.8*a["accepted"]["mean"]
            and s[name]["mean"]<=0
            and outcomes["already_strict"][name+"_minus_accepted"]["percentile95"][1]<0
            and geometry[name]["new_losses"]==0)
    result=dict(status="SSDD_FIT_TEACHER_DIRECTION_PROBE_COMPLETE",protocol=args.protocol,
        FIT_only=True,AP=False,DEV=False,TEST=False,holdout_images_used=0,optimizer_updates=0,
        fit_images=484,encoded_groups=len(keys),gradient_rows=dict(grad_rows),relative_step=REL_STEP,
        anchor_parameter_l2=param_l2,plans=plans,outcomes=outcomes,strict_geometry=geometry,
        mean_all_query_logit_rms={k:float(np.mean(v)) for k,v in rms.items()},
        cross_fitted_conflict=bool(conflict),direction_gates=gates,seconds=time.perf_counter()-began,
        cuda_peak_bytes=t.cuda.max_memory_allocated())
    args.output.mkdir()
    with (args.output/"RESULT.json").open("x",encoding="utf-8") as f:
        json.dump(result,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fidelity-source",type=Path,required=True)
    p.add_argument("--fidelity-result",type=Path,required=True)
    for n in ("student-result","teacher-result","students","teachers"):
        p.add_argument("--"+n,type=Path,required=True)
    basic,_=p.parse_known_args()
    fi=module(basic.fidelity_source,FIDELITY_SHA,"_closed_teacher_fidelity")
    student=fi.dependency()
    source=student.teacher_source()
    source.common_args(p)
    run(p.parse_args(),fi,student,source)
