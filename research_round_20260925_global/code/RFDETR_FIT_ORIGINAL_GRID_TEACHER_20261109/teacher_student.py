"""Same-support direct and evidence-filtered coefficient-teacher students."""
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

TEACHER_SOURCE_SHA="67cd6687d1882889bc671fda2207ed88150627772b0f137ff97b29faec82bf3d"
SCOUT_RESULT_SHA="fb985286825a3c878f66744edee99ddaeebf50edade5255ec8324b3317da3e28"
SEED=5201
ARMS=("direct_all","direct_selected","teacher")


def supervised_targets(native, hard, fitted, selected):
    import torch
    stock=native.sigmoid()
    gate=selected[:,None]
    return hard,torch.where(gate,hard,stock),torch.where(gate,fitted.sigmoid(),stock)


def weighted_bce(logits,target,weights):
    from torch.nn.functional import binary_cross_entropy_with_logits
    return (binary_cross_entropy_with_logits(logits,target,reduction="none")*weights).sum(1).mean()


def sampled(z,queries,grid):
    from torch.nn.functional import grid_sample
    return grid_sample(z[queries,None],grid,mode="bilinear",padding_mode="border",align_corners=False)[:,0,0]


def teacher_source():
    path=Path(__file__).with_name("teacher_fit.py")
    raw=path.read_bytes()
    if sha256(raw).hexdigest()!=TEACHER_SOURCE_SHA:
        raise ValueError("Exact closed teacher source")
    mod=ModuleType("_original_grid_teacher_source")
    mod.__file__=str(path)
    sys.modules[mod.__name__]=mod
    exec(compile(raw,str(path),"exec"),mod.__dict__)
    return mod


def locked_supervision(env,source,item,c,record):
    t=env.torch
    z,q,f,ctx,hw=(c[k] for k in ("z","q","basis","ctx","hw"))
    anns,truth,matches,_=env.matches(item,c)
    active=[(j,k) for j,k in matches if bool(ctx["active"][c["query"][j]])]
    rows=record["rows"]
    env.s.need(record["image_id"]==item["image_id"] and record["dataset_index"]==item["dataset_index"]
               and len(active)==len(rows),"Exact teacher image and matching count")
    if not rows:
        return None
    count=max(row["index"].numel() for row in rows)
    coordinates=t.zeros((len(rows),1,count,2),device=z.device)
    labels=t.zeros((len(rows),count),device=z.device)
    weights=t.zeros_like(labels)
    delta=t.zeros_like(q)
    q_ids=[]
    selected=[]
    for n,(row,(j,k)) in enumerate(zip(rows,active,strict=True)):
        qi=c["query"][j]
        idx=row["index"].numpy()
        stock=c["native"][j]
        gt=truth[k]
        env.s.need(qi==row["query"] and c["rank"][j]==row["native_rank"] and anns[k]["id"]==row["annotation_id"]
                   and sha256(stock.tobytes()).hexdigest()==row["stock_mask_sha256"]
                   and row["strict_before"]==env.geometry.signed_counts(stock,gt)["strict"]
                   and np.array_equal(gt.ravel()[idx],row["labels"].numpy())
                   and t.isclose(row["weights"].sum(),t.tensor(1.)),"Exact frozen original-grid teacher identity")
        env.s.need(row["delta"].shape==(256,) and bool(t.isfinite(row["delta"]).all())
                   and (row["keep"] or not bool(row["delta"].any())), "Identity targets remain exact")
        points=source.pixel_grid(idx,hw,t,z.device)
        end=len(idx)
        coordinates[n,0,:end]=points
        labels[n,:end]=row["labels"].to(device=z.device,dtype=z.dtype)
        weights[n,:end]=row["weights"].to(device=z.device,dtype=z.dtype)
        delta[qi]=row["delta"].to(z.device)
        q_ids.append(qi)
        selected.append(row["keep"])
    chosen=t.tensor(q_ids,dtype=t.int64,device=z.device)
    gate=t.tensor(selected,dtype=t.bool,device=z.device)
    with t.no_grad():
        fitted=env.head.bounded_readout(f,q,z,ctx,delta)
        if any(selected):
            edited=source.render(fitted[chosen],hw).gt(0).cpu().numpy()
            for n,((j,k),row) in enumerate(zip(active,rows,strict=True)):
                if row["keep"]:
                    gt=truth[k]
                    stock=c["native"][j]
                    support=env.geometry.sampling_support(gt,stock,SEED+item["dataset_index"]*100+row["query"])
                    held=support["held"]
                    st=env.geometry.signed_counts(stock,gt)
                    out=env.geometry.signed_counts(edited[n],gt)
                    held_delta=env.geometry.signed_counts(edited[n][held],gt[held])["error"]-env.geometry.signed_counts(
                        stock[held],gt[held])["error"]
                    env.s.need(env.geometry.accepted(st,out,held_delta,st["area"]),"Accepted teacher survives exact replay")
        a=sampled(z,chosen,coordinates)
        b=sampled(fitted,chosen,coordinates)
        env.s.need(t.equal(a[~gate],b[~gate]),"Native rejected targets are exact on all points")
        targets=supervised_targets(a,labels,b,gate)
    return chosen,coordinates,weights,targets


def evaluate(env,source,students,scout):
    t=env.torch
    names=("native",)+ARMS
    predictions={k:[] for k in names}
    footprint={k:Counter() for k in names}
    all_geometry={k:Counter() for k in names}
    grouped=defaultdict(list)
    begin=time.perf_counter()
    for i,item in enumerate(env.hold,1):
        c=env.image(item)
        anns,truth,pairs,_=env.matches(item,c)
        iid=item["image_id"]
        hw,z,q,f,ctx=(c[k] for k in ("hw","z","q","basis","ctx"))
        matched={j for j,_ in pairs}
        reference=[env.geometry.signed_counts(c["native"][j],truth[k]) for j,k in pairs]
        group_metrics={}
        for arm in names:
            native=c["native"]
            if arm=="native" or not c["query"]:
                planes=native
            else:
                with t.no_grad():
                    candidate=env.head.bounded_readout(f,q,z,ctx,students[arm](ctx["contrast"]))[c["query"]]
                    planes=np.concatenate([source.render(v,hw).gt(0).cpu().numpy() for v in candidate.split(32)])
            env.s.need(planes.shape==native.shape,"Every original mask retained")
            rles=env.s.coerce_rles(planes,env.codec)
            predictions[arm].extend(dict(image_id=iid,category_id=1,score=score,segmentation=rle)
                                    for score,rle in zip(c["scores"],rles,strict=True))
            for j in range(len(planes)):
                diff=int(np.count_nonzero(planes[j]!=native[j]))
                footprint[arm].update(changed_rows=int(diff>0),pixels=diff,
                    unmatched_rows=int(diff>0 and j not in matched),unmatched_pixels=diff if j not in matched else 0)
            counter=Counter()
            errors=[]
            small_errors=[]
            for (j,k),base in zip(pairs,reference,strict=True):
                counts=env.geometry.signed_counts(planes[j],truth[k])
                small=base["area"]<1024
                change=(counts["error"]-base["error"])/base["area"]
                errors.append(change)
                if small:
                    small_errors.append(change)
                counter.update(fp=counts["fp"],fn=counts["fn"],strict=counts["strict"],
                    small_strict=counts["strict"]*small,stock_strict=base["strict"],stock_strict_small=base["strict"]*small,
                    strict_losses=int(base["strict"] and not counts["strict"]),
                    strict_losses_small=int(base["strict"] and not counts["strict"] and small))
            all_geometry[arm].update(counter)
            group_metrics[arm]=dict(small_strict=counter["small_strict"],
                signed_error=float(np.mean(errors)) if errors else 0.,
                small_signed_error=float(np.mean(small_errors)) if small_errors else 0.)
        grouped[item["image_sha256"]].append(group_metrics)
        if i%25==0 or i==len(env.hold):
            print(json.dumps(dict(event="TEACHER_STUDENT_HOLDOUT_PROGRESS",images=i)),flush=True)
    ids=[x["image_id"] for x in env.hold]
    idset=set(ids)
    gt=env.gt
    gt_eval=dict(info=gt["info"],images=[x for x in gt["images"] if x["id"] in idset],
                 annotations=[x for x in gt["annotations"] if x["image_id"] in idset],categories=gt["categories"])
    stats={k:env.s.fit_stats(gt_eval,v,ids) for k,v in predictions.items()}
    env.s.need(stats["native"]==scout["metrics"]["native"],"Exact same-environment native 111-image anchor")
    identity=lambda v:[(x["image_id"],x["category_id"],x["score"].hex()) for x in v]
    env.s.need(all(identity(v)==identity(predictions["native"]) for v in predictions.values()),
               "All original score bits and rank order")
    boot=np.random.default_rng(SEED).integers(0,len(grouped),size=(2048,len(grouped)))
    compared={}
    for control in ("native","direct_all","direct_selected"):
        compared[control]={}
        for metric in ("small_strict","signed_error","small_signed_error"):
            vals=np.array([np.mean([x["teacher"][metric]-x[control][metric] for x in grouped[key]])
                           for key in sorted(grouped)])
            compared[control][metric]=dict(mean=float(vals.mean()),
                percentile95=np.quantile(vals[boot].mean(1),[.025,.975]).tolist())
    actual=all_geometry["teacher"]
    preserves=(20*actual["strict_losses"]<=actual["stock_strict"]
               and 20*actual["strict_losses_small"]<=actual["stock_strict_small"])
    gate=preserves and all(stats["teacher"]["AP"]-stats[k]["AP"]>=.001
        and stats["teacher"]["AP75"]>=stats[k]["AP75"] and stats["teacher"]["AP90"]>stats[k]["AP90"]
        and compared[k]["small_strict"]["percentile95"][0]>0
        for k in ("native","direct_all","direct_selected"))
    gate=gate and all(compared["native"][k]["mean"]<=0 for k in ("signed_error","small_signed_error"))
    return dict(metrics=stats,geometry=all_geometry,paired=compared,edit_footprint=footprint,
                retention_guard=preserves,gate=gate,holdout_seconds=time.perf_counter()-begin)


def run(args,source):
    started=time.perf_counter()
    env=source.Environment(args)
    t=env.torch
    raw=args.teacher_result.read_bytes()
    env.s.need(sha256(raw).hexdigest()==args.teacher_result_sha256,"Exact closed verified teacher report")
    report=json.loads(raw)
    env.s.need(report["teacher_gate"] is True and report["images"]==484 and report["holdout_images_used"]==0,
               "Teacher evidence admitted without holdout")
    old=args.scout_result.read_bytes()
    env.s.need(sha256(old).hexdigest()==SCOUT_RESULT_SHA,"Exact prior same-run native anchor")
    scout=json.loads(old)
    with args.teachers.open("rb") as f:
        env.s.need(env.cap.file_digest(f)==report["teacher_archive_sha256"],"Exact private teacher archive")
        f.seek(0)
        data=t.load(f,map_location="cpu",weights_only=True)
    env.s.need(data["protocol"]==report["protocol"] and data["gate"] is True
               and len(data["records"])==len(env.fit),"Exact teacher archive lineage")
    t.manual_seed(SEED)
    first=env.head.MaskCoefficientHead().cuda()
    students={k:deepcopy(first) for k in ARMS}
    optim={k:t.optim.AdamW(h.parameters(),lr=3e-4,weight_decay=1e-4) for k,h in students.items()}
    updates=Counter()
    counts=Counter()
    losses=defaultdict(list)
    begin=time.perf_counter()
    for i,(item,record) in enumerate(zip(env.fit,data["records"],strict=True),1):
        c=env.image(item)
        data_row=locked_supervision(env,source,item,c,record)
        counts.update(images=1,rows=len(record["rows"]),verified_targets=sum(x["keep"] for x in record["rows"]))
        if data_row:
            query,grid,weights,targets=data_row
            f,q,z,ctx=(c[k] for k in ("basis","q","z","ctx"))
            for arm,target in zip(ARMS,targets,strict=True):
                optim[arm].zero_grad(set_to_none=True)
                fitted=env.head.bounded_readout(f,q,z,ctx,students[arm](ctx["contrast"]))
                loss=weighted_bce(sampled(fitted,query,grid),target,weights)+.05*(fitted-z).square().mean()
                env.s.need(bool(t.isfinite(loss)),"Finite common supervised loss")
                loss.backward()
                optim[arm].step()
                updates[arm]+=1
                losses[arm].append(float(loss.detach()))
        if i%100==0 or i==len(env.fit):
            print(json.dumps(dict(event="TEACHER_STUDENT_TRAIN_PROGRESS",images=i,updates=dict(updates))),flush=True)
    t.cuda.synchronize()
    training_seconds=time.perf_counter()-begin
    env.s.need(len(set(updates.values()))==1 and set(updates)==set(ARMS) and min(updates.values())>=300
       and counts["rows"]==report["totals"]["active_matched"]
       and counts["verified_targets"]==report["totals"]["coeff_accepted"],"Equal actual updates and full replay")
    for arm in ARMS:
        states=optim[arm].state.values()
        env.s.need(len(states)==4 and all(int(v["step"])==updates[arm] for v in states),"Exact AdamW counts")
        students[arm].eval()
    outcome=evaluate(env,source,students,scout)
    env.close()
    result=dict(status="SSDD_FIT_ORIGINAL_GRID_TEACHER_STUDENT_COMPLETE",protocol=args.protocol,
        teacher_protocol=report["protocol"],FIT_only=True,DEV=False,TEST=False,AP=True,formal=False,
        seed=SEED,updates=dict(updates),training_counts=dict(counts),head_parameters=sum(x.numel() for x in first.parameters()),
        loss_mean={k:float(np.mean(v)) for k,v in losses.items()},teacher_generation_seconds=report["seconds"],
        training_seconds=training_seconds,total_seconds=time.perf_counter()-started,
        cuda_peak_bytes=t.cuda.max_memory_allocated(),**outcome)
    args.output.mkdir()
    with (args.output/"STUDENTS_PRIVATE.pt").open("xb") as f:
        t.save(dict(protocol=args.protocol,steps=dict(updates),
                    students={k:{n:v.detach().cpu() for n,v in h.state_dict().items()} for k,h in students.items()}),f)
        f.flush()
        os.fsync(f.fileno())
    with (args.output/"STUDENTS_PRIVATE.pt").open("rb") as f:
        result["student_archive_sha256"]=env.cap.file_digest(f)
    with (args.output/"RESULT.json").open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    src=teacher_source()
    parser=argparse.ArgumentParser(description=__doc__)
    src.common_args(parser)
    for name in ("teacher-result","teachers","scout-result"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--teacher-result-sha256",required=True)
    run(parser.parse_args(),src)
