"""Measure whether the closed students learned the verified score targets on FIT."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time
from types import ModuleType

import numpy as np

STUDENT_SOURCE_SHA="0ced3814c286c9833dedb195e68595a49e46018799cd8f48f346483da35801c4"
STUDENT_RESULT_SHA="d598aab9e2cbb5260fc0cd1a14f8e16e938516245d3354a55880f613f7708de2"
TEACHER_RESULT_SHA="5fae0fdd33100923f433fef9cfb8148867ff6368705f3234088d855e1143b293"
NAMES=("native","direct_all","direct_selected","teacher")


def dependency():
    p=Path(__file__).with_name("teacher_student.py")
    raw=p.read_bytes()
    if sha256(raw).hexdigest()!=STUDENT_SOURCE_SHA:
        raise ValueError("Exact closed student source")
    m=ModuleType("_closed_original_grid_student")
    m.__file__=str(p)
    sys.modules[m.__name__]=m
    exec(compile(raw,str(p),"exec"),m.__dict__)
    return m


def regret(z,target,weights):
    from torch.nn.functional import binary_cross_entropy, binary_cross_entropy_with_logits
    z,target,weights=(v.double() for v in (z,target,weights))
    entropy=binary_cross_entropy(target,target,reduction="none")
    return ((binary_cross_entropy_with_logits(z,target,reduction="none")-entropy)*weights).sum(1).clamp_min(0)


def run(args,student,source):
    start=time.perf_counter()
    env=source.Environment(args)
    t=env.torch
    need=env.s.need
    student_bytes=args.student_result.read_bytes()
    teacher_bytes=args.teacher_result.read_bytes()
    need(sha256(student_bytes).hexdigest()==STUDENT_RESULT_SHA and sha256(teacher_bytes).hexdigest()==TEACHER_RESULT_SHA,
         "Exact accepted closed results")
    st,te=json.loads(student_bytes),json.loads(teacher_bytes)
    for path,digest in ((args.students,st["student_archive_sha256"]),(args.teachers,te["teacher_archive_sha256"])):
        with path.open("rb") as f:
            need(env.cap.file_digest(f)==digest,"Exact closed fitted archive")
    heads=t.load(args.students,map_location="cpu",weights_only=True)
    data=t.load(args.teachers,map_location="cpu",weights_only=True)
    need(heads["protocol"]==st["protocol"] and heads["steps"]==st["updates"]
         and data["protocol"]==te["protocol"] and len(data["records"])==484,"Exact closed training lineage")
    modules={}
    for key in NAMES[1:]:
        model=env.head.MaskCoefficientHead()
        model.load_state_dict(heads["students"][key],strict=True)
        modules[key]=model.requires_grad_(False).eval().cuda()
    summaries=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    visits=0
    for i,(row,record) in enumerate(zip(env.fit,data["records"],strict=True),1):
        c=env.image(row)
        locked=student.locked_supervision(env,source,row,c,record)
        if locked is not None:
            qi,grid,weights,targets=locked
            z,q,f,ctx=(c[k] for k in ("z","q","basis","ctx"))
            labels=targets[2]
            with t.no_grad():
                base=student.sampled(z,qi,grid)
                out={"native":base}
                for k,m in modules.items():
                    edit=env.head.bounded_readout(f,q,z,ctx,m(ctx["contrast"]))
                    out[k]=student.sampled(edit,qi,grid)
                losses={k:regret(v,labels,weights).cpu().numpy() for k,v in out.items()}
            for j,entry in enumerate(record["rows"]):
                stratum="accepted" if entry["keep"] else "already_strict" if entry["strict_before"] else "rejected_miss"
                for k in NAMES:
                    summaries[stratum][row["image_sha256"]][k].append(float(losses[k][j]))
                visits+=1
        if i%100==0 or i==len(env.fit):
            print(json.dumps(dict(event="TEACHER_FIT_FIDELITY_PROGRESS",images=i,rows=visits)),flush=True)
    env.close()
    output={}
    for name,groups in summaries.items():
        order=sorted(groups)
        a=np.array([[np.mean(groups[g][k]) for k in NAMES] for g in order])
        n=len(a)
        mean=a.mean(0)
        boot=np.random.default_rng(5201).integers(0,n,size=(2048,n))
        sample=a[boot].mean(1)
        count=sum(len(groups[g]["native"]) for g in order)
        output[name]=dict(images=n,rows=count,mean=dict(zip(NAMES,mean.tolist(),strict=True)),
            teacher_minus_selected_95=np.quantile(sample[:,3]-sample[:,2],[.025,.975]).tolist())
        if name=="accepted":
            need(mean[0]>0 and bool((sample[:,0]>0).all()),"Native teacher target differs on accepted support")
            ratio=1.-sample[:,3]/sample[:,0]
            output[name]["fraction_of_native_regret_removed"]=float(1.-mean[3]/mean[0])
            output[name]["fraction_percentile95"]=np.quantile(ratio,[.025,.975]).tolist()
    need(visits==1082 and output["accepted"]["rows"]==752 and output["already_strict"]["rows"]==277,
         "Exact training strata and scope")
    accepted=output["accepted"]
    result=dict(status="SSDD_FIT_TEACHER_FIDELITY_COMPLETE",protocol=args.protocol,FIT_only=True,DEV=False,TEST=False,
        AP=False,holdout_images_used=0,optimizer_updates=0,fit_images=484,fit_rows=visits,strata=output,
        inadequate_current_target_fit=bool(accepted["fraction_of_native_regret_removed"]<.25
                                           and accepted["fraction_percentile95"][1]<.50),
        seconds=time.perf_counter()-start,cuda_peak_bytes=t.cuda.max_memory_allocated())
    args.output.mkdir()
    with (args.output/"RESULT.json").open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    student=dependency()
    source=student.teacher_source()
    p=argparse.ArgumentParser(description=__doc__)
    source.common_args(p)
    for n in ("student-result","teacher-result","students","teachers"):
        p.add_argument("--"+n,type=Path,required=True)
    run(p.parse_args(),student,source)
