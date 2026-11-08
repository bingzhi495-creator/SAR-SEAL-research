"""Measure attained original-grid coefficient actions under fixed SSDD COCO ranks."""
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
REFERENCE_SHA="d598aab9e2cbb5260fc0cd1a14f8e16e938516245d3354a55880f613f7708de2"


def pinned_teacher(path):
    source=path.read_bytes()
    if sha256(source).hexdigest()!=TEACHER_SHA:
        raise ValueError("Exact original bounded teacher source")
    module=ModuleType("_p150_teacher_fit")
    module.__file__=str(path)
    sys.modules[module.__name__]=module
    exec(compile(source,str(path),"exec"),module.__dict__)
    return module


def run(args,teacher,policy,fitter):
    started=time.perf_counter()
    reference_raw=args.reference.read_bytes()
    if sha256(reference_raw).hexdigest()!=REFERENCE_SHA:
        raise ValueError("Exact closed P144 reference report")
    reference=json.loads(reference_raw)
    env=teacher.Environment(args)
    solver=teacher.module(args.solver,teacher.SOLVER_SHA,"_p150_bounded_solver")
    names=("native","offset","coeff","matched_gt")
    pred={k:[] for k in names}
    geom={k:Counter() for k in names}
    totals=Counter()
    fixed_thresholds=tuple(range(50,100,5))
    for i,item in enumerate(env.hold,1):
        readout=policy.Readout()
        rows,counts=fitter.build_image(env,item,solver,readout)
        totals.update(counts)
        env.s.need(len(rows)==counts["active_matched"],"Complete active fixed-match accounting")
        c=readout.candidate
        eligible={row["query"] for row in rows if not row["strict_before"]}
        protected=[j for j,q in enumerate(c["query"]) if q not in eligible]
        env.s.need(len(c["query"])==len(c["rank"])==len(c["scores"])==len(c["native"]),
                   "Exact candidate identity before replacement")
        for name,planes in readout.masks.items():
            env.s.need(np.array_equal(planes[protected],c["native"][protected]),"Only active nonstrict native matches may change")
            rles=readout.native_rles if name=="native" else env.s.coerce_rles(planes,env.codec)
            g=geom[name]
            for j,k in readout.matches:
                gt=readout.truth[k]
                m=env.geometry.signed_counts(planes[j],gt)
                stock=env.geometry.signed_counts(c["native"][j],gt)
                denom=m["area"]+m["fp"]
                flips=int(np.count_nonzero(planes[j]!=c["native"][j]))
                g.update(fp=m["fp"],fn=m["fn"],strict=m["strict"],strict_losses=int(stock["strict"] and not m["strict"]),
                         small_strict=int(m["strict"] and m["area"]<1024),changed_rows=int(flips>0),changed_pixels=flips,
                         both_error_sides_reduced=int(m["fp"]<stock["fp"] and m["fn"]<stock["fn"]))
                g.update({f"fixed_match_iou_{thr}":int(denom>0 and m["tp"]*100>=thr*denom) for thr in fixed_thresholds})
            pred[name].extend(dict(image_id=item["image_id"],category_id=1,score=s,segmentation=r)
                              for s,r in zip(c["scores"],rles,strict=True))
        if i%25==0 or i==len(env.hold):
            print(json.dumps(dict(event="BOUNDED_HEADROOM_PROGRESS",images=i)),flush=True)
    env.close()
    ids=[item["image_id"] for item in env.hold]
    idset=set(ids)
    gt=dict(info=env.gt["info"],images=[x for x in env.gt["images"] if x["id"] in idset],
            annotations=[x for x in env.gt["annotations"] if x["image_id"] in idset],categories=env.gt["categories"])
    identity=lambda rows:[(x["image_id"],x["category_id"],x["score"].hex()) for x in rows]
    baseline_identity=identity(pred["native"])
    env.s.need(len(baseline_identity)==9082 and all(identity(p)==baseline_identity for p in pred.values()),
               "Exact common candidate order and score bits")
    metrics={k:env.s.fit_stats(gt,p,ids) for k,p in pred.items()}
    env.s.need(metrics["native"]==reference["metrics"]["native"]
               and geom["native"]["strict"]==reference["geometry"]["native"]["strict"]
               and geom["native"]["small_strict"]==reference["geometry"]["native"]["small_strict"],
               "Exact closed original candidate and AP replay")
    env.s.need(all(g["strict_losses"]==0 for g in geom.values()),"Original strict fixed matches unchanged")
    env.s.need(geom["offset"]["both_error_sides_reduced"]==0,"Uniform bias must yield nested masks")
    diffs={name:{k:metrics[name][k]-metrics["native"][k] for k in ("AP","AP75","AP90","APsmall","AR100")}
           for name in names if name!="native"}
    incremental=metrics["coeff"]["AP"]-metrics["offset"]["AP"]
    gate=bool(diffs["coeff"]["AP"]>=.005 and incremental>=.003 and diffs["coeff"]["AP75"]>=0
              and diffs["coeff"]["AP90"]>=.01)
    result=dict(status="SSDD_FIT_BOUNDED_ACTION_HEADROOM_COMPLETE",protocol=args.protocol,
                reference_protocol=reference["protocol"],metrics=metrics,geometry=geom,delta_vs_native=diffs,
                coeff_minus_offset_AP=incremental,gate=gate,exact_native_endpoint_replay=True,
                images=len(ids),encoded_groups=len({x["image_sha256"] for x in env.hold}),support_counts=totals,
                elapsed_s=time.perf_counter()-started,cuda_peak_bytes=env.torch.cuda.max_memory_allocated(),
                FIT_only=True,AP=True,DEV=False,TEST=False,optimizer_updates=0,uses_oracle_GT=True)
    args.output.mkdir()
    with (args.output/"RESULT.json").open("x",encoding="utf-8") as f:
        json.dump(result,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(result,sort_keys=True,allow_nan=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher",type=Path,required=True)
    args,_=parser.parse_known_args()
    teacher=pinned_teacher(args.teacher)
    teacher.common_args(parser)
    for name in ("solver","reference","action-policy","observed-fitter"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--action-policy-sha",required=True)
    parser.add_argument("--observed-fitter-sha",required=True)
    args=parser.parse_args()
    policy=teacher.module(args.action_policy,args.action_policy_sha,"_p150_action_policy")
    fitter=teacher.module(args.observed_fitter,args.observed_fitter_sha,"_p150_observed_fitter")
    run(args,teacher,policy,fitter)

