"""Swap fixed-eight quota policies across pre-existing matched partitions."""
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

ASYMMETRIC_SHA="81d90df0a5862f87351a9590c3cbf125738df4d8665f652294e6613f5f547767"
MATH_SHA="258e9d1aec9a9c9b6ffee60627d737161b2aee5fb07fd5bb0eda6546a7228c70"
RESULT_SHA="f1b73c89f1ea25e17fc1ded67ad79aa05ea823dcc76a7f645f108491df6c5775"
ARMS=("native","balanced","signed8","signed_matched_only","signed_unmatched_only")


def splice(first,second,positions):
    return [b if j in positions else a for j,(a,b) in enumerate(zip(first,second,strict=True))]


def need(ok,message):
    if not ok:
        raise ValueError(message)


def load_base(path):
    raw=path.read_bytes()
    need(sha256(raw).hexdigest()==ASYMMETRIC_SHA,"Exact published P162 code")
    mod=ModuleType("_asymmetric_crossread_base")
    mod.__file__=str(path)
    sys.modules[mod.__name__]=mod
    exec(compile(raw,str(path),"exec"),mod.__dict__)
    return mod


def run(args,a,base,teacher,prior,local,policy,math):
    start=time.perf_counter()
    raw=args.asymmetric_result.read_bytes()
    need(sha256(raw).hexdigest()==RESULT_SHA,"Exact closed P162 result")
    ref=json.loads(raw)
    env=teacher.Environment(args)
    roles=base.parts(env.fit)
    pt=base.point_training(env,teacher,prior,local,roles["point_fit"])
    point,loss=base.train_model(env.torch,np.concatenate([r["x"] for r in pt]),
                  np.concatenate([r["y"] for r in pt]),prior.weights(pt),True)
    need(len(pt)==623 and sum(len(r["y"]) for r in pt)==142170 and loss==0.3621605634689331,
         "Exact closed point-order replay")
    cal=a.calibration(env,teacher,base,local,policy,roles["gate_fit"],point,"crossread_probability_fit")
    need(len(cal)==ref["calibration_fit_matches"],"Exact matched fit calibration population")
    model=[math.fit_reliability(*a.calibrated_rows(cal,k)) for k in (0,1)]
    predictions={k:[] for k in ARMS}
    partition={k:Counter() for k in ("matched","unmatched")}
    geo={k:Counter() for k in ("balanced","signed8")}
    identity=[]
    for i,item in enumerate(env.hold,1):
        c=env.image(item)
        rows=a.prefixes(c,item["dataset_index"],point,teacher,base,local,policy)
        stock=env.s.coerce_rles(c["native"],env.codec)
        edited={k:{} for k in ("balanced","signed8")}
        ratios={}
        for row in rows:
            j=row["j"]
            u,v=row["indices"]
            xy=row["pixels"]
            p=model[0](row["scores"][0])
            q=model[1](row["scores"][1])
            ratio,_=math.choose_split(math.signed_gain(p,False),math.signed_gain(q,True))
            ratios[j]=ratio
            for k,n in (("balanced",4),("signed8",ratio)):
                edited[k][j]=math.edit_mask(c["native"][j],xy[u[:n]],xy[v[:8-n]])
        outputs={}
        for k,m in edited.items():
            rle=list(stock)
            for j,z in m.items():
                rle[j]=env.codec.encode(np.asfortranarray(z.astype(np.uint8)))
            outputs[k]=rle
        # The two prediction-side actions are complete before reading fixed matches.
        _,truth,matches,_=env.matches(item,c)
        matched=dict(matches)
        eligible=set(ratios)
        m=eligible&set(matched)
        u=eligible-m
        for group,js in (("matched",m),("unmatched",u)):
            for j in js:
                ratio=ratios[j]
                partition[group].update(rows=1,changed_ratio=int(ratio!=4),pure_deletion=int(ratio==0),
                                        signed_add=ratio,signed_delete=8-ratio)
        for k,r in edited.items():
            geo[k].update(edited_rows=len(r),matched_edits=len(m),unmatched_edits=len(u))
            for j in m:
                geo[k].update(a.stats(env.geometry,c["native"][j],r[j],truth[matched[j]]))
        fixed=outputs["balanced"]
        signed=outputs["signed8"]
        streams=dict(native=stock,balanced=fixed,signed8=signed,
                     signed_matched_only=splice(fixed,signed,m),signed_unmatched_only=splice(fixed,signed,u))
        iid=item["image_id"]
        identity.extend((iid,rank,q,score.hex()) for rank,q,score in zip(c["rank"],c["query"],c["scores"],strict=True))
        for k,rle in streams.items():
            predictions[k].extend(dict(image_id=iid,category_id=1,score=z,segmentation=r)
                                  for z,r in zip(c["scores"],rle,strict=True))
        if i%25==0 or i==len(env.hold):
            print(json.dumps(dict(event="ASYMMETRIC_CROSSREAD_PROGRESS",images=i)),flush=True)
    env.close()
    digest=sha256(json.dumps(identity,separators=(",",":")).encode()).hexdigest()
    need(digest==ref["original_candidates_sha256"] and len(identity)==9082
         and all(all(v==ref["geometry"][k][m] for m,v in counter.items()) for k,counter in geo.items()),
         "Exact original candidate and balanced/signed geometry")
    ids=[x["image_id"] for x in env.hold]
    idset=set(ids)
    gt=dict(info=env.gt["info"],images=[x for x in env.gt["images"] if x["id"] in idset],
            annotations=[x for x in env.gt["annotations"] if x["image_id"] in idset],categories=env.gt["categories"])
    metrics={k:env.s.fit_stats(gt,v,ids) for k,v in predictions.items()}
    need(all(metrics[k]==ref["metrics"][k] for k in ("native","balanced","signed8")),"Exact three original endpoints")
    delta={k:{m:metrics[k][m]-metrics["balanced"][m] for m in ("AP","AP75","AP90","APsmall","AR100")}
           for k in ARMS if k!="balanced"}
    d=delta["signed_matched_only"]
    matched_gate=(d["AP"]>=.003 and d["AP75"]>=0 and d["AP90"]>=.01
                  and geo["signed8"]["strict_loss"]<=geo["balanced"]["strict_loss"])
    result=dict(status="SSDD_FIT_ASYMMETRIC_CROSSREAD_COMPLETE",protocol=args.protocol,original_protocol=ref["protocol"],
                metrics=metrics,delta_vs_balanced=delta,partition=partition,geometry=geo,
                AP_interaction=delta["signed8"]["AP"]-d["AP"]-delta["signed_unmatched_only"]["AP"],
                matched_gate=bool(matched_gate),identity_sha256=digest,exact_endpoints=True,
                point_replay_updates=96,detector_updates=0,uses_GT_for_attribution=True,
                AP=True,DEV=False,TEST=False,elapsed_s=time.perf_counter()-start,
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
    for k in ("asymmetric-source","selective-source","teacher"):
        p.add_argument("--"+k,type=Path,required=True)
    first,_=p.parse_known_args()
    a=load_base(first.asymmetric_source)
    base=a.pin(first.selective_source,a.SELECTIVE_SHA,"_quota_crossread_selective")
    teacher=base.teacher_module(first.teacher)
    teacher.common_args(p)
    for k in ("asymmetric-result","asymmetric-math","local-source","local-math","policy"):
        p.add_argument("--"+k,type=Path,required=True)
    args=p.parse_args()
    math=a.pin(args.asymmetric_math,MATH_SHA,"_quota_crossread_math")
    prior=a.pin(args.local_source,base.LOCAL_SHA,"_quota_crossread_prior")
    local=a.pin(args.local_math,base.MATH_SHA,"_quota_crossread_local")
    policy=a.pin(args.policy,a.POLICY_SHA,"_quota_crossread_policy")
    run(args,a,base,teacher,prior,local,policy,math)

