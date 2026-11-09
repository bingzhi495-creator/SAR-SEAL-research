"""Separate point-location and quota opportunity at the same eight edits."""
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
ASYM_MATH_SHA="258e9d1aec9a9c9b6ffee60627d737161b2aee5fb07fd5bb0eda6546a7228c70"
OUTCOME_MATH_SHA="093319769bbf86ab88b8bcd454884d99b48ef5a7dd0d98eddb95fec69fdbcdd5"
REFERENCE_SHA="74d4cd18d59b231f400a57e2ac5e06b2974b00711c4f325f13601e36df342311"
ARMS=("native","balanced","quota_oracle","rank_oracle","rank_and_quota_oracle")


def perfect_order(pixel,inside,correct):
    out=[]
    for side in (False,True):
        idx=np.flatnonzero(inside==side)
        order=np.lexsort((pixel[idx],-correct[idx].astype(np.int32)))
        out.append(idx[order[:8]])
    return tuple(out)


def best_quota(area,fp,fn,c1,c2,math):
    r=math.realized(area,fp,fn,c1,c2)
    ordered=sorted(range(len(r["k"])),key=lambda i:(abs(int(r["k"][i])-4),int(r["k"][i])))
    best=ordered[0]
    for i in ordered[1:]:
        left=(area-int(r["fn"][i]))*(area+int(r["fp"][best]))
        right=(area-int(r["fn"][best]))*(area+int(r["fp"][i]))
        if left>right:
            best=i
    return int(r["k"][best])


def need(ok,message):
    if not ok:
        raise ValueError(message)


def module(path,digest,name):
    raw=path.read_bytes()
    need(sha256(raw).hexdigest()==digest,"Exact source")
    m=ModuleType(name)
    m.__file__=str(path)
    sys.modules[name]=m
    exec(compile(raw,str(path),"exec"),m.__dict__)
    return m


def read_role(env,items,reference,role,point,a,base,teacher,local,policy,asym,math):
    pred={k:[] for k in ARMS}
    geo={k:Counter() for k in ARMS}
    identity=[]
    ids=[]
    for i,item in enumerate(items,1):
        c=env.image(item)
        native=c["native"]
        proposed=a.prefixes(c,item["dataset_index"],point,teacher,base,local,policy)
        balanced={}
        for r in proposed:
            j=r["j"]
            u,v=r["indices"]
            x=r["pixels"]
            balanced[j]=asym.edit_mask(native[j],x[u[:4]],x[v[:4]])
        # The deployed balanced action has already been fixed on all candidates.
        _,truth,matches,_=env.matches(item,c)
        matched=dict(matches)
        sets={k:dict(balanced) for k in ARMS[2:]}
        sets.update(native={},balanced=balanced)
        nstrict=sum(env.geometry.signed_counts(native[j],truth[k])["strict"] for j,k in matches)
        for r in proposed:
            j=r["j"]
            if j not in matched:
                continue
            gt=truth[matched[j]]
            before=env.geometry.signed_counts(native[j],gt)
            pixel=r["pixels"]
            correct=gt.ravel()[pixel]!=r["inside"]
            a0,d0=r["indices"]
            p,q=perfect_order(pixel,r["inside"],correct)
            def quota(v,w):
                return best_quota(before["area"],before["fp"],before["fn"],correct[v],correct[w],math)
            rules=(("quota_oracle",a0,d0,quota(a0,d0)),("rank_oracle",p,q,4),
                   ("rank_and_quota_oracle",p,q,quota(p,q)))
            for name,add,delete,k in rules:
                sets[name][j]=asym.edit_mask(native[j],pixel[add[:k]],pixel[delete[:8-k]])
                geo[name].update(oracle_matched_rows=1,net_oracle_area=2*k-8)
        iid=item["image_id"]
        ids.append(iid)
        identity.extend((iid,rank,q,z.hex()) for rank,q,z in zip(c["rank"],c["query"],c["scores"],strict=True))
        stock=env.s.coerce_rles(native,env.codec)
        for arm,rows in sets.items():
            rle=list(stock)
            geo[arm].update(edited_rows=len(rows),original_matches=len(matched),original_strict=nstrict)
            for j,value in rows.items():
                rle[j]=env.codec.encode(np.asfortranarray(value.astype(np.uint8)))
                if j in matched:
                    geo[arm].update(a.stats(env.geometry,native[j],value,truth[matched[j]]))
            pred[arm].extend(dict(image_id=iid,category_id=1,score=s,segmentation=r)
                             for s,r in zip(c["scores"],rle,strict=True))
        if i%50==0 or i==len(items):
            print(json.dumps(dict(event="RANK_QUOTA_ORACLE_PROGRESS",role=role,images=i)),flush=True)
    dig=sha256(json.dumps(identity,separators=(",",":")).encode()).hexdigest()
    need(dig==reference["identity_sha256"] and len(identity)==reference["predictions"]
         and all(geo[k]["edited_rows"]==reference["eligible"] for k in ARMS[1:]),
         "Exact old candidate identity and eight-edit coverage")
    idset=set(ids)
    gt=dict(info=env.gt["info"],images=[x for x in env.gt["images"] if x["id"] in idset],
            annotations=[x for x in env.gt["annotations"] if x["image_id"] in idset],categories=env.gt["categories"])
    m={k:env.s.fit_stats(gt,v,ids) for k,v in pred.items()}
    need(all(m[k]==reference["metrics"][k] for k in ("native","balanced")),"Exact old native/paired AP endpoints")
    return dict(images=len(ids),predictions=len(identity),eligible=reference["eligible"],identity_sha256=dig,
                metrics=m,geometry=geo)


def run(args,a,base,teacher,prior,local,policy,asym,math):
    start=time.perf_counter()
    raw=args.reference.read_bytes()
    need(sha256(raw).hexdigest()==REFERENCE_SHA,"Exact closed 114 and 111 fit results")
    ref=json.loads(raw)
    env=teacher.Environment(args)
    pt=base.point_training(env,teacher,prior,local,base.parts(env.fit)["point_fit"])
    point,loss=base.train_model(env.torch,np.concatenate([r["x"] for r in pt]),
                                np.concatenate([r["y"] for r in pt]),prior.weights(pt),True)
    need(len(pt)==623 and loss==0.3621605634689331,"Exact learned point order")
    allfit=json.loads(env.cap.exact_bytes(args.fit_index,348469,env.cap.FIT_INDEX_SHA))["items"]
    cal=[x for x in allfit if x["partition"]=="calibration"]
    main=read_role(env,cal,ref["primary114"],"historical_calibration114",point,a,base,teacher,local,policy,asym,math)
    second=read_role(env,env.hold,ref["secondary111"],"historical_holdout111",point,a,base,teacher,local,policy,asym,math)
    env.close()
    need([second["geometry"][k]["strict_gain"] for k in ARMS[2:]]==[33,59,68],
         "Exact known original fixed-match opportunity counts")
    m=main["metrics"]
    g=main["geometry"]
    rank=m["rank_oracle"]
    paired=m["balanced"]
    gate=(rank["AP"]-paired["AP"]>=.005 and rank["AP75"]>=paired["AP75"]
          and rank["AP90"]-paired["AP90"]>=.01 and rank["AP"]-m["quota_oracle"]["AP"]>=.002
          and g["rank_oracle"]["strict_loss"]<=g["balanced"]["strict_loss"])
    result=dict(status="SSDD_FIT_RANK_QUOTA_ORACLE_AP_COMPLETE",protocol=args.protocol,
                original_protocol=ref["protocol"],primary114=main,secondary111=second,
                rank_gate=bool(gate),point_replay_updates=96,detector_updates=0,
                AP=True,DEV=False,TEST=False,uses_GT_for_attribution=True,
                elapsed_s=time.perf_counter()-start,cuda_peak_bytes=env.torch.cuda.max_memory_allocated())
    args.output.mkdir()
    with (args.output/"RESULT.json").open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(result,sort_keys=True,allow_nan=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("asymmetric-source","selective-source","teacher"):
        p.add_argument("--"+n,type=Path,required=True)
    first,_=p.parse_known_args()
    a=module(first.asymmetric_source,ASYMMETRIC_SHA,"_rank_quota_asym")
    base=module(first.selective_source,a.SELECTIVE_SHA,"_rank_quota_selective")
    teacher=base.teacher_module(first.teacher)
    teacher.common_args(p)
    for n in ("asymmetric-math","quota-math","local-source","local-math","policy","reference"):
        p.add_argument("--"+n,type=Path,required=True)
    args=p.parse_args()
    asym=module(args.asymmetric_math,ASYM_MATH_SHA,"_rank_quota_asym_math")
    math=module(args.quota_math,OUTCOME_MATH_SHA,"_rank_quota_math")
    prior=module(args.local_source,base.LOCAL_SHA,"_rank_quota_prior")
    local=module(args.local_math,base.MATH_SHA,"_rank_quota_local")
    policy=module(args.policy,a.POLICY_SHA,"_rank_quota_policy")
    run(args,a,base,teacher,prior,local,policy,asym,math)

