"""Exact replay and symmetric common-count rendered-intensity diagnostic."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from types import ModuleType

import numpy as np

PROTOCOL="c8e4fae15d6d3dd92b4f992b72230d536369cd33a91f86a7060d8112d12c47c3"
GRAY="b5ebca62babc8c131e4f7cbc64f9b9b8e0c838d25a61cb450556b9aa4648498a"
PREVIOUS="6334218f6e2c93ac6555fc81e9bad9afbff06aea07ca5a3b025c8e2dc0c5a066"
ACTIVE=("gray_zero","gray_shuffled","full")
MODES=("raw",*ACTIVE)


def need(ok,reason):
    if not ok:
        raise ValueError(reason)


def pin(path,digest):
    raw=path.read_bytes()
    need(sha256(raw).hexdigest()==digest,"Exact original immutable source/result")
    return raw


def load(path,digest,name):
    obj=ModuleType(name)
    obj.__file__=str(path)
    exec(compile(pin(path,digest),str(path),"exec"),obj.__dict__)
    return obj


def load_pair(path,digest):
    raw=pin(path,digest).decode()
    line='slot=dict(group=b["group"],fold=p["fold"],area=area,row=r,role=role,data=data,truth=None,raw=None)'
    need(raw.count(line)==1,"Exact matched identity capture seam")
    raw=raw.replace(line,line[:-1]+',gt_id=identity)')
    obj=ModuleType("_fixed_geometry_with_original_identity")
    obj.__file__=str(path)
    exec(compile(raw,str(path),"exec"),obj.__dict__)
    return obj


def common_actions(natural):
    counts=[min(map(len,picks)) for picks in zip(*(natural[m] for m in ACTIVE),strict=True)]
    result={"raw":natural["raw"]}
    for arm in ACTIVE:
        result[arm]=[v[:k] for v,k in zip(natural[arm],counts,strict=True)]
    return result


def transition(before,after,edited):
    if before and after:
        return "edited_retained" if edited else "clean_retained"
    if before:
        return "lost"
    if after:
        return "new"
    return "stable_miss"


def threshold(slots,actions):
    status={arm:Counter() for arm in MODES}
    records=[]
    for i,slot in enumerate(slots):
        if slot["raw"] is None:
            continue
        g,raw=slot["raw"]
        p=slot["area"]
        before=100*raw/(p+g-raw)>=90
        seen={}
        for arm in MODES:
            ids=actions[arm][i]
            q=slot["truth"]
            extra=float((q[ids,0]-q[ids,1]).sum()) if ids else 0.
            inter=float(np.clip(raw+extra,0,min(p,g)))
            after=100*inter/(p+g-inter)>=90
            state=transition(before,after,bool(ids))
            status[arm][state]+=1
            seen[arm]=dict(state=state,swaps=len(ids))
        records.append(dict(image_id=slot["row"]["image_id"],native_rank=i%100+1,
                            original_gt_id=slot["gt_id"],raw_strict90=bool(before),arms=seen))
    keys=("clean_retained","edited_retained","lost","new","stable_miss")
    return {a:{k:int(status[a][k]) for k in keys} for a in MODES},records


def summarize(pair,slots,chosen,math):
    values=pair.fixed_geometry(slots,chosen,math)
    values["full_minus"]=values.pop("paired_minus")
    del values["full_minus"]["full"]
    return values


def validate_transitions(scope,fixed):
    for m,counts in scope.items():
        need(sum(counts.values())==fixed["pairs"] and
             counts["clean_retained"]+counts["edited_retained"]+counts["new"]==fixed["metrics"][m]["strict_90"] and
             counts["clean_retained"]+counts["edited_retained"]+counts["lost"]==fixed["metrics"]["raw"]["strict_90"],
             "Exact same original matches across all strict transitions")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for f in ("protocol","gray","previous","pair","pair-math","training","old","base","capture","accepted","prelabel",
              "wait","fit-gt","code","shapely-site","official","output"):
        p.add_argument("--"+f,type=Path,required=True)
    a=p.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="-1","Exact CPU-only scope")
    pin(a.protocol,PROTOCOL)
    gray=load(a.gray,GRAY,"_exact_previous_gray")
    pair=load_pair(a.pair,gray.PAIR)
    base=load(a.base,pair.BASE,"_accepted_fit109")
    math=load(a.pair_math,gray.PAIR_MATH,"_accepted_point_pair")
    learner=gray.ablation_training(a.training)
    official=load(a.official,pair.OFFICIAL,"_stock_coco")
    old=json.loads(pin(a.old,gray.OLD))
    previous=json.loads(pin(a.previous,PREVIOUS))
    need(previous["full_exact_replay"] and previous["advance"] is False and not previous["DEV"] and not previous["TEST"],
         "Exact prior gray result and preserved original decision")
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    _,blocks=base.read_core(a)
    a.prelabel_sha=gray.OLD_PRELABEL
    a.wait_sha=gray.OLD_WAIT
    receipt,planned=pair.sealed_points(a,base,math,blocks)
    prepared=gray.point_prepare(planned)
    need(prepared==previous["counts"] and prepared["pairs"]==receipt["counts"]["pairs"],
         "Exact same frozen gray point and permutation support")
    a.output.mkdir()
    pre=base.seal(a.output/"PRELABEL_SAFE.json",base.encoded(dict(status="DFINE_FIT109_GRAY_COMMON_COUNT_PRELABEL",
                  protocol_sha256=PROTOCOL,prior_gray_sha256=PREVIOUS,old_points_sha256=gray.OLD_PRELABEL,
                  FIT=True,DEV=False,TEST=False,GT_opened=False,AP=False,counts=prepared)))
    gt,anns=pair.read_labels(a,base,blocks)
    slots,scope=pair.fixed_slots(a,base,blocks,planned,gt,anns)
    natural,training=gray.train_and_choose(slots,math,learner,old)
    pair.MODES=MODES
    pair.GEOMETRY=(*MODES,"oracle")
    native_coco,native_budget=gray.materialize(slots,natural,base,official,gt)
    native_geom=summarize(pair,slots,natural,math)
    need((native_coco,native_budget,native_geom,training)==
         (previous["cohorts"],previous["budgets"],previous["fixed_geometry"],previous["training"]),
         "Exact complete previous natural arms, geometry and training replay")
    common=common_actions(natural)
    coco,budgets=gray.materialize(slots,common,base,official,gt)
    fixed=summarize(pair,slots,common,math)
    need(all(budgets[m]==budgets["full"] for m in ACTIVE) and all(
         len(a)==len(b)==len(c) for a,b,c in zip(*(common[m] for m in ACTIVE),strict=True)),
         "Exact symmetric per-candidate common-count equality")
    native_transitions,nrows=threshold(slots,natural)
    common_transitions,crows=threshold(slots,common)
    validate_transitions(native_transitions,native_geom)
    validate_transitions(common_transitions,fixed)
    witness=base.seal(a.output/"STRICT_TRANSITIONS_PRIVATE.jsonl",b"".join(
        base.encoded(dict(natural=n,common=c)) for n,c in zip(nrows,crows,strict=True)))
    location={}
    for arm in ("gray_zero","gray_shuffled"):
        d=fixed["full_minus"][arm]
        location["AP_vs_"+arm]=coco["full"]["AP"]-coco[arm]["AP"]>=.30
        location["geometry_vs_"+arm]=d["delta_points"]>=.20 and d["interval95_points"][0]>0.
    useful=dict(AP_gain=coco["full"]["AP"]-coco["raw"]["AP"]>=.50,
                geometry_gain=fixed["full_minus"]["raw"]["delta_points"]>=.20,
                AP75_retained=coco["full"]["AP75"]>=coco["raw"]["AP75"]-1e-10,
                strict90_retained=fixed["metrics"]["full"]["strict_90"]>=fixed["metrics"]["raw"]["strict_90"])
    result=dict(status="DFINE_FIT109_GRAY_COMMON_COUNT_ACTUAL",FIT=True,DEV=False,TEST=False,GT_opened=True,AP=True,
                detector_forwards=0,detector_optimizer_updates=0,point_optimizer_updates=3000,exact_natural_replay=True,
                protocol_sha256=PROTOCOL,prior_gray_sha256=PREVIOUS,prelabel_sha256=pre,
                transition_witness_sha256=witness,natural_transitions=native_transitions,
                common_transitions=common_transitions,common_budgets=budgets,common_coco=coco,common_geometry=fixed,
                match_scope=scope,training=training,counts=prepared,location_gates=location,usefulness_gates=useful,
                location_supported=all(location.values()),usefulness_supported=all(useful.values()),
                advance=all(location.values()) and all(useful.values()))
    digest=base.seal(a.output/"RESULT_SAFE.json",base.encoded(result))
    print(json.dumps(dict(status=result["status"],result_sha256=digest,advance=result["advance"],
                          location_supported=result["location_supported"],usefulness_supported=result["usefulness_supported"]),
                     sort_keys=True),flush=True)


if __name__=="__main__":
    main()