"""Nested source calibration of unchanged whole-candidate balanced swaps."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import math as scalar
import os
from pathlib import Path
from types import ModuleType

import numpy as np

PROTOCOL="690bbd6032f095956f197529a388ce768d1ebbba976d072488eebbe46e72d09c"
COMMON="f26da339c97bb4e22044d85e1cc4abb78e117a6bc3fde69500c08e2fca155ae5"
COMMON_RESULT="7786fee336530a155a73bb982e1aaf502a0b273e76b6b915dd7de3cc0292945e"
MODES=("raw","unguarded","guarded")


def need(ok,reason):
    if not ok:
        raise ValueError(reason)


def pin(path,digest):
    data=path.read_bytes()
    need(sha256(data).hexdigest()==digest,"Exact immutable source or aggregate")
    return data


def load(path,digest,name):
    out=ModuleType(name)
    out.__file__=str(path)
    exec(compile(pin(path,digest),str(path),"exec"),out.__dict__)
    return out


def geometry_adapter(path,digest,pair):
    raw=pin(path,digest).decode()
    start=raw.index("def fixed_geometry(")
    end=raw.index("\n\ndef score(",start)
    section=raw[start:end]
    for before,after in (('if m=="paired":','if m=="guarded":'),
                         ('groups[:,3,0]','groups[:,MODES.index("guarded"),0]'),
                         ('paired_minus=effects','guarded_minus=effects')):
        need(section.count(before)==1,"Exact fixed geometry readout adapter")
        section=section.replace(before,after,1)
    pair.MODES=MODES
    pair.GEOMETRY=(*MODES,"oracle")
    exec(compile(section,str(path),"exec"),pair.__dict__)


def calibrated_cutoff(residuals):
    need(all(scalar.isfinite(x) for x in residuals),"Finite source maxima")
    n=len(residuals)
    k=(9*(n+1)+9)//10
    return None if k>n else max(0.,sorted(residuals)[k-1])


def keep_edit(h,cutoff):
    return cutoff is not None and h>cutoff


def get_actions(slots,fold,head,center,scale,learner,geometry):
    result={}
    for i,s in enumerate(slots):
        if s["fold"]!=fold or s["data"] is None:
            continue
        f=s["data"]["features"]
        gains=learner.gains(head,f,f,center,scale)
        actions=geometry.choose_pairs(gains,min(8,s["area"]//10,len(gains)))
        if not actions:
            continue
        h=float(np.mean(gains[actions],dtype=np.float64))
        q=s["truth"]
        actual=None if q is None else float(np.mean(q[actions,0]-q[actions,1],dtype=np.float64))
        need(h>0 and scalar.isfinite(h) and (actual is None or scalar.isfinite(actual)),
             "Finite same natural action gain")
        result[i]=(actions,h,actual)
    return result


def source_residuals(slots,groups,actions):
    residuals={group:[] for group in groups}
    for i,(_,predicted,actual) in actions.items():
        if actual is not None:
            residuals[slots[i]["group"]].append(predicted-actual)
    return [max(v) if v else 0. for v in residuals.values()],sum(bool(v) for v in residuals.values())


def quality(actions,kept):
    result=Counter()
    prediction,truth=[],[]
    for i,(_,h,actual) in actions.items():
        if kept is not None and i not in kept:
            continue
        result["all"]+=1
        if actual is not None:
            result.update(valid=1,negative=int(actual<0),zero=int(actual==0),positive=int(actual>0))
            prediction.append(h)
            truth.append(actual)
    return {k:result[k] for k in ("all","valid","negative","zero","positive")} | {
        "mean_predicted_valid":float(np.mean(prediction)) if prediction else None,
        "mean_true_valid":float(np.mean(truth)) if truth else None}


def nested(slots,learner,geometry):
    chosen={m:[[] for _ in slots] for m in MODES}
    groups=[{s["group"] for s in slots if s["fold"]==f} for f in range(5)]
    need(sum(map(len,groups))==len(set.union(*groups))==109,"Exact disjoint source folds")
    learner.HEADS=("full",)
    audits=[]
    for fold in range(5):
        cal=(fold+1)%5
        train=[s for s in slots if s["fold"]!=cal]
        models,center,scale,audit=learner.train(train,fold)
        need(audit["heads"]["full"]["steps"]==200,"Exact point update budget")
        cal_actions=get_actions(slots,cal,models["full"],center,scale,learner,geometry)
        cal_residuals,cal_eligible=source_residuals(slots,sorted(groups[cal]),cal_actions)
        cutoff=calibrated_cutoff(cal_residuals)
        test_actions=get_actions(slots,fold,models["full"],center,scale,learner,geometry)
        test_residuals,test_eligible=source_residuals(slots,sorted(groups[fold]),test_actions)
        kept=set()
        for i,(actions,h,_) in test_actions.items():
            chosen["unguarded"][i]=actions
            if keep_edit(h,cutoff):
                chosen["guarded"][i]=actions
                kept.add(i)
        audits.append(dict(outer_fold=fold,calibration_fold=cal,train_folds=[j for j in range(5) if j not in (fold,cal)],
                           training=audit,calibration_sources=len(cal_residuals),calibration_eligible_sources=cal_eligible,
                           cal_residual_min=min(cal_residuals),cal_residual_max=max(cal_residuals),
                           cutoff=cutoff,abstain_all=cutoff is None,test_sources=len(test_residuals),
                           test_eligible_sources=test_eligible,
                           test_source_exceedances=sum(cutoff is not None and r>cutoff for r in test_residuals),
                           unguarded_quality=quality(test_actions,None),guarded_quality=quality(test_actions,kept)))
        print(json.dumps(dict(event="DFINE_FIT109_STRICT_GUARD_FOLD_COMPLETE",fold=fold,steps=200,
                              cutoff=cutoff,eligible_calibration_sources=cal_eligible,retained=len(kept)),
                         sort_keys=True),flush=True)
    return chosen,audits


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("protocol","common","common-result","gray","pair","pair-math","training","base","capture","accepted",
                "prelabel","wait","fit-gt","code","shapely-site","official","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    a=parser.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="-1","Exact CPU-only scope")
    pin(a.protocol,PROTOCOL)
    common=load(a.common,COMMON,"_closed_common_guard")
    prev=json.loads(pin(a.common_result,COMMON_RESULT))
    need(not prev["advance"] and prev["exact_natural_replay"] and not prev["DEV"] and not prev["TEST"],
         "Exact stopped prior scope")
    gray=common.load(a.gray,common.GRAY,"_closed_gray_guard")
    pair=common.load_pair(a.pair,gray.PAIR)
    geometry_adapter(a.pair,gray.PAIR,pair)
    base=common.load(a.base,pair.BASE,"_accepted_fit109_guard")
    geometry=common.load(a.pair_math,gray.PAIR_MATH,"_closed_pair_math_guard")
    learner=gray.ablation_training(a.training)
    official=common.load(a.official,pair.OFFICIAL,"_stock_coco_guard")
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    _,blocks=base.read_core(a)
    a.prelabel_sha=gray.OLD_PRELABEL
    a.wait_sha=gray.OLD_WAIT
    old,planned=pair.sealed_points(a,base,geometry,blocks)
    a.output.mkdir()
    pre=base.seal(a.output/"PRELABEL_SAFE.json",base.encoded(dict(status="DFINE_FIT109_STRICT_GUARD_PRELABEL",
                  FIT=True,DEV=False,TEST=False,GT_opened=False,AP=False,protocol_sha256=PROTOCOL,
                  prior_common_sha256=COMMON_RESULT,old_prelabel_sha256=gray.OLD_PRELABEL,counts=old["counts"])))
    gt,anns=pair.read_labels(a,base,blocks)
    slots,scope=pair.fixed_slots(a,base,blocks,planned,gt,anns)
    chosen,audits=nested(slots,learner,geometry)
    gray.MODES=common.MODES=MODES
    coco,budgets=gray.materialize(slots,chosen,base,official,gt)
    fixed=pair.fixed_geometry(slots,chosen,geometry)
    transitions,rows=common.threshold(slots,chosen)
    common.validate_transitions(transitions,fixed)
    witness=base.seal(a.output/"STRICT_TRANSITIONS_PRIVATE.jsonl",b"".join(base.encoded(r) for r in rows))
    need(coco["raw"]==prev["common_coco"]["raw"] and
         fixed["metrics"]["raw"]==prev["common_geometry"]["metrics"]["raw"] and
         fixed["metrics"]["oracle"]==prev["common_geometry"]["metrics"]["oracle"] and scope==prev["match_scope"],
         "Exact original score, geometry, oracle and matching witnesses")
    change=fixed["guarded_minus"]["raw"]
    gates=dict(AP_gain=coco["guarded"]["AP"]-coco["raw"]["AP"]>=.50,
               geometry_gain=change["delta_points"]>=.20 and change["interval95_points"][0]>0.,
               AP75_retained=coco["guarded"]["AP75"]>=coco["raw"]["AP75"]-1e-10,
               zero_original_strict90_loss=transitions["guarded"]["lost"]==0)
    result=dict(status="DFINE_FIT109_STRICT_GUARD_ACTUAL",FIT=True,DEV=False,TEST=False,GT_opened=True,AP=True,
                detector_forwards=0,detector_optimizer_updates=0,point_optimizer_updates=1000,
                protocol_sha256=PROTOCOL,prelabel_sha256=pre,prior_common_sha256=COMMON_RESULT,
                transition_witness_sha256=witness,cohorts=coco,budgets=budgets,fixed_geometry=fixed,
                strict90_transitions=transitions,match_scope=scope,folds=audits,gates=gates,
                advance=all(gates.values()),no_distribution_free_claim=True)
    digest=base.seal(a.output/"RESULT_SAFE.json",base.encoded(result))
    print(json.dumps(dict(status=result["status"],result_sha256=digest,gates=gates,advance=result["advance"]),
                     sort_keys=True),flush=True)


if __name__=="__main__":
    main()
