"""Compare identical point heads with BCE, Brier and direct utility losses."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

PROTOCOL="cf36ab557b0a7c59f57e197c0f05da750a25c8e14c12c7152455a804e78d02d5"
COMMON="f26da339c97bb4e22044d85e1cc4abb78e117a6bc3fde69500c08e2fca155ae5"
COMMON_RESULT="7786fee336530a155a73bb982e1aaf502a0b273e76b6b915dd7de3cc0292945e"
ACTIVE=("bce","brier","direct")
MODES=("raw",*ACTIVE)


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


def training_loss(arm,logits,targets):
    if arm=="bce":
        return torch.nn.functional.binary_cross_entropy_with_logits(logits,targets)
    errors=logits.sigmoid()-targets
    if arm=="brier":
        return errors.square().mean()
    if arm=="direct":
        return .5*(errors[:,0]-errors[:,1]).square().mean()
    raise ValueError("Unknown point loss")


def adapt_train(path,digest):
    raw=pin(path,digest).decode()
    seams=(
        ('HEADS=("independent","shuffled","paired")','HEADS=("bce","brier","direct")'),
        ('        partner=np.concatenate([s["data"]["contexts"][arm] for s in train_slots])\n'
         '        partner_t=torch.from_numpy(np.asarray((partner-center)/scale,dtype=np.float32))','        partner_t=own_t'),
        ('            loss=loss_fn(logits,targets[indices])',
         '            loss=loss_fn(logits,targets[indices]) if arm=="bce" else training_loss(arm,logits,targets[indices])'),
    )
    for before,after in seams:
        need(raw.count(before)==1,"Exact loss-only original trainer seam")
        raw=raw.replace(before,after,1)
    learner=ModuleType("_matched_point_utility")
    learner.__file__=str(path)
    learner.training_loss=training_loss
    exec(compile(raw,str(path),"exec"),learner.__dict__)
    return learner


def fit(slots,learner,math,previous):
    actions={m:[[] for _ in slots] for m in MODES}
    readouts={}
    audits=[]
    for fold in range(5):
        models,center,scale,audit=learner.train(slots,fold)
        old=previous["training"][fold]
        need(audit["heads"]["bce"]==old["heads"]["full"] and
             audit["groups"]==old["groups"] and audit["unique_train_pairs"]==old["unique_train_pairs"] and
             all(v["steps"]==200 and v["parameters"]==1321 and
                 v["batch_sha256"]==audit["heads"]["bce"]["batch_sha256"] for v in audit["heads"].values()),
             "Exact BCE replay and same source-weighted training budget")
        for i,s in enumerate(slots):
            if s["fold"]!=fold or s["data"] is None:
                continue
            f=s["data"]["features"]
            gains={arm:learner.gains(model,f,f,center,scale) for arm,model in models.items()}
            cap=min(8,s["area"]//10,len(f))
            for arm,g in gains.items():
                actions[arm][i]=math.choose_pairs(g,cap)
            if s["truth"] is not None:
                q=s["truth"]
                readouts[i]=(q[:,0]-q[:,1],gains)
        audits.append(dict(fold=fold,**audit))
        print(json.dumps(dict(event="DFINE_FIT109_PAIR_UTILITY_FOLD_COMPLETE",fold=fold,steps_per_arm=200),
                         sort_keys=True),flush=True)
    return actions,readouts,audits


def error_readout(readouts,arm,actions=None):
    ps,qs=[],[]
    for i,(q,gains) in readouts.items():
        selection=slice(None) if actions is None else actions[arm][i]
        ps.append(gains[arm][selection])
        qs.append(q[selection])
    p=np.concatenate(ps).astype(np.float64)
    q=np.concatenate(qs)
    if not len(p):
        return dict(pairs=0)
    e=p-q
    return dict(pairs=len(p),mean_predicted=float(p.mean()),mean_true=float(q.mean()),bias=float(e.mean()),
                mae=float(np.abs(e).mean()),rmse=float(np.sqrt(np.square(e).mean())),
                true_negative=int((q<0).sum()),true_zero=int((q==0).sum()),true_positive=int((q>0).sum()))


def score(pair,gray,common,slots,actions,math,base,official,gt):
    coco,budgets=gray.materialize(slots,actions,base,official,gt)
    geometry=pair.fixed_geometry(slots,actions,math)
    geometry["direct_minus"]=geometry.pop("paired_minus")
    del geometry["direct_minus"]["direct"]
    transitions,witness=common.threshold(slots,actions)
    common.validate_transitions(transitions,geometry)
    return dict(cohorts=coco,budgets=budgets,fixed_geometry=geometry,strict90_transitions=transitions),witness


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("protocol","common","common-result","gray","previous","pair","pair-math","training","base","capture",
                "accepted","prelabel","wait","fit-gt","code","shapely-site","official","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    a=parser.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="-1","Exact CPU-only scope")
    pin(a.protocol,PROTOCOL)
    common=load(a.common,COMMON,"_utility_common")
    old_common=json.loads(pin(a.common_result,COMMON_RESULT))
    prior=json.loads(pin(a.previous,common.PREVIOUS))
    need(old_common["exact_natural_replay"] and not old_common["advance"] and prior["full_exact_replay"] and
         not prior["advance"] and not prior["DEV"] and not prior["TEST"],"Preserve prior scope and decisions")
    gray=common.load(a.gray,common.GRAY,"_utility_gray")
    pair=common.load_pair(a.pair,gray.PAIR)
    base=common.load(a.base,pair.BASE,"_utility_fit109")
    math=common.load(a.pair_math,gray.PAIR_MATH,"_utility_pairs")
    learner=adapt_train(a.training,gray.TRAIN)
    official=common.load(a.official,pair.OFFICIAL,"_utility_coco")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    _,blocks=base.read_core(a)
    a.prelabel_sha=gray.OLD_PRELABEL
    a.wait_sha=gray.OLD_WAIT
    old,planned=pair.sealed_points(a,base,math,blocks)
    a.output.mkdir()
    pre=base.seal(a.output/"PRELABEL_SAFE.json",base.encoded(dict(status="DFINE_FIT109_PAIR_UTILITY_PRELABEL",
                  FIT=True,DEV=False,TEST=False,GT_opened=False,AP=False,protocol_sha256=PROTOCOL,
                  prior_common_sha256=COMMON_RESULT,prior_gray_sha256=common.PREVIOUS,
                  old_prelabel_sha256=gray.OLD_PRELABEL,counts=old["counts"])))
    gt,anns=pair.read_labels(a,base,blocks)
    slots,scope=pair.fixed_slots(a,base,blocks,planned,gt,anns)
    actions,readouts,audits=fit(slots,learner,math,prior)
    pair.MODES=gray.MODES=common.MODES=MODES
    common.ACTIVE=ACTIVE
    pair.GEOMETRY=(*MODES,"oracle")
    natural,nrows=score(pair,gray,common,slots,actions,math,base,official,gt)
    need(natural["cohorts"]["bce"]==prior["cohorts"]["full"] and
         natural["cohorts"]["raw"]==prior["cohorts"]["raw"] and
         natural["budgets"]["bce"]==prior["budgets"]["full"] and
         all(natural["fixed_geometry"]["metrics"][key]==prior["fixed_geometry"]["metrics"][prev]
             for key,prev in (("bce","full"),("raw","raw"),("oracle","oracle"))) and
         natural["strict90_transitions"]["bce"]==old_common["natural_transitions"]["full"] and
         scope==prior["match_scope"],"Exact prior natural BCE replay")
    shared=common.common_actions(actions)
    same,srows=score(pair,gray,common,slots,shared,math,base,official,gt)
    need(all(same["budgets"][arm]==same["budgets"]["bce"] for arm in ACTIVE),"Exact same realized action budget")
    witness=base.seal(a.output/"STRICT_TRANSITIONS_PRIVATE.jsonl",b"".join(
        base.encoded(dict(natural=n,common=c)) for n,c in zip(nrows,srows,strict=True)))
    masks=natural["cohorts"]
    geometry=natural["fixed_geometry"]["direct_minus"]
    gates=dict(AP75_retained=masks["direct"]["AP75"]>=masks["raw"]["AP75"]-1e-10,
               zero_original_strict90_loss=natural["strict90_transitions"]["direct"]["lost"]==0)
    for arm in ("raw","bce","brier"):
        gates["AP_vs_"+arm]=masks["direct"]["AP"]-masks[arm]["AP"]>=.50
        d=geometry[arm]
        gates["geometry_vs_"+arm]=d["delta_points"]>=.20 and d["interval95_points"][0]>0
    errors={arm:dict(all_pairs=error_readout(readouts,arm),
                     natural=error_readout(readouts,arm,actions),
                     common=error_readout(readouts,arm,shared)) for arm in ACTIVE}
    result=dict(status="DFINE_FIT109_PAIR_UTILITY_ACTUAL",FIT=True,DEV=False,TEST=False,GT_opened=True,AP=True,
                detector_forwards=0,detector_optimizer_updates=0,point_optimizer_updates=3000,
                protocol_sha256=PROTOCOL,prelabel_sha256=pre,prior_common_sha256=COMMON_RESULT,
                prior_gray_sha256=common.PREVIOUS,transition_witness_sha256=witness,exact_natural_bce_replay=True,
                natural=natural,common=same,utility_error=errors,training=audits,match_scope=scope,
                gates=gates,advance=all(gates.values()))
    digest=base.seal(a.output/"RESULT_SAFE.json",base.encoded(result))
    print(json.dumps(dict(status=result["status"],result_sha256=digest,gates=gates,advance=result["advance"]),
                     sort_keys=True),flush=True)


if __name__=="__main__":
    main()
