"""Shared-capacity comparison of continuous occupancy and decision-focused supervision."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
from pycocotools import mask as codec


ARMS = ("bce","pairwise","spo")
OLD_PAPER = "c791b3eb637acf5e30341768a53fa1acb8a3c8db5d6153717474b0b6958fc319"
VERIFIER_SHA = "daa3f6dc7751fe546700b8297b89c0f7295238b3876631f073c8f863d72d449f"


def load_local(path):
    raw=path.read_bytes()
    mod=ModuleType("_fit109_rank_"+path.stem)
    mod.__file__=str(path)
    exec(compile(raw,str(path),"exec"),mod.__dict__)
    return mod,sha256(raw).hexdigest()


def tensor_bank(rows):
    # Keep the original endpoint pairing in the feature tensor. Independently
    # sort the two output score sides only after the paired forward.
    x=np.zeros((len(rows),32,2,14),np.float32)
    y=np.zeros((len(rows),32,2),np.float32)
    order=np.tile(np.arange(32).reshape(1,32,1),(len(rows),1,2))
    lengths=np.array([len(r["coordinates"]) for r in rows],np.int64)
    for i,r in enumerate(rows):
        n=lengths[i]
        x[i,:n]=r["old14"]
        if r["truth"] is not None:
            y[i,:n]=r["truth"]
        order[i,:n]=np.argsort(r["coordinates"],axis=0,kind="stable")
    return x,y,order,lengths


def fit(rows,folds,protocol,objective,head_cls):
    x,y,order,lengths=tensor_bank(rows)
    opts=protocol
    pred={a:np.zeros(x.shape[:3],np.float32) for a in ARMS}
    timing=Counter()
    logs=[]
    for fold in range(5):
        training=defaultdict(list)
        held=[]
        for i,row in enumerate(rows):
            if folds[row["group"]]==fold:
                held.append(i)
            elif row["role"]=="valid" and lengths[i]:
                assert row["truth"] is not None
                training[row["group"]].append(i)
        names=sorted(training)
        assert held and names and all(folds[g]!=fold for g in names)
        first=np.stack([np.concatenate([x[i,:lengths[i]] for i in training[g]],axis=0).reshape(-1,14).mean(axis=0,dtype=np.float64) for g in names])
        second=np.stack([np.square(np.concatenate([x[i,:lengths[i]] for i in training[g]],axis=0).reshape(-1,14),dtype=np.float64).mean(axis=0) for g in names])
        center=first.mean(axis=0)
        scale=np.sqrt(np.maximum(second.mean(axis=0)-center**2,1e-8))
        feature=torch.from_numpy(((x-center)/scale).astype(np.float32))
        probs=torch.from_numpy(x[:,:,:,0].copy())
        ys=torch.from_numpy(y)
        sorting=torch.from_numpy(order)
        rng=np.random.default_rng(5217+fold)
        schedule=[]
        for _ in range(opts["updates_per_head"]):
            ix=np.array([rng.choice(training[names[j]]) for j in rng.integers(len(names),size=opts["candidate_rows_per_step"])])
            k=np.array([rng.integers(1,min(8,lengths[i])+1) for i in ix])
            assert not any(folds[rows[i]["group"]]==fold for i in ix)
            schedule.append((torch.from_numpy(ix),torch.from_numpy(lengths[ix]),torch.from_numpy(k)))
        torch.manual_seed(5217+fold)
        initial=head_cls(14).state_dict()
        audit=dict(fold=fold,train_groups=len(names),fit_matched_candidates=sum(map(len,training.values())),held_candidates=len(held),arms={})
        for arm in ARMS:
            model=head_cls(14)
            model.load_state_dict(deepcopy(initial))
            optimizer=torch.optim.AdamW(model.parameters(),lr=opts["lr"],weight_decay=opts["weight_decay"])
            start=perf_counter()
            for i,n,k in schedule:
                logits=model(feature[i],feature[i].flip(2),probs[i])
                losses=objective.rank_objectives(logits.gather(1,sorting[i]),ys[i].gather(1,sorting[i]),n,k)
                loss=losses[arm]
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            took=perf_counter()-start
            timing[arm]+=took
            model.eval()
            with torch.inference_mode():
                for pos in range(0,len(held),256):
                    i=held[pos:pos+256]
                    pred[arm][i]=model(feature[i],feature[i].flip(2),probs[i]).sigmoid().numpy()
            audit["arms"][arm]=dict(updates=len(schedule),loss_final=float(loss.detach()),fit_seconds=took)
        logs.append(audit)
        print(json.dumps(dict(event="RANK_SUPERVISION_FOLD_COMPLETE",fold=fold)),flush=True)
    return pred,timing,logs


def evaluate(gt,pools,rows,budgets,pred,rank,helpers,baseline,ctl):
    names=("N",*ARMS)
    lookup={helpers.geometry.identity(r):j for j,r in enumerate(rows)}
    assert len(lookup)==len(rows)==len(budgets)==6738
    groups=defaultdict(list)
    cp={a:[] for a in names}
    tally=Counter()
    for image in gt["images"]:
        iid=image["id"]
        for candidate in pools[iid]:
            key=(iid,candidate["rank"],candidate["query_id"])
            src=candidate["rle"]
            masks=dict.fromkeys(names,src)
            if key in lookup:
                j=lookup[key]
                r=rows[j]
                c=r["coordinates"]
                n=len(c)
                k=budgets[key]["k"]
                row={}
                for a in names:
                    choice=rank.balanced(c,pred[ARMS[0] if a=="N" else a][j,:n],k if a!="N" else 0,k if a!="N" else 0)
                    helpers.geometry.sparse_edit(r,choice)
                    masks[a]=helpers.sparse.xor_rle(src,np.asarray(choice["flipped"],dtype=np.int64))
                    assert int(codec.area(masks[a]))==candidate["area"]
                    tally[a+"_edits"]+=choice["edits"]
                    if r["role"]=="valid":
                        g=helpers.geometry.candidate_geometry(r,choice)
                        row[a]=g["iou"]
                        tally[a+"_strict90_losses"]+=int(g["raw_iou"]>.90 and g["iou"]<=.90)
                if row:
                    groups[r["group"]].append([row[a] for a in names])
            for a in names:
                cp[a].append(dict(image_id=iid,category_id=1,score=candidate["score"],segmentation=masks[a]))
    assert all(tally[a+"_edits"]==97964 for a in ARMS) and tally["N_edits"]==0
    assert all(len(v)==10488 for v in cp.values()) and len(groups)==92 and sum(map(len,groups.values()))==392
    ap={a:baseline.pooled(gt,cp[a],helpers.official) for a in names}
    assert ap["N"]["all12"]==ctl["coco_nonoracle"]["N"]["all12"]
    values=100*np.stack([np.mean(groups[g],axis=0) for g in sorted(groups)])
    means=dict(zip(names,map(float,values.mean(axis=0)),strict=True))
    rng=np.random.default_rng(5217)
    draws=rng.integers(len(values),size=(4096,len(values)))
    delta={}
    checks={}
    for a in names[:-1]:
        d=values[:,-1]-values[:,names.index(a)]
        ci=np.quantile(d[draws].mean(axis=1),[.025,.975]).tolist()
        effect=float(d.mean())
        ap_gain=100*(ap["spo"]["AP"]-ap[a]["AP"])
        delta[a]=dict(ap_points=ap_gain,iou_points=effect,interval=ci)
        checks[a]=dict(AP=ap_gain>=.30,AP75=ap["spo"]["AP75"]>=ap[a]["AP75"])
        if a in ("bce","pairwise"):
            checks[a].update(geometry=effect>=.20,interval=ci[0]>0,
                             strict=tally["spo_strict90_losses"]<=tally[a+"_strict90_losses"])
    checks["prior_Z"]=dict(AP=100*ap["spo"]["AP"]-55.752>=.30)
    return dict(coco=ap,continuous_iou_points=means,spo_minus=delta,counts=dict(tally),checks=checks,
                advance=all(all(p.values()) for p in checks.values()))


def main():
    p=ArgumentParser(allow_abbrev=False)
    for n in ("code","work","run","owner","accepted","old-support","old-score","raw-capture","raw-acceptance",
              "source-cohort","fit-gt","shapely-site","protocol","ranking-source","ranking-result","output"):
        p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--owner-sha256",required=True)
    p.add_argument("--protocol-sha256",required=True)
    p.add_argument("--objective-sha256",required=True)
    a=p.parse_args()
    assert sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1" and not a.output.exists()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    start=perf_counter()
    here=Path(__file__).parent
    v,v_sha=load_local(a.work/"verify_point_fit109.py")
    assert v_sha==VERIFIER_SHA
    obj,obj_sha=load_local(here/"rank_objectives.py")
    assert obj_sha==a.objective_sha256
    rank,rank_sha=load_local(a.ranking_source)
    lock=v.read_doc(a.protocol,a.protocol_sha256)
    assert lock["FIT"] and not lock["DEV"] and not lock["TEST"]
    ctl=v.read_doc(a.ranking_result,lock["prior_rank_result_sha256"])
    assert ctl["source_sha256"]==rank_sha and ctl["contrasts"]["O_B_minus_B"]["points"]>3
    owner=v.checked_terminal(a.owner,a.owner_sha256)
    oldpeer=v.read_doc(a.accepted,OLD_PAPER)
    assert oldpeer["result_sha256"]==owner["result_sha256"]==rank.PRIOR_RESULT and oldpeer["COCO12_bit_exact_all_six_arms"]
    pins=v.read_doc(a.work/"POINTREND_FIT109_MODULE_PINS.json",owner["module_manifest_sha256"])
    runner=v.checked_module(a.work/"run_point_fit109.py",pins["run_point_fit109.py"])
    orig=v.read_doc(a.work/"POINTREND_FIT109_PRELOCK.json",runner.PROTOCOL_SHA)
    previous=a.code/"RFDETR_QUERY_FEATURE_FIT109_20261121"
    old=v.checked_module(previous/"run_fit109_query.py","97fdc3987462aabbcd7d4fbe672c9788a3b969a221243e13d82473093170a15a")
    spot=old.load(a.code/"RFDETR_CORE36_MATCHED_POINT_SPOT_20261120/run_spot.py")
    base_dir=a.code/"RFDETR_MATCHED_POINT_FORMAL_20261120"
    prior,plan=(old.load(base_dir/n) for n in ("run_fit109.py","fit_plan.py"))
    a.capture,a.acceptance,a.acceptance_sha256=a.raw_capture,a.raw_acceptance,old.OLD_RAW_ACCEPTANCE
    roster,_=prior.read_population(a,spot)
    baseline=old.load(previous/"fit109_score.py")
    helpers=baseline.load_helpers(a.code,a.shapely_site)
    pools,rows=runner.collect_old_support(a,SimpleNamespace(old=old,spot=spot),roster,orig)
    budgets,folds=runner.read_budgets(a.old_score,orig,rows,plan,roster,helpers.geometry)
    gt,data=old.attach_fit(a,roster,rows,pools,helpers,spot)
    assert len(gt["images"])==109 and len(gt["annotations"])==496
    headpath=a.code/"DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/fit109_paired_point/train_point.py"
    trainer,trainer_sha=load_local(headpath)
    pred,time,logs=fit(data,folds,lock,obj,trainer.PointHead)
    metric=evaluate(gt,pools,data,budgets,pred,rank,helpers,baseline,ctl)
    digest=runner.write(a.output,dict(status="FIT109_EXPECTED_INTERSECTION_SHARED_CAPACITY_ACTUAL",FIT=True,DEV=False,TEST=False,
                  detector_forwards=0,updates_per_arm={ar:sum(x["arms"][ar]["updates"] for x in logs) for ar in ARMS},
                  past_oracle_sha256=lock["prior_rank_result_sha256"],protocol_sha256=a.protocol_sha256,
                  source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),loss_source_sha256=obj_sha,
                  head_source_sha256=trainer_sha,train_seconds=dict(time),fold_log=logs,wall_seconds=perf_counter()-start,metrics=metric))
    print(json.dumps(dict(status="FIT109_EXPECTED_INTERSECTION_SHARED_CAPACITY_ACTUAL",sha256=digest,advance=metric["advance"])),flush=True)


if __name__=="__main__":
    main()
