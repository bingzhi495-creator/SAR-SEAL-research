"""Equal-capacity, equal-budget probe of directional query-logit context."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
from io import BytesIO, StringIO
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
from pycocotools import mask as codec
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


ARMS = ("zero","radial","ordered")
OLD_PAPER = "c791b3eb637acf5e30341768a53fa1acb8a3c8db5d6153717474b0b6958fc319"
VERIFIER_SHA = "daa3f6dc7751fe546700b8297b89c0f7295238b3876631f073c8f863d72d449f"


def load_local(path):
    raw=path.read_bytes()
    mod=ModuleType("_fit109_rank_"+path.stem)
    mod.__file__=str(path)
    exec(compile(raw,str(path),"exec"),mod.__dict__)
    return mod,sha256(raw).hexdigest()


def capture_bank(args, runner, original_scope, rows, pools, sampler):
    accepted, index=runner.admit_capture(SimpleNamespace(capture=args.context_capture,
        capture_readback=args.capture_readback,capture_readback_sha=args.capture_readback_sha),original_scope)
    grouped=defaultdict(list)
    for r in rows:
        grouped[r["image_id"]].append(r)
    assert len(index)==109 and set(index)==set(pools) and grouped.keys()<=index.keys()
    audit=sha256()
    start=perf_counter()
    count=0
    for iid in sorted(index):
        rows_here=grouped.get(iid,[])
        candidate={r["rank"]:r for r in pools[iid]}
        ent=index[iid]["row_archive"]
        raw=runner.pinned(args.context_capture/ent["name"],ent["sha256"],12*1024**2)
        with np.load(BytesIO(raw),allow_pickle=False) as item:
            assert int(item["image_id"])==iid
            ranks,queries,areas,scores,planes=(item[key] for key in ("ranks","query_ids","areas","scores","raw_logits"))
            lookup={int(r):i for i,r in enumerate(ranks)}
        assert len(lookup)==len(ranks)==len(rows_here)
        if not rows_here:
            continue
        chosen=[]
        points=np.zeros((len(rows_here),32,2),dtype=np.int64)
        lengths=[]
        for j,r in enumerate(rows_here):
            i=lookup[r["rank"]]
            assert int(queries[i])==r["query_id"] and float(scores[i])==candidate[r["rank"]]["score"]
            assert int(areas[i])==candidate[r["rank"]]["area"]
            chosen.append(i)
            c=r["coordinates"]
            points[j,:len(c)]=c
            lengths.append(len(c))
        context=sampler.sample_context(planes[np.asarray(chosen)],points,np.array(lengths)).numpy()
        assert context.shape==(len(rows_here),32,2,32) and np.isfinite(context).all()
        for j,r in enumerate(rows_here):
            r["logit_context"]=context[j]
        audit.update(context.tobytes(order="C"))
        count+=len(rows_here)
    assert count==6738
    return dict(candidates=count,capture_readback_sha256=args.capture_readback_sha,
                export_result_sha256=accepted["result_sha256"],context_digest=audit.hexdigest(),
                prep_seconds=perf_counter()-start)


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
    ordered=np.stack([r["logit_context"] for r in rows])
    radial=np.concatenate((np.sort(ordered[...,:8],axis=-1),np.sort(ordered[...,8:],axis=-1)),axis=-1)
    all_features={a:np.concatenate((x,v),axis=-1) for a,v in
                  (("zero",np.zeros_like(ordered)),("radial",radial),("ordered",ordered))}
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
        state14=head_cls(14).state_dict()
        torch.manual_seed(265217+fold)
        initial=head_cls(46).state_dict()
        for key,w in state14.items():
            if key=="network.0.weight":
                initial[key][:,:14]=w[:,:14]
                initial[key][:,46:60]=w[:,14:]
            else:
                initial[key].copy_(w)
        audit=dict(fold=fold,train_groups=len(names),fit_matched_candidates=sum(map(len,training.values())),held_candidates=len(held),arms={})
        for arm in ARMS:
            data=all_features[arm]
            per_group=[np.concatenate([data[i,:lengths[i]] for i in training[g]],axis=0).reshape(-1,46) for g in names]
            center=np.stack([p.mean(axis=0,dtype=np.float64) for p in per_group]).mean(axis=0)
            moment=np.stack([np.square(p,dtype=np.float64).mean(axis=0) for p in per_group]).mean(axis=0)
            scale=np.sqrt(np.maximum(moment-center**2,1e-8))
            feature=torch.from_numpy(((data-center)/scale).astype(np.float32))
            model=head_cls(46)
            model.load_state_dict(deepcopy(initial))
            optimizer=torch.optim.AdamW(model.parameters(),lr=opts["lr"],weight_decay=opts["weight_decay"])
            start=perf_counter()
            for i,n,k in schedule:
                logits=model(feature[i],feature[i].flip(2),probs[i])
                losses=objective.rank_objectives(logits.gather(1,sorting[i]),ys[i].gather(1,sorting[i]),n,k)
                loss=losses["pairwise"]
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
        print(json.dumps(dict(event="CONTEXT_FOLD_COMPLETE",fold=fold)),flush=True)
    return pred,timing,logs


def coco(gt,predictions):
    ref=COCO()
    ref.dataset=deepcopy(gt)
    ref.dataset.setdefault("info",{})
    with redirect_stdout(StringIO()):
        ref.createIndex()
        state=COCOeval(ref,ref.loadRes(deepcopy(predictions)),"segm")
        state.params.imgIds=[im["id"] for im in gt["images"]]
        state.params.catIds=[1]
        state.params.maxDets=[1,10,100]
        state.evaluate()
        state.accumulate()
        state.summarize()
    ap90=state.eval["precision"][8,:,0,0,-1]
    return dict(AP=float(state.stats[0]),AP75=float(state.stats[2]),AP90=float(ap90[ap90>=0].mean()),
                all12=state.stats.tolist())


def evaluate(gt,pools,rows,budgets,pred,rank,helpers,ctl):
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
    ap={a:coco(gt,cp[a]) for a in names}
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
        ap_gain=100*(ap["ordered"]["AP"]-ap[a]["AP"])
        delta[a]=dict(ap_points=ap_gain,iou_points=effect,interval=ci)
        if a!="N":
            checks[a]=dict(AP=ap_gain>=.30,AP75=ap["ordered"]["AP75"]>=ap[a]["AP75"],
                           AP90=ap["ordered"]["AP90"]>=ap[a]["AP90"],
                           geometry=effect>=.20,interval=ci[0]>0,
                           strict=tally["ordered_strict90_losses"]<=tally[a+"_strict90_losses"])
    return dict(coco=ap,continuous_iou_points=means,ordered_minus=delta,counts=dict(tally),checks=checks,
                advance=all(all(p.values()) for p in checks.values()))


def main():
    p=ArgumentParser(allow_abbrev=False)
    for n in ("code","work","run","owner","accepted","old-support","old-score","raw-capture","raw-acceptance",
              "source-cohort","fit-gt","shapely-site","protocol","ranking-source","ranking-result","output",
              "context-capture","capture-readback","feature-receipt"):
        p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--owner-sha256",required=True)
    p.add_argument("--protocol-sha256",required=True)
    p.add_argument("--objective-sha256",required=True)
    p.add_argument("--capture-readback-sha",required=True)
    p.add_argument("--sampler-sha256",required=True)
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
    sampler,sampler_sha=load_local(here/"context_sampler.py")
    assert sampler_sha==a.sampler_sha256
    rank,rank_sha=load_local(a.ranking_source)
    lock=v.read_doc(a.protocol,a.protocol_sha256)
    assert lock["FIT"] and not lock["DEV"] and not lock["TEST"] and lock["capture_readback_sha256"]==a.capture_readback_sha
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
    prep=capture_bank(a,runner,orig,rows,pools,sampler)
    prep_sha=runner.write(a.feature_receipt,dict(status="FIT109_CONTEXT_FEATURES_PRELABEL_COMPLETE",FIT=True,
              DEV=False,TEST=False,GT_opened=False,sampler_source_sha256=sampler_sha,protocol_sha256=a.protocol_sha256,**prep))
    print(json.dumps(dict(event="FIT109_CONTEXT_FEATURES_PRELABEL_COMPLETE",sha256=prep_sha)),flush=True)
    gt,data=old.attach_fit(a,roster,rows,pools,helpers,spot)
    assert len(gt["images"])==109 and len(gt["annotations"])==496
    headpath=a.code/"DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/fit109_paired_point/train_point.py"
    trainer,trainer_sha=load_local(headpath)
    pred,time,logs=fit(data,folds,lock,obj,trainer.PointHead)
    metric=evaluate(gt,pools,data,budgets,pred,rank,helpers,ctl)
    digest=runner.write(a.output,dict(status="FIT109_QUERY_LOGIT_ORDER_CONTEXT_ACTUAL",FIT=True,DEV=False,TEST=False,seed=5217,
                  detector_forwards=0,updates_per_arm={ar:sum(x["arms"][ar]["updates"] for x in logs) for ar in ARMS},
                  past_oracle_sha256=lock["prior_rank_result_sha256"],protocol_sha256=a.protocol_sha256,
                  source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),loss_source_sha256=obj_sha,
                  head_source_sha256=trainer_sha,feature_receipt_sha256=prep_sha,
                  train_seconds=dict(time),fold_log=logs,wall_seconds=perf_counter()-start,metrics=metric))
    print(json.dumps(dict(status="FIT109_QUERY_LOGIT_ORDER_CONTEXT_ACTUAL",sha256=digest,advance=metric["advance"])),flush=True)


if __name__=="__main__":
    main()
