"""Candidate-conditional realized quota outcomes on frozen SSDD FIT prefixes."""
import argparse
from collections import Counter,defaultdict
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
REFERENCE_SHA="f1b73c89f1ea25e17fc1ded67ad79aa05ea823dcc76a7f645f108491df6c5775"
ARMS=("native","balanced","static_ratio","raw_top8","calibrated_top8","signed8","margin_head","iou_head","multi_head")


def need(ok,message):
    if not ok:
        raise ValueError(message)


def load_source(path,digest,name):
    raw=path.read_bytes()
    need(sha256(raw).hexdigest()==digest,"Exact ordinary source digest")
    m=ModuleType(name)
    m.__file__=str(path)
    sys.modules[m.__name__]=m
    exec(compile(raw,str(path),"exec"),m.__dict__)
    return m


def row_context(c,row):
    j=row["j"]
    return (int(np.count_nonzero(c["native"][j])),c["scores"][j],c["rank"][j],
            float((~row["inside"]).mean()*2-1))


def matched_rows(env,teacher,base,prior,local,policy,a,items,point,role):
    records=[]
    for i,item in enumerate(items,1):
        c=env.image(item)
        proposals=a.prefixes(c,item["dataset_index"],point,teacher,base,local,policy)
        _,truth,pairs,_=env.matches(item,c)
        match=dict(pairs)
        for row in proposals:
            j=row["j"]
            if j not in match:
                continue
            gt=truth[match[j]]
            correct=gt.ravel()[row["pixels"]]!=row["inside"]
            labels=tuple(correct[x] for x in row["indices"])
            stock=env.geometry.signed_counts(c["native"][j],gt)
            records.append(dict(group=item["image_sha256"],scores=row["scores"],labels=labels,
                                stock=stock,context=row_context(c,row)))
        if i%50==0 or i==len(items):
            print(json.dumps(dict(event="QUOTA_OUTCOME_PROGRESS",role=role,images=i)),flush=True)
    return records


def outcome(row,math):
    s=row["stock"]
    return math.realized(s["area"],s["fp"],s["fn"],*row["labels"])


def feature(row,context,models,math):
    a,d=row["scores"]
    return math.action_features(a,d,models[0](a),models[1](d),*context)


def fit_heads(rows,models,math):
    g=Counter(r["group"] for r in rows)
    x=[]
    y=[]
    weight=[]
    for r in rows:
        k,v=feature(r,r["context"],models,math)
        label=outcome(r,math)
        need(np.array_equal(k,label["k"]),"Exact feasible quota choices")
        x.append(v)
        b=np.flatnonzero(k==4).item()
        y.append(np.column_stack((label["margin"],label["iou"]-label["iou"][b],label["multi"])))
        weight.extend([1/(len(g)*g[r["group"]]*len(k))]*len(k))
    weight=np.array(weight)
    need(abs(weight.sum()-1)<1e-10,"Equal group-to-action regression weights")
    return math.fit_ridge(np.concatenate(x),np.concatenate(y),weight),int(len(weight)),len(g)


def decisions(row,context,models,ridge,a,asym,math,static):
    k,f=feature(row,context,models,math)
    scores=ridge(f)
    raw=a.decisions(row,models,asym,static)
    del raw["signed_or_keep"]
    raw["margin_head"]=math.pick(k,scores[:,0])
    raw["iou_head"]=math.pick(k,scores[:,1])
    raw["multi_head"]=math.pick(k,scores[:,2])
    return raw


def check_heads(rows,models,ridge,a,asym,math,static):
    totals={k:Counter() for k in ARMS[1:]}
    group=defaultdict(list)
    for r in rows:
        label=outcome(r,math)
        k=label["k"]
        ds=decisions(r,r["context"],models,ridge,a,asym,math,static)
        for arm,n in ds.items():
            i=np.flatnonzero(k==n).item()
            totals[arm].update(rows=1,utility=float(label["multi"][i]),margin=float(label["margin"][i]),
                               regret=float(label["multi"].max()-label["multi"][i]),
                               strict_loss=int(label["native_strict"] and not label["strict"][i]),
                               pass75=int(label["pass75"][i]),strict=int(label["strict"][i]))
            if arm=="multi_head":
                group[r["group"]].append(float(label["multi"][i]))
    vals=np.array([np.mean(group[k]) for k in sorted(group)])
    boot=np.random.default_rng(5201).integers(0,len(vals),size=(2048,len(vals)))
    result=dict(rows=len(rows),groups=len(vals),multi_vs_balanced_group_mean=float(vals.mean()),
                percentile95=np.quantile(vals[boot].mean(1),[.025,.975]).tolist(),by_arm=totals)
    return result


def score_role(env,items,role,teacher,base,local,policy,point,models,ridge,a,asym,math,static):
    pred={k:[] for k in ARMS}
    geometry={k:Counter() for k in ARMS}
    ids=[]
    identity=[]
    for i,item in enumerate(items,1):
        c=env.image(item)
        native=c["native"]
        proposed=a.prefixes(c,item["dataset_index"],point,teacher,base,local,policy)
        original=env.s.coerce_rles(native,env.codec)
        edits={k:{} for k in ARMS}
        for row in proposed:
            j=row["j"]
            add,delete=row["indices"]
            px=row["pixels"]
            for arm,n in decisions(row,row_context(c,row),models,ridge,a,asym,math,static).items():
                edits[arm][j]=asym.edit_mask(native[j],px[add[:n]],px[delete[:8-n]])
                geometry[arm].update({"n_add":n,"n_delete":8-n,f"n_add_{n}":1,
                                      "newly_empty":int(native[j].any() and not edits[arm][j].any())})
        iid=item["image_id"]
        ids.append(iid)
        identity.extend((iid,r,q,z.hex()) for r,q,z in zip(c["rank"],c["query"],c["scores"],strict=True))
        for arm,rows in edits.items():
            rle=list(original)
            for j,x in rows.items():
                rle[j]=env.codec.encode(np.asfortranarray(x.astype(np.uint8)))
            pred[arm].extend(dict(image_id=iid,category_id=1,score=s,segmentation=r)
                             for s,r in zip(c["scores"],rle,strict=True))
        # All choices above depend only on model outputs.
        _,truth,pairs,_=env.matches(item,c)
        matched=dict(pairs)
        native_strict=sum(env.geometry.signed_counts(native[j],truth[k])["strict"] for j,k in pairs)
        for arm,rows in edits.items():
            geometry[arm].update(original_matches=len(pairs),original_strict=native_strict,edited_rows=len(rows),
                                 matched_edits=sum(j in matched for j in rows),unmatched_edits=sum(j not in matched for j in rows))
            for j,x in rows.items():
                if j in matched:
                    geometry[arm].update(a.stats(env.geometry,native[j],x,truth[matched[j]]))
        if i%50==0 or i==len(items):
            print(json.dumps(dict(event="QUOTA_OUTCOME_PROGRESS",role=role,images=i)),flush=True)
    keys=[(i,1,z) for i,_,_,z in identity]
    need(all([(r["image_id"],r["category_id"],r["score"].hex()) for r in v]==keys
             for v in pred.values()),"Frozen complete candidate scores/order")
    elig=geometry["balanced"]["edited_rows"]
    need(all(geometry[k]["edited_rows"]==elig and geometry[k]["n_add"]+geometry[k]["n_delete"]==8*elig
             for k in ARMS[1:]),"Exact per-row and population-wide eight-edit budget")
    chosen=set(ids)
    gt=dict(info=env.gt["info"],images=[r for r in env.gt["images"] if r["id"] in chosen],
            annotations=[r for r in env.gt["annotations"] if r["image_id"] in chosen],categories=env.gt["categories"])
    metrics={k:env.s.fit_stats(gt,v,ids) for k,v in pred.items()}
    return dict(images=len(ids),groups=len({x["image_sha256"] for x in items}),predictions=len(identity),eligible=elig,
                metrics=metrics,geometry=geometry,
                identity_sha256=sha256(json.dumps(identity,separators=(",",":")).encode()).hexdigest())


def run(args,a,base,teacher,prior,local,policy,asym,math):
    start=time.perf_counter()
    raw=args.reference.read_bytes()
    need(sha256(raw).hexdigest()==REFERENCE_SHA,"Exact closed P162 reference")
    ref=json.loads(raw)
    env=teacher.Environment(args)
    roles=base.parts(env.fit)
    pt=base.point_training(env,teacher,prior,local,roles["point_fit"])
    point,loss=base.train_model(env.torch,np.concatenate([r["x"] for r in pt]),
                                np.concatenate([r["y"] for r in pt]),prior.weights(pt),True)
    need(len(pt)==623 and sum(len(r["y"]) for r in pt)==142170 and loss==0.3621605634689331,
         "Exact original point ranking and fit loss")
    data=matched_rows(env,teacher,base,prior,local,policy,a,roles["gate_fit"],point,"outcome_fit")
    check=matched_rows(env,teacher,base,prior,local,policy,a,roles["gate_cal"],point,"outcome_check")
    need(len(data)==294 and len(check)==165,"Exact old fit and check populations")
    calibration=[(r["group"],r["scores"],r["labels"]) for r in data]
    models=[asym.fit_reliability(*a.calibrated_rows(calibration,k)) for k in (0,1)]
    static=a.fit_global_ratio(calibration,asym)
    need(static==ref["global_ratio"],"Exact old static quota")
    ridge,action_count,group_count=fit_heads(data,models,math)
    checked=check_heads(check,models,ridge,a,asym,math,static)
    allfit=json.loads(env.cap.exact_bytes(args.fit_index,348469,env.cap.FIT_INDEX_SHA))["items"]
    cal=[x for x in allfit if x["partition"]=="calibration"]
    g=lambda v:{r["image_sha256"] for r in v}
    need(len(cal)==114 and len(g(cal))==109 and not g(cal)&(g(env.fit)|g(env.hold)),"Head-disjoint 114 role")
    first=score_role(env,cal,"historical_calibration114",teacher,base,local,policy,point,models,ridge,a,asym,math,static)
    old=score_role(env,env.hold,"historical_holdout111",teacher,base,local,policy,point,models,ridge,a,asym,math,static)
    env.close()
    need(old["identity_sha256"]==ref["original_candidates_sha256"] and
         all(old["metrics"][k]==ref["metrics"][k] for k in ARMS[:-3]),"Exact original 111 endpoints")
    best=max(m["AP"] for k,m in first["metrics"].items() if k!="multi_head")
    main=first["metrics"]["multi_head"]
    balanced=first["metrics"]["balanced"]
    geom=first["geometry"]
    gain=main["AP"]-best
    gate=(gain>=.003 and main["AP75"]>=balanced["AP75"] and main["AP90"]-balanced["AP90"]>=.01
          and geom["multi_head"]["strict_loss"]<=geom["balanced"]["strict_loss"]
          and geom["multi_head"]["strict_loss"]*20<=geom["multi_head"]["original_strict"]
          and checked["percentile95"][0]>0)
    result=dict(status="SSDD_FIT_QUOTA_OUTCOME_COMPLETE",protocol=args.protocol,original_protocol=ref["protocol"],
                primary114=first,secondary111=old,out_of_fit_check96=checked,fit_matches=len(data),
                fit_actions=action_count,fit_encoded_groups=group_count,ridge_coefficients_per_head=30,
                primary_gain_fraction=gain,gate=bool(gate),point_replay_updates=96,detector_updates=0,
                AP=True,DEV=False,TEST=False,truth_used_for_holdout_decisions=False,
                elapsed_s=time.perf_counter()-start,cuda_peak_bytes=env.torch.cuda.max_memory_allocated())
    args.output.mkdir()
    with (args.output/"RESULT.json").open("x",encoding="utf-8") as f:
        json.dump(result,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(result,sort_keys=True,allow_nan=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("asymmetric-source","selective-source","teacher"):
        p.add_argument("--"+n,type=Path,required=True)
    first,_=p.parse_known_args()
    a=load_source(first.asymmetric_source,ASYMMETRIC_SHA,"_quota_outcome_asymmetric")
    base=load_source(first.selective_source,a.SELECTIVE_SHA,"_quota_outcome_base")
    teacher=base.teacher_module(first.teacher)
    teacher.common_args(p)
    for n in ("asymmetric-math","quota-math","local-source","local-math","policy","reference"):
        p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--quota-math-sha",required=True)
    args=p.parse_args()
    prior=load_source(args.local_source,base.LOCAL_SHA,"_quota_outcome_prior")
    local=load_source(args.local_math,base.MATH_SHA,"_quota_outcome_local")
    policy=load_source(args.policy,a.POLICY_SHA,"_quota_outcome_policy")
    asym=load_source(args.asymmetric_math,ASYM_MATH_SHA,"_quota_outcome_asym_math")
    math=load_source(args.quota_math,args.quota_math_sha,"_quota_outcome_math")
    run(args,a,base,teacher,prior,local,policy,asym,math)

