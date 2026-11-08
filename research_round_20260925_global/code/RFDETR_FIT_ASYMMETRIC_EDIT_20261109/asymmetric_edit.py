"""Same-eight-edit signed allocation on fixed SSDD FIT candidates."""
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

SELECTIVE_SHA="4840f5511d4225ef1701e88f0b96da2669149ed529b7c071b4dd908d882b074b"
POLICY_SHA="44c79d7459a59d9dd601d5192c9743ecce815725dd24e1456ba14c70573e0031"
REFERENCE_SHA="17c79b4645e70d585d48ee1fab33048fc4f65bac0e4ed48a44fadb0bf0ec36a4"
ARMS=("native","balanced","static_ratio","raw_top8","calibrated_top8","signed8","signed_or_keep")
ORACLES=("balanced_prefix","free_split_prefix","balanced_perfect","free_split_perfect","free_split_or_keep")


def need(ok,message):
    if not ok:
        raise ValueError(message)


def pin(path,digest,name):
    raw=path.read_bytes()
    need(sha256(raw).hexdigest()==digest,"Exact published source digest")
    module=ModuleType(name)
    module.__file__=str(path)
    sys.modules[name]=module
    exec(compile(raw,str(path),"exec"),module.__dict__)
    return module


def prefixes(c,index,point,teacher,base,local,policy):
    """Proposes fixed candidate prefixes using only the original outputs."""
    records=[]
    for j,q in enumerate(c["query"]):
        if not bool(c["ctx"]["active"][q]):
            continue
        xy,_,_,distance,inside=local.boundary_samples(c["native"][j],128,5201+100*index+q)
        if int(inside.sum())<4 or int((~inside).sum())<4:
            continue
        records.append((j,q,xy,distance,inside))
    if not records:
        return []
    planes=[]
    queries=[x[1] for x in records]
    for k in range(0,len(queries),16):
        planes.extend(teacher.render(c["z"][queries[k:k+16]],c["hw"]).detach().cpu().numpy())
    features=[]
    for (j,_,xy,distance,inside),plane in zip(records,planes,strict=True):
        area=float(np.log1p(np.count_nonzero(c["native"][j])))
        features.append(base.logit_inputs(plane,xy,distance,inside,area,c["scores"][j],c["rank"][j]))
    values=np.split(point(np.concatenate(features)),np.cumsum([len(x) for x in features])[:-1])
    out=[]
    for (j,_,xy,_,inside),z in zip(records,values,strict=True):
        pixel=xy[:,1]*c["hw"][1]+xy[:,0]
        indices=policy.choose_points(pixel,inside,z,8)
        out.append(dict(j=j,pixels=pixel,inside=inside,indices=indices))
        out[-1]["scores"]=tuple(z[i] for i in indices)
    return out


def calibration(env,teacher,base,local,policy,items,point,role):
    rows=[]
    for i,item in enumerate(items,1):
        c=env.image(item)
        r=prefixes(c,item["dataset_index"],point,teacher,base,local,policy)
        _,truth,matches,_=env.matches(item,c)
        lookup=dict(matches)
        for a in r:
            if a["j"] not in lookup:
                continue
            y=truth[lookup[a["j"]]].ravel()[a["pixels"]]
            labels=y!=a["inside"]
            rows.append((item["image_sha256"],a["scores"],tuple(labels[k] for k in a["indices"])))
        if i%50==0 or i==len(items):
            print(json.dumps(dict(event="ASYMMETRIC_FIT_PROGRESS",role=role,images=i)),flush=True)
    return rows


def calibrated_rows(rows,side):
    g=Counter(x[0] for x in rows)
    z=np.concatenate([x[1][side] for x in rows])
    y=np.concatenate([x[2][side] for x in rows])
    w=np.concatenate([np.full(len(x[1][side]),1/(len(g)*g[x[0]]*len(x[1][side]))) for x in rows])
    return z,y,w


def fit_calibrators(fit,hold,math):
    models=[]
    stats=[]
    for side in (0,1):
        train=calibrated_rows(fit,side)
        apply=math.fit_reliability(*train)
        z,y,w=calibrated_rows(hold,side)
        p=apply(z)
        raw=1/(1+np.exp(-np.clip(z,-50,50)))
        stats.append(dict(fit_points=len(train[0]),held_points=len(z),held_mean_prediction=float(w@p),
                          held_observed=float(w@y),held_brier=float(w@((p-y)**2)),
                          uncalibrated_brier=float(w@((raw-y)**2))))
        models.append(apply)
    return models,stats


def feasible_count(k,add,delete):
    return min(max(k,max(0,8-len(delete))),min(8,len(add)))


def fit_global_ratio(rows,math):
    values=[]
    for k in range(9):
        total=0
        for _,_,labels in rows:
            a,d=labels
            use=feasible_count(k,a,d)
            total+=math.signed_gain(a[:use],False).sum()+math.signed_gain(d[:8-use],True).sum()
        values.append((k,total))
    return min(values,key=lambda x:(-x[1],abs(x[0]-4),x[0]))[0]


def decisions(record,models,math,global_ratio):
    a,d=record["scores"]
    p=models[0](a)
    q=models[1](d)
    k_raw,_=math.choose_split(a,d)
    k_prob,_=math.choose_split(p,q)
    k_signed,value=math.choose_split(math.signed_gain(p,False),math.signed_gain(q,True))
    return dict(balanced=4,static_ratio=feasible_count(global_ratio,a,d),raw_top8=k_raw,calibrated_top8=k_prob,signed8=k_signed,
                signed_or_keep=k_signed if value>0 else None)


def oracle(record,truth,slack,math):
    y=truth.ravel()[record["pixels"]]
    correct=y!=record["inside"]
    i,j=record["indices"]
    a=math.signed_gain(correct[i],False)
    d=math.signed_gain(correct[j],True)
    p=np.sort(math.signed_gain(correct[~record["inside"]],False))[::-1]
    q=np.sort(math.signed_gain(correct[record["inside"]],True))[::-1]
    _,prefix=math.choose_split(a,d)
    _,perfect=math.choose_split(p,q)
    gains=(a[:4].sum()+d[:4].sum(),prefix,p[:4].sum()+q[:4].sum(),perfect,max(0,prefix))
    return {k:(int(v),int(slack+v>=0)) for k,v in zip(ORACLES,gains,strict=True)}


def stats(geometry,mask,edited,truth):
    before=geometry.signed_counts(mask,truth)
    after=geometry.signed_counts(edited,truth)
    return dict(fp_delta=after["fp"]-before["fp"],fn_delta=after["fn"]-before["fn"],
                strict_loss=int(before["strict"] and not after["strict"]),
                strict_gain=int(after["strict"] and not before["strict"]),
                correct_edits=int(np.count_nonzero((mask!=edited)&(edited==truth))),
                margin=before["error"]-after["error"],both_reduced=int(after["fp"]<before["fp"] and after["fn"]<before["fn"]))


def run(args,base,teacher,prior,local,policy,math):
    start=time.perf_counter()
    raw=args.reference.read_bytes()
    need(sha256(raw).hexdigest()==REFERENCE_SHA,"Exact closed FIT baseline and balanced endpoint")
    reference=json.loads(raw)
    env=teacher.Environment(args)
    roles=base.parts(env.fit)
    point_rows=base.point_training(env,teacher,prior,local,roles["point_fit"])
    point,loss=base.train_model(env.torch,np.concatenate([x["x"] for x in point_rows]),
                                np.concatenate([x["y"] for x in point_rows]),prior.weights(point_rows),True)
    need(loss==reference["point_fit_loss"] and len(point_rows)==reference["point_fit_pairs"]
         and sum(len(r["y"]) for r in point_rows)==reference["point_fit_points"],"Exact point ranking replay")
    cal=calibration(env,teacher,base,local,policy,roles["gate_fit"],point,"probability_fit")
    check=calibration(env,teacher,base,local,policy,roles["gate_cal"],point,"probability_check")
    models,reliability=fit_calibrators(cal,check,math)
    global_ratio=fit_global_ratio(cal,math)
    pred={k:[] for k in ARMS}
    diag={k:Counter() for k in ARMS}
    oracle_totals={k:Counter() for k in ORACLES}
    group_diffs=defaultdict(list)
    n_oracle_gain=0
    n_eligible_matched=0
    identity=[]
    for i,item in enumerate(env.hold,1):
        c=env.image(item)
        proposed=prefixes(c,item["dataset_index"],point,teacher,base,local,policy)
        native=c["native"]
        original_rle=env.s.coerce_rles(native,env.codec)
        edited={k:{} for k in ARMS}
        for row in proposed:
            j=row["j"]
            a,d=row["indices"]
            pixels=row["pixels"]
            for arm,k in decisions(row,models,math,global_ratio).items():
                if k is None:
                    continue
                edited[arm][j]=math.edit_mask(native[j],pixels[a[:k]],pixels[d[:8-k]])
                diag[arm].update({"n_add":k,"n_delete":8-k,f"n_add_{k}":1,
                                  "newly_empty":int(native[j].any() and not edited[arm][j].any())})
        iid=item["image_id"]
        identity.extend((iid,rank,q,score.hex()) for rank,q,score in zip(c["rank"],c["query"],c["scores"],strict=True))
        for arm,m in edited.items():
            rle=list(original_rle)
            for j,p in m.items():
                rle[j]=env.codec.encode(np.asfortranarray(p.astype(np.uint8)))
            pred[arm].extend(dict(image_id=iid,category_id=1,score=s,segmentation=x)
                             for s,x in zip(c["scores"],rle,strict=True))
        # The GT below does not influence any exported action.
        _,truth,matches,_=env.matches(item,c)
        matched=dict(matches)
        for arm,m in edited.items():
            diag[arm].update(edited_rows=len(m),matched_edits=sum(j in matched for j in m),
                             unmatched_edits=sum(j not in matched for j in m))
            for j,p in m.items():
                if j in matched:
                    diag[arm].update(stats(env.geometry,native[j],p,truth[matched[j]]))
        for row in proposed:
            j=row["j"]
            if j not in matched:
                continue
            gt=truth[matched[j]]
            before=env.geometry.signed_counts(native[j],gt)
            slack=before["area"]-before["error"]
            fixed=oracle(row,gt,slack,math)
            n_eligible_matched+=1
            n_oracle_gain+=int(fixed["free_split_prefix"][0]-fixed["balanced_prefix"][0]>=19)
            for arm,(gain,strict) in fixed.items():
                oracle_totals[arm].update(margin=gain,strict_gain=int(strict and not before["strict"]),
                                          strict_loss=int(before["strict"] and not strict))
            actual=stats(env.geometry,native[j],edited["signed8"][j],gt)["margin"]
            paired=stats(env.geometry,native[j],edited["balanced"][j],gt)["margin"]
            group_diffs[item["image_sha256"]].append(actual-paired)
        if i%25==0 or i==len(env.hold):
            print(json.dumps(dict(event="ASYMMETRIC_FIT_PROGRESS",role="fit_holdout",images=i)),flush=True)
    env.close()
    digest=sha256(json.dumps(identity,separators=(",",":")).encode()).hexdigest()
    need(digest==reference["all_candidate_identities_sha256"] and len(identity)==9082,"Exact original candidate population")
    keys=[(a,1,d) for a,_,_,d in identity]
    need(all([(r["image_id"],r["category_id"],r["score"].hex()) for r in v]==keys
             for v in pred.values()),"Original score bits and row order")
    ids=[x["image_id"] for x in env.hold]
    keep=set(ids)
    gt=dict(info=env.gt["info"],images=[x for x in env.gt["images"] if x["id"] in keep],
            annotations=[x for x in env.gt["annotations"] if x["image_id"] in keep],categories=env.gt["categories"])
    metrics={k:env.s.fit_stats(gt,v,ids) for k,v in pred.items()}
    need(metrics["native"]==reference["metrics"]["native"] and metrics["balanced"]==reference["metrics"]["forced_all"]
         and n_eligible_matched==reference["diagnostics"]["forced_all"]["matched_edits"]
         and all(diag[k]["edited_rows"]==5684 and diag[k]["n_add"]+diag[k]["n_delete"]==45472
                 for k in ("balanced","static_ratio","raw_top8","calibrated_top8","signed8")),"Exact replay and identical eight-edit budgets")
    ds=np.array([np.mean(group_diffs[k]) for k in sorted(group_diffs)])
    bootstrap=np.random.default_rng(5201).integers(0,len(ds),size=(2048,len(ds)))
    ci=np.quantile(ds[bootstrap].mean(1),[.025,.975]).tolist()
    oracle_gate=(10*n_oracle_gain>=n_eligible_matched and
                 oracle_totals["free_split_prefix"]["strict_gain"]-oracle_totals["balanced_prefix"]["strict_gain"]>=5)
    m=metrics["signed8"]
    b=metrics["balanced"]
    gate=(oracle_gate and m["AP"]-b["AP"]>=.003 and m["AP75"]>=b["AP75"] and m["AP90"]-b["AP90"]>=.01
          and all(m["AP"]-metrics[k]["AP"]>=.001 for k in ("static_ratio","raw_top8","calibrated_top8"))
          and diag["signed8"]["strict_loss"]<=diag["balanced"]["strict_loss"] and ci[0]>0)
    result=dict(status="SSDD_FIT_ASYMMETRIC_EDIT_COMPLETE",protocol=args.protocol,reference_protocol=reference["protocol"],
                metrics=metrics,geometry=diag,calibration=reliability,calibration_fit_matches=len(cal),
                calibration_check_matches=len(check),global_ratio=global_ratio,oracle=oracle_totals,oracle_gain_rows=n_oracle_gain,
                oracle_gate=bool(oracle_gate),eligible_matched=n_eligible_matched,
                signed_minus_balanced_margin=dict(groups=len(ds),mean=float(ds.mean()),percentile95=ci),
                gate=bool(gate),images=len(ids),point_replay_updates=96,detector_updates=0,
                original_candidates_sha256=digest,uses_GT_for_oracle=True,truth_used_for_holdout_decisions=False,AP=True,DEV=False,TEST=False,
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
    p.add_argument("--selective-source",type=Path,required=True)
    p.add_argument("--teacher",type=Path,required=True)
    first,_=p.parse_known_args()
    base=pin(first.selective_source,SELECTIVE_SHA,"_asymmetric_selective")
    teacher=base.teacher_module(first.teacher)
    teacher.common_args(p)
    for n in ("local-source","local-math","policy","asymmetric-math","reference"):
        p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--asymmetric-sha",required=True)
    a=p.parse_args()
    prior=pin(a.local_source,base.LOCAL_SHA,"_asymmetric_prior")
    local=pin(a.local_math,base.MATH_SHA,"_asymmetric_local")
    policy=pin(a.policy,POLICY_SHA,"_asymmetric_policy")
    math=pin(a.asymmetric_math,a.asymmetric_sha,"_asymmetric_math")
    run(a,base,teacher,prior,local,policy,math)

