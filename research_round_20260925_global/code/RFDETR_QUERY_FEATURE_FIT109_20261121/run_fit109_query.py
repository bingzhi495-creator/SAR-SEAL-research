"""Run the prelocked equal-count FIT109 query feature comparison after capture admission."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
import torch
from pycocotools import mask as codec

PINS = {
    "run_fit109.py": "9f989c3a976642d9e790d65199dda43da4ae6b946d649fa13d1106258e99f84e",
    "fit_plan.py": "1fd0fe8f815b5ad4e01d0d8eeca0aa0e1480433292fc48f94fee17e5837e5c35",
    "run_spot.py": "75ff72a3177e99f06d212a878c8d8921de4db6da8f4249722fb762718083d0ee",
    "fit109_heads.py": "d336e55ad0d9ebfc27e5028aeddec65c6c50174e742f7161bad82c52948f4d5a",
    "fit109_score.py": "31ed8deda184abc44d047dcf24d99c6d3979676f3df299a54d49c871bff2955e",
    "channel_residual.py": "4ca7f2345fd4bc3d0faee39745292df07757a083dcb7f123af3643057a5d21ce",
}
ARMS = dict(zero="Z", ordinary_centered="F", query_conditioned_centered="Q",
            query_conditioned_permuted="QP")
OLD_RAW_ACCEPTANCE = "7f3f6dd112d287623360c8e3a174bae3a12a039c7f5505e4bf7c5d76b2e5ed6f"
FEATURE_SHA = "1cea85eb8c30b11bfa847292c55d2231fb4d16dd2a551c3da52ddffdd8d3830e"
CAPTURE_WORKER_SHA = "9f25e0bfcac25d44258b0b75ec94ad33f03a65feb4a20e896fb0381a3d21dc20"
PROTOCOL = "a96a1d6d2455414f204be22711c5ea0de47841484a843b6e4b3287a1c0f0e967"
PRELABEL_PRIOR = "33fc0e63fa5deff68aa384e03a3f00fd78430e63a7680b7a5a8025e33422acf4"
OLD_HEAD_RESULT = "114ac529e2196fa4f60d8fe56e07381fb6814e0fdb09e7d5615626fd74a85f64"
OLD_HEAD_ACCEPTED = "476b0f3fa748d1d3fd2393bdeb1af2d83e1532995d58b12625d57222e3f2638a"


def need(ok, text):
    if not ok:
        raise ValueError(text)


def pinned(path, digest, limit=1024**2):
    with path.open("rb") as stream:
        buf=stream.read(limit+1)
    need(0<len(buf)<=limit and sha256(buf).hexdigest()==digest,"Pinned input "+path.name)
    return buf


def load(path):
    module=ModuleType("_new_fit109_"+path.stem)
    module.__file__=str(path)
    sys.modules[module.__name__]=module
    raw=pinned(path,PINS[path.name])
    exec(compile(raw,str(path),"exec"),module.__dict__)
    return module


def read_acceptance(args):
    receipt=json.loads(pinned(args.new_capture/"INDEPENDENT_READBACK_SAFE.json",args.capture_acceptance_sha))
    need(receipt["status"]=="RFDETR_FIT109_QUERY_BASIS_INDEPENDENTLY_ACCEPTED"
         and receipt["independent_readback_accepted"] is True and receipt["job_drained"] is True
         and receipt["original_os_exit"]==0 and receipt["entry_sha256"]==CAPTURE_WORKER_SHA
         and receipt["model_forwards"]==105 and receipt["reused_core4"]==4
         and receipt["private_file_count"]==109 and receipt["feature_sha256"]==FEATURE_SHA
         and receipt["prelabel_sha256"]==PRELABEL_PRIOR
         and receipt["FIT"] and not any(receipt[k] for k in ("DEV","TEST","AP","GT_opened")),
         "Actual closed FIT vector capture accepted")
    report=json.loads(pinned(args.new_capture/"RESULT_SAFE.json",receipt["result_sha256"]))
    need(report["private_output_sha256"]==receipt["private_output_sha256"]
         and report["protocol_sha256"]==PROTOCOL and report["head_updates"]==0
         and all(report[k] is True for k in ("exact_original_eligible_logits","full_basis_logits_exact",
                                            "original_candidate_and_mask_exact","exact_dense_sample_witness")),
         "Exact original carrier observations")
    pinned(args.new_capture/"PRELABEL_SAFE.json",PRELABEL_PRIOR)
    return receipt,report


def read_vectors(args, roster, spot, residual, receipt):
    records, entries = [], []
    by_image={}
    features=sha256()
    counts=Counter()
    for item, ordinal, block in roster:
        image_id=item["image_id"]
        pop=spot.native_candidate_rows(block)
        for c in pop:
            c["area"]=int(codec.area(c["rle"]))
        by_image[image_id]=pop
        counts["mapped"]+=len(pop)
        name=f"NATIVE_ENDPOINT_BASIS_{ordinal:02d}_PRIVATE.npz"
        with (args.new_capture/name).open("rb") as f:
            raw=f.read(48*1024**2+1)
        need(0<len(raw)<=48*1024**2,"Bounded vector archive")
        entries.append(dict(ordinal=ordinal,sha256=sha256(raw).hexdigest(),bytes=len(raw)))
        with np.load(BytesIO(raw),allow_pickle=False) as archive:
            need(int(archive["image_id"])==image_id,"Vector archive native image ID")
            ranks, qid, offsets=[archive[k] for k in ("ranks","query_ids","offsets")]
            x, old, layers, spatial, query=[archive[k] for k in ("coords","old14","layers","spatial","projected_queries")]
            need(len(ranks)==len(qid)==len(query) and np.all(np.diff(ranks)>0)
                 and offsets.shape==(len(ranks)+1,) and offsets[0]==0 and offsets[-1]==len(x),
                 "Ordinal native vector archive")
            vector_rows={int(v):i for i,v in enumerate(ranks)}
            seen=set()
            for c in pop:
                area=c["area"]
                if not 32 <= area < 1024:
                    continue
                rank, native_query=c["rank"], c["query_id"]
                extra=dict(image_id=image_id,rank=rank,query_id=native_query,group=item["group"])
                if rank in vector_rows:
                    i=vector_rows[rank]
                    need(qid[i]==native_query,"Original query ID for each vector row")
                    z=slice(offsets[i],offsets[i+1])
                    coords=x[z]
                    v=old[z]
                    need(0<len(coords)<=32, "Nonempty original vector rows")
                    expanded=residual.point_feature_controls(v,spatial[z],query[i],coords,layers[z],
                                                               candidate_id=(image_id,rank,native_query))
                    arms=expanded["features"]
                    permutation=expanded["permutation_source_flat_indices"].reshape(-1)
                    counts["moved_endpoint_ids"]+=int(np.count_nonzero(permutation!=np.arange(2*len(coords))))
                    q=arms["query_conditioned_centered"][:,:,14:]
                    qp=arms["query_conditioned_permuted"][:,:,14:]
                    counts["moved_endpoint_values"]+=int(np.count_nonzero(np.any(q!=qp,axis=-1)))
                    features.update(spot.encode(extra))
                    features.update(coords.tobytes())
                    features.update(v.tobytes())
                    counts["point_sets"]+=1
                    counts["point_pairs"]+=len(coords)
                    seen.add(rank)
                else:
                    coords=np.empty((0,2),dtype=np.int32)
                    v=np.empty((0,2,14),dtype=np.float32)
                    arms={a:np.empty((0,2,30),dtype=np.float32) for a in ARMS}
                records.append(dict(extra,coordinates=coords,old14=v,features=arms,area=area,
                                    cap=min(8,area//10,len(coords))))
            need(seen==set(vector_rows), "All original nonempty supports represented")
    need({k:counts[k] for k in ("mapped","point_sets","point_pairs")}==dict(mapped=10488,point_sets=6421,point_pairs=203903) and len(records)==6738,
         "Complete original 109 native and unchanged support populations")
    need(sha256(spot.encode(entries)).hexdigest()==receipt["private_output_sha256"]
         and features.hexdigest()==FEATURE_SHA, "Closed arrays and historical feature order")
    return records,by_image,dict(endpoints=2*counts["point_pairs"], moved_ids=counts["moved_endpoint_ids"], moved_values=counts["moved_endpoint_values"])


def restore_old(args,plan,trainer,step):
    audit=json.loads(pinned(args.old_heads/"INDEPENDENT_SAFE.json",OLD_HEAD_ACCEPTED))
    result=json.loads(pinned(args.old_heads/"RESULT_SAFE.json",OLD_HEAD_RESULT))
    need(audit["status"]=="RFDETR_MATCHED_POINTS_FIT109_INDEPENDENT_ACCEPTED"
         and audit["fit_result_sha256"]==OLD_HEAD_RESULT and audit["run_natural_exit"]==0 and audit["exact_schedule"]
         and not any(audit[k] for k in ("DEV","TEST","AP")) and result["prelabel_sha256"]==PRELABEL_PRIOR,
         "Original trained point head closure")
    old={}
    for d in result["all_initializations"]:
        if d["seed"]!=5217 or d["fold"] is None:
            continue
        fold=d["fold"]
        need(d["relative_weights"]==f"5217_fold_{fold}.npz" and d["steps"]==200 and d["batch"]==64,
             "Exact old held-fold endpoint")
        models,center,scale=plan.restore_bundle(args.old_heads/"POINT_HEADS_PRIVATE"/d["relative_weights"],
                                               d["weights_sha256"],trainer,step)
        need(step.tensor_hash(models["S"])==d["heads"]["S"]["final_tensor_sha256"],
             "Exact old S restored final tensor hash")
        old[fold]=dict(model=models["S"],center=center,scale=scale,seed=5217,fold=fold,
                       common_batch_sha256=d["common_batch_sha256"],
                       initial_tensor_sha256=d["heads"]["S"]["initial_tensor_sha256"])
    need(set(old)==set(range(5)), "Original five old held-fold heads")
    return old


def seal_native_budgets(output,budgets,spot):
    path=output/"BUDGETS_PRIVATE.jsonl"
    expected={}
    with path.open("xb") as f:
        for key,row in budgets.items():
            record=dict(key=list(key),fold=row["fold"],cap=row["cap"],k=row["k"],inputs_sha256=row["inputs_sha256"],
                        old_q_sha256=sha256(row["old_q"].tobytes()).hexdigest())
            expected[key]=record
            f.write(spot.encode(record))
        f.flush()
        os.fsync(f.fileno())
    digest=sha256(path.read_bytes()).hexdigest()
    with path.open("rb") as f:
        actual=[json.loads(line) for line in f]
    need(actual==list(expected.values()), "Exact original budget replay bytes before annotations")
    return digest


def attach_fit(args,roster,records,by_image,helpers,spot):
    source=json.loads(pinned(args.fit_gt,spot.GT_SHA,32*1024**2))
    need(len(source["images"])==2937 and len(source["annotations"])==9260 and
         [x["id"] for x in source["categories"]]==[1],"Original authorized FIT index")
    source_images={im["id"]:im for im in source["images"]}
    ids=[item["image_id"] for item,_,_ in roster]
    selected=set(ids)
    anns=defaultdict(list)
    rows=defaultdict(list)
    for ann in source["annotations"]:
        if ann["image_id"] in selected:
            anns[ann["image_id"]].append(ann)
    for row in records:
        rows[row["image_id"]].append(row)
    actual=[]
    for item,_,_ in roster:
        i=item["image_id"]
        img=source_images[i]
        need(img["file_name"]==item["native_basename"] and img["height"]==img["width"]==800,
             "FIT label maps to exact source image")
        joined,_=helpers.fit.attach_truth(img,anns[i],by_image[i],rows[i],helpers.truth)
        actual.extend(joined)
    need([helpers.geometry.identity(r) for r in actual]==[helpers.geometry.identity(r) for r in records],
         "Fixed matching retains all eligible identities")
    need(sum(r["truth"] is not None and len(r["old14"])>0 for r in actual)==392,
         "Same original supported supervised candidates")
    gt=dict(images=[source_images[i] for i in ids],
            annotations=[ann for ann in source["annotations"] if ann["image_id"] in selected],
            categories=source["categories"])
    need(len(gt["annotations"])==496,"Exact original source-subset labels")
    return gt,actual


def write_fold(output,fold,held):
    offsets=np.concatenate(([0],np.cumsum([len(r["coordinates"]) for r in held])))
    output_file=output/f"OOF_FOLD_{fold}_PRIVATE.npz"
    payload=dict(keys=np.asarray([[r["image_id"],r["rank"],r["query_id"]] for r in held],dtype=np.int64),
                 offsets=offsets, k=np.asarray([r["k"] for r in held],dtype=np.int8))
    for a, name in ARMS.items():
        payload[name]=np.concatenate([r["probabilities"][a] for r in held])
    with output_file.open("xb") as f:
        np.savez(f,**payload)
        f.flush()
        os.fsync(f.fileno())
    with np.load(output_file,allow_pickle=False) as f:
        need(set(f.files)==set(payload) and all(f[k].tobytes()==v.tobytes() for k,v in payload.items()),
             "Exact held predictions read back at fold closure")
    return sha256(output_file.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ("code","heads","scorer","new-capture","old-capture","old-acceptance","source-cohort",
                 "old-heads","fit-gt","shapely-site","output","protocol"):
        ap.add_argument("--"+name,type=Path,required=True)
    ap.add_argument("--capture-acceptance-sha",required=True)
    a=ap.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1"
         and sys.version_info[:3]==(3,10,13) and torch.__version__=="2.10.0+cu130"
         and np.__version__=="1.26.4","Original pinned CPU runtime")
    need(a.output.is_absolute() and not a.output.exists() and a.output.parent.is_dir(),
         "Exclusive new fit work directory")
    pinned(a.protocol,PROTOCOL,64*1024)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    clock=perf_counter()
    code=a.code
    spot=load(code/"RFDETR_CORE36_MATCHED_POINT_SPOT_20261120/run_spot.py")
    old_root=code/"RFDETR_MATCHED_POINT_FORMAL_20261120"
    prev=load(old_root/"run_fit109.py")
    plan=load(old_root/"fit_plan.py")
    heads=load(a.heads)
    score=load(a.scorer)
    residual=load(code/"RFDETR_QUERY_CHANNEL_RESIDUAL_20261120/channel_residual.py")
    need(tuple(heads.ARMS)==tuple(ARMS) and tuple(score.ARMS)==tuple(ARMS.values()),"Exact arm interface mapping")
    receipt,capture_result=read_acceptance(a)
    a.capture,a.acceptance,a.acceptance_sha256=a.old_capture,a.old_acceptance,OLD_RAW_ACCEPTANCE
    roster,_=prev.read_population(a,spot)
    sources=[item["group"] for item,_,_ in roster]
    helpers=score.load_helpers(code,a.shapely_site)
    trainer=helpers.fit.load_trainer(code/"DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/fit109_paired_point/train_point.py")
    record,by_image,permutation=read_vectors(a,roster,spot,residual,receipt)
    need(permutation['moved_values']==capture_result['moved_points'], 'Reproduced real permutation coverage')
    old=restore_old(a,plan,trainer,helpers.fit)
    budgets=heads.seal_oof_budgets(record,sources,old,helpers.fit,plan)
    a.output.mkdir()
    budget_sha=seal_native_budgets(a.output,budgets,spot)
    pre=spot.write(a.output/"PRELABEL_SAFE.json",dict(status="RFDETR_QUERY_FEATURE_FIT109_OOF_BUDGET_SEALED",
                   old_feature_sha256=FEATURE_SHA,new_capture_accepted_sha256=a.capture_acceptance_sha,
                   protocol_sha256=PROTOCOL,native_budget_sha256=budget_sha,groups=len(sources),candidate_budgets=len(budgets),
                   total_swaps=sum(row["k"] for row in budgets.values()),seed=5217,permutation=permutation,
                   FIT=True,DEV=False,TEST=False,GT_opened=False,AP=False))
    print(json.dumps(dict(event="FIT109_PRELABEL_COMPLETE",receipt_sha256=pre)),flush=True)
    gt,training=attach_fit(a,roster,record,by_image,helpers,spot)
    predictions,flips,folds={},{},[]
    for fold,held,audit in heads.fit_oof(training,sources,old,trainer,helpers.fit,plan,budgets):
        saved=write_fold(a.output,fold,held)
        folds.append(dict(fold=fold,heads=audit["heads"],batch=audit["batch"],steps=audit["steps"],
                     shared_pairs=audit["shared_pairs"],train_groups=len(audit["training_groups"]),
                     held_groups=len(audit["held_groups"]),held_candidates=len(held),
                     common_batch_sha256=audit["common_batch_sha256"],prediction_sha256=saved))
        for row in held:
            key=heads.identity(row)
            need(key not in predictions,"One held prediction per candidate")
            predictions[key]={short:row["probabilities"][arm] for arm,short in ARMS.items()}
            flips[key]={short:row["flipped"][arm] for arm,short in ARMS.items()}
        print(json.dumps(dict(event="FIT109_SOURCE_FOLD_CLOSED",fold=fold,candidates=len(held))),flush=True)
    need(len(folds)==5 and len(predictions)==len(budgets)==6738 and not torch.cuda.is_initialized(),
         "All four heads × five held folds")
    result=score.evaluate_fit109(gt,by_image,record,predictions,{key:b["k"] for key,b in budgets.items()},
                                 helpers,expected_flipped=flips)
    need(result["counts"]["mapped_candidates"]==10488 and result["counts"]["eligible"]==6738,
         "Complete original output population scored")
    updates=sum(sum(x["optimizer_steps"] for x in f["heads"].values()) for f in folds)
    need(updates==4000,"Exactly 4000 demonstrated head optimizer updates")
    report=dict(status="RFDETR_QUERY_FEATURE_FIT109_OOF_EVALUATION_COMPLETE",FIT=True,DEV=False,TEST=False,
                GT_opened=True,AP=True,model_forward_calls=0,head_updates=updates,seed=5217,
                prelabel_sha256=pre,old_feature_sha256=FEATURE_SHA,new_capture_accepted_sha256=a.capture_acceptance_sha,
                protocol_sha256=PROTOCOL,native_budget_sha256=budget_sha,
                original_detector_training_exposure=True,source_group_holdout_for_point_head_only=True,
                folds=folds,metrics=result,permutation=permutation,wall_seconds=perf_counter()-clock,
                frozen_basis_capture=dict(forwards=105,reused=4,seconds=capture_result['wall_seconds']),
                entry_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    digest=spot.write(a.output/"RESULT_SAFE.json",report)
    print(json.dumps(dict(status=report["status"],result_sha256=digest,advances=result["gate"]["advance"])),flush=True)


if __name__=="__main__":
    main()

