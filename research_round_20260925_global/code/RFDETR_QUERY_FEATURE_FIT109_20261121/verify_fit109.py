"""Post-exit readback of the fixed FIT109 point-head OOF comparison.

Replays the old small-head held-fold budget before opening FIT annotations.
Consumes five saved OOF predictions and independently reconstructs their fold
membership and reruns the complete four-arm COCO and frozen geometry scorer.
"""
import argparse
from collections import defaultdict
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from types import ModuleType

import numpy as np


STATUS = "RFDETR_QUERY_FEATURE_FIT109_OOF_EVALUATION_COMPLETE"
VERIFIED = "RFDETR_QUERY_FEATURE_FIT109_OOF_INDEPENDENTLY_ACCEPTED"
MAIN_SHA = "97fdc3987462aabbcd7d4fbe672c9788a3b969a221243e13d82473093170a15a"
SESSION = 50833
ARMS = ("Z","F","Q","QP")


def need(ok, message):
    if not ok:
        raise ValueError(message)


def pinned(path,digest,limit=1024**2):
    with path.open("rb") as f:
        data=f.read(limit+1)
    need(0 < len(data) <= limit and sha256(data).hexdigest() == digest,"Pinned closed artifact "+path.name)
    return data


def exact(actual,expected,encode,message):
    need(encode(actual) == encode(expected),message)


def accept_footer(receipt,digest):
    launch, terminal, footer = (receipt[k] for k in ("launch","terminal","observed_footer"))
    need(launch["session_id"] == SESSION and terminal["exit_code"] == 0
         and terminal.get("session_id") is None and receipt["result_sha256"] == digest
         and footer["status"] == STATUS and footer["result_sha256"] == digest
         and type(footer["advances"]) is bool,"Original closed training handle, result, decision")
    if "output" in terminal:
        lines=terminal["output"].strip().splitlines()
        need(lines and json.loads(lines[-1]) == footer,"Observed footer agrees with original terminal output")
    return footer["advances"]


def budget_check(raw,budgets,encode):
    expected=[]
    total=0
    for key,row in budgets.items():
        q=row["old_q"]
        cap=row["cap"]
        # Marginal signs from sorted values independently confirm the old natural SU count;
        # ordinal ties affect endpoint identities, but never these ordered scalar differences.
        add=np.sort(q[:,0])[::-1][:cap]
        drop=np.sort(q[:,1])[:cap]
        k=int(np.count_nonzero(add-drop > 0))
        need(row["k"] == k, "Independent old held-fold positive-marginal budget")
        expected.append(encode(dict(key=list(key),fold=row["fold"],cap=cap,k=k,
                                    inputs_sha256=row["inputs_sha256"],
                                    old_q_sha256=sha256(q.tobytes()).hexdigest())))
        total+=k
    need(b"".join(expected) == raw, "Exact complete ordered historical budget file")
    return len(expected),total


def fold_predictions(raw,expected,budgets):
    with np.load(BytesIO(raw),allow_pickle=False) as f:
        need(set(f.files) == {"keys","offsets","k",*ARMS} and len(f.files)==7,
             "Exact saved OOF array schema")
        keys=f["keys"]
        need(keys.dtype == np.int64 and keys.shape == (len(expected),3)
             and np.array_equal(keys,np.asarray(expected,dtype=np.int64)), "Ordered original held candidate IDs")
        offsets=f["offsets"]
        lengths=[len(budgets[key]["coordinates"]) for key in expected]
        need(offsets.dtype in (np.dtype("int32"),np.dtype("int64")) and offsets.shape == (len(expected)+1,)
             and np.array_equal(offsets,np.cumsum([0,*lengths],dtype=np.int64)), "Ordered OOF endpoint offsets")
        k=f["k"]
        need(k.dtype == np.int8 and k.shape == (len(expected),)
             and np.array_equal(k,[budgets[key]["k"] for key in expected]), "Exact prelabel k per held row")
        out={key:{} for key in expected}
        for arm in ARMS:
            q=f[arm]
            need(q.dtype == np.float32 and q.shape == (offsets[-1],2) and np.isfinite(q).all()
                 and ((q>=0)&(q<=1)).all(), "Exact finite held occupancy for "+arm)
            for i,key in enumerate(expected):
                out[key][arm]=q[offsets[i]:offsets[i+1]]
    return out


def selected_labels(args,spot,roster):
    gt=json.loads(pinned(args.fit_gt,spot.GT_SHA,32*1024**2))
    need(len(gt["images"])==2937 and len(gt["annotations"])==9260 and [c["id"] for c in gt["categories"]]==[1],
         "Original FIT-only annotations")
    img={r["id"]:r for r in gt["images"]}
    chosen=[]
    for item,_,_ in roster:
        r=img[item["image_id"]]
        need(r["file_name"]==item["native_basename"] and r["height"]==r["width"]==800,
             "Native original FIT roster for pooled score")
        chosen.append(r)
    ids={r["id"] for r in chosen}
    anns=[r for r in gt["annotations"] if r["image_id"] in ids]
    need(len(ids)==len(chosen)==109 and len(anns)==496, "Exact FIT109 score population including empty rasters")
    return dict(images=chosen,categories=gt["categories"],annotations=anns)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("code","heads","scorer","new-capture","old-capture","old-acceptance","source-cohort",
              "old-heads","fit-gt","shapely-site","run-output","protocol","receipt","main-source"):
        p.add_argument("--"+k,required=True,type=Path)
    for k in ("capture-acceptance-sha","result-sha","receipt-sha"):
        p.add_argument("--"+k,required=True)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    output=a.output or a.run_output/"INDEPENDENT_SCORE_SAFE.json"
    need(not output.exists() and output.parent.is_dir(), "Exclusive independent readback receipt")
    receipt=json.loads(pinned(a.receipt,a.receipt_sha,4*1024**2))
    decision=accept_footer(receipt,a.result_sha)
    report=json.loads(pinned(a.run_output/"RESULT_SAFE.json",a.result_sha,4*1024**2))
    need(report["status"]==STATUS and report["entry_sha256"]==MAIN_SHA and report["FIT"] and report["AP"]
         and report["GT_opened"] and not report["DEV"] and not report["TEST"] and report["seed"]==5217
         and report["model_forward_calls"]==0 and report["head_updates"]==4000
         and report["original_detector_training_exposure"] and report["source_group_holdout_for_point_head_only"]
         and report["metrics"]["gate"]["advance"] is decision, "Closed source-only training result and decision")

    # Do not load the closed training worker until its original terminal and hash are bound.
    source=pinned(a.main_source,MAIN_SHA)
    m=ModuleType("_closed_fit109_query_run")
    m.__file__=str(a.main_source)
    exec(compile(source,str(a.main_source),"exec"),m.__dict__)
    torch=m.torch
    need(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1"
         and sys.version_info[:3]==(3,10,13) and torch.__version__=="2.10.0+cu130"
         and np.__version__=="1.26.4", "Original isolated CPU runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    pinned(a.protocol,m.PROTOCOL,64*1024)
    spot=m.load(a.code/"RFDETR_CORE36_MATCHED_POINT_SPOT_20261120/run_spot.py")
    closed=a.code/"RFDETR_MATCHED_POINT_FORMAL_20261120"
    previous=m.load(closed/"run_fit109.py")
    plan=m.load(closed/"fit_plan.py")
    heads=m.load(a.heads)
    score=m.load(a.scorer)
    channel=m.load(a.code/"RFDETR_QUERY_CHANNEL_RESIDUAL_20261120/channel_residual.py")
    need(tuple(m.ARMS.values()) == tuple(score.ARMS) == ARMS and tuple(m.ARMS)==tuple(heads.ARMS),
         "Independent four-arm identity agreement")
    capture,_=m.read_acceptance(a)
    need(report["protocol_sha256"]==m.PROTOCOL and report["old_feature_sha256"]==m.FEATURE_SHA
         and report["new_capture_accepted_sha256"]==a.capture_acceptance_sha, "Original prelocked source bindings")
    a.capture,a.acceptance,a.acceptance_sha256=a.old_capture,a.old_acceptance,m.OLD_RAW_ACCEPTANCE
    roster,_=previous.read_population(a,spot)
    groups=[item["group"] for item,_,_ in roster]
    records,by_image,permutation=m.read_vectors(a,roster,spot,channel,capture)
    exact(permutation,report["permutation"],spot.encode,"Original QP endpoint exposure")
    helpers=score.load_helpers(a.code,a.shapely_site)
    trainer=helpers.fit.load_trainer(
        a.code/"DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/fit109_paired_point/train_point.py")
    old=m.restore_old(a,plan,trainer,helpers.fit)
    budgets=heads.seal_oof_budgets(records,groups,old,helpers.fit,plan)
    bsha=report["native_budget_sha256"]
    bud=pinned(a.run_output/"BUDGETS_PRIVATE.jsonl",bsha,8*1024**2)
    n,swaps=budget_check(bud,budgets,spot.encode)
    pre=json.loads(pinned(a.run_output/"PRELABEL_SAFE.json",report["prelabel_sha256"]))
    need(pre["status"]=="RFDETR_QUERY_FEATURE_FIT109_OOF_BUDGET_SEALED" and pre["FIT"]
         and not any(pre[k] for k in ("DEV","TEST","GT_opened","AP"))
         and pre["seed"]==5217 and pre["groups"]==109 and pre["candidate_budgets"]==n==6738
         and pre["total_swaps"]==swaps and pre["native_budget_sha256"]==bsha
         and pre["old_feature_sha256"]==m.FEATURE_SHA and pre["protocol_sha256"]==m.PROTOCOL
         and pre["new_capture_accepted_sha256"]==a.capture_acceptance_sha,
         "Sealed original candidate budgets and source provenance before FIT labels")
    exact(permutation,pre["permutation"],spot.encode,"Identical prelabel QP intervention")

    # Rebuild held identities in original record order, independently of NPZ ordering.
    assignments=plan.source_folds(groups)
    held=defaultdict(list)
    for row in records:
        key=heads.identity(row)
        f=assignments[row["group"]]
        need(budgets[key]["fold"]==f,"Old fold assignment")
        held[f].append(key)
    folds=report["folds"]
    need([f["fold"] for f in folds]==list(range(5)),"Complete five source folds")
    observations={}
    for fold,summary in enumerate(folds):
        keys=held[fold]
        need(summary["steps"]==200 and summary["batch"]==64 and summary["held_candidates"]==len(keys)
             and summary["held_groups"]==sum(f==fold for f in assignments.values())
             and summary["common_batch_sha256"]==old[fold]["common_batch_sha256"]
             and set(summary["heads"])==set(m.ARMS), "Matched old schedule and held groups")
        need(all(h["optimizer_steps"]==200 and h["parameters"]==2089
                 and h["common_batch_sha256"]==summary["common_batch_sha256"]
                 and h["initial_tensor_sha256"]!=h["final_tensor_sha256"] for h in summary["heads"].values())
             and len({h["initial_tensor_sha256"] for h in summary["heads"].values()})==1,
             "Recorded equal matched four-head training budgets")
        fold_raw=pinned(a.run_output/f"OOF_FOLD_{fold}_PRIVATE.npz",summary["prediction_sha256"],32*1024**2)
        preds=fold_predictions(fold_raw,keys,budgets)
        need(not observations.keys() & preds.keys(), "Each candidate held exactly once")
        observations.update(preds)
    need(observations.keys()==budgets.keys(),"Every held row including zero-support rows")
    print(json.dumps(dict(event="FIT109_INDEPENDENT_PRELABEL_REPLAY",rows=n,
                          folds=len(folds),total_swaps=swaps)),flush=True)

    gt=selected_labels(a,spot,roster)
    expected_k={k:b["k"] for k,b in budgets.items()}
    reconstructed=score.evaluate_fit109(gt,by_image,records,observations,expected_k,helpers)
    exact(reconstructed,report["metrics"],spot.encode,"Exact full re-score of four arms including matched interval and gate")
    need(all(x["edits"]==2*swaps for x in reconstructed["arms"].values())
         and reconstructed["gate"]["advance"] is decision and not torch.cuda.is_initialized(),
         "Exact four-arm realized binary edit count and original decision")
    safe=dict(status=VERIFIED,independent_readback_accepted=True,original_session_id=SESSION,original_os_exit=0,
              job_drained=True,tool_wait_sha256=a.receipt_sha,result_sha256=a.result_sha,
              main_source_sha256=MAIN_SHA,protocol_sha256=m.PROTOCOL,prelabel_sha256=report["prelabel_sha256"],
              capture_acceptance_sha256=a.capture_acceptance_sha,native_budget_sha256=bsha,
              original_old_k_before_GT_replayed=True,five_held_files_exact=True,
              complete_metrics_exact=True,exact_four_arm_same_count=True,
              folds=5,arms=list(ARMS),eligible_candidates=n,mapped_candidates=reconstructed["counts"]["mapped_candidates"],
              eligible_zero_budget=reconstructed["counts"]["zero_budget"],swaps_per_arm=swaps,
              metrics_sha256=sha256(spot.encode(reconstructed)).hexdigest(),gate_advance=decision,
              FIT=True,DEV=False,TEST=False,AP=True,GT_opened=True,detector_forward_calls=0,
              replayed_old_head_predictions=n,head_updates=0)
    digest=spot.write(output,safe)
    print(json.dumps(dict(status=VERIFIED,accepted_sha256=digest,advances=decision)),flush=True)


if __name__=="__main__":
    main()
