"""Label-free selective replay of the exact archived BCE action and fixed static fallback."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np
from pycocotools import mask as codec
import torch

LOCK = "4eba0ba9d289ba7544df104aadb899a6056af9e04d04e13bc02a5f284dd2a85e"
POLICY = "cb3d44dd5ca432d1c6190a09db8f839cb45e78e28e9b29c05b00a7be0a71d749"
GROUP = "4030a3cb5684a6dac0ea1e7765a48d05caec113c5c6e0e6de2c82b991e9d5f15"
ARCH_RENDER = "7aab5753eafe1e18464d0740a6b440fe09997dca2020c64a0d48180c228728fd"
ARCH_SCORE = "b93e0f4d355f538ae751b8622bf359541dd5705f25d1dfe6eca836984521b052"
ARCH_RENDERER = "c4841925513aeee64078064d483c73557b48c352594dd0d1edbe67a15be4b5b8"
ARCH_SCORER = "f65219d4677f93fa0347989ac3f3b5d7b51c4b2342d6aa6ab645c472d8dc3ad7"
ARCH_FIT = "ae1b165a111a6b88939eb37b5741421259f0f39531c859a7c7663ad9116d1b05"
ARCH_BCE = "a2c169f895b1435e689480836ebf4f04fd6e05d84652f51d74e0448ed099faff"
REFERENCE_ARMS = ("native","dynamic","static","bce_robust","bce_mean","bce_cell_pessimistic",
                  "action_robust","action_mean","action_cell_pessimistic")
NEW_ARMS = ("selective","margin_ranked")
ARMS = (*REFERENCE_ARMS,*NEW_ARMS)


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path,expected):
    raw=path.read_bytes()
    need(digest(raw)==expected,"Exact digest: "+path.name)
    return raw


def load(path,expected):
    module=ModuleType(path.stem)
    module.__file__=str(path)
    exec(compile(pin(path,expected),str(path),"exec"),module.__dict__)
    return module


def archive(a):
    historical=load(a.archive/"render_gate.py",ARCH_RENDERER)
    scoring=load(a.archive/"score_gate.py",ARCH_SCORER)
    root=load(a.gate/"render_gate.py",historical.RENDER_WORKER)
    root.terminal(a.archive/"RENDER_TERMINAL.json","render",ARCH_RENDER,ARCH_RENDERER)
    term=root.terminal(a.archive/"SCORE_TERMINAL.json","score",ARCH_SCORE,ARCH_SCORER)
    score=json.loads(pin(a.archive/"score/RESULT_SAFE.json",ARCH_SCORE))
    arms,reader,_,render_sha,render_term,base,_,rows,groups=scoring.accepted(
        SimpleNamespace(gate=a.gate),a.archive)
    need(tuple(arms)==REFERENCE_ARMS and render_sha==ARCH_RENDER==score["render_result_sha256"]
         and render_term==score["render_terminal_sha256"] and score["render_stream_sha256"]==base["render_stream_sha256"]
         and score["fit_result_sha256"]==base["fit_result_sha256"]==ARCH_FIT
         and score["fit_labels_opened"] is score["historical_test_opened"] is False,
         "Exact archived candidate-balanced BCE writer and old natural scorer")
    return historical,root,scoring,reader,base,rows,groups,score,term


def selective_fit(a, root):
    worker=a.fit/"fit_selective.py"
    pin(worker,a.fit_worker_sha)
    pin(a.fit/"PRELOCK.md",a.fit_lock_sha)
    end=root.terminal(a.fit/"observed_fit/TERMINAL.json","prepare-edits",a.fit_sha,a.fit_worker_sha)
    result=json.loads(pin(a.fit/"actual/RESULT_SAFE.json",a.fit_sha))
    readback=json.loads(pin(a.fit/"READBACK_SAFE.json",a.fit_readback_sha))
    need(result["status"]=="SPSD_PSEG_SELECTIVE_WRITER_FIT_COMPLETE" and
         result["worker_sha256"]==a.fit_worker_sha and result["protocol_sha256"]==a.fit_lock_sha
         and result["frozen_parent_fit_sha256"]==ARCH_FIT and result["frozen_parent_bce_tensor_sha256"]==ARCH_BCE
         and result["feature_worker_sha256"]==a.helper_sha and result["fit_feature_count"]==9
         and result["ridge_alpha"]==.02 and result["closed_linear_solves"]==1 and result["fitted_selective_models"]==1
         and result["optimizer_updates"]==result["detector_forwards"]==0 and result["shared_oof_updates"]==5000
         and result["source_gate_labels_opened"] is result["historical_test_opened"] is False,
         "Closed FIT-only selective scorer of the exact archived BCE writer")
    need(readback["status"]=="SPSD_PSEG_SELECTIVE_WRITER_READBACK_COMPLETE"
         and readback["fit_result_sha256"]==a.fit_sha and readback["terminal_sha256"]==end
         and readback["worker_sha256"]==a.fit_worker_sha and readback["protocol_sha256"]==a.fit_lock_sha
         and readback["completed"] is True and readback["natural_exit"]==0
         and readback["input_sha256"]==result["input_receipt_sha256"]
         and all(readback[key]==result[key] for key in ("ridge_checkpoint_sha256","feature_worker_sha256",
                       "frozen_parent_fit_sha256","frozen_parent_bce_tensor_sha256")),
         "Exact checkpoint, protocol and parent independently observed and closed")
    admitted=json.loads(pin(a.fit/"actual/INPUT_SAFE.json",result["input_receipt_sha256"]))
    need(admitted["status"]=="FIT_SELECTIVE_WRITER_OOF_PACKETS_ADMITTED" and admitted["fit_only"] is True
         and admitted["source_gate_labels_opened"] is admitted["historical_test_opened"] is False
         and all(admitted[key]==result[key] for key in ("frozen_parent_fit_sha256","frozen_parent_bce_tensor_sha256",
                       "oof_authority_fit_sha256","out_of_fold_probability_sha256","out_of_fold_group_sha256")),
         "Exact fit-only packet and shared out-of-fold lineage")
    helper=load(a.fit/"selective_writer.py",a.helper_sha)
    coefficients=json.loads(pin(a.fit/"actual/RIDGE_PRIVATE.json",result["ridge_checkpoint_sha256"]))
    need(np.isfinite(helper.score(coefficients,np.zeros(9))),"Observed exact fixed-width ridge coefficients")
    return helper,coefficients,end


def eligible(row):
    return row["quota"]>0 and row["changes"]["bce_robust"]!=row["changes"]["static"]


def selection(policy, roster):
    ranks,quotas,learned,margin = ([z[i] for z in roster] for i in range(4))
    chosen=policy.choose(ranks,quotas,learned)
    control=policy.choose(ranks,quotas,margin)
    k=dict(zip(ranks,quotas,strict=True))
    counts=Counter(k[r] for r in chosen)
    need(len(chosen)==len(control)==len(roster)//3 and counts==Counter(k[r] for r in control),
         "Exact per-image adoption rates and original quota histogram")
    return dict(selective=chosen,margin_ranked=control),counts


def render(a):
    start=perf_counter()
    old,root,_,_,base,archive_rows,groups,_,arch_term=archive(a)
    helper,coef,fit_term=selective_fit(a,root)
    policy=load(Path(__file__).with_name("policy.py"),POLICY)
    core,models,center,scale,doc,old_fit_term=old.fitted(
        SimpleNamespace(fit=a.original_fit, fit_sha=ARCH_FIT, core=a.core),root)
    need(doc["train_audit"]["heads"]["bce"]["final_tensor_sha256"]==ARCH_BCE
         and doc["output_checkpoint_sha256"]==base["fit_checkpoint_sha256"]
         and old_fit_term==base["fit_terminal_sha256"],"Exact old candidate-balanced BCE model")
    selector=load(a.selector,old.SELECTOR)
    pt=json.loads(pin(a.points/"actual_retry/RESULT_SAFE.json",old.POINTS))
    pt_end=root.terminal(a.points/"observed_points_retry/TERMINAL.json","prepare-edits",old.POINTS,old.POINT_WORKER)
    need(pt_end==base["point_terminal_sha256"] and pt["train_labels_opened"] is pt["test_labels_opened"] is False,
         "Original gate point authority")
    point_root=a.points/"actual_retry/source_gate"
    index=[json.loads(s) for s in pin(point_root/"POINT_INDEX_PRIVATE.jsonl",
           pt["partitions"]["source_gate"]["point_index_sha256"]).splitlines()]
    need(len(index)==len(archive_rows)==183 and set(archive_rows)=={r["image_id"] for r in index},
         "Exact label-free gate image roster")
    totals=Counter()
    quota_totals=Counter()
    row_digest=sha256()
    a.output.mkdir()
    with (a.output/"GATE_RENDER_PRIVATE.jsonl").open("xb") as stream:
        for item in index:
            iid=item["image_id"]
            previous=archive_rows[iid]
            need(item["source_group"]==previous["source_group"]==groups[iid]
                 and item["original_hw"]==previous["original_hw"],"Frozen original group and geometry")
            raw=pin(point_root/item["filename"],item["sha256"])
            need(len(raw)==item["bytes"],"Exact closed packet length")
            with np.load(BytesIO(raw),allow_pickle=False) as f:
                ranks,offset,coords,features,quotas,booked=(f[n] for n in ("native_ranks","point_offsets",
                     "signed_coordinates","features","static_offsets","static_side_indices"))
            n=len(ranks)
            need(n==item["supported_candidates"]==len(set(ranks)) and len(coords)==item["paired_sites"]
                 and len(booked)==item["realized_static_exchanges"] and quotas.shape==offset.shape==(n+1,)
                 and np.all((1<=ranks)&(ranks<=100)) and features.shape==(len(coords),2,14)
                 and offset[0]==quotas[0]==0 and offset[-1]==len(coords) and quotas[-1]==len(booked),
                 "Exact original supported rank and endpoint span geometry")
            native=previous["candidates"]
            per_image=core.predict(models["bce"],features,center,scale)
            scores=[]
            verified=[]
            for j,r in enumerate(ranks):
                rank=int(r)
                p=native[rank-1]
                k=int(quotas[j+1]-quotas[j])
                need(k==p["quota"],"Exact old candidate edit quota")
                if not k:
                    continue
                verified.append(rank)
                span=slice(offset[j],offset[j+1])
                pp=per_image[span]
                y=coords[span]
                inc=root.actions(booked[quotas[j]:quotas[j+1]])
                options=selector.select(pp[:,0,:],pp[:,1,:],*inc,k=k)
                state=codec.decode(p["native"]).astype(bool,order="C")
                for key in ("static","robust","mean","cell_pessimistic"):
                    arm=key if key=="static" else "bce_"+key
                    need(root.edited_mask(p["native"],state,y,options[key],k)==p["changes"][arm],
                         "Exact rebuilt original BCE writer RLE parity")
                if eligible(p):
                    features_for_score,minimum=helper.features(features[span],pp,options["robust"],inc,k)
                    margin=min(p["margin_estimates"]["bce"]["bce"])/(2*k)
                    need(np.isclose(minimum,margin,atol=1e-8,rtol=0),"Exact archived signed worst gain heuristic")
                    score_value=float(helper.score(coef,features_for_score))
                    need(np.isfinite(score_value),"Finite FIT-only estimated gain")
                    scores.append([rank,k,score_value,float(margin)])
            need(set(verified)=={j for j,p in enumerate(native,1) if p["quota"]} and
                 {s[0] for s in scores}=={j for j,p in enumerate(native,1) if eligible(p)},
                 "All original native positive quotas and adoption-eligible proposals retained")
            chose,assigned=selection(policy,scores)
            rows=[]
            for rank,p in enumerate(native,1):
                q=dict(p)
                q["changes"]=dict(p["changes"])
                if p["quota"]:
                    for arm,members in chose.items():
                        q["changes"][arm]=p["changes"]["bce_robust" if rank in members else "static"]
                rows.append(q)
            line=(json.dumps(dict(previous,candidates=rows,eligible_scores=scores),sort_keys=True,
                             separators=(",",":"),allow_nan=False)+"\n").encode()
            need(stream.write(line)==len(line),"Complete exact label-free output row")
            row_digest.update(line)
            totals.update(images=1,candidates=100,positive=len(verified),eligible=len(scores),
                          adopted=len(chose["selective"]),exchanges=int(quotas[-1]),
                          adopted_overlap=len(chose["selective"]&chose["margin_ranked"]),
                          images_with_adoptions=int(bool(chose["selective"])))
            quota_totals.update(assigned)
    need(totals["images"]==183 and totals["candidates"]==18300 and totals["positive"]==8785
         and totals["exchanges"]==57692,"Exact original complete global signed edit budget")
    report=dict(status="SPSD_PSEG_SELECTIVE_BCE_GATE_RENDER_COMPLETE",protocol_sha256=LOCK,
         worker_sha256=digest(Path(__file__).read_bytes()),policy_sha256=POLICY,fit_result_sha256=a.fit_sha,
         fit_terminal_sha256=fit_term,fit_readback_sha256=a.fit_readback_sha,fit_worker_sha256=a.fit_worker_sha,
         fit_protocol_sha256=a.fit_lock_sha,fit_helper_sha256=a.helper_sha,
         archive_render_sha256=ARCH_RENDER,archive_score_sha256=ARCH_SCORE,archive_score_terminal_sha256=arch_term,
         archive_bce_tensor_sha256=ARCH_BCE,point_terminal_sha256=pt_end,render_stream_sha256=row_digest.hexdigest(),
         counts=totals,adopted_same_quota_histogram=quota_totals,decoded_groups=180,
         exact_bce_archive_rle_parity=True,source_gate_labels_opened=False,fit_labels_opened=False,
         historical_test_opened=False,detector_forwards=0,optimizer_updates=0,elapsed_seconds=perf_counter()-start)
    raw=(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    with (a.output/"RESULT_SAFE.json").open("xb") as out:
        out.write(raw)
    print(json.dumps(dict(status=report["status"],result_sha256=digest(raw))),flush=True)


def main():
    parser=ArgumentParser(description=__doc__)
    for x in ("gate","archive","fit","original-fit","points","core","selector","protocol","group-protocol","output"):
        parser.add_argument("--"+x,type=Path,required=True)
    for x in ("fit-sha","fit-worker-sha","fit-lock-sha","fit-readback-sha","helper-sha"):
        parser.add_argument("--"+x,required=True)
    a=parser.parse_args()
    pin(a.protocol,LOCK)
    pin(a.group_protocol,GROUP)
    torch.set_num_threads(1)
    render(a)


if __name__=="__main__":
    main()
