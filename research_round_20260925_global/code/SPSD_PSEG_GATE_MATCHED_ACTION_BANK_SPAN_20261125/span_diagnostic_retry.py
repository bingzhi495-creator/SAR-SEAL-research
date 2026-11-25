"""Geometry-only coverage of sealed source-gate decisions by the original FIT action bank."""
from argparse import ArgumentParser
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
from pycocotools import mask

LOCK = "d175f0ccefc0a32a0a81c5f3bf03c773fa895ea1b5486cf355f797bb99962964"
STATUS = "SPSD_MATCHED_ACTION_GATE_BANK_SPAN_COMPLETE"
RENDER = "7aab5753eafe1e18464d0740a6b440fe09997dca2020c64a0d48180c228728fd"
SCORE = "b93e0f4d355f538ae751b8622bf359541dd5705f25d1dfe6eca836984521b052"
RENDER_WORKER = "c4841925513aeee64078064d483c73557b48c352594dd0d1edbe67a15be4b5b8"
SCORE_WORKER = "f65219d4677f93fa0347989ac3f3b5d7b51c4b2342d6aa6ab645c472d8dc3ad7"
FIT = "011914954482cd505343a6421296ae754b1ea3098f5ad3fe37cdaa14e3c05fd3"
FIT_RESULT = "ae1b165a111a6b88939eb37b5741421259f0f39531c859a7c7663ad9116d1b05"
ATTRACTION = "8eb162a3bd544929843d7f2aeef174f1f1ba03a2c593b6bcda49289cd94ad3c6"
POINTS = "87693897f873a2129764781611b0f7c35cadde51c0c4e830d7f8187c3b684049"
POINT_WORKER = "926f87b220c4ba9f1a7074088a52641cd66c17d4016bac91fba20c84f450f766"
CAPTURE = "6f58ba8b486950431f4aebebf237e60029b447e5ec21a7f51ad2202d065f904a"
READBACK = "e461828b04f166d98030210fe42df0d590d24e2452ea5da8630fb0fc3219d795"
READBACK_WORKER = "12b289e1307de56151cf016b8f9f8100eaffab8f6ae3da4a3b890647029dd449"
MANIFEST = "701c89d7350f6a078337d6074d69960799ade868a4f64e546897e07137c92bf4"
ARMS = ("bce_robust", "action_robust")
RTOL = 1e-10
QUANTILES = (0, 10, 25, 50, 75, 90, 95, 100)
METRICS = ("out_of_span_fraction", "nearest_bank_squared_ratio", "nearest_bank_xor_pixels",
           "outside_bank_coordinate_energy_fraction", "support_delta_pixels")


def need(test, message):
    if not test:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, expected):
    raw = path.read_bytes()
    need(digest(raw) == expected, "Exact closed file digest: " + path.name)
    return raw


def module(path, expected):
    source = ModuleType(path.stem)
    source.__file__ = str(path)
    exec(compile(pin(path, expected),str(path),"exec"),source.__dict__)
    return source


def terminal(path, phase, result, worker):
    raw=path.read_bytes()
    doc=json.loads(raw)
    need(doc["phase"]==phase and doc["actual_exit"]==0 and doc["output_footer_verified"] is True
         and doc["result_sha256"]==result and doc["worker_sha256"]==worker and bool(doc["ended_utc"]),
         "Exact observed natural exit and result footer: " + phase)
    return digest(raw)


def admitted(a):
    gate_end = terminal(a.gate/"RENDER_TERMINAL.json","render",RENDER,RENDER_WORKER)
    score_end = terminal(a.gate/"SCORE_TERMINAL.json","score",SCORE,SCORE_WORKER)
    gate = json.loads(pin(a.gate/"render/RESULT_SAFE.json",RENDER))
    score = json.loads(pin(a.gate/"score/RESULT_SAFE.json",SCORE))
    need(gate["status"]=="SPSD_MATCHED_SIGNED_ACTION_GATE_RENDER_COMPLETE" and gate["worker_sha256"]==RENDER_WORKER
         and gate["source_gate_labels_opened"] is gate["fit_labels_opened"] is gate["historical_test_opened"] is False
         and gate["optimizer_updates"]==gate["detector_forwards"]==0 and gate["fit_result_sha256"]==FIT_RESULT
         and gate["point_result_sha256"]==POINTS and gate["images"]==183 and gate["decoded_groups"]==180
         and score["render_result_sha256"]==RENDER and score["render_stream_sha256"]==gate["render_stream_sha256"],
         "Exact closed and scored matched-action render binding without entering label scorer")
    fit_end = terminal(a.fit/"observed_fit/TERMINAL.json","prepare-edits",FIT_RESULT,FIT)
    need(fit_end==gate["fit_terminal_sha256"], "Exact FIT action-bank authority")
    fit = json.loads(pin(a.fit/"actual/RESULT_SAFE.json",FIT_RESULT))
    need(fit["worker_sha256"]==FIT and fit["train_audit"]["seed"]==5297
         and fit["protocol_sha256"]==gate["fit_protocol_sha256"], "Exact closed action FIT source")
    p_end=terminal(a.points/"observed_points_retry/TERMINAL.json","prepare-edits",POINTS,POINT_WORKER)
    points=json.loads(pin(a.points/"actual_retry/RESULT_SAFE.json",POINTS))
    need(p_end==gate["point_terminal_sha256"] and points["capture_readback_sha256"]==READBACK
         and points["test_images_opened"] is points["train_labels_opened"] is points["test_labels_opened"] is False,
         "Exact closed label-free full source point preparation")
    c_end=terminal(a.capture/"observed_readback/TERMINAL.json","verify",READBACK,READBACK_WORKER)
    read=json.loads(pin(a.capture/"READBACK_SAFE.json",READBACK))
    need(read["manifest_sha256"]==MANIFEST and read["export_result_sha256"]==CAPTURE
         and read["FIT"] is read["original_os_wait"] is read["job_drained"] is True
         and read["TEST"] is read["GT_opened"] is read["AP"] is False,
         "Exact complete closed author capture")
    m = json.loads(pin(a.capture/"intake_retry/TRAIN_IMAGE_ONLY_MANIFEST_PRIVATE.json",MANIFEST))["items"]
    native=pin(a.capture/"capture/DEV_MASK_PREDICTIONS_PRIVATE.jsonl",read["predictions_sha256"]).splitlines()
    audit=pin(a.capture/"capture/DEV_CANDIDATE_AUDIT_PRIVATE.jsonl",read["candidate_audit_sha256"]).splitlines()
    root=a.points/"actual_retry/source_gate"
    index=[json.loads(x) for x in pin(root/"POINT_INDEX_PRIVATE.jsonl",
             points["partitions"]["source_gate"]["point_index_sha256"]).splitlines()]
    decisions=[json.loads(x) for x in pin(a.gate/"render/GATE_RENDER_PRIVATE.jsonl",gate["render_stream_sha256"]).splitlines()]
    need(len(m)==len(native)==len(audit)==915 and len(index)==len(decisions)==183
         and len({x["image_id"] for x in index})==183 and len({x["source_group"] for x in index})==180,
         "Original exact full gate roster and decoded grouping")
    receipts=dict(render_result_sha256=RENDER,render_terminal_sha256=gate_end,
                  score_result_sha256=SCORE,score_terminal_sha256=score_end,
                  original_fit_terminal_sha256=fit_end,point_result_sha256=POINTS,point_terminal_sha256=p_end,
                  capture_readback_sha256=READBACK,capture_readback_terminal_sha256=c_end,
                  render_stream_sha256=gate["render_stream_sha256"],manifest_sha256=MANIFEST,
                  source_gate_point_index_sha256=points["partitions"]["source_gate"]["point_index_sha256"])
    return gate,m,native,audit,root,index,decisions,receipts


def encoded(plane):
    rle = mask.encode(np.asfortranarray(plane,dtype=np.uint8))
    rle["counts"]=rle["counts"].decode("ascii")
    return rle


def apply(stock, coords, side_actions, k):
    out=stock.copy(order="C")
    flat=out.ravel(order="C")
    for side, replacement in enumerate((True,False)):
        picks=np.asarray(side_actions[:,side],dtype=np.intp)
        expected=coords[picks,side]
        need(len(picks)==len(np.unique(picks))==k and np.all((picks>=0)&(picks<len(coords)))
             and np.all(flat[expected] != replacement), "Exact signed distinct write")
        flat[expected]=replacement
    return out


def selected_vector(raw, static, stock, coords, k):
    chosen=mask.decode(raw).astype(bool,order="C")
    need(chosen.shape==stock.shape,"Same sealed native output shape")
    raster=chosen.ravel()
    split = (raster[coords[:,0]], ~raster[coords[:,1]])
    actions=np.column_stack([np.flatnonzero(x) for x in split])
    need(actions.shape==(k,2),"All original independently signed edits")
    need(raw==encoded(apply(stock,coords,actions,k)),"Sealed selected RLE is exactly the signed-pool write")
    signed=raster[coords].astype(np.int8)-static.ravel()[coords].astype(np.int8)
    need(np.all(signed.sum(axis=0)==0)
         and int(np.count_nonzero(signed))==int(np.count_nonzero(chosen!=static)),
         "Complete signed final mask difference on original pool")
    return signed.ravel(order="C")


def span_metrics(weights, vectors):
    n_action,width=weights.shape
    if n_action:
        _,singular,directions=np.linalg.svd(weights.astype(np.float64),full_matrices=False)
        rank=int(np.count_nonzero(singular > RTOL * singular[0]))
        supported=np.any(weights!=0,axis=0)
        need(1<=rank<=n_action and singular[0]>0,"Nonzero integer bank span")
    else:
        rank=0
        directions=np.zeros((0,width))
        supported=np.zeros(width,dtype=bool)
    basis=directions[:rank,:]
    values={}
    for mode,v in vectors.items():
        energy=int(np.count_nonzero(v))
        if not energy:
            values[mode]=[0.0,0.0,0.0,0.0,0.0]
            continue
        need(n_action>0,"A nontrivial written action requires a nontrivial bank")
        vf=v.astype(np.float64)
        residual=vf-(basis@vf)@basis
        ratio=float(residual@residual / energy)
        need(-1e-12 <= ratio <= 1+1e-12,"Orthonormal projection energy identity")
        delta=weights.astype(np.int16)-v.astype(np.int16)
        need(np.all(np.abs(delta)<=1),"Same-static-origin vectors differ by binary final raster state")
        nearest=int(np.min(np.count_nonzero(delta,axis=1)))
        values[mode]=[float(np.clip(ratio,0,1)),nearest/energy,float(nearest),
                      int(np.count_nonzero(v[~supported]))/energy,float(energy)]
    return rank,int(np.count_nonzero(supported)),values


def packed(rows):
    arr=np.array(rows,dtype=np.float64).reshape((-1,len(METRICS)))
    need(len(arr)>0,"Nonempty aggregate cohort")
    return arr


def quant(values):
    return {str(q): float(v) for q,v in zip(QUANTILES,np.percentile(values,QUANTILES),strict=True)}


def numeric_summary(values, with_quantiles):
    result={}
    for name, v in zip(METRICS,values.T,strict=True):
        d=dict(mean=float(v.mean()))
        if with_quantiles:
            d["quantiles"]=quant(v)
        result[name]=d
    return result


def aggregate(arm_results, groups):
    output={}
    for arm,per_image in arm_results.items():
        all_values=packed([v for image in per_image.values() for v in image])
        changed=all_values[all_values[:,-1]>0.0]
        need(len(changed)>0,"Actual changed-candidate subpopulation exists")
        whole_groups,changed_groups, indicators=defaultdict(list),defaultdict(list),defaultdict(list)
        for iid, values in per_image.items():
            g=groups[iid]
            arr=np.asarray(values,dtype=np.float64).reshape((-1,len(METRICS)))
            whole_groups[g].append(arr.mean(axis=0) if len(arr) else np.zeros(len(METRICS)))
            indicators[g].append(float((arr[:,0]>.5).mean()) if len(arr) else 0.0)
            changed_only=arr[arr[:,-1]>0]
            if len(changed_only):
                changed_groups[g].extend(changed_only.tolist())
        gvalues=packed([np.asarray(v).mean(axis=0) for v in whole_groups.values()])
        gchanged=packed([np.asarray(v).mean(axis=0) for v in changed_groups.values()])
        need(len(gvalues)==180 and len(all_values)==8785,"All decoded groups and positive candidates included")
        output[arm]=dict(positive_candidates=len(all_values),changed_candidates=len(changed),
          changed_groups=len(gchanged),unchanged_candidates=len(all_values)-len(changed),
          exact_existing_nonstatic_bank_matches_among_changed=int((changed[:,2]==0).sum()),
          conditional_out_of_span_fraction_gt_half=dict(count=int((changed[:,0]>.5).sum()),
            candidate_fraction=float((changed[:,0]>.5).mean()),
            equal_changed_groups=float(np.mean([np.mean(np.asarray(v)[:,0]>.5) for v in changed_groups.values()]))),
          whole_group_equal_out_of_span_fraction_gt_half=float(np.mean([np.mean(v) for v in indicators.values()])),
          changed_only_candidate_weighted=numeric_summary(changed,True),
          changed_only_group_equal=numeric_summary(gchanged,True),
          whole_positive_candidate_weighted=numeric_summary(all_values,True),
          whole_group_equal=numeric_summary(gvalues,True))
    return output


def run(a):
    started_at=perf_counter()
    gate,manifest,subs,checks,root,index,decisions,receipts=admitted(a)
    bank_fn=module(a.fit/"fit_action.py",FIT)
    attraction=module(a.attraction,ATTRACTION)
    counts=Counter()
    structural={
      name:Counter() for name in ("bank_unique_actions","linear_rank","endpoint_pairs","quota",
                                   "ambient_signed_subspace_dimension")}
    arrays=defaultdict(list)
    outcome={k:{} for k in ARMS}
    groups={}
    for entry, frame in zip(index,decisions,strict=True):
        n_arc=entry["native_archive_ordinal"]
        i=entry["image_id"]
        pred,audit=(json.loads(lines[n_arc-1]) for lines in (subs,checks))
        row=manifest[n_arc-1]
        group=entry["source_group"]
        hw=entry["original_hw"]
        need(row["image_id"]==pred["image_id"]==audit["image_id"]==frame["image_id"]==i
             and row["original_hw"]==hw==frame["original_hw"] and row["source_group"]==group==frame["source_group"]
             and row["rgb_sha256"]==entry["rgb_sha256"] and row["image_ref"]["path"]==frame["filename"]
             and len(frame["candidates"])==100 and len(pred["coco"])>=100
             and entry["filename"]==f"POINTS_PRIVATE/{n_arc:04}.npz", "Exact native image roster and rank alignment")
        groups[i]=group
        for arm in ARMS: outcome[arm][i]=[]
        for j,(rendered,old) in enumerate(zip(frame["candidates"],pred["coco"][:100],strict=True),1):
            need(rendered["native"]==old["segmentation"] and rendered["score"]==old["score"]
                 and bool(rendered["quota"])==bool(rendered["changes"]),
                 "Original exact native stock, score and pass-through census")
            counts["native_candidates"] += 1
        raw=pin(root/entry["filename"],entry["sha256"])
        need(len(raw)==entry["bytes"],"Exact original packet bytes")
        with np.load(BytesIO(raw),allow_pickle=False) as p:
            ranks,off,coord,feature,offs_k,recorded=(p[x] for x in ("native_ranks","point_offsets",
               "signed_coordinates","features","static_offsets","static_side_indices"))
        need(len(ranks)==entry["supported_candidates"] and len(ranks)==len(set(ranks))
             and len(off)==len(offs_k)==len(ranks)+1 and off[0]==offs_k[0]==0
             and off[-1]==len(coord)==entry["paired_sites"] and coord.shape==(len(coord),2)
             and offs_k[-1]==len(recorded)==entry["realized_static_exchanges"] and recorded.shape==(len(recorded),2)
             and feature.shape==(len(coord),2,14) and np.isfinite(feature).all(),
             "Complete original source gate signed feature packet")
        changed_ranks=set()
        for j,r0 in enumerate(ranks):
            rank=int(r0)
            r=frame["candidates"][rank-1]
            area=audit["candidates"][rank-1]["mask_area"]
            start,end=off[j:j+2]
            qlo,qhi=offs_k[j:j+2]
            k=int(qhi-qlo)
            counts["supported_candidates"] += 1
            need(k==r["quota"] and 1<=end-start<=32 and k<=min(8,end-start,area//10)
                 and 32<=area<1024,"Exact original admissible quota and stock size")
            if not k:
                continue
            changed_ranks.add(rank)
            counts.update(positive_candidates=1,exchanges=k)
            positions=coord[start:end]
            n=len(positions)
            stock=mask.decode(r["native"]).astype(bool,order="C")
            dyn,sta=attraction.select(stock,positions,min(8,area//10,n))
            dynamic=np.asarray(dyn,dtype=np.int16).reshape((-1,2))
            old_static=np.asarray(sta,dtype=np.int16).reshape((-1,2))
            incumbent=apply(stock,positions,old_static,k)
            need(len(dyn)==len(sta)==k and np.array_equal(old_static,recorded[qlo:qhi])
                 and encoded(incumbent)==r["changes"]["static"]
                 and encoded(apply(stock,positions,dynamic,k))==r["changes"]["dynamic"],
                 "Exact original attraction proposal quotas and all final anchor RLEs")
            vectors={arm:selected_vector(r["changes"][arm],incumbent,stock,positions,k) for arm in ARMS}
            counts["bce_action_render_disagree"]+=int(r["changes"][ARMS[0]]!=r["changes"][ARMS[1]])
            weights=bank_fn.action_bank(positions,feature[start:end],stock,k,old_static,
                                        attraction,f"{n_arc}|{rank}")
            if k==n:
                need(not weights and not any(np.any(x) for x in vectors.values()), "No action freedom at k=n")
                weights=np.empty((0,n,2),dtype=np.int8)
                counts["no_free_action_candidates"]+=1
            else:
                need(weights.ndim==3 and weights.shape[1:]==(n,2) and weights.dtype==np.int8
                     and 1<=len(weights)<=6 and np.all(weights.sum(axis=1)==0)
                     and np.all(np.any(weights!=0,axis=(1,2)))
                     and len(np.unique(weights.reshape((-1,2*n)),axis=0))==len(weights),
                     "Exact unique frozen FIT-style proposals relative to static")
            shape=weights.reshape((len(weights),2*n))
            dim,supported,val=span_metrics(shape,vectors)
            for arm, result in val.items(): outcome[arm][i].append(result)
            bound=2*(n-1) if k<n else 0
            need(0<=dim<=bound,"Signed action-count rank constraint")
            for key, v in (("bank_unique_actions",len(weights)),("linear_rank",dim),("endpoint_pairs",n),
                           ("quota",k),("ambient_signed_subspace_dimension",bound)):
                structural[key][v]+=1
                arrays[key].append(v)
            arrays["coordinate_support_count"].append(supported)
            arrays["coordinate_support_fraction"].append(supported/(2*n))
            arrays["rank_over_feasible_dimension"].append(dim/bound if bound else 0.0)
        need(changed_ranks=={j for j,r in enumerate(frame["candidates"],1) if r["quota"]},
             "All positive signed candidates visited once")
        counts["images"]+=1
        if counts["images"]%50==0 or counts["images"]==183:
            print(json.dumps(dict(progress_images=counts["images"],positive_candidates=counts["positive_candidates"])),flush=True)
    need(counts["native_candidates"]==18300 and counts["positive_candidates"]==8785
         and counts["supported_candidates"]==8821 and counts["exchanges"]==57692
         and counts["bce_action_render_disagree"]==gate["counts"]["changed_action_bce"]==7417
         and len(set(groups.values()))==180,"Exact all-arm and whole cohort census")
    return dict(status=STATUS,protocol_sha256=LOCK,worker_sha256=digest(Path(__file__).read_bytes()),
                fit_action_source_sha256=FIT,attraction_source_sha256=ATTRACTION,receipts=receipts,
                svd_relative_tolerance=RTOL,counts=dict(counts),decoded_groups=180,
                structure=dict(histograms={k:dict(v) for k,v in structural.items()},
                  summaries={k:dict(mean=float(np.mean(v)),quantiles=quant(v)) for k,v in arrays.items()}),
                arms=aggregate(outcome,groups),train_labels_opened=False,gate_labels_opened=False,
                test_images_opened=False,test_labels_opened=False,detector_forwards=0,head_forwards=0,
                optimizer_updates=0,AP=False,altered_predictions=False,elapsed_seconds=perf_counter()-started_at)


def main():
    p=ArgumentParser(description=__doc__,allow_abbrev=False)
    for option in ("gate","fit","points","capture","attraction","protocol","output"):
        p.add_argument("--"+option,required=True,type=Path)
    a=p.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1",
         "Exact isolated CPU geometry-only replay")
    pin(a.protocol,LOCK)
    need(not a.output.exists(),"Unique safe output path")
    doc=run(a)
    raw=(json.dumps(doc,indent=2,sort_keys=True,allow_nan=False)+"\n").encode("utf-8")
    with a.output.open("xb") as file:
        file.write(raw)
        file.flush()
        os.fsync(file.fileno())
    print(json.dumps(dict(status=STATUS,result_sha256=digest(raw))),flush=True)


if __name__=="__main__":
    main()
