"""Attribution of frozen V−F posterior errors on the reused dual-reference source gate."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np
from pycocotools import mask as codec
import torch


HERE=Path(__file__).resolve().parent
LOCK="c0c0379a939bb9fafc8249f928709a2299cab9c41f4d007ce7bf03ecd5fb86b5"
SHARED="9bbf19f7dc3cc42a8dcc05c51f8f04a4c95c7b9ba4b861ee0a532866803a5772"
AGGREGATOR="f3e6d7a668e3ea54012f482851fdce982c475c84d72e165afe4132b32ac58a24"
BASE_RENDER="d12a193bf80e6fb24b3c853c8da7d342f6408aa22e9f3920b873f2d5eb3de1e5"
BASE_RENDERER="a0c11c5c37c3a8569e1a54047fbc09598604710861bcd19f4096a0e3abe35959"
BASE_RENDER_LOCK="90151f99805ec9baaf8a08236b8f9981f86a43247943bc7a03ecf7ce3bc25fa4"
BASE_RENDER_EXIT="cb84320a93db76c5d11156c40aca0b2b1bd4d0fa2bd746a7d68c6c86ccaef9d2"
BASE_SCORE="f86390623ffd658e91c17e8a7b99e0859bcc3918b9fce0afda2ff332d72d023a"
BASE_SCORER="b5c61ebbc9d02994b2d9cac210a19e4091693055ec97e257cc8063095c4b86f2"
BASE_SCORE_EXIT="a3ad0ea5274ecfd7c11c7baf1d4ad2047b1926790871268922c5b90fc97edeb4"
ACTIONS=("static","fixed_iou","variable_iou")
REFERENCES=("pseg","spsd")
BUCKETS=tuple(f"{side}_{sign}" for side in ("outside","inside") for sign in ("plus","minus"))
PIXEL_TERMS=("area_delta","pred_intersection_gain","true_intersection_gain","overlay_error","slope_error",
             "score_error","pred_margin","true_margin","outside_overlay","inside_overlay","outside_minus_inside_overlay",
             "outside_true_gain","inside_true_gain","outside_fp_delta","inside_fp_delta")
OTHER_TERMS=("pred_iou_delta","true_iou_delta","native_area")


def need(ok,message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path,expected):
    raw=path.read_bytes()
    need(digest(raw)==expected,"Exact immutable bytes "+path.name)
    return raw


def load(path,checksum):
    m=ModuleType(path.stem)
    m.__file__=str(path)
    exec(compile(pin(path,checksum),str(path),"exec"),m.__dict__)
    return m


def read_archives(a):
    old_score=load(a.base_score/"score_gate.py",BASE_SCORER)
    authority=load(a.authority,old_score.ARCH_AUTHORITY)
    prior,terminal,scorer,reader,old_render,archived,groups,archive_score,_=authority.archive(a)
    geometry=load(a.base_score/"mechanism.py",old_score.MECHANISM)
    old_score.pin(a.group_protocol,old_score.GROUP)
    end=json.loads(pin(a.base_score/"SCORE_TERMINAL.json",BASE_SCORE_EXIT))
    trusted=json.loads(pin(a.base_score/"score/RESULT_SAFE.json",BASE_SCORE))
    need(end["actual_exit"]==0 and end["phase"]=="score" and end["output_footer_verified"] is True
         and end["worker_sha256"]==BASE_SCORER and end["result_sha256"]==BASE_SCORE
         and trusted["render_result_sha256"]==BASE_RENDER and trusted["render_terminal_sha256"]==BASE_RENDER_EXIT
         and trusted["exact_all_archive_score_parity"] is True and trusted["exact_original_support_and_actual_total_budget"] is True
         and trusted["fit_labels_opened"] is trusted["historical_test_opened"] is False,
         "Exact independent source outcome and natural closed observer")
    proof,point=old_score.read_receipt(SimpleNamespace(render_root=a.base_render,points=a.points,
                  render_sha=BASE_RENDER,renderer_sha=BASE_RENDERER,renderer_lock_sha=BASE_RENDER_LOCK,
                  renderer_terminal_sha=BASE_RENDER_EXIT),prior,old_render)
    raw=pin(a.base_render/"render/GATE_RENDER_PRIVATE.jsonl",proof["render_stream_sha256"])
    rows=[json.loads(line) for line in raw.splitlines()]
    records={x["image_id"]:x for x in rows}
    need(len(rows)==len(records)==183 and len(set(groups.values()))==180,"Original unique source development")
    totals,area=old_score.signed_support(a,point,records,archived,groups,proof,authority.REFERENCE_ARMS,geometry)
    need(dict(area)==trusted["net_area_change_histograms"],"Exact previous signed-area census")
    return prior,terminal,scorer,reader,authority,old_render,archive_score,records,groups,trusted,point,totals


def read_action(stock,mask,endpoints,k):
    final=codec.decode(mask).astype(bool,order="C")
    need(stock.shape==final.shape,"Exact native geometry")
    x,y=stock.ravel(),final.ravel()
    expected=(np.flatnonzero(y & ~x),np.flatnonzero(x & ~y))
    answer=tuple(tuple(np.flatnonzero(np.isin(endpoints[:,j],actual)).tolist())
                 for j,actual in enumerate(expected))
    need(sum(map(len,answer))==2*k and all(len(found)==len(actual)
         and np.array_equal(np.sort(endpoints[np.asarray(found,dtype=np.intp),side]),actual)
         for side,(found,actual) in enumerate(zip(answer,expected,strict=True))),
         "Exact realized archived F/V/static orientation, 2k count and signed support")
    return answer


def predicted_utility(p,action,area,ref_area):
    add,drop=action
    hit=area-len(p)+p[:,1,:].sum(axis=0)
    hit+=p[np.asarray(add,dtype=np.intp),0,:].sum(axis=0)-p[np.asarray(drop,dtype=np.intp),1,:].sum(axis=0)
    edited_area=area+len(add)-len(drop)
    return hit,hit/(edited_area+ref_area-hit)


def predict_frozen(a,prior,terminal,authority,older,records,point):
    core,models,center,scale,fit_receipt,fit_end=prior.fitted(SimpleNamespace(fit=a.fit,fit_sha=authority.ARCH_FIT,core=a.core),terminal)
    need(fit_receipt["train_audit"]["heads"]["bce"]["final_tensor_sha256"]==authority.ARCH_BCE and
         fit_receipt["output_checkpoint_sha256"]==older["fit_checkpoint_sha256"] and fit_end==older["fit_terminal_sha256"],
         "Original frozen BCE head and exact training artifact lineage")
    point_root=a.points/"actual_retry/source_gate"
    idx=[json.loads(line) for line in pin(point_root/"POINT_INDEX_PRIVATE.jsonl",
            point["partitions"]["source_gate"]["point_index_sha256"]).splitlines()]
    need(len(idx)==183 and {m["image_id"] for m in idx}==set(records),"Exact frozen 183-image feature admission")
    kept={}
    totals=Counter()
    for meta in idx:
        iid=meta["image_id"]
        raw=pin(point_root/meta["filename"],meta["sha256"])
        need(len(raw)==meta["bytes"],"Exact sealed point packet")
        with np.load(BytesIO(raw),allow_pickle=False) as f:
            ranks,offsets,coords,features,quotas,static=(f[k] for k in ("native_ranks","point_offsets",
                "signed_coordinates","features","static_offsets","static_side_indices"))
        probabilities=core.predict(models["bce"],features,center,scale)
        need(probabilities.shape==(len(features),2,2) and probabilities.dtype==np.float32
             and np.isfinite(probabilities).all() and np.all((probabilities>=0)&(probabilities<=1)),
             "Exact archived FP32 2-reference posterior format")
        positive=set()
        for j,r in enumerate(ranks):
            slot=records[iid]["candidates"][int(r)-1]
            k=int(quotas[j+1]-quotas[j])
            need(k==slot["quota"],"Exact original quota")
            if not k:
                continue
            positive.add(int(r))
            lo,hi=int(offsets[j]),int(offsets[j+1])
            sites=coords[lo:hi]
            post=probabilities[lo:hi].astype(np.float64)
            state=codec.decode(slot["native"]).astype(bool,order="C")
            area=int(state.sum())
            actions={key:read_action(state,slot["changes"][key],sites,k) for key in ACTIONS}
            inc=terminal.actions(static[quotas[j]:quotas[j+1]])
            need(inc==actions["static"] and len(actions["fixed_iou"][0])==len(actions["fixed_iou"][1])==k,
                 "Exact static incumbent and original fixed action")
            predicted_reference=area-len(post)+post.sum(axis=(0,1))
            bi,bj=predicted_utility(post,inc,area,predicted_reference)
            for key in ACTIONS[1:]:
                pi,pj=predicted_utility(post,actions[key],area,predicted_reference)
                need(np.allclose(pi-bi,slot["variable_predicted_intersection_gain"][key],rtol=0,atol=1e-12)
                     and np.allclose(pj-bj,slot["variable_predicted_iou_gain"][key],rtol=0,atol=1e-12),
                     "Both exact reconstructed frozen-posterior gain vectors agree with the archived actions")
            totals.update(positive=1,exchanges=k,differing=int(slot["changes"]["fixed_iou"]!=slot["changes"]["variable_iou"]))
            kept[(iid,int(r))]=(post,sites,actions["fixed_iou"],actions["variable_iou"],area)
        need(positive=={r for r,v in enumerate(records[iid]["candidates"],1) if v["quota"]},"Complete positive image subset")
        totals.update(images=1,candidates=100,supported=len(ranks))
    need(totals["images"]==183 and totals["candidates"]==18300 and totals["supported"]==8821 and
         (totals["positive"],totals["exchanges"])==(8785,57692) and len(kept)==8785,
         "Every original source-development slot accounted for")
    return kept,totals


def extend(raw):
    for side in ("outside","inside"):
        plus,minus=(f"{side}_{s}_" for s in ("plus","minus"))
        raw[side+"_true_gain"]=raw[plus+"y_sum"]-raw[minus+"y_sum"]
        raw[side+"_overlay"]=(raw[plus+"p_sum"]-raw[plus+"y_sum"])-(raw[minus+"p_sum"]-raw[minus+"y_sum"])
        raw[side+"_fp_delta"]=raw[plus+"count"]-raw[minus+"count"]-raw[side+"_true_gain"]
    raw["outside_minus_inside_overlay"]=raw["outside_overlay"]-raw["inside_overlay"]
    need(abs(raw["outside_overlay"]+raw["inside_overlay"]-raw["overlay_error"])<=1e-10
         and raw["outside_true_gain"]+raw["inside_true_gain"]==raw["true_intersection_gain"],
         "Four subgroup occupancy counts reconstruct both signed intersections")
    return raw


def raster_counts(native,fixed,variable,ref):
    return tuple(int(codec.area(codec.merge([w,ref],intersect=True))) for w in (native,fixed,variable))


def summarize(a,stats,truth,native,matches,groups,trials,posterior,side,ledger,aggregate):
    basis_names=(*PIXEL_TERMS,*OTHER_TERMS)
    row_count=defaultdict(list)
    rates=defaultdict(list)
    raw_counts=Counter()
    exposure=Counter()
    edited_groups=set()
    for gt in truth["annotations"]:
        iid=gt["image_id"]
        group=groups[iid]
        key=(iid,group)
        item=matches.get(gt["id"])
        if item is None:
            raw={}
            truth_area=1
            exposure["native_miss"]+=1
        else:
            need(item[0]==iid,"Original native match owns author hull image")
            slot=native[iid][item[1]-1]
            native_rle=slot["native"]
            area=int(codec.area(native_rle))
            ref=gt["segmentation"]
            truth_area=int(codec.area(ref))
            need(truth_area>0,"Nonempty author raster held fixed")
            raw=dict(native_area=area)
            exposure["native_match"]+=1
            if slot["quota"]:
                exposure["positive_original_quota"]+=1
                p,sites,f,v,old_area=posterior[item]
                need(old_area==area,"Frozen native geometric area")
                bits=codec.decode(ref).ravel(order="C")[sites]
                m0,m1=slot["changes"]["fixed_iou"],slot["changes"]["variable_iou"]
                truth_native_hit,i0,i1=raster_counts(native_rle,m0,m1,ref)
                raw=extend(ledger.audit(p[:,:,side],bits,f,v,area,truth_area,truth_native_hit))
                actual_iou=codec.iou([m0,m1],[ref],[0])[:,0]
                need(abs(raw["true_iou_delta"]-(actual_iou[1]-actual_iou[0]))<=1e-12 and
                     raw["true_intersection_gain"]==i1-i0,
                     "Frozen signed posterior ledger independently equals exact raster overlap geometry")
                if m0!=m1:
                    exposure["changed_native_matches"]+=1
                    edited_groups.add(group)
                exposure["positive_or_zero_signed_margin_overcredit"]+=int(raw["score_error"]>=0)
        row_count[key].append(np.asarray([100.*raw.get(name,0.)/truth_area for name in PIXEL_TERMS]+
                             [100.*raw.get("pred_iou_delta",0.),100.*raw.get("true_iou_delta",0.),
                              raw.get("native_area",0.)]))
        channels=[]
        for bucket in BUCKETS:
            channels.extend(raw.get(f"{bucket}_{term}",0.) for term in ("p_sum","count","y_sum"))
        rates[key].append(np.asarray(channels))
        raw_counts.update({key:val for key,val in raw.items() if key in PIXEL_TERMS or any(key.startswith(b+"_") and
                         key.rsplit("_",1)[-1] in ("count","sum") for b in BUCKETS)})
    order=sorted(set(groups.values()))
    matrix=aggregate.group_table(row_count,groups,order,len(basis_names))
    r_scores=aggregate.group_table(rates,groups,order,3*len(BUCKETS))
    pts,ci=aggregate.intervals(matrix,trials)
    summaries={key:dict(points=value,percentile95=conf) for key,value,conf in zip(basis_names,pts,ci,strict=True)}
    subgroup={}
    for j,b in enumerate(BUCKETS):
        cols=r_scores[:,j*3:j*3+3]
        rate,range_=aggregate.rate(cols,trials)
        subgroup[b]=dict(raw_count=int(raw_counts[b+"_count"]),raw_probability_sum=raw_counts[b+"_p_sum"],
                  raw_author_occupancy_sum=int(raw_counts[b+"_y_sum"]),group_equal_probability_minus_label_per_exposed_cell=rate,
                  percentile95=range_)
    need(exposure["native_match"]==stats["original_matches"] and len(truth["annotations"])==stats["author_hulls"] and
         abs(summaries["true_iou_delta"]["points"]-stats["variable_iou_minus"]["fixed_iou"]["iou_points"])<=1e-12 and
         np.allclose(summaries["true_iou_delta"]["percentile95"],
                     stats["variable_iou_minus"]["fixed_iou"]["percentile95_points"],atol=1e-12,rtol=0),
         "Exact all-hull original outcome and paired interval")
    return dict(estimates=summaries,subgroups=subgroup,exposure=dict(exposure,changed_groups=len(edited_groups),
                      author_hulls=len(truth["annotations"])),raw_pixel_sums={k:v for k,v in raw_counts.items() if k in PIXEL_TERMS})


def main():
    parser=ArgumentParser(description=__doc__)
    for name in ("base-render","base-score","gate","archive","authority","fit","points","labels","core",
                 "group-protocol","ledger","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    a=parser.parse_args()
    start=perf_counter()
    need(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1"
         and not a.output.exists(),"Isolated fresh CPU-only source diagnostic")
    pin(HERE/"PRELOCK.md",LOCK)
    ledger=load(a.ledger,SHARED)
    aggregate=load(HERE/"aggregate.py",AGGREGATOR)
    torch.set_num_threads(1)
    prior,term,scorer,reader,auth,older,archive_score,records,groups,trusted,point,usage=read_archives(a)
    posterior,confirmed=predict_frozen(a,prior,term,auth,older,records,point)
    print("Exact sealed source 18,300 slots, V/F/static signed actions and both predicted gain vectors verified before labels.",flush=True)
    truths=reader.labels(a,records)
    native={iid:row["candidates"] for iid,row in records.items()}
    results,matches={},{ }
    draws=np.random.Generator(np.random.PCG64(5297)).integers(180,size=(4096,180))
    for side,key in enumerate(REFERENCES):
        truth=truths[key]
        scores={}
        for arm in ("native","fixed_iou","variable_iou"):
            scores[arm],matching=scorer.coco_score(truth,native,arm)
            need(all(abs(scores[arm][field]-val)<=1e-12 for field,val in trusted["references"][key]["metrics"][arm].items()),
                 "Original mask AP and companion metrics independently rerun")
            if arm=="native":
                matches[key]=matching
        results[key]=summarize(a,trusted["references"][key],truth,native,matches[key],groups,draws,
                              posterior,side,ledger,aggregate)
        results[key]["metrics"]=scores
        print("Exact all-hull source attribution completed under",key,flush=True)
    old=load(a.base_score/"score_gate.py",BASE_SCORER)
    _,common=old.matched_pairs(truths,matches,native,groups)
    need(common==trusted["shared_hull_diagnostic"],"Exact original common-reference exposure and paired geometry")
    out=dict(status="SPSD_PSEG_SIGNED_POSTERIOR_ATTRIBUTION_COMPLETE",
             protocol_sha256=LOCK,worker_sha256=digest(Path(__file__).read_bytes()),
             shared_ledger_sha256=SHARED,aggregation_worker_sha256=AGGREGATOR,prior_render_sha256=BASE_RENDER,
             prior_render_terminal_sha256=BASE_RENDER_EXIT,prior_score_sha256=BASE_SCORE,prior_score_terminal_sha256=BASE_SCORE_EXIT,
             exact_all_native_and_signed_action_parity=True,exact_arithmetic_posterior_gain_parity=True,
             applied_author_masks=("pseg","spsd"),repeated_source_development=True,images=183,decoded_groups=180,
             bootstrap_draws=4096,seed=5297,original_counts=confirmed,common_native_exposure=common,
             references=results,signed_usage=usage,fit_labels_opened=False,external_acquisitions_opened=False,
             historical_test_opened=False,detector_forwards=0,optimizer_updates=0,elapsed_seconds=perf_counter()-start)
    a.output.mkdir()
    data=(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with (a.output/"RESULT_SAFE.json").open("xb") as f:
        f.write(data)
    print(json.dumps(dict(status=out["status"],result_sha256=digest(data))),flush=True)


if __name__=="__main__":
    main()
