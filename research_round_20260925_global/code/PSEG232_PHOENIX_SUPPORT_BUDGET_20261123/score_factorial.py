"""Score a frozen four-policy Phoenix replay on both predeclared PSeg-SSDD TEST232 populations."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from time import perf_counter
from types import ModuleType

import numpy as np
from PIL import Image
from pycocotools import mask as codec
import shapely


ARMS=("nc","nt","wc","wt")
PAIRS=(("wc","wt"),("nc","wc"),("nc","nt"),("nt","wt"))
COMPARE="633da1df76185bb32177422eba0c3b51f2c0b1a01931ce432b993641c10aa6c7"
PHASES={
    "prepare-edits":("fae27e53ed3ca8253b8f1768e6d541525b2ea79dcb931d0edfe6c94ab4a837f3","cc2873bfe3a30bd1de1ad93e6c8d70125b05a998714371ecf65450ae851c69d0"),
    "verify-edits":("6042fe2ef69c30c9ae4567c9deae7c0982425cc07576f79b7733259b0a1071b7","39ce2d6b0aefeef9a9447342e69d05a250c2cb61beafaaecb39298ad896b53aa"),
    "score":("3efc20ab1218d2f36b21e5967a9f178488af32d0a3e06c78e64a1ca8062b7365","25a9d2709a8b53ea24a53d8067a8f22777008b451eec16a41a718c99eca25901")}
BASE_INFER="ea98a849ae181f06f3ffc0e41b29d2a91a7f867b4a4bcba6909c957f27db2b99"
FACTORIAL_INFER="8d46b1d16bcba274576f03448e7cf74b9d332a9bf8848b090aa659409fa91c80"
REQUEST_STATUS="PSEG232_PHOENIX_SUPPORT_BUDGET_REQUEST"
SCORE_STATUS="PSEG232_PHOENIX_SUPPORT_BUDGET_SCORE_COMPLETE"
INFER_STATUS="PSEG232_PHOENIX_SUPPORT_BUDGET_INFERENCE_COMPLETE"
INFER_WORKER="de5fd5a831192d6b98b042381dcbf55688a99d329318b8cdc892dbca8013d95e"
REQUEST_WORKER="e4d87156e29d966569b9854af7d44aa67a41ad9b0408934da676ac182ebb2e69"
LOCK="5307bcbc32f94f11d27092945e1993692b6ef8347e85a2d3231080dae54c81df"
LOCK_NAME="PSEG232_PHOENIX_SUPPORT_BUDGET_PRELOCK_20261123.md"


def require(value,reason):
    if not value:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path,expected):
    data=path.read_bytes()
    require(digest(data)==expected,"Input digest differs for "+path.name)
    return data


def load(path,sha):
    unit=ModuleType(path.stem)
    unit.__file__=str(path)
    exec(compile(pin(path,sha),str(path),"exec"),unit.__dict__)
    return unit


def observed(folder,terminal,phase,status,worker,pins=None):
    result=(folder/"RESULT_SAFE.json").read_bytes()
    term=terminal.read_bytes()
    r,t=json.loads(result),json.loads(term)
    rs,ts=digest(result),digest(term)
    require((pins is None or pins==(rs,ts)) and t["worker_sha256"]==worker and t["result_sha256"]==rs
            and type(t["actual_exit"]) is int and t["actual_exit"]==0 and t["output_footer_verified"] is True
            and t["phase"]==phase and r["status"]==status and r.get("worker_sha256",r.get("source_sha256"))==worker,
            "Successful observed "+phase+" required")
    return r,rs,ts


def encode(a):
    value=codec.encode(np.asfortranarray(a,dtype=np.uint8))
    return dict(size=value["size"],counts=value["counts"].decode("ascii"))


def decoded(rle,hw):
    require(isinstance(rle,dict) and set(rle)=={"size","counts"} and rle["size"]==list(hw)
            and isinstance(rle["counts"],str) and rle["counts"],"Native compressed mask required")
    runs=[]
    nbits=value=0
    total=int(hw[0])*int(hw[1])
    for char in rle["counts"]:
        digit=ord(char)-48
        require(0<=digit<64 and nbits<=25,"Malformed COCO RLE digit")
        value|=(digit&31)<<nbits
        nbits+=5
        if digit&32:
            continue
        if digit&16:
            value-=1<<nbits
        if len(runs)>2:
            value+=runs[-2]
        require(0<=value<=total,"Malformed COCO RLE run")
        runs.append(value)
        value=nbits=0
    require(nbits==0 and sum(runs)==total,"Incomplete native canvas")
    a=codec.decode(rle).astype(bool)
    require(a.shape==tuple(hw) and encode(a)==rle,"Expected canonical original canvas mask")
    return a


def verify_candidate(candidate,post,entries,hw):
    require(set(entries)==set(ARMS) and all(set(row)=={"rle"} for row in entries.values())
            and set(post)=={"position","probabilities","half_foreground"} and post["position"]==candidate["position"],
            "Candidate position or endpoint schema differs")
    native=decoded(candidate["segmentation"],hw)
    coords=np.asarray(candidate["coords"])
    q=np.asarray(post["probabilities"],dtype=np.float64)
    k=candidate["k"]
    require(q.ndim==2 and coords.shape==q.shape==(len(q),2) and np.isfinite(q).all() and np.all((0<=q)&(q<=1))
            and np.array_equal(q,q.astype(np.float32).astype(np.float64)) and coords.dtype.kind in "iu"
            and np.all((0<=coords)&(coords<native.size)) and np.unique(coords).size==coords.size
            and not native.flat[coords[:,0]].any() and native.flat[coords[:,1]].all()
            and 32<=native.sum()<1024 and type(k) is int and 0<k<=min(8,int(native.sum())//10,len(q)),
            "Invalid finite native endpoints, sign, frozen area or dynamic budget")
    half=post["half_foreground"]
    require(isinstance(half,list) and all(type(x) is int and 0<=x<q.size and q.flat[x]==.5 for x in half)
            and sorted(set(half))==half,"Invalid binary foreground tie annotation")
    add=sorted(range(len(q)),key=lambda i:-q[i,0])
    drop=sorted(range(len(q)),key=lambda i:q[i,1])
    # Adding a common 1/2 to every signed utility preserves the optimum at fixed 2k.
    # Rational complements retain subnormal probability differences even near 1.
    add_gain=[Fraction.from_float(float(v)) for v in q[:,0]]
    drop_gain=[1-Fraction.from_float(float(v)) for v in q[:,1]]
    cutoff=sorted(add_gain+drop_gain,reverse=True)[2*k-1]
    # These bounds include all splits that realize the global top-2k cutoff.
    min_add=max(sum(v>cutoff for v in add_gain),2*k-sum(v>=cutoff for v in drop_gain))
    max_add=min(sum(v>=cutoff for v in add_gain),2*k-sum(v>cutoff for v in drop_gain))
    require(min_add<=max_add,"Empty exact signed allocation")
    a=min(max(k,min_add),max_add)
    capped=native.copy(order="C")
    capped.flat[coords[add[:a],0]],capped.flat[coords[drop[:2*k-a],1]]=True,False
    threshold=q>.5
    threshold.flat[half]=True
    local=native.copy(order="C")
    local.flat[coords]=threshold
    masks={arm:decoded(entries[arm]["rle"],hw) for arm in ARMS}
    require(np.array_equal(masks["nc"],capped) and np.array_equal(masks["nt"],local)
            and np.array_equal(masks["wt"].flat[coords],threshold),"Narrow rank, threshold or original wide projection differs")
    signs=np.stack((np.sign(q[:,0]-.5),np.sign(.5-q[:,1])),axis=1)
    wc=masks["wc"]^native
    seen=wc.flat[coords]
    for side,idx in enumerate((add,drop)):
        order=seen[idx,side]
        require(not np.any(order[1:] & ~order[:-1]),"Whole capped mask violates sampled within-sign ranks")
    by_side=(add_gain,drop_gain)
    selected=[[v for v,on in zip(gains,seen[:,side]) if on] for side,gains in enumerate(by_side)]
    skipped=[[v for v,on in zip(gains,seen[:,side]) if not on] for side,gains in enumerate(by_side)]
    chosen=selected[0]+selected[1]
    omitted=skipped[0]+skipped[1]
    if chosen and omitted:
        require(min(chosen)>=max(omitted),"Whole cap skipped a strictly better observed signed endpoint")
    wide_add=int(np.count_nonzero(wc & ~native))
    if wide_add>k and selected[0] and skipped[1]:
        require(min(selected[0])>max(skipped[1]),"Whole cap permits a same-gain more balanced observed allocation")
    if wide_add<k and selected[1] and skipped[0]:
        require(min(selected[1])>max(skipped[0]),"Whole cap permits a same-gain more balanced observed allocation")
    out={}
    wt=masks["wt"]^native
    for arm,m in masks.items():
        flip=m^native
        added=int(np.count_nonzero(m&~native))
        dropped=int(np.count_nonzero(native&~m))
        n=added+dropped
        require(arm not in ("nc","wc") or n==2*k,"Exact 2k flips required")
        known=signs[flip.flat[coords]]
        out[arm]=dict(adds=added,drops=dropped,flips=n,signed_area_delta=added-dropped,empty_outputs=int(not m.any()),
            changed_candidates=int(n>0),inside_narrow_flips=len(known),outside_narrow_flips=n-len(known),
            candidates_above_2k=int(n>2*k),candidates_below_2k=int(n<2*k),candidates_equal_2k=int(n==2*k),
            narrow_positive_selected=int(np.count_nonzero(known>0)),narrow_negative_selected=int(np.count_nonzero(known<0)),
            narrow_zero_selected=int(np.count_nonzero(known==0)),also_threshold_flips=int(np.count_nonzero(wt&flip)))
    left,right=int(np.count_nonzero(wc&~wt)),int(np.count_nonzero(wt&~wc))
    out["bounds"]=dict(required_half_cells=left+right if left and right else 0,narrow_half_cells=int(np.count_nonzero(q==.5)))
    return out


def historical(a):
    comp=load(a.code/"PSEG_TEST232_CLASSICAL_ATTRACTION_20261122/compare.py",COMPARE)
    scorer=load(a.code/"PSEG_TEST232_DFINE_FROZEN_SCORE_20261121/score_test232.py",comp.SCORE_SHA)
    geometry=load(a.code/"PSEG_TEST232_PAIRED_GEOMETRY_20261121/paired_geometry.py",comp.GEOMETRY_SHA)
    stages={}
    for phase,status in (("prepare-edits",comp.PREPARED),("verify-edits",comp.VERIFIED),("score",comp.SCORED)):
        stages[phase]=observed(a.classical_root/phase,a.classical_root/("observed_"+phase)/"TERMINAL.json",
                              phase,status,COMPARE,PHASES[phase])[0]
    prep,verified,old=(stages[k] for k in ("prepare-edits","verify-edits","score"))
    require(prep["edits_sha256"]==verified["edits_sha256"]==old["edits_sha256"] and
            prep["counts"]==verified["counts"]==old["counts"] and prep["GT_opened"] is verified["GT_opened"] is False,
            "Previously closed classical edit identities differ")
    path=a.original_readback or a.capture.parent/"PSEG_TEST232_DFINE_FROZEN_REPLAY_READBACK_ACTUAL_20261121_01/RESULT_SAFE.json"
    capture=json.loads(pin(path,comp.READBACK_SHA))
    scorer.check_acceptance(capture)
    manifest=json.loads(pin(a.owner/"TEST232_IMAGE_ONLY_MANIFEST_PRIVATE.json",capture["manifest_sha256"]))
    roster=scorer.validate_manifest(manifest)
    def lines(path,sha):
        return [json.loads(line) for line in pin(path,sha).splitlines()]
    preds=lines(a.capture/"DEV_MASK_PREDICTIONS_PRIVATE.jsonl",capture["input_predictions_sha256"])
    audits=lines(a.capture/"DEV_CANDIDATE_AUDIT_PRIVATE.jsonl",capture["candidate_audit_sha256"])
    edits=lines(a.classical_root/"prepare-edits/EDITS_PRIVATE.jsonl",prep["edits_sha256"])
    hist=(*scorer.ARMS,*comp.NEW_ARMS)
    scorer.ARMS=hist
    blocks=scorer.join_blocks(roster,preds,audits,edits)
    require(len(blocks)==232 and all(len(v[0])==100 for v in blocks.values()),"Original ranked TEST232 detections required")
    scorer.ARMS=(*hist,*ARMS)
    return comp,scorer,geometry,old,capture,roster,audits,edits,blocks


def crosscheck_classical(stock,candidate,reference,attraction,names):
    coords=np.asarray(candidate["coords"])
    native=codec.decode(stock).astype(bool)
    dynamic,static=attraction.select(native,coords,min(8,int(native.sum())//10,len(coords)))
    require(len(dynamic)==len(static)==candidate["k"],"Original dynamic quota differs")
    for name,order in zip(names[:2],(dynamic,static),strict=True):
        bitmap=native.copy(order="C")
        ai,di=np.asarray(order).T
        bitmap.flat[coords[ai,0]],bitmap.flat[coords[di,1]]=True,False
        require(reference[name]["rle"]==encode(bitmap),"Native sampled support fails historical classical parity")
    learned=codec.decode(reference[names[2]]["rle"]).astype(bool)
    aa,dd=np.flatnonzero(learned&~native),np.flatnonzero(native&~learned)
    require(len(aa)==len(dd)==len(dynamic) and np.isin(aa,coords[:,0]).all() and np.isin(dd,coords[:,1]).all(),
            "Historical learned policy differs from signed support")


def confirm_counts(by_arm,stats,infer):
    bound=infer["counts"]
    require(stats["required_half_cells"]<=bound["threshold_half_cells"]
            and stats["narrow_half_cells"]==bound["threshold_half_narrow_endpoint_cells"]
            and bound["threshold_disagreements_away_from_half"]==0
            and bound["exact_nt_projections"]==bound["exact_wt_author_masks"]==bound["whole_count_rank_certificates"]
            ==bound["exact_dyadic_rank_certificates"]==11054,
            "Threshold ties, observed projection or whole-grid optimization certification differs")
    margins={}
    for arm,c in by_arm.items():
        reported=infer["counts_by_arm"][arm]
        require(all(reported[key]==value for key,value in c.items() if not key.startswith("narrow_") and key!="also_threshold_flips"),
                "Independently counted endpoint edits, sign, support or area disagree")
        pos,zero,neg=(reported[k+"_selected_flips"] for k in ("positive","zero","negative"))
        overlap=c["also_threshold_flips"]
        wt_zero=bound["threshold_zero_utility_flipped_cells"]
        require(all(type(n) is int and n>=0 for n in (pos,zero,neg)) and pos+zero+neg==c["flips"]
                and zero+neg==reported["nonpositive_selected_flips"] and (arm not in ("nc","wc") or c["flips"]==147540)
                and (arm not in ("nt","wt") or neg==0) and zero<=bound["threshold_half_cells"]
                and overlap-min(overlap,wt_zero)<=pos<=overlap-max(0,wt_zero-(by_arm["wt"]["flips"]-overlap))
                and (arm!="wt" or zero==wt_zero),"Invalid full-posterior signed-utility bounds")
        counts=(pos,zero,neg)
        local=tuple(c["narrow_"+k+"_selected"] for k in ("positive","zero","negative"))
        require(all(t>=v for t,v in zip(counts,local,strict=True)) and (arm not in ("nc","nt") or counts==local),
                "Returned marginal utilities disagree with available native posteriors")
        margins[arm]=dict(positive=pos,zero=zero,negative=neg,nonpositive=neg+zero)
    return margins


def build_inputs(a,hist):
    comp,scorer,geom,old,capture,roster,audits,original,blocks=hist
    req,rs,rt=observed(a.request,a.request_terminal,"request",REQUEST_STATUS+"_COMPLETE",REQUEST_WORKER)
    infer,ifs,ift=observed(a.inference,a.inference_terminal,"inference",INFER_STATUS,INFER_WORKER)
    require(req["protocol_sha256"]==infer["protocol_sha256"]==LOCK and req["phase"]=="request" and infer["phase"]=="inference"
            and req["target_labels_opened"] is infer["target_labels_opened"] is False and req["worker_sha256"]==infer["request_worker_sha256"]
            and infer["request_result_sha256"]==rs and infer["request_terminal_sha256"]==rt
            and req["request_sha256"]==infer["request_sha256"] and req["input_lineage"]==infer["input_lineage"]
            and req["source_replay"]==infer["source_replay"] and req["detector_forwards"]==req["optimizer_updates"]==0
            and infer["native_detector_forwards"]==infer["optimizer_updates"]==0 and infer["factorial_worker_sha256"]==FACTORIAL_INFER
            and infer["baseline_worker_sha256"]==BASE_INFER and infer["author_commit"]=="13ee3996ef3d0292fab144316cc8e03b69c0ecfa"
            and infer["weight_sha256"]=="9b6ba211e0d6d434085d3d5e379eb527d8b56ef413628f42cb587153cc51af7d"
            and infer["threshold_projection_exact"] is infer["whole_binary_exact"] is True
            and infer["signed_split_objective"]=="exact_binary32_dyadic"
            and infer["runtime"]=={"FastGeodis":"1.0.5","numpy":"1.26.4","torch":"2.10.0+cu130",
                                   "Pillow":"12.3.0","pycocotools":"2.0.11"}
            and tuple(infer[k] for k in ("workers","threads","seed","batch_size","refine_iters","smooth_kernel"))==(4,2,5202,2,5,1),
            "Frozen request, Phoenix settings or native lineage differs")
    payload=json.loads(pin(a.request/"REQUEST_PRIVATE.json",req["request_sha256"]))
    acc=json.loads(pin(a.owner/"CAPTURE_READBACK/INDEPENDENT_SAFE.json",capture["acceptance_sha256"]))
    lineage=dict(manifest_sha256=capture["manifest_sha256"],acceptance_sha256=capture["acceptance_sha256"],
                 **{k:acc[k] for k in ("ema_sha256","predictions_sha256","candidate_audit_sha256","probability_index_sha256",
                                        "export_result_sha256","export_terminal_sha256","original_tool_wait_sha256")})
    origins=dict(classical_worker_sha256=COMPARE,edits_sha256=old["edits_sha256"],prepare_result_sha256=PHASES["prepare-edits"][0],
        prepare_terminal_sha256=PHASES["prepare-edits"][1],verify_result_sha256=PHASES["verify-edits"][0],
        verify_terminal_sha256=PHASES["verify-edits"][1],original_readback_sha256=comp.READBACK_SHA,
        original_edits_sha256=capture["edits_sha256"],original_replay_sha256=capture["replay_sha256"],
        original_replay_terminal_sha256=capture["original_tool_wait_sha256"])
    require(req["input_lineage"]==lineage and req["source_replay"]==origins and acc["GT_opened"] is False
            and acc["predictions_sha256"]==capture["input_predictions_sha256"] and
            acc["candidate_audit_sha256"]==capture["candidate_audit_sha256"],"Observed source and native capture disagree")
    require(payload["status"]==REQUEST_STATUS and payload["protocol_sha256"]==LOCK
            and all(payload[k]==req[k] for k in ("input_lineage","source_replay")),"Exact image-only request required")
    masks=pin(a.masks or a.inference/"FACTORIAL_EDITS_PRIVATE.jsonl",infer["masks_sha256"])
    sparse=pin(a.posteriors or a.inference/"NARROW_POSTERIORS_PRIVATE.jsonl",infer["narrow_posteriors_sha256"])
    stats=Counter()
    per_arm={arm:Counter() for arm in ARMS}
    attraction=load(a.code/"FIT109_CLASSICAL_ATTRACTION_20261121/spatial_attraction.py",
                    "8eb162a3bd544929843d7f2aeef174f1f1ba03a2c593b6bcda49289cd94ad3c6")
    for j,(item,archive,img,line,post) in enumerate(zip(roster,original,payload["images"],map(json.loads,masks.splitlines()),
                                                        map(json.loads,sparse.splitlines()),strict=True),1):
        raw,edits,hw=blocks[item["image_ref"]["path"]]
        require(item["image_id"]==archive["image_id"]==img["image_id"]==line["image_id"]==post["image_id"]
                and type(line["ordinal"]) is type(post["ordinal"]) is int and j==line["ordinal"]==post["ordinal"]
                and set(line)=={"ordinal","image_id","changes"} and set(post)=={"ordinal","image_id","candidates"}
                and img["hw"]==hw and img["image_file"]==f"image_{j:03d}.png"
                and img["decoded_rgb_sha256"]==archive["rgb_sha256"],"Image identity, order or canvas differs")
        with Image.open(BytesIO(pin(a.request/img["image_file"],img["encoded_sha256"]))) as picture:
            require(picture.mode=="RGB" and picture.format=="PNG" and picture.size==tuple(hw[::-1])
                    and digest(picture.tobytes())==archive["rgb_sha256"],"PNG differs from archived decoded RGB")
        positions=[p for p in range(1,len(raw)+1) if comp.validate_extra_budget(edits.get(str(p),{}))>0]
        require(sorted(map(int,line["changes"]))==positions and set(line["changes"])==set(map(str,positions)),
                "Only the original positive-budget candidates may change")
        for p,cand,witness in zip(positions,img["candidates"],post["candidates"],strict=True):
            old_edits=edits[str(p)]
            require(p==cand["position"]==witness["position"] and cand["segmentation"]==raw[p-1]["segmentation"]
                    and comp.validate_extra_budget(old_edits)==cand["k"],"Original candidate or quota differs")
            new=line["changes"][str(p)]
            observed_counts=verify_candidate(cand,witness,new,hw)
            crosscheck_classical(cand["segmentation"],cand,old_edits,attraction,comp.NEW_ARMS)
            for arm in ARMS:
                per_arm[arm].update(observed_counts[arm])
            stats.update(observed_counts["bounds"])
            stats.update(positive_budget_candidates=1,swaps_per_arm=cand["k"],endpoint_pairs=len(cand["coords"]))
            old_edits.update(new)
        stats.update(images=1,mapped_candidates=len(raw),decoder_batches=(len(positions)+1)//2,
                     area_eligible=sum(32<=d["mask_area"]<1024 for d in audits[j-1]["candidates"][:100]))
    require(all(stats[k]==req["counts"][k] for k in ("images","mapped_candidates","area_eligible","positive_budget_candidates",
                "swaps_per_arm","endpoint_pairs")) and (stats["images"],stats["mapped_candidates"],stats["area_eligible"],
                stats["positive_budget_candidates"],stats["swaps_per_arm"])==(232,23200,11209,11054,73770),
            "Historical quota, eligible support or population differs")
    real=infer["counts"]
    require(real["images"]==232 and real["decoded_candidates"]==11054 and real["decoder_batches"]==stats["decoder_batches"]
            and real["endpoint_pairs"]==stats["endpoint_pairs"] and real["endpoint_probabilities"]==2*stats["endpoint_pairs"],
            "Accepted inference and independent mask/readout populations differ")
    margins=confirm_counts(per_arm,stats,infer)
    chain=dict(request_result_sha256=rs,request_terminal_sha256=rt,request_sha256=req["request_sha256"],
               request_worker_sha256=REQUEST_WORKER,inference_result_sha256=ifs,inference_terminal_sha256=ift,
               inference_worker_sha256=INFER_WORKER,masks_sha256=infer["masks_sha256"],narrow_posteriors_sha256=infer["narrow_posteriors_sha256"],
               input_lineage=req["input_lineage"],source_replay=req["source_replay"],historical_score_sha256=PHASES["score"][0],
               historical_score_terminal_sha256=PHASES["score"][1])
    return dict(census=stats,per_arm=per_arm,selected_margins=margins),chain,infer


def interaction_summary(means,arms,fields):
    nc,nt,wc,wt=(means[:,arms.index(a)] for a in ARMS)
    diff=(wc-wt)-(nc-nt)
    n=len(diff)
    sample=np.random.Generator(np.random.PCG64(5217)).integers(n,size=(4096,n))
    intervals=np.percentile(diff[sample].mean(axis=1),[2.5,97.5],axis=0)
    return {key:dict(delta=float(diff[:,j].mean()),percentile95=intervals[:,j].tolist()) for j,key in enumerate(fields)}


def fixed_view(gt,ids,records,metrics,prior,geom,base,modules,truth,measured,reference_arms,all_anchors):
    native,diagnostic=geom.evaluate_raw(gt,records,ids,metrics["raw"])
    if all_anchors is not None:
        retained=set(ids)
        require(native=={k:v for k,v in all_anchors.items() if v[0] in retained},
                "Subset .50 native anchors differ from official cohort")
    annotations={v["id"]:v for v in gt["annotations"]}
    geom.ARMS=(*reference_arms,*ARMS)
    geom.PAIRS=PAIRS
    exposures={pair:Counter() for pair in PAIRS}
    kept=Counter({a:0 for a in ARMS})
    rows=[]
    for gid,(iid,rank) in native.items():
        rec=records[iid]
        raw=rec["native"][rank-1]["segmentation"]
        edits=rec["edit"].get(str(rank),{})
        masks={arm:edits.get(arm,{}).get("rle",raw) for arm in geom.ARMS}
        raster=codec.merge(codec.frPyObjects(annotations[gid]["segmentation"],*raw["size"]))
        overlap=codec.iou([raw]+[masks[arm] for arm in ARMS],[raster],[False])[:,0]
        require(overlap[0]>=.50,"Reconstructed author-raster original pair differs")
        kept.update({arm:int(q>=.50) for arm,q in zip(ARMS,overlap[1:],strict=True)})
        native_area=rec["audit"][rank-1]["mask_area"]
        if not 32<=native_area<1024:
            continue
        if gid not in measured:
            cache={}
            for rle in masks.values():
                key=rle["counts"]
                if key not in cache:
                    area=int(codec.area(rle))
                    xyz=base.continuous(rle,area,truth[gid][0],modules) if area else (0.,0.,100.)
                    cache[key]=(*xyz,100.-xyz[2])
            measured[gid]=np.asarray([cache[masks[arm]["counts"]] for arm in geom.ARMS])
        rows.append((rec["rgb"],measured[gid]))
        for a,b in PAIRS:
            if masks[a]["counts"]!=masks[b]["counts"]:
                exposures[a,b][rec["rgb"]]+=1
    geom.FIELDS=("continuous_iou_points","fp_pixels_per_gt_area_pct","fn_pixels_per_gt_area_pct","intersection_per_gt_area_pct")
    g=geom.summarize(rows)
    require(len(native)==prior["original_native_matches"] and len(rows)==prior["eligible_pairs"]
            and g["decoded_image_groups_with_pairs"]==prior["decoded_image_groups_with_pairs"],
            "Original native pairs or decoded image units differ")
    g["original_native_matches"]=len(native)
    g["raw_matches_outside_eligible_area"]=len(native)-len(rows)
    for arm in reference_arms:
        for field,expected in prior["arms"][arm].items():
            require(abs(g["arms"][arm][field]-expected)<1e-9,"Historical fixed polygon geometry differs for "+arm)
    groups=defaultdict(list)
    for rgb,value in rows:
        groups[rgb].append(value)
    means=np.stack([np.mean(groups[k],axis=0) for k in sorted(groups)])
    ix=interaction_summary(means,geom.ARMS,geom.FIELDS)
    exp={a+"_minus_"+b:dict(different_hulls=sum(c.values()),different_decoded_groups=len(c))
         for (a,b),c in exposures.items()}
    counts={}
    for arm in ARMS:
        adjusted={}
        for iid in ids:
            rec=records[iid]
            revised=[dict(d,segmentation=rec["edit"].get(str(k),{}).get(arm,{}).get("rle",d["segmentation"]))
                     for k,d in enumerate(rec["native"],1)]
            adjusted[iid]=dict(native=revised)
        matches,details=geom.evaluate_raw(gt,adjusted,ids,metrics[arm])
        counts[arm]=dict(original_pairs_retained_50=kept[arm],original_pairs=len(native),
                         rematched_hulls_50=len(matches),ground_truth_hulls=details["gt_instances"])
    counts["raw"]=dict(original_pairs_retained_50=len(native),original_pairs=len(native),
                       rematched_hulls_50=len(native),ground_truth_hulls=diagnostic["gt_instances"])
    return g,exp,ix,counts,native


def difference(left,right):
    return {key:(100*(v-right[key]) if min(v,right[key])>=0 else None) for key,v in left.items()}


def run_score(a,hist):
    comp,scorer,geom,prior,_,roster,audits,original,blocks=hist
    counts,chain,inference=build_inputs(a,hist)
    print(json.dumps(dict(phase="all_native_masks_verified",positive_candidates=counts["census"]["positive_budget_candidates"])),flush=True)
    # All input receipts, ordered masks and sparse posterior witnesses have closed and passed before author truth is opened.
    gt=json.loads(pin(a.author_labels,scorer.GT_SHA))
    require(len(gt["images"])==232 and len(gt["annotations"])==546 and
            digest(("\n".join(scorer.EXCLUDED)+"\n").encode("ascii"))==scorer.EXCLUSION_SHA,
            "Frozen official truth or declared exclusion roster differs")
    views=scorer.score_populations(gt,blocks,scorer.EXCLUDED)
    base=geom.load(a.code/"DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/dev705_free_endpoint/score_endpoint.py",geom.OLD)
    shared=geom.load(a.code/"MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py",geom.GEOM)
    modules=shared.geometry_modules(a.code)
    truth,neutral,_=modules.geometry.polygon_support(gt,modules.vector)
    require(len(truth)==546 and not neutral,"All original author polygons required")
    ids={im["file_name"]:im["id"] for im in gt["images"]}
    records={}
    for r,k,e in zip(roster,audits,original,strict=True):
        name=r["image_ref"]["path"]
        stock,arms,hw=blocks[name]
        records[ids[name]]=dict(name=name,rgb=e["rgb_sha256"],native=stock,audit=k["candidates"][:100],edit=arms)
    sets={"official":sorted(records),"exclude_exact_fit_overlap":sorted(i for i,r in records.items() if r["name"] not in scorer.EXCLUDED)}
    anchors=None
    measured={}
    old_geo=("raw",*comp.NEW_ARMS)
    for view,subset in sets.items():
        results=views[view]
        metrics=results["metrics_fraction"]
        ref=prior["views"][view]
        require(all(results[f]==ref[f] for f in ("images","annotations","candidates")) and
                all(metrics[arm]==values for arm,values in ref["metrics_fraction"].items()),
                "Archived category-zero COCO vector or original dataset dimensions differ")
        geo,e,ix,coverage,current=fixed_view(gt,subset,records,metrics,prior["geometry"][view],geom,base,modules,
                                            truth,measured,old_geo,anchors)
        if anchors is None:
            anchors=current
        require(geo["eligible_pairs"]==(293 if view=="official" else 279) and
                geo["decoded_image_groups_with_pairs"]==(143 if view=="official" else 138),"Fixed matched population differs")
        effects={}
        for a1,b in PAIRS:
            name=a1+"_minus_"+b
            coco=difference(metrics[a1],metrics[b])
            paired=geo["pairwise"][a1+" minus "+b]
            iou=paired["continuous_iou_points"]
            floor=1.00 if (a1,b)==PAIRS[0] else .30
            exposed=e[name]["different_hulls"]>=20 and e[name]["different_decoded_groups"]>=8
            ap=coco["AP"]
            gate=dict(**e[name],exposure_supported=exposed,absolute_AP_threshold=floor,
                      supported_advantage=bool(exposed and ap is not None and ap>=floor and iou["percentile95"][0]>0),
                      supported_loss=bool(exposed and ap is not None and ap<=-floor and iou["percentile95"][1]<0))
            effects[name]=dict(COCO_points=coco,criteria=gate,paired_continuous_iou_points=iou)
        first=effects["wc_minus_wt"]["COCO_points"]
        second=effects["nc_minus_nt"]["COCO_points"]
        interaction={f:(first[f]-second[f] if first[f] is not None and second[f] is not None else None) for f in first}
        results.update(geometry=geo,coverage=coverage,contrasts=effects,
                       interaction=dict(definition="(wc-wt)-(nc-nt)",COCO_points=interaction,paired_geometry=ix))
        print(json.dumps(dict(phase="scored_view",view=view,images=len(subset))),flush=True)
    official=views["official"]
    return dict(status=SCORE_STATUS,phase="score",worker_sha256=digest(Path(__file__).read_bytes()),protocol_sha256=LOCK,
        **chain,counts=counts,views=views,primary_view="official",primary_contrast="wc_minus_wt",primary=official["contrasts"]["wc_minus_wt"],
        secondary={a1+"_minus_"+b:official["contrasts"][a1+"_minus_"+b] for a1,b in PAIRS[1:]},interaction=official["interaction"],
        historical_COCO_arms_exact=8,historical_fixed_geometry_arms_exact=4,narrow_signed_rank_check_exact=True,
        signed_split_objective=inference["signed_split_objective"],
        exact_dyadic_rank_certificates=inference["counts"]["exact_dyadic_rank_certificates"],
        legacy_allocation_diagnostics={k:inference["counts"][k] for k in
                                      ("legacy_rounded_split_differs","legacy_rank_violation","legacy_true_objective_inferior")},
        native_category=1,author_category=0,author_labels_sha256=scorer.GT_SHA,target_labels_opened=True,
        detector_forwards=0,phoenix_forwards=0,optimizer_updates=0,bootstrap_draws=4096,bootstrap_seed=5217,
        cluster_unit="decoded RGB groups with eligible fixed matches",inference_seconds=inference["seconds"])


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("code","owner","capture","classical-root","request","request-terminal","inference","inference-terminal",
                 "author-labels","protocol","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--masks",type=Path)
    p.add_argument("--posteriors","--narrow-posteriors",dest="posteriors",type=Path)
    p.add_argument("--original-readback",type=Path)
    p.add_argument("--protocol-sha256",required=True)
    a=p.parse_args()
    for key,path in vars(a).items():
        if isinstance(path,Path):
            setattr(a,key,path.resolve(strict=key!="output"))
    require(not a.output.exists() and a.protocol.name==LOCK_NAME and digest(a.protocol.read_bytes())==a.protocol_sha256==LOCK,
            "Fresh output and the exact locked protocol required")
    require(shapely.__version__=="2.0.2" and shapely.geos_version_string=="3.11.2","Historical geometry runtime required")
    start=perf_counter()
    answer=run_score(a,historical(a))
    answer["score_seconds"]=perf_counter()-start
    raw=(json.dumps(answer,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
    a.output.mkdir()
    with (a.output/"RESULT_SAFE.json").open("xb") as f:
        require(f.write(raw)==len(raw),"Incomplete safe score")
    print(json.dumps(dict(status=SCORE_STATUS,result_sha256=digest(raw))),flush=True)


if __name__=="__main__":
    main()
