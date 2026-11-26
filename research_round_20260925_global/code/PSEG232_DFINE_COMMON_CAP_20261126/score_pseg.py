"""Score an independently replayed D-FINE common-cap comparison on PSeg and SPSD."""
from argparse import ArgumentParser
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask as codec


OLD_SCORE_CODE="dd8e435976d2331146c8f83d70f368ee4e0954c05520c54fa9a3b1205838d40f"
OLD_SCORE="9bc6411cf35ee355f142a5fa5588c8e5820127b8101acf4da0e653e70553d3d1"
OLD_GEOM_CODE="107f31864bbb7e807a4194e552c7a1778d410f0e564f80437c5717fe6b47098d"
OLD_GEOM="5daed34a1df0c93d5c379423191bd8bc993561171ea021cbe22178783bbcd660"
OLD_REFERENCE_CODE="9baf865d510b3af95510ca55017943e5147a9c9cc68384665f14cb40b07434a9"
OLD_DUAL="dd152bd587d64453b874a9a34e6531adacbd7bf163de4f9b22ffb25c4dea52fc"
OLD_DUAL_TERMINAL="3db31ce1be04638f80fd227b8c6656925ba580c2e12c54d6b7a4a0c5188964c3"
OLD_REPLAY="531b979c316c58f608e48205a2f86e66e666eae61c1029de59c8229a941c937e"
OLD_READBACK="e99d0f2bbd9e1839e260a88075f6af187e05ffc2b86abaf6cc33b788372b7b71"
SPARSE="cb9b8e839de85588fdf116ce5554ed6cadbc20bdbe76e38180561528bfd870d5"
SOLVER="99817f192798f52f03efe5e3ac422a453019bf458d33096a03844cb60360436c"
SCORE_STATUS="PSEG232_DFINE_COMMON_CAP_SCORE_COMPLETE"
ARMS=("raw","fixed","bounded","free_matched_count")
LEGACY=("raw","fixed","free_matched_count")
PAIRS=(("free_matched_count","bounded"),("bounded","fixed"),("free_matched_count","fixed"))
QUALIFIED_SOURCE=dict(
    manifest_sha256="9c0b9bc07eade8b9de4349f187a80837f6a1c8449e596a2c1234f3827356b65e",
    capture_result_sha256="6107389d36848d8d0e24db2cc82518f73494cd990a9f71fe18e473da0f1201c9",
    capture_terminal_sha256="194e1a52763773dee4c369fd5952cc3332148bf67ef208d6ef55cd36c37942f7",
    capture_readback_sha256="dc48e18c0b988fda3eada68dc06b1404fd5c3806502e32d455a08a2e96d0121c",
    old_replay_result_sha256=OLD_REPLAY,old_readback_sha256=OLD_READBACK,
    old_replay_terminal_sha256="1f11f07bfe772ba27dfc35387ce5592ab635405abf8bf311bc78d94aac6b0b2c",
    old_edits_sha256="4c9638e9b502cf7048e0cb6e52d88b14ff71d78b1d6ec728dc533728ae1220bc",
    solver_sha256=SOLVER,weight_sha256="eca85da01c53545485928ae3c502844839083d4ebff896e3df1f91a072d9f725")
QUALIFIED_COUNTS=dict(images=232,candidates=23200,positive=6404,swaps=29304)


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, sha):
    raw=Path(path).read_bytes()
    need(digest(raw)==sha,"Exact immutable input "+Path(path).name)
    return raw


def module(path, sha, name):
    m=ModuleType(name)
    m.__file__=str(path)
    exec(compile(pin(path,sha),str(path),"exec"),vars(m))
    return m


def closed(path,terminal,sha,term_sha,worker,phase):
    result=json.loads(pin(path,sha))
    seen=json.loads(pin(terminal,term_sha))
    need(seen["phase"]==phase and seen["actual_exit"]==0 and seen["output_footer_verified"] is True
         and seen["worker_sha256"]==worker and seen["result_sha256"]==sha
         and result.get("worker_sha256",result.get("source_sha256"))==worker,
         "Observed successful closure for "+phase)
    return result


def stream(path, sha):
    raw=pin(path,sha)
    need(raw.endswith(b"\n"),"Complete newline-delimited ordered stream")
    return [json.loads(line) for line in raw.splitlines()]


def validate_qualification(a, render, verify):
    for current, phase, independent in ((render,"RENDER",False),(verify,"VERIFY",True)):
        need(current["status"]=="PSEG232_DFINE_COMMON_CAP_"+phase+"_COMPLETE"
             and current["independent_replay"] is independent
             and current["protocol_sha256"]==a.protocol_sha
             and current["prelock_commit"]==a.prelock_commit
             and all(current["counts"][k]==v for k,v in QUALIFIED_COUNTS.items()),
             "Original cohort and executed independent qualification")
        need(all(current[k]==v for k,v in QUALIFIED_SOURCE.items())
             and all(current[k] is True for k in
                     ("eof_complete","exact_archived_five_arm_bytes","original_f_i_reconstructed",
                      "exact_signed_cap_readback"))
             and all(current[k]==0 for k in ("label_files_opened","detector_forwards","optimizer_updates")),
             "Original model, signed counts, complete ledger and label-free closure")
    need(render["render_result_sha256"] is None and render["render_terminal_sha256"] is None
         and render["edits_sha256"]==verify["edits_sha256"]
         and render["counts"]==verify["counts"] and render["solver_totals"]==verify["solver_totals"]
         and verify["render_result_sha256"]==a.render_sha
         and verify["render_terminal_sha256"]==a.render_terminal_sha,
         "Independent receipt binds exact originally observed write and edit tape")


def selected_geometry(sparse, original, item):
    coords=np.asarray(item["coordinates"])
    h,w=original["size"]
    n=len(coords)
    k=item["fixed"]["swaps"]
    need(set(item)=={"coordinates","fixed_pairs","bounded_pairs","independent_pairs","cap_squared",*ARMS[1:]}
         and coords.shape==(n,2) and coords.dtype.kind in "iu" and len(np.unique(coords))==coords.size
         and np.all((0<=coords)&(coords<h*w)) and type(k) is int and 0<k<=min(8,n) and n<=32,
         "Exact scope, support and original signed quota")
    endpoints=np.cumsum(sparse.decode_runs(original["counts"]),dtype=np.int64)
    foreground=np.searchsorted(endpoints,(coords%w)*h+coords//w,side="right")%2
    need(not foreground[:,0].any() and foreground[:,1].all(),"Original two signed native supports")
    for arm,key in zip(ARMS[1:],("fixed_pairs","bounded_pairs","independent_pairs"),strict=True):
        indices=np.asarray(item[key])
        need(indices.dtype.kind in "iu" and indices.shape==(k,2) and np.all((0<=indices)&(indices<n))
             and all(np.unique(indices[:,s]).size==k for s in range(2)),"Disjoint selected endpoint slots")
        mask=item[arm]
        selected=coords[indices,np.arange(2)]
        need(mask["swaps"]==k and sparse.xor_rle(original,selected.ravel())==mask["rle"],
             "Exact same-geometry full RLE reconstructs on signed support")
        y,x=np.divmod(selected,w)
        squared=(y[:,0]-y[:,1])**2+(x[:,0]-x[:,1])**2
        if arm=="fixed":
            need(int(squared.max())==item["cap_squared"] and np.array_equal(indices[:,0],indices[:,1]),
                 "F original selected paired rows define exact common maximum edge")
        elif arm=="bounded":
            need(np.all(squared<=item["cap_squared"]),"Every L pair is within F's cap")
    return k


def source_rows(a,old):
    render=closed(a.render_root/"render/RESULT_SAFE.json",a.render_root/"observed_render/TERMINAL.json",
                  a.render_sha,a.render_terminal_sha,a.render_worker_sha,"prepare-edits")
    verify=closed(a.render_root/"verify/RESULT_SAFE.json",a.render_root/"observed_verify/TERMINAL.json",
                  a.verify_sha,a.verify_terminal_sha,a.render_worker_sha,"verify-edits")
    validate_qualification(a,render,verify)
    readback=json.loads(pin(a.old_readback,OLD_READBACK))
    old.check_acceptance(readback)
    previous=json.loads(pin(a.original_replay/"RESULT_SAFE.json",OLD_REPLAY))
    need(previous["edits_sha256"]==readback["edits_sha256"] and readback["replay_sha256"]==OLD_REPLAY,
         "Exact previous original writer closure")
    old_terminal=json.loads(pin(a.owner/"REPLAY_TERMINAL_PRIVATE.json",readback["original_tool_wait_sha256"]))
    need(old_terminal["original_process_terminal"] is True and old_terminal["original_exit"]==0
         and old_terminal["result_sha256"]==OLD_REPLAY,"Closed previous original writer")
    roster=old.validate_manifest(json.loads(pin(a.owner/"TEST232_IMAGE_ONLY_MANIFEST_PRIVATE.json",
                                                readback["manifest_sha256"])))
    captures=stream(a.capture/"DEV_MASK_PREDICTIONS_PRIVATE.jsonl",readback["input_predictions_sha256"])
    audits=stream(a.capture/"DEV_CANDIDATE_AUDIT_PRIVATE.jsonl",readback["candidate_audit_sha256"])
    archive=stream(a.original_replay/"EDITS_PRIVATE.jsonl",readback["edits_sha256"])
    fresh=stream(a.render_root/"render/EDITS_PRIVATE.jsonl",verify["edits_sha256"])
    # Validate original stable native IDs, mask canvas, rank, confidence and budgets.
    stock=old.join_blocks(roster,captures,audits,archive)
    sparse=module(a.code/"RFDETR_SPARSE_RLE_EQUIVALENCE_20261120/sparse_rle.py",SPARSE,"_pseg_sparse")
    rgbs={}
    rows,swaps,candidate_total=0,0,0
    for item, past, change in zip(roster,archive,fresh,strict=True):
        iid=item["image_id"]
        name=item["image_ref"]["path"]
        name_stock, options, _=stock[name]
        need(set(change)=={"image_id","rgb_sha256","changes"} and
             change["image_id"]==past["image_id"]==iid and change["rgb_sha256"]==past["rgb_sha256"],
             "Exact image identity and decoded group preserved")
        expected={str(pos) for pos,op in options.items() if "fixed" in op}
        need(set(change["changes"])==expected and len(name_stock)==100,"Exactly the original F-positive native slots")
        for rank,entry in change["changes"].items():
            original=options[rank]
            need(entry["fixed"]==original["fixed"] and entry["free_matched_count"]==original["free_matched_count"],
                 "Archived F/I reproduce byte-for-byte while L alone is new")
            native=name_stock[int(rank)-1]["segmentation"]
            swaps+=selected_geometry(sparse,native,entry)
            rows+=1
            original["bounded"]=entry["bounded"]
        rgbs[name]=past["rgb_sha256"]
        candidate_total+=len(name_stock)
    need(len(stock)==len(set(rgbs.values()))==232 and (candidate_total,rows,swaps)==(23200,6404,29304),
         "Complete original candidate, editing and exact RGB groups")
    return roster,stock,rgbs,verify


def compare(current, former, scope):
    if isinstance(former,dict):
        need(set(current)==set(former),"Same fields for "+scope)
        for k in former:
            compare(current[k],former[k],scope+"/"+k)
    elif isinstance(former,list):
        need(len(current)==len(former),"Same field length for "+scope)
        for a,b in zip(current,former,strict=True):
            compare(a,b,scope)
    elif isinstance(former,float):
        need(abs(current-former)<=1e-12,"Exact within arithmetic guard: "+scope)
    else:
        need(current==former,"Exact baseline "+scope)


def decision(ap, interval, count, groups):
    covered=count>=20 and groups>=8
    if covered and ap>=.30 and interval[0]>0:
        return "positive"
    if covered and ap<=-.30 and interval[1]<0:
        return "negative"
    return "inconclusive"


def primary_geometry(a, pseg, stock, rgbs, old, geo, official, former):
    source=geo.load(a.code/"DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/dev705_free_endpoint/score_endpoint.py",geo.OLD)
    geometry=geo.load(a.code/"MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py",geo.GEOM)
    modules=geometry.geometry_modules(a.code)
    truth,neutral,_=modules.geometry.polygon_support(pseg,modules.vector)
    need(len(truth)==546 and not neutral,"Exact full original author continuous geometry")
    images={r["id"]:r["file_name"] for r in pseg["images"]}
    records={iid:dict(native=stock[name][0],rgb=rgbs[name]) for iid,name in images.items()}
    geo.ARMS=ARMS
    geo.PAIRS=PAIRS
    result, full_anchors, cached = {}, None, {}
    for view,d in official.items():
        ids=sorted(i for i,name in images.items() if view=="official" or name not in old.EXCLUDED)
        anchors,raw_stats=geo.evaluate_raw(pseg,records,ids,d["metrics_fraction"]["raw"])
        if full_anchors is None:
            full_anchors=anchors
        else:
            need(anchors=={k:v for k,v in full_anchors.items() if v[0] in ids},
                 "Nested exact-duplicate sensitivity retains original raw match identities")
        grouped=[]
        exposed={p:(0,set()) for p in PAIRS}
        for gid,(iid,rank) in anchors.items():
            name=images[iid]
            native,edits,_=stock[name]
            raw=native[rank-1]["segmentation"]
            area=int(codec.area(raw))
            if not 32<=area<1024:
                continue
            updates=edits.get(str(rank),{})
            masks=[updates.get(arm,{}).get("rle",raw) for arm in ARMS]
            if gid not in cached:
                measured={}
                for rle in masks:
                    key=rle["counts"]
                    if key not in measured:
                        measured[key]=source.continuous(rle,area,truth[gid][0],modules)
                cached[gid]=np.asarray([measured[rle["counts"]] for rle in masks])
            grouped.append((rgbs[name],cached[gid]))
            for pair in PAIRS:
                if masks[ARMS.index(pair[0])]!=masks[ARMS.index(pair[1])]:
                    count,groups=exposed[pair]
                    groups.add(rgbs[name])
                    exposed[pair]=count+1,groups
        summary=geo.summarize(grouped)
        old_geo=former["views"][view]
        need(summary["eligible_pairs"]==old_geo["eligible_pairs"]
             and summary["decoded_image_groups_with_pairs"]==old_geo["decoded_image_groups_with_pairs"]
             and len(anchors)==old_geo["stock_matches_at_50"],"Exact historical 293/143 and 279/138 cohorts")
        for arm in LEGACY:
            previous=old_geo["arms"][arm]
            compare(summary["arms"][arm],previous,view+"/"+arm)
        key="free_matched_count minus fixed"
        previous=old_geo["pairwise"][key]
        compare(summary["pairwise"][key],previous,view+"/I-F")
        compare(raw_stats,old_geo["raw_mask_score_diagnostics"],view+"/raw score diagnostics")
        estimates={}
        for left,right in PAIRS:
            label=left+" minus "+right
            ap=100*(d["metrics_fraction"][left]["AP"]-d["metrics_fraction"][right]["AP"])
            changed,groups=exposed[left,right]
            values=summary["pairwise"][label]
            estimates[label]=dict(ap_points=ap,geometry=values,changed_eligible_matches=changed,
                                  changed_decoded_groups=len(groups))
            if view=="official" and (left,right)==PAIRS[0]:
                estimates[label]["decision"]=decision(ap,values["continuous_iou_points"]["percentile95"],
                                                       changed,len(groups))
        result[view]=dict(images=d["images"],annotations=d["annotations"],candidates=d["candidates"],
                          metrics_fraction={arm:d["metrics_fraction"][arm] for arm in ARMS},
                          original_native_matches=len(anchors),
                          original_unmatched_author_hulls=d["annotations"]-len(anchors),
                          fixed_geometry=summary,contrasts=estimates)
    return result


def dual_measure(pseg,spsd,stock,rgbs,old,geo,reference,former):
    old.ARMS=ARMS
    reference.PAIRS=PAIRS
    draws=np.random.Generator(np.random.PCG64(5297)).integers(232,size=(4096,232))
    references,changes={},{}
    for kind,gt,expected_matches in (("PSeg_SSDD",pseg,402),("SPSD",spsd,403)):
        out,delta=reference.score_reference(gt,stock,rgbs,old,geo,draws)
        prev=former["references"][kind]
        need(out["matched_hulls"]==expected_matches and out["matched_hulls"]==prev["matched_hulls"]
             and out["image_groups"]==prev["image_groups"]==232
             and out["hulls"]==prev["hulls"],"Both original author denominators and fixed matches")
        for arm in LEGACY:
            compare(out["metrics"][arm],prev["metrics"][arm],kind+"/"+arm)
        key="free_matched_count_minus_fixed"
        compare(out["pairwise"][key],prev["pairwise"][key],kind+"/I-F")
        del out["prior_score_parity"]
        out["legacy_n_f_i_parity"]=True
        references[kind]=out
        changes[kind]=delta
    difference={}
    for left,right in PAIRS:
        key=left+"_minus_"+right
        first=references["PSeg_SSDD"]["pairwise"][key]
        second=references["SPSD"]["pairwise"][key]
        difference[key]=dict(AP=second["AP"]-first["AP"],geometry=reference.interval(
            changes["SPSD"][key]-changes["PSeg_SSDD"][key],draws))
    return dict(references=references,spsd_minus_pseg_effect=difference,
                bootstrap_seed=5297,bootstrap_draws=4096,unique_same_image_groups=232)


def main():
    p=ArgumentParser(description=__doc__,allow_abbrev=False)
    for key in ("code","protocol","render-root","owner","capture","old-readback","original-replay",
                "old-dual-terminal","official","slgsea","output"):
        p.add_argument("--"+key,type=Path,required=True)
    for key in ("protocol-sha","prelock-commit","render-sha","render-terminal-sha",
                "verify-sha","verify-terminal-sha","render-worker-sha"):
        p.add_argument("--"+key,required=True)
    a=p.parse_args()
    started=perf_counter()
    need(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1"
         and np.__version__=="1.26.4" and version("pycocotools")=="2.0.11"
         and a.output.parent.is_dir() and not a.output.exists()
         and re.fullmatch("[0-9a-f]{40}",a.prelock_commit) is not None,"Original isolated CPU environment")
    pin(a.protocol,a.protocol_sha)
    old=module(a.code/"PSEG_TEST232_DFINE_FROZEN_SCORE_20261121/score_test232.py",OLD_SCORE_CODE,"_pseg_original_coco")
    geo=module(a.code/"PSEG_TEST232_PAIRED_GEOMETRY_20261121/paired_geometry.py",OLD_GEOM_CODE,"_pseg_original_geometry")
    reference=module(a.code/"SPSD_PSEG232_ANNOTATION_SENSITIVITY_20261125/score_reference.py",OLD_REFERENCE_CODE,"_pseg_dual")
    roster,stock,rgbs,verified=source_rows(a,old)
    prior_score=json.loads(pin(a.code.parent/"results/PSEG_SSDD_TEST232_DFINE_FROZEN_SCORE_ACTUAL_20261121.json",OLD_SCORE))
    prior_geo=json.loads(pin(a.code.parent/"results/PSEG_TEST232_PAIRED_GEOMETRY_SAFE_20261121.json",OLD_GEOM))
    prior_dual=closed(a.code.parent/"results/SPSD_PSEG232_ANNOTATION_SENSITIVITY_20261125_SAFE.json",
                      a.old_dual_terminal,OLD_DUAL,OLD_DUAL_TERMINAL,OLD_REFERENCE_CODE,"score")
    need(prior_geo["score_sha256"]==OLD_SCORE and prior_dual["identity"]["source_readback_sha256"]==OLD_READBACK,
         "All original published scores use the accepted native carrier")
    prior_protocol=a.code/"SPSD_PSEG232_ANNOTATION_SENSITIVITY_20261125"
    pin(prior_protocol/"PRELOCK.md",reference.LOCK)
    pin(prior_protocol/"ERRATUM.md",reference.ERRATUM)
    import shapely
    need(shapely.__version__=="2.0.2" and shapely.geos_version_string=="3.11.2",
         "Exact source continuous polygon engine")
    print("Exact label-free render, independent replay and original signed masks verified.",flush=True)
    pseg,spsd,pan_sha,disagreement=reference.read_references(a,old,roster)
    need(pan_sha==prior_dual["spsd_annotation_json_sha256"]
         and disagreement==prior_dual["spsd_metadata_disagreement"]
         and digest(("\n".join(old.EXCLUDED)+"\n").encode("ascii"))==old.EXCLUSION_SHA,
         "Exactly the two original author releases and ten exclusions")
    original_arms=old.ARMS
    old.ARMS=(*original_arms,"bounded")
    official=old.score_populations(pseg,stock,old.EXCLUDED)
    for view,current in official.items():
        parent=prior_score["views"][view]
        for field in ("images","annotations","candidates"):
            need(current[field]==parent[field],"No changed historical benchmark denominator")
        for arm in original_arms:
            compare(current["metrics_fraction"][arm],parent["metrics_fraction"][arm],view+" original "+arm)
    primary=primary_geometry(a,pseg,stock,rgbs,old,geo,official,prior_geo)
    dual=dual_measure(pseg,spsd,stock,rgbs,old,geo,reference,prior_dual)
    result=dict(status=SCORE_STATUS,phase="score",worker_sha256=digest(Path(__file__).read_bytes()),
                protocol_sha256=a.protocol_sha,prelock_commit=a.prelock_commit,
                render_result_sha256=a.render_sha,render_terminal_sha256=a.render_terminal_sha,
                verify_result_sha256=a.verify_sha,verify_terminal_sha256=a.verify_terminal_sha,
                edits_sha256=verified["edits_sha256"],original_coco_and_geometry_parity=True,
                original_population_and_signed_edit_counts=QUALIFIED_COUNTS,
                primary_paired_bootstrap_draws=4096,primary_paired_bootstrap_seed=5217,
                primary_paired_bootstrap_unit="decoded input JPEG with a valid original fixed match",
                views=primary,same_image_reference_sensitivity=dual,labels_opened_after_signed_verify=True,
                original_author_views_historical=True,scene_independence_unverified=True,
                detector_forwards=0,optimizer_updates=0,elapsed_seconds=perf_counter()-started)
    raw=(json.dumps(result,sort_keys=True,allow_nan=False,indent=2)+"\n").encode()
    a.output.mkdir()
    with (a.output/"RESULT_SAFE.json").open("xb") as h:
        h.write(raw)
        h.flush()
        os.fsync(h.fileno())
    print(json.dumps(dict(status=SCORE_STATUS,result_sha256=digest(raw))),flush=True)


if __name__=="__main__":
    main()

