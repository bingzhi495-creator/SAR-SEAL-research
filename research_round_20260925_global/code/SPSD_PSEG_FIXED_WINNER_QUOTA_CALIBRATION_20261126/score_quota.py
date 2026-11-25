"""Verify actual signed masks, then measure the source-development quota comparison."""
from argparse import ArgumentParser
from collections import Counter
import json
import os
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import HERE, LOCK, QUOTA, closed, closeout, load, need, pin, settings
from render_quota import NEW, STATUS as RENDER_STATUS
import numpy as np
from pycocotools import mask as codec

BASE_SCORE = "b5c61ebbc9d02994b2d9cac210a19e4091693055ec97e257cc8063095c4b86f2"
BASE_SCORE_RESULT = "f86390623ffd658e91c17e8a7b99e0859bcc3918b9fce0afda2ff332d72d023a"
BASE_SCORE_EXIT = "a3ad0ea5274ecfd7c11c7baf1d4ad2047b1926790871268922c5b90fc97edeb4"
BASE_RENDER_RESULT = "d12a193bf80e6fb24b3c853c8da7d342f6408aa22e9f3920b873f2d5eb3de1e5"
BASE_RENDER_EXIT = "cb84320a93db76c5d11156c40aca0b2b1bd4d0fa2bd746a7d68c6c86ccaef9d2"
BASE_RENDERER = "a0c11c5c37c3a8569e1a54047fbc09598604710861bcd19f4096a0e3abe35959"
KEYS = ("gate","archive","authority","points","labels","group_protocol","base_score","base_render","mechanism")
RIVALS = ("variable_iou","fixed_iou","reversed_quota","near_iou","bce_robust","bce_mean","static")


def exposure_worker(source):
    anchor='new=masks["variable_iou"]'
    need(source.count(anchor)==1,"Expected exact common positive matched-hull geometry")
    return source.replace(anchor,'new=masks["calibrated_quota"]')


def read_records(run_dir, sha):
    rows = [json.loads(line) for line in pin(run_dir/"GATE_RENDER_PRIVATE.jsonl",sha).splitlines()]
    out={i["image_id"]:i for i in rows}
    need(len(rows)==len(out)==183, "Full ordered source-development roster")
    return out


def check_base(current,original):
    need(current.keys()==original.keys(),"Exact original image order")
    for img in current:
        a,b = current[img],original[img]
        need(all(a[key]==value for key,value in b.items() if key!="candidates"), "Original image identity")
        for new,old in zip(a["candidates"],b["candidates"],strict=True):
            for key,value in old.items():
                if key=="changes" or key.startswith("variable_") and isinstance(value,dict):
                    need(all(new[key][arm]==z for arm,z in value.items()), "Exact existing writer RLE/support/estimate")
                else:
                    need(new[key]==value, "Exact original score, native mask and metadata")
            if new.get("quota") and new["variable_signed_counts"][NEW[0]][0] == new["quota"]:
                need(new["changes"][NEW[0]]==new["changes"]["fixed_iou"],"The calibrated fixed decision keeps fixed endpoints")
            if new.get("quota") and new["variable_signed_counts"][NEW[1]][0] == new["quota"]:
                need(new["changes"][NEW[1]]==new["changes"]["fixed_iou"],"The control fixed decision keeps fixed endpoints")
            if new.get("quota") and new["variable_signed_counts"][NEW[0]]==new["variable_signed_counts"][NEW[1]]:
                need(new["changes"][NEW[0]]==new["changes"][NEW[1]],"Learned and control share each within-split mask")


def error_counts(truth,native,matches,arms):
    totals={arm:Counter() for arm in arms}
    for hull in truth["annotations"]:
        truth_rle=hull["segmentation"]
        ref_area=int(codec.area(truth_rle))
        match=matches.get(hull["id"])
        if match is None:
            for arm in arms:
                totals[arm]["missed_area_cells"]+=ref_area
            continue
        iid,rank=match
        need(iid==hull["image_id"],"Exact original image and native association")
        candidate=native[iid][rank-1]
        for arm in arms:
            pred=candidate["changes"].get(arm,candidate["native"])
            hit=int(codec.area(codec.merge([truth_rle,pred],intersect=True)))
            totals[arm].update(excess_area_cells=int(codec.area(pred))-hit,missed_area_cells=ref_area-hit)
    return totals


def run(cfg,root,out):
    start=perf_counter()
    render,render_sha=closed(root/"render/RESULT_SAFE.json",root/"observed_render/TERMINAL.json",HERE/"render_quota.py")
    need(render["status"]==RENDER_STATUS and render["protocol_sha256"]==LOCK and render["quota_core_sha256"]==QUOTA
         and render["source_gate_labels_opened"] is render["fit_labels_opened"] is render["historical_test_opened"] is False
         and render["optimizer_updates"]==render["detector_forwards"]==0,"Frozen successful render before label access")
    fit,fit_sha=closed(root/"fit/RESULT_SAFE.json",root/"observed_fit/TERMINAL.json",HERE/"fit_quota.py")
    need(render["calibration_fit_result_sha256"]==fit_sha and render["calibrator_sha256"]==fit["calibrator_sha256"],
         "Exact prepared FIT calibrator identity")
    previous_render,old_render_sha=closed(cfg.base_render/"render/RESULT_SAFE.json",
         cfg.base_render/"RENDER_TERMINAL.json",cfg.base_render/"render_gate.py",BASE_RENDER_RESULT,BASE_RENDER_EXIT,BASE_RENDERER)
    historical,old_score_sha=closed(cfg.base_score/"score/RESULT_SAFE.json",
         cfg.base_score/"SCORE_TERMINAL.json",cfg.base_score/"score_gate.py",BASE_SCORE_RESULT,BASE_SCORE_EXIT,BASE_SCORE)
    need(historical["render_result_sha256"]==old_render_sha and historical["exact_all_archive_score_parity"],
         "Complete original 5297 matched-baseline comparison")
    old_score=load(cfg.base_score/"score_gate.py",BASE_SCORE,exposure_worker)
    auth=load(cfg.authority,old_score.ARCH_AUTHORITY)
    mech=load(cfg.mechanism,old_score.MECHANISM)
    old_driver, _, scorer, reader, base, archive, groups, _, _=auth.archive(cfg)
    pin(cfg.group_protocol,old_score.GROUP)
    points=json.loads(pin(cfg.points/"actual_retry/RESULT_SAFE.json",old_driver.POINTS))
    need(render["point_terminal_sha256"]==base["point_terminal_sha256"]==previous_render["point_terminal_sha256"]
         and render["archive_render_sha256"]==old_score.ARCH_RENDER and render["archive_score_sha256"]==old_score.ARCH_SCORE
         and render["archive_bce_tensor_sha256"]==old_score.ARCH_BCE and render["exact_bce_archive_rle_parity"],
         "Same detector, head, point packets and archive as historical score")
    records=read_records(root/"render",render["render_stream_sha256"])
    reference_records=read_records(cfg.base_render/"render",previous_render["render_stream_sha256"])
    check_base(records,reference_records)
    del reference_records
    old_score.CUES=(*old_score.CUES,*NEW)
    sizes,net_area=old_score.signed_support(cfg,points,records,archive,groups,render,auth.REFERENCE_ARMS,mech)
    print("Every historical mask and all real signed edits verified before label access",flush=True)
    truths=reader.labels(cfg,records)
    native={iid:row["candidates"] for iid,row in records.items()}
    arms=(*auth.REFERENCE_ARMS,*old_score.CUES)
    order=sorted(set(groups.values()))
    draws=np.random.Generator(np.random.PCG64(5297)).integers(180,size=(4096,180))
    refs={}
    matches={}
    for ref,truth in truths.items():
        history=historical["references"][ref]
        metrics={}
        for arm in arms:
            metrics[arm],hit=scorer.coco_score(truth,native,arm)
            if arm in history["metrics"]:
                need(all(abs(metrics[arm][key]-v)<=1e-12 for key,v in history["metrics"][arm].items()),
                     "Exact old mask AP, AP75, AP90, APs and recalls")
            if arm=="native":
                matches[ref]=hit
                need(len(hit)==history["original_matches"] and len(truth["annotations"])==history["author_hulls"],
                     "Exact original native match and hull denominator")
        values=scorer.geometry(arms,truth,native,matches[ref],groups,order)
        fixed=dict(zip(arms,values.mean(axis=0).tolist(),strict=True))
        need(all(abs(fixed[arm]-v)<=1e-12 for arm,v in history["fixed_iou_points"].items()),
             "Exact old fixed native-match group-weighted IoU")
        compared={arm:old_score.contrast(arms,values,metrics,draws,NEW[0],arm) for arm in RIVALS}
        refs[ref]=dict(metrics=metrics,fixed_iou_points=fixed,calibrated_minus=compared,
          normalized_scope="all author hulls, unmatched counted at zero",
          pixel_error_counts=error_counts(truth,native,matches[ref],("variable_iou","fixed_iou","near_iou",*NEW)),
          original_matches=len(matches[ref]),author_hulls=len(truth["annotations"]))
    old_score.CONTROLS=RIVALS
    exposure,shared=old_score.matched_pairs(truths,matches,native,groups)
    utility={}
    for ref,doc in refs.items():
        compared=doc["calibrated_minus"]
        utility[ref]=dict(main={arm:compared[arm]["iou_points"]>=.15 and compared[arm]["percentile95_points"][0]>0
                                        and compared[arm]["ap_points"]>=-.10 for arm in ("variable_iou","fixed_iou")},
                          learned_correspondence=compared[NEW[1]]["iou_points"]>0)
    exposure_pass=exposure["variable_iou"][0]>=20 and exposure["variable_iou"][1]>=8
    pass_all=exposure_pass and all(all(doc["main"].values()) and doc["learned_correspondence"] for doc in utility.values())
    out.mkdir()
    closeout(out/"RESULT_SAFE.json",Path(__file__),"SPSD_PSEG_QUOTA_CALIBRATION_SCORE_COMPLETE",
             before_labels_full_replay_verified=True,all_historical_masks_and_scores_exact=True,
             equal_total_actual_flips=True,source_development_only=True,external_target_opened=False,
             historical_test_opened=False,fit_labels_opened=False,optimizer_updates=0,detector_forwards=0,
             fit_result_sha256=fit_sha,render_result_sha256=render_sha,original_render_sha256=old_render_sha,
             original_score_sha256=old_score_sha,refs=refs,shared_hulls=shared,realized_signed_usage=sizes,
             net_area_change_histograms=net_area,source_counts=render["counts"],bootstrap_draws=4096,
             criteria=dict(utility=utility,exposure_pass=exposure_pass,pass_all=pass_all),elapsed_seconds=perf_counter()-start)


def main():
    p=ArgumentParser(description=__doc__)
    for arg in ("inputs","root","output"):
        p.add_argument("--"+arg,type=Path,required=True)
    a=p.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1"
         and not a.output.exists(),"Fresh isolated author scoring")
    pin(HERE/"PRELOCK.md",LOCK)
    run(settings(a.inputs,KEYS),a.root,a.output)

if __name__=="__main__":
    main()
