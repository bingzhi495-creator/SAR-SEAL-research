"""Record threshold AP and transitions inside the archived SRNet COCO scorer."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np


SCORE_SHA = "8f26062eed2f56501f1a50a656961535b9fff3aa7ed493b7d4fe758f44b1f77a"
TERMINAL_SHA = "799047e9123044aff40ef1bead0698ba6a7d103aa744d9d4934d766d562012e9"
AUTHORITY_SHA = "7121fd1b7c1c98ee87ce6cf23963c09fabc437af458f57fffa43fb99a75c6339"
SCORE_WORKER = "6ba17a0bcaa897f6f56bca354c22b2db57b79e38b18f60f50cc4ae3498613889"
STATES = ("TP","FP","ignore")
ARMS = ("raw","fixed","free_matched_count")
CONTRASTS = ((ARMS[2],ARMS[1]),(ARMS[2],ARMS[0]),(ARMS[1],ARMS[0]))
STATUS = "SRNET_RSDD74_THRESHOLD_AP_TRANSITIONS_COMPLETE"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(data):
    return sha256(data).hexdigest()


def pin(path, expected):
    raw=path.read_bytes()
    require(digest(raw)==expected,"Exact archived SHA-256: "+path.name)
    return raw


def module(path, expected):
    mod=ModuleType(path.stem)
    mod.__file__=str(path)
    exec(compile(pin(path,expected),str(path),"exec"),mod.__dict__)
    return mod


def detection_transitions(old, new):
    require(old.keys()==new.keys(),"Identical included detections under maxDets 100")
    counts=Counter({f"{x} to {y}":0 for x in STATES for y in STATES})
    same=changed=0
    for key, (before,old_gt) in old.items():
        after,new_gt=new[key]
        counts[f"{before} to {after}"]+=1
        if before==after=="TP":
            same+=old_gt==new_gt
            changed+=old_gt!=new_gt
    return dict(transition_counts=dict(counts),TP_to_same_reference=same,TP_to_different_reference=changed)


def fixed_crossings(old, new, threshold):
    require(len(old)==len(new),"Identical original .50 native assignments")
    before,after=np.asarray(old)>=threshold,np.asarray(new)>=threshold
    return dict(above_to_above=int(np.sum(before&after)),below_to_above=int(np.sum(~before&after)),
                above_to_below=int(np.sum(before&~after)),below_to_below=int(np.sum(~before&~after)))


def tape_states(observer, native):
    by_id={}
    for image, row in native.items():
        for rank, original in zip(row["ranks"],row["stock"],strict=True):
            key=(image,rank)
            idx=len(by_id)+1
            truth=observer.cocoDt.anns[idx]
            require(truth["image_id"]==image and truth["category_id"]==original["category_id"] and
                    truth["score"]==original["score"],"Exact deterministic original candidate mapping")
            by_id[idx]=key
    require(observer.params.maxDets==[1,10,100] and observer.params.catIds==[1] and
            len(by_id)==len(observer.cocoDt.anns),"Canonical all-candidate COCO evaluator")
    states=[{} for _ in observer.params.iouThrs]
    for row in observer.evalImgs:
        if row is None or row["maxDet"]!=100 or list(row["aRng"])!=list(observer.params.areaRng[0]):
            continue
        for j, result in enumerate(states):
            for det_id, gt_id, ignored in zip(row["dtIds"],row["dtMatches"][j],row["dtIgnore"][j],strict=True):
                key=by_id[int(det_id)]
                require(key[0]==row["image_id"] and key not in result,"Unique candidate in all-area evaluation")
                status="ignore" if ignored else "TP" if gt_id else "FP"
                result[key]=(status,int(gt_id) if status=="TP" else 0)
    require(all(len(r)==len(by_id) for r in states),"All original candidates represented at each threshold")
    return states


def check_precision_envelope(observer):
    rows=[r for r in observer.evalImgs if r is not None and r["maxDet"]==100 and
          list(r["aRng"])==list(observer.params.areaRng[0])]
    oracle=np.concatenate([r["dtMatches"] for r in rows],axis=1)>0
    ignore=np.concatenate([r["dtIgnore"] for r in rows],axis=1)
    scores=np.concatenate([r["dtScores"] for r in rows])
    gts=sum(int(np.count_nonzero(~r["gtIgnore"])) for r in rows)
    require(gts>0,"Positive nonignored reference count for independent envelope")
    order=np.argsort(-scores,kind="mergesort")
    trajectory_tp=np.cumsum((oracle & ~ignore)[:,order],axis=1)
    trajectory_fp=np.cumsum((~oracle & ~ignore)[:,order],axis=1)
    for threshold,(tp,fp) in enumerate(zip(trajectory_tp,trajectory_fp,strict=True)):
        recall=tp/gts
        pr=tp/(tp+fp+np.spacing(1))
        reverse=np.maximum.accumulate(pr[::-1])[::-1]
        indices=np.searchsorted(recall,observer.params.recThrs,side="left")
        accepted=indices<len(pr)
        precision=np.zeros(len(observer.params.recThrs))
        precision[accepted]=reverse[indices[accepted]]
        require(np.array_equal(precision,observer.eval["precision"][threshold,:,0,0,-1]),
                "Stable score/ignore/TP series reconstitutes the exact original precision envelope")


def old_anchor_ious(audit, anchors, native, polygons, evaluator, core, shapely):
    raster={arm:[] for arm in ARMS}
    raster_eligible={arm:[] for arm in ARMS}
    continuous={arm:[] for arm in ARMS}
    for reference,(image,rank,stock) in anchors.items():
        original=stock["segmentation"]
        annotation=audit.cocoGt.anns[reference]
        require(annotation["image_id"]==image and not annotation["iscrowd"],"Same original nonignored .50 reference")
        gt_rle=audit.cocoGt.annToRLE(annotation)
        options=native[image]["changes"].get(rank,{})
        poly=polygons[reference]
        area=int(core.codec.area(original))
        eligible=poly.is_valid and poly.area>0 and 32<=area<1024
        for arm in ARMS:
            mask=options.get(arm,{}).get("rle",original)
            iou=float(core.codec.iou([mask],[gt_rle],[0])[0,0])
            raster[arm].append(iou)
            if eligible:
                raster_eligible[arm].append(iou)
                continuous[arm].append(float(core.geometric_iou(mask,poly,shapely)))
    return raster,raster_eligible,continuous


def summarize_view(traces, anchors, gt, ids, blocks, polygons, evaluator, core, shapely, old):
    native={i:r for i,r in blocks.items() if i in ids}
    arrays={arm:tape_states(traces[arm],native) for arm in ARMS}
    for arm in ARMS:
        check_precision_envelope(traces[arm])
    thresholds=traces["raw"].params.iouThrs
    require(np.allclose(thresholds,np.linspace(.50,.95,10),rtol=0,atol=1e-12),
            "Original COCO ten IoU thresholds")
    fixed=old_anchor_ious(traces["raw"],anchors,native,polygons,evaluator,core,shapely)
    require(len(fixed[0]["raw"])==old["geometry"]["counts"]["original_matches"] and
            len(fixed[1]["raw"])==len(fixed[2]["raw"])==old["geometry"]["counts"]["eligible_matches"],
            "Original fixed-match and eligible denominators")
    by_threshold={}
    for k,t in enumerate(thresholds):
        scores = {}
        states = {}
        for arm in ARMS:
            precision=traces[arm].eval["precision"][k,:,0,0,-1]
            require(len(precision)==101 and np.isfinite(precision).all() and np.any(precision>=0),
                    "Author 101-point interpolated precision of a nonempty cohort")
            scores[arm]=100*float(precision[precision>=0].mean())
            states[arm]=dict(Counter(r[0] for r in arrays[arm][k]))
        contrasts={}
        for new,previous in CONTRASTS:
            contrasts[new+" minus "+previous]=dict(ap_delta_points=scores[new]-scores[previous],
                authoritative_maxdets100=detection_transitions(arrays[previous][k],arrays[new][k]),
                original_native_match_raster_all=fixed_crossings(fixed[0][previous],fixed[0][new],t),
                original_native_match_raster_eligible=fixed_crossings(fixed[1][previous],fixed[1][new],t),
                original_native_match_continuous_eligible=fixed_crossings(fixed[2][previous],fixed[2][new],t))
        by_threshold[f"{t:.2f}"]=dict(ap_points=scores,authoritative_detection_status=states,contrasts=contrasts)
    for arm in ARMS:
        metric=old["metrics_fraction"][arm]
        require(abs(np.mean([v["ap_points"][arm] for v in by_threshold.values()])-100*metric["AP"])<1e-10 and
                all(abs(by_threshold[t]["ap_points"][arm]-100*metric[name])<1e-10
                    for t,name in (("0.50","AP50"),("0.75","AP75"),("0.90","AP90"))),
                "Individual thresholds reconstitute original AP, AP50, AP75, AP90")
    for new,previous in CONTRASTS:
        name=new+" minus "+previous
        average=np.mean([v["contrasts"][name]["ap_delta_points"] for v in by_threshold.values()])
        require(abs(average-old["paired_contrasts_points"][name]["AP"])<1e-10,
                "Exact arithmetic closure of the original AP contrast")
    return dict(images=len(ids),original_candidates=sum(len(r["stock"]) for r in native.values()),
                author_annotations=old["annotations"],original_native_matches=len(anchors),
                original_eligible_native_matches=len(fixed[1]["raw"]),by_threshold=by_threshold)


def run(cmd):
    start=perf_counter()
    require(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1"
            and np.__version__=="1.26.4" and version("pycocotools")=="2.0.11" and not cmd.output.exists(),
            "Fresh qualified historical CPU diagnostic")
    pin(cmd.protocol,cmd.protocol_sha256)
    historic=json.loads(pin(cmd.previous/"score/RESULT_SAFE.json",SCORE_SHA))
    old_terminal=json.loads(pin(cmd.previous/"observed_score/TERMINAL.json",TERMINAL_SHA))
    require(historic["status"]=="SRNET_SSDD_E24_RSDD74_PAIRED_SCORE_COMPLETE" and
            historic["worker_sha256"]==old_terminal["worker_sha256"]==SCORE_WORKER and
            historic["authority_sha256"]==AUTHORITY_SHA and old_terminal["actual_exit"]==0 and
            old_terminal["phase"]=="score" and old_terminal["result_sha256"]==SCORE_SHA and
            old_terminal["output_footer_verified"] is True and historic["author_test_opened"] is False,
            "Exact naturally closed original scoring, reference and worker")
    old=module(cmd.previous/"score.py",SCORE_WORKER)
    authority=json.loads(pin(cmd.previous/"AUTHORITY_score.json",AUTHORITY_SHA))
    values={key:Path(value) if isinstance(value,str) and Path(value).is_absolute() else value for key,value in authority.items()}
    a=SimpleNamespace(**values)
    require(a.prepare.resolve()==cmd.previous.resolve() and a.protocol_sha==old.PROTOCOL_SHA and
            a.verify_result_sha==historic["independent_edits_result_sha256"] and
            a.verify_terminal_sha==historic["independent_edits_terminal_sha256"] and
            a.capture_result_sha==historic["capture_result_sha256"],
            "Original bound capture/replay to score authority")
    pin(a.protocol,a.protocol_sha)
    replay=old.load(a.prepare/"replay.py",a.replay_worker_sha)
    old.audit_source(a,replay)
    edit,blocks=old.read_closed_stream(a,replay)
    require(edit["edits_sha256"]==historic["edits_sha256"] and len(blocks)==74 and
            sum(len(r["stock"]) for r in blocks.values())==89,"Exact archived candidate collection without labels")
    reserve=old.load(a.code/old.RESERVE,old.RESERVE_SHA)
    historic_adapter=old.load(a.code/old.HISTORIC,reserve.HISTORIC)
    evaluator=historic_adapter.module(a.code/"PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/score.py",historic_adapter.SCORE_SOURCE)
    old.support_empty_predictions(evaluator)
    adapter=historic_adapter.module(a.code/"UEIKAP_RFDETR_CARRIER_20261122/score_rf.py",historic_adapter.GEOMETRY_SOURCE)
    core=adapter.load_geometry(a.code/"UEIKAP_RESERVED_HULL_FROZEN_20261122/score.py")
    sys.path.append(str(a.shapely_site))
    import shapely
    require(shapely.__version__=="2.0.2" and shapely.geos_version_string=="3.11.2","Exact archived continuous geometry")
    census=json.loads(pin(a.census,reserve.CENSUS))
    dates={p["paper_scene"]:(p["sensor"],p["calendar_date"]) for p in census["per_parent"]}
    nested_parents=set(census["date_separated_sensitivity"]["parent_scene_ids"])
    nested={i for i,r in blocks.items() if r["group"] in nested_parents}
    require(len(nested)==42 and set(dates)=={r["group"] for r in blocks.values()},"Locked historical nested subset")
    print(json.dumps(dict(stage="PRELABEL_QUALIFIED")),flush=True)

    gt,rekeyed=historic_adapter.coco_annotation_ids(reserve.select_labels(pin(a.author_train,reserve.TRAIN),blocks))
    require(rekeyed==historic["annotation_ids_rekeyed"],"Original canonical closed author IDs")
    polygons={v["id"]:shapely.Polygon(np.asarray(v["segmentation"][0]).reshape(-1,2)) for v in gt["annotations"]}
    reserve.ARMS,reserve.PAIRS=ARMS,CONTRASTS
    builtin_score=evaluator.score_one
    builtin_eval=evaluator.COCOeval
    captured=[]

    class RecordedCOCOeval(builtin_eval):
        def summarize(self):
            super().summarize()
            captured.append(self)

    evaluator.COCOeval=RecordedCOCOeval
    views={}
    for name,ids in (("full74",set(blocks)),("nested42_sensor_date",nested)):
        traces,anchors={},{}
        def remember(gt_view, group_rows, arm, **kwargs):
            require(not captured,"One original evaluation trace per invocation")
            metrics,match=builtin_score(gt_view,group_rows,arm,**kwargs)
            require(len(captured)==1 and arm not in traces,"Recorded original COCO result")
            traces[arm]=captured.pop()
            if arm=="raw":
                anchors.update(match)
            return metrics,match
        evaluator.score_one=remember
        observed=old.score_view(reserve,evaluator,core,shapely,gt,ids,blocks,polygons,dates)
        observed.pop("criteria")
        observed.pop("within_recipe_mechanism_criterion")
        historic_view=historic["views"][name]
        require(observed==historic_view,"Exact original COCO and continuous geometry aggregate parity")
        views[name]=summarize_view(traces,anchors,gt,ids,blocks,polygons,evaluator,core,shapely,historic_view)

    result=dict(status=STATUS,phase="score",protocol_sha256=cmd.protocol_sha256,
                worker_sha256=digest(Path(__file__).read_bytes()),old_score_sha256=SCORE_SHA,
                old_score_terminal_sha256=TERMINAL_SHA,old_score_worker_sha256=SCORE_WORKER,
                old_score_authority_sha256=AUTHORITY_SHA,old_arm_metrics_and_geometry_exact=True,independent_precision_envelope_exact=True,
                fixed_anchor_rule="previously verified original score-ordered COCO .50 raw match to same reference",
                authoritative_status_rule="individual original candidate COCO dtMatches and dtIgnore, all area maxDets100",
                threshold_AP="equal-weight mean over ten author IoU thresholds, independently aggregated per threshold",
                original_author_annotation_scope="exact old 74 author TRAIN and unchanged nested42",
                target_fit=False,author_test_opened=False,detector_forwards=0,optimizer_updates=0,
                candidate_edits_or_rematching_algorithms_changed=False,views=views,elapsed_seconds=perf_counter()-start)
    cmd.output.mkdir()
    raw=(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    (cmd.output/"RESULT_SAFE.json").write_bytes(raw)
    print(json.dumps(dict(status=STATUS,result_sha256=digest(raw))),flush=True)


def main():
    p=ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("previous","protocol","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--protocol-sha256",required=True)
    run(p.parse_args())


if __name__=="__main__":
    main()
