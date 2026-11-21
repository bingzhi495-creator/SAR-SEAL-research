"""Independent FIT-only check of the two new, closed ratio-writer policies."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


REPORT = "0972c906c20c0fbd8d5fe8d30ad6ead960e783c8e046c2ccab7f0b1fa6b347d2"
PREVIOUS_VERIFIER = "daa3f6dc7751fe546700b8297b89c0f7295238b3876631f073c8f863d72d449f"
PRIOR_PEER = "c791b3eb637acf5e30341768a53fa1acb8a3c8db5d6153717474b0b6958fc319"
PRIOR_OWNER = "f60dad00524ab0b25bf5728c43ae1c42ee3970e8cf00470d8d2b6bb568442aa8"


def need(ok, why):
    if not ok:
        raise ValueError(why)


def prior_helper(work):
    file = work / "verify_point_fit109.py"
    raw = file.read_bytes()
    need(sha256(raw).hexdigest() == PREVIOUS_VERIFIER, "Previous accepted verifier source")
    module = ModuleType("_independent_point_fit109")
    module.__file__ = str(file)
    exec(compile(raw,str(file),"exec"),module.__dict__)
    return module


def dense_toggle(base, changes, coordinates, k):
    need(type(changes) is list and len(changes)==len(set(changes))==2*k and all(type(i) is int for i in changes),
         "Unique 2k edits at original native raster indices")
    allowed = np.asarray(coordinates,dtype=np.int64).reshape(-1,2)
    need(k <= min(8,len(allowed),int(np.count_nonzero(base))//10)
         and all(i in set(allowed.ravel().tolist()) for i in changes),"Admitted candidate support and cap")
    native = np.array(base,order="C",copy=True)
    view = native.ravel(order="C")
    indices = np.asarray(changes,dtype=np.int64)
    need(np.all(view[allowed[:,0]]==0) and np.all(view[allowed[:,1]]==1),"Original add and drop sides")
    drops = int(np.count_nonzero(view[indices]))
    adds = len(indices)-drops
    view[indices] ^= 1
    need(np.count_nonzero(native!=base)==2*k and int(native.sum())-int(base.sum())==adds-drops,
         "Realized dense flips and net area")
    enc = mask.encode(np.asfortranarray(native))
    return dict(size=enc["size"],counts=enc["counts"].decode("ascii")), adds,drops


def official(gt, stream):
    coco = COCO()
    coco.dataset = deepcopy(gt)
    coco.dataset.setdefault("info",{})
    with redirect_stdout(StringIO()):
        coco.createIndex()
        ev = COCOeval(coco,coco.loadRes(deepcopy(stream)),"segm")
        ev.params.imgIds=[im["id"] for im in gt["images"]]
        ev.params.catIds=[1]
        ev.params.maxDets=[1,10,100]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    vals=ev.eval["precision"][:,:,0,ev.params.areaRngLbl.index("all"),ev.params.maxDets.index(100)]
    curve=[float(p[p>=0].mean()*100) for p in vals]
    threshold90,=np.flatnonzero(np.isclose(ev.params.iouThrs,.90))
    return dict(AP=float(100*ev.stats[0]), AP75=float(100*ev.stats[2]), AP90=curve[threshold90], AP_thresholds=curve)


def main():
    p=ArgumentParser()
    for name in ("work","code","raw-capture","raw-acceptance","source-cohort","old-support","old-score","fit-gt","shapely-site"):
        p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args()
    folder=Path(__file__).parent
    prior=prior_helper(args.work)
    score=prior.read_doc(folder/"IOU_WRITER_ACTUAL_SCORE_SAFE.json",REPORT)
    owner=prior.read_doc(folder/"RATIO_ORIGINAL_TOOL_WAIT_PRIVATE.json",score["prediction_terminal_sha256"])
    need(str(owner["original_tool_handle"])=="93023" and owner["original_exit"]==1 and owner["stage"]=="PREDICTIONS_CLOSED_SCORING_DISPATCH_FAILED"
         and owner["images_completed"]==109 and score["recovered_without_regeneration"] is True and score["FIT"] is True
         and score["DEV"] is score["TEST"] is False, "Real prelabel witness and independent score-only recovery")
    pre=prior.read_doc(folder/"IOU_WRITER_ACTUAL_PRIVATE/PRELABEL_SAFE.json",owner["prelabel_sha256"])
    need(pre["GT_opened"] is pre["DEV"] is pre["TEST"] is False and pre["FIT"] is True and pre["protocol_sha256"]==score["protocol_sha256"]
         and pre["witness_sha256"]==owner["witness_sha256"]==score["predictions_witness_sha256"],"Frozen FIT data and policies")
    lock=prior.read_doc(folder/"IOU_WRITER_PRELOCK.json",score["protocol_sha256"])
    edits=prior.read_doc(folder/"IOU_WRITER_ACTUAL_PRIVATE/EDITS_PRIVATE.json",pre["witness_sha256"],20*2**20)
    choices={tuple(k):v for k,v in edits}
    need(len(choices)==len(edits)==6738 and all(set(v)=={"T","R"} for v in choices.values()),"Full unique frozen T and R edits")
    peer=prior.read_doc(args.work/"INDEPENDENT_OOF_READBACK_SAFE.json",PRIOR_PEER)
    original_owner=prior.checked_terminal(args.work/"BENCHMARK_ORIGINAL_TOOL_WAIT_PRIVATE.json",PRIOR_OWNER)
    need(peer["result_sha256"]==lock["source_result_sha256"]==score["old_result_sha256"]==original_owner["result_sha256"],"Closed unchanged native benchmark")
    manifest=prior.read_doc(args.work/"POINTREND_FIT109_MODULE_PINS.json",original_owner["module_manifest_sha256"])
    runner=prior.checked_module(args.work/"run_point_fit109.py",manifest["run_point_fit109.py"])
    scope=prior.read_doc(args.work/"POINTREND_FIT109_PRELOCK.json",runner.PROTOCOL_SHA)
    old_dir=args.code/"RFDETR_QUERY_FEATURE_FIT109_20261121"
    old=prior.checked_module(old_dir/"run_fit109_query.py","97fdc3987462aabbcd7d4fbe672c9788a3b969a221243e13d82473093170a15a")
    spot=old.load(args.code/"RFDETR_CORE36_MATCHED_POINT_SPOT_20261120/run_spot.py")
    formal=args.code/"RFDETR_MATCHED_POINT_FORMAL_20261120"
    base,plan=(old.load(formal/n) for n in ("run_fit109.py","fit_plan.py"))
    args.capture,args.acceptance,args.acceptance_sha256=args.raw_capture,args.raw_acceptance,old.OLD_RAW_ACCEPTANCE
    roster,_=base.read_population(args,spot)
    helpers=old.load(old_dir/"fit109_score.py").load_helpers(args.code,args.shapely_site)
    pools,rows=runner.collect_old_support(args,SimpleNamespace(old=old,spot=spot),roster,scope)
    budgets,_=runner.read_budgets(args.old_score,scope,rows,plan,roster,helpers.geometry)
    need(choices.keys()==budgets.keys(),"Each and every old OOF budget-bearing candidate")
    gt,attached=old.attach_fit(args,roster,rows,pools,helpers,spot)
    mapped={(r["image_id"],r["rank"],r["query_id"]):r for r in attached}
    need(len(gt["images"])==109 and len(gt["annotations"])==496 and mapped.keys()==choices.keys(),"Exact original GT and fixed joins")
    polys,_,_=helpers.truth.modules.geometry.polygon_support(gt,helpers.truth.modules.vector)
    streams={"T":[],"R":[]}
    groups=defaultdict(list)
    stats={a:Counter() for a in streams}
    zero=0
    for image in gt["images"]:
        iid=image["id"]
        for cand in pools[iid]:
            key=(iid,cand["rank"],cand["query_id"])
            pred=dict.fromkeys(streams,cand["rle"])
            if key in choices:
                k=budgets[key]["k"]
                zero+= k==0
                rec=mapped[key]
                native=mask.decode(cand["rle"])
                for arm in streams:
                    pred[arm],adds,drops=dense_toggle(native,choices[key][arm],rec["coordinates"],k)
                    stats[arm].update(adds=adds,drops=drops,net_shrink=int(drops>adds))
                if rec["role"]=="valid":
                    polygon=polys[rec["matched_gt_id"]][0]
                    raw_iou=prior.footprint(cand["rle"],polygon,helpers)
                    values=[]
                    for arm in streams:
                        realized=prior.footprint(pred[arm],polygon,helpers)
                        values.append(realized)
                        stats[arm]["strict_losses"]+=int(raw_iou>.9 and realized<=.9)
                    groups[rec["group"]].append(values)
            for arm,stream in streams.items():
                stream.append(dict(image_id=iid,score=cand["score"],category_id=1,segmentation=pred[arm]))
    need(all(len(stream)==10488 for stream in streams.values()) and zero==317 and len(groups)==92
         and sum(len(r) for r in groups.values())==392,"Mapped census, eligible no-ops and valid geometry")
    means=100*np.asarray([np.mean(groups[g],axis=0) for g in sorted(groups)])
    result={}
    max_error=0.
    for pos,(arm,stream) in enumerate(streams.items()):
        s=stats[arm]
        need(s["adds"]+s["drops"]==97964,"Common realized total")
        measures=dict(**official(gt,stream),geometry=float(means[:,pos].mean()),**dict(s))
        previous=score["metrics"]["arms"][arm]
        for key,v in measures.items():
            err=float(np.max(np.abs(np.asarray(v)-np.asarray(previous[key]))))
            need(err<2e-8,"Independent mismatch in "+arm+" "+key)
            max_error=max(max_error,err)
        result[arm]=measures
    delta=means[:,1]-means[:,0]
    samples=np.random.Generator(np.random.PCG64(5217)).integers(len(delta),size=(4096,len(delta)))
    ci=np.quantile(delta[samples].mean(axis=1),[.025,.975]).tolist()
    contrast=dict(geometry_delta_points=float(delta.mean()),geometry_ci95_points=ci,AP_delta_points=result["R"]["AP"]-result["T"]["AP"])
    for k,v in contrast.items():
        need(np.allclose(v,score["metrics"]["R_minus"]["T"][k],rtol=0,atol=2e-8),"Independent R/T paired contrast")
    print(json.dumps(dict(status="FIT109_T_R_CLOSED_EDITS_INDEPENDENTLY_VERIFIED",recovered_score_sha256=REPORT,
          score_recovery_handle_observed_by_original_owner=54111,score_recovery_exit_observed_by_original_owner=0,
          original_prediction_handle=93023,original_prediction_exit=1,witness_sha256=pre["witness_sha256"],
          verifier_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),images=109,original_annotations=496,
          mapped_rows=10488,eligible_rows=6738,zero_budget=zero,immutable_edit_total_per_arm=97964,
          fixed_matches=392,contributing_groups=92,arms=result,R_minus_T=contrast,
          max_arm_measure_disagreement_points=max_error,FIT=True,DEV=False,TEST=False,model_forwards=0,head_updates=0),sort_keys=True))


if __name__=="__main__":
    main()
