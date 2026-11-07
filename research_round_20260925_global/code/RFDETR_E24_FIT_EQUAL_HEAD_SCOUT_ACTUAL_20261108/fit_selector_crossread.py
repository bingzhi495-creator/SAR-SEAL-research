"""Read the existing selector's effects in fixed matched and unmatched cohorts."""
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import time

from fit_selector_materialize import digest,doc,load,need,original_gt,put,runtime

BRANCHES=("uncertainty","mean","rank")


def run(path,expected):
    started=time.monotonic()
    raw=path.read_bytes()
    need(digest(raw)==expected,"Exact published cross-readout request")
    q=json.loads(raw)
    need(q["schema"]=="Root-FIT-selector-fixed-match-cross-readout-v1" and q["DEV"] is q["TEST"] is False,
         "Locked FIT-only cross-readout")
    scout=doc(q["scout_acceptance"])
    result=doc(scout["original_result_ref"])
    need(scout["status"]=="ACTUAL_FIT741_EQUAL_HEAD_SCOUT_ACCEPTED" and scout["gate"]["eligible_for_dev"] is False
         and result["gate"]==scout["gate"],"Exact accepted stopped scout")
    _,coco,cocoeval,_=runtime(q)
    crossread=load(q["sources"]["crossread"],"_fit_scout_crossread")
    evaluator=load(q["sources"]["holdout_eval"],"_fit_scout_crossread_official")
    fit=doc(q["image_manifest"])
    accepted=doc(scout["material_acceptance_ref"])
    material=doc(accepted["material_index_ref"])
    predictions=doc(result["predictions_ref"])
    gt=original_gt(q,fit["items"])
    need(len(material)==len(fit["items"])==741,"Exact accepted whole materialization")
    statuses={}
    for m,f in zip(material,fit["items"],strict=True):
        need(all(m[k]==f[k] for k in ("dataset_index","image_id","partition")),"Same exact FIT partitions")
        if m["partition"]=="head_holdout":
            statuses[m["image_id"]]=[r["status"] for r in doc(m["match_ref"])["rows"]]
    ids=list(statuses)
    need(len(ids)==111 and all(set(arm)==set(map(str,ids)) for arm in predictions.values()),
         "Original complete holdout prediction populations")
    base={i:predictions["no_edit"][str(i)] for i in ids}
    outcome={}
    crossed={}
    out=path.parent
    with (out/"OFFICIAL_COCO_CROSSREAD.stdout").open("x",encoding="utf-8") as log,redirect_stdout(log):
        for branch in BRANCHES:
            arm={i:predictions[branch][str(i)] for i in ids}
            combined=crossread.build_crossread(base,arm,statuses,ids,require_no_ignored=True)
            metrics={}
            for group in ("matched_only","unmatched_only"):
                values=evaluator.evaluate_holdout(gt,base,combined[group],ids,
                    coco_class=coco.COCO,cocoeval_class=cocoeval.COCOeval)
                need(values["original"]==result["official_holdout_metrics"]["no_edit"],
                     "Same official accepted unedited endpoint")
                metrics[group]=values["edited"]
            crossed[branch]={k:combined[k] for k in ("matched_only","unmatched_only")}
            baseline=result["official_holdout_metrics"]["no_edit"]
            full=result["official_holdout_metrics"][branch]
            def contrasts(key):
                old=baseline[key]; match=metrics["matched_only"][key]; unmatch=metrics["unmatched_only"][key]
                both=full[key]
                return dict(matched_only_delta_pp=(match-old)*100,unmatched_only_delta_pp=(unmatch-old)*100,
                            both_delta_pp=(both-old)*100,interaction_pp=(both-match-unmatch+old)*100)
            outcome[branch]=dict(summary=combined["summary"],original=baseline,all_edits=full,**metrics,
                                 contrasts={key:contrasts(key) for key in ("AP","AP75")})
    refs=put(out/"CROSS_READ_PREDICTIONS_PRIVATE.json",crossed)
    put(out/"RESULT.json",dict(status="ACTUAL_FIT_FIXED_MATCH_CROSS_READ_COMPLETE_PENDING_ROOT",
        original_gate_preserved=scout["gate"],holdout_images=len(ids),GT_conditioned=True,
        training=False,calibration_repeated=False,DEV=False,TEST=False,metrics=outcome,
        crossed_predictions_ref=refs,seconds=time.monotonic()-started))


if __name__=="__main__":
    need(len(sys.argv)==3,"Exact cross-readout spec")
    run(Path(sys.argv[1]),sys.argv[2])
