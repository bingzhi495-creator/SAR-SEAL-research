"""Verify the frozen E24/F/S COCO and strict-coverage results independently on CPU."""
from contextlib import redirect_stdout
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

D=Path(__file__).resolve().parent.parent
PAIR=D/"RFDETR_E24_STRICT_OPTIMIZER_ACTUAL_PRIVATE_20261109_01"
OWNER=D/"RFDETR_E24_STRICT_OPTIMIZER_OWNER_PRIVATE_20261109_01"
CONTROL=PAIR/"DEV187_CONTROL"
OUT=PAIR/"DEV187_EVAL"
EVAL=Path(__file__).resolve().with_name("evaluate_optimizer_pair.py")
HISTORIC=D/"RFDETR_E24_WARM_BD_SCOUT_20261108_02/PAIR_S5201/DEV187_EVAL/RESULT.json"
HISTORIC_SHA="cec66e5202fe9106bd9c7e1a4868f67cf21a8229ba2c068fd4e356583ebd10ab"
GT="5657600d6ea198f79bd5d557d2a1aadce5bec031d2ecf72a98716248d1183a55"


def need(ok,message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def raw_ref(ref,limit):
    path=Path(ref["path"])
    need(path.resolve(strict=True)==path and path.stat().st_size==ref["bytes"]<=limit,"Closed artifact scope")
    b=path.read_bytes()
    need(len(b)==ref["bytes"] and sha256(b).hexdigest()==ref["sha256"],"Closed artifact hash")
    return b


def read_worker_result(stdout, expected):
    terminal=json.loads(stdout.splitlines()[-1])
    need(type(terminal) is dict and set(terminal)=={"result_ref"}, "Exact evaluator terminal envelope")
    ref=terminal["result_ref"]
    need(type(ref) is dict and set(ref)=={"path","bytes","sha256"} and Path(ref["path"])==expected,
         "Exact original end report")
    return json.loads(raw_ref(ref,1024**2))


def main():
    import numpy as np
    import torch
    from importlib.metadata import version
    from pycocotools import mask
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from scipy.optimize import linear_sum_assignment

    need(not torch.cuda.is_available() and version("pycocotools")=="2.0.11","Exact CPU official scorer")
    cli=read(OWNER/"EVAL_CONTROLLER_TERMINAL_PRIVATE.json")
    control=read(CONTROL/"TERMINAL_PRIVATE.json")
    request=read(CONTROL/"REQUEST_PRIVATE.json")
    need(cli["original_OS_wait"] is control["original_OS_wait"] is True and cli["exit"]==control["exit"]==0
         and cli["automatic_retry"] is request["automatic_retry"] is False and control["guard"] is None
         and Path(request["evaluator"]["path"])==EVAL
         and cli["evaluator_source_sha256"]==request["evaluator"]["sha256"],
         "Genuine completed evaluation and exact owner/source")
    raw_ref(request["evaluator"],1024**2)
    need(Path(control["stdout_ref"]["path"])==CONTROL/"WORKER.stdout"
         and Path(control["stderr_ref"]["path"])==CONTROL/"WORKER.stderr","Exact worker log files")
    so=raw_ref(control["stdout_ref"],1024**2)
    raw_ref(control["stderr_ref"],1024**2)
    value=read_worker_result(so,OUT/"RESULT.json")
    acceptance=request["closed_pair"]
    need(Path(acceptance["path"])==PAIR/"ROOT_CLOSED_PAIR_ACCEPTANCE_PRIVATE.json",
         "Exact originally admitted acceptance receipt")
    closed=json.loads(raw_ref(acceptance,1024**2))
    train=read(PAIR/"ROOT_SAFE_TRAINING_RESULT.json")
    need(value["status"]=="OPTIMIZER_STATE_FS_DEV187_COMPLETE_PENDING_ROOT_ACCEPTANCE" and value["images"]==187
         and value["GT_instances"]==380 and value["TEST"] is value["new_training"] is False
         and value["acceptance_ref"]==request["closed_pair"] and value["protocol_commit"]==closed["protocol_commit"]
         ==train["protocol_commit"],"Fixed DEV187 S/F scope")
    need(value["integer_GT_ref"]["sha256"]==GT,"Pinned original integer GT")
    dataset=json.loads(raw_ref(value["integer_GT_ref"],1024**2))
    img_ids=[row["id"] for row in dataset["images"]]
    hws={row["id"]:(row["height"],row["width"]) for row in dataset["images"]}
    truths={i:[] for i in img_ids}
    for row in dataset["annotations"]:
        i=row["image_id"]
        h,w=hws[i]
        seg=row["segmentation"]
        if isinstance(seg,list):
            seg=mask.merge(mask.frPyObjects(seg,h,w))
        elif isinstance(seg["counts"],list):
            seg=mask.frPyObjects(seg,h,w)
        truths[i].append(seg)
    results={}
    for name in ("E24","F","S"):
        a=value["arms"][name]
        need(a["strict_tensors"]==545 and a["native_rows"]==18700 and a["images"]==187,
             "Complete same native queries")
        if name!="E24":
            need(a["state_ref"]==closed["arms"][name]["checkpoint_ref"],"Actual original trained endpoint")
        prediction=json.loads(raw_ref(a["predictions_ref"],256*1024**2))
        need(len(prediction)==a["mapped_rows"],"Full independent scoring input")
        with redirect_stdout(sys.stderr):
            coco=COCO()
            coco.dataset=dataset
            coco.createIndex()
            dt=coco.loadRes(prediction)
            ev=COCOeval(coco,dt,"segm")
            ev.params.imgIds=img_ids
            ev.params.catIds=[1]
            ev.params.maxDets=[1,10,100]
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        metrics=a["metrics"]
        stats=[float(x) for x in ev.stats]
        need(stats==metrics["stats"] and all(metrics[k]==stats[i] for k,i in (("AP",0),("AP75",2),("APs",3),("AR100",8))),
             "Independent direct COCO all12 reproduction")
        preds={i:[] for i in img_ids}
        for r in prediction:
            preds[r["image_id"]].append(r["segmentation"])
        strict=0
        for i in img_ids:
            if not truths[i] or not preds[i]:
                continue
            overlaps=mask.iou(preds[i],truths[i],[0]*len(truths[i]))
            binary=(overlaps>=.9).astype(np.int32)
            row,col=linear_sum_assignment(binary,maximize=True)
            strict+=int(binary[row,col].sum())
        geometry=value["retained_score_free_geometry"]["coverage"][name]["0.90"]
        need(strict==geometry["max_injective"],"Independent strict injective match")
        results[name]=dict(AP=stats[0],AP75=stats[2],APs=stats[3],AR100=stats[8],all12=stats,
            retained_strict_coverage=strict,retained_strict_small=geometry["small_only_max_injective"],
            raw_queries=a["native_rows"],retained_predictions=a["mapped_rows"],strict_tensors=a["strict_tensors"],
            cuda_forward_seconds=a["cuda_forward_seconds"],cuda_peak_reserved_bytes=a["cuda_peak_reserved_bytes"],
            export_postprocess_seconds=a["stock_postprocess_and_rle_seconds"])
    raw=HISTORIC.read_bytes()
    need(sha256(raw).hexdigest()==HISTORIC_SHA,"Exact previously accepted Windows E24 result")
    anchor=json.loads(raw)["arms"]["E24"]
    exact_anchor=(anchor["predictions_ref"]["sha256"]==value["arms"]["E24"]["predictions_ref"]["sha256"]
         and anchor["metrics"]["stats"]==results["E24"]["all12"])
    s,f,e=results["S"],results["F"],results["E24"]
    delta=s["retained_strict_coverage"]-f["retained_strict_coverage"]
    pass_mechanism=delta>=5 and s["AP"]>f["AP"] and s["AP"]>=e["AP"]
    safe=dict(status="ACTUAL_FIXED_OPTIMIZER_STATE_FS_OFFICIAL_COCO_INDEPENDENTLY_VERIFIED",
        decision=("MECHANISM_SCOUT_PASS_RAW_QUERY_FOLLOWUP" if pass_mechanism else "STOP_FIXED_OPTIMIZER_STATE_RECIPE"),
        protocol_commit=value["protocol_commit"],images=187,GT_instances=380,DEV_scope=value["population_role"],
        inference_precision=value["inference_precision"],exact_windows_E24_reproduction=exact_anchor,
        AP_points_S_minus_F=100*(s["AP"]-f["AP"]),AP_points_S_minus_E24=100*(s["AP"]-e["AP"]),
        S_minus_F_retained_strict_coverage=delta,arms=results,training=train,
        retained_geometry=value["retained_score_free_geometry"],dgx_E100_context=value["dgx_E100_context"],
        GPU_original_exit=0,CPU_recompute=True,TEST=False)
    target=PAIR/"ROOT_SAFE_FIXED_EVAL_RESULT.json"
    with target.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(safe,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=safe["status"],decision=safe["decision"],E24_exact=exact_anchor,
        AP={n:100*r["AP"] for n,r in results.items()},retained_strict={n:r["retained_strict_coverage"] for n,r in results.items()})))


if __name__=="__main__":
    main()
