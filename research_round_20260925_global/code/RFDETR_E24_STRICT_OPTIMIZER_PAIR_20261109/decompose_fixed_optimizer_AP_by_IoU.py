"""Decompose unchanged official DEV AP by its ten IoU thresholds."""
from contextlib import redirect_stdout
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from verify_optimizer_eval_cpu import CONTROL, OUT, PAIR, need, raw_ref, read, read_worker_result


def main():
    need(not torch.cuda.is_available(),"Independent CPU-only fixed-output analysis")
    closed=PAIR/"ROOT_SAFE_FIXED_EVAL_RESULT.json"
    need(sha256(closed.read_bytes()).hexdigest()=="a0ea5a22c60162c3076269d1d01ab18be53ab7de838420da7a6658cd90b452e7",
         "Exact independently accepted fixed DEV result")
    safe=read(closed)
    wait=read(CONTROL/"TERMINAL_PRIVATE.json")
    need(wait["exit"]==0 and wait["original_OS_wait"] is True and safe["TEST"] is False,"Genuine closed internal DEV")
    v=read_worker_result(raw_ref(wait["stdout_ref"],1024**2),OUT/"RESULT.json")
    g=json.loads(raw_ref(v["integer_GT_ref"],1024**2))
    img_ids=[row["id"] for row in g["images"]]
    need(len(img_ids)==187 and len(g["annotations"])==380,"Fixed original DEV population")
    ap={}
    for name in ("E24","F","S"):
        a=v["arms"][name]
        p=json.loads(raw_ref(a["predictions_ref"],256*1024**2))
        with redirect_stdout(sys.stderr):
            gt=COCO()
            gt.dataset=g
            gt.createIndex()
            dt=gt.loadRes(p)
            ev=COCOeval(gt,dt,"segm")
            ev.params.imgIds=img_ids
            ev.params.catIds=[1]
            ev.params.maxDets=[1,10,100]
            ev.evaluate()
            ev.accumulate()
        precision=ev.eval["precision"][:,:,:,0,-1]
        keys=[format(x,".2f") for x in ev.params.iouThrs]
        scores=[float(x[x>=0].mean()) for x in precision]
        need(abs(float(np.mean(scores))-safe["arms"][name]["AP"])<1e-14,"Threshold average must reproduce original AP")
        ap[name]=dict(zip(keys,scores,strict=True))
    result=dict(status="EXPLORATORY_OFFICIAL_DEV_FIXED_PREDICTION_AP_BY_IOU",
        protocol_commit=safe["protocol_commit"],same_original_predictions_and_coco_rules=True,
        images=187,GT_instances=380,TEST=False,new_training=False,AP_by_IoU=ap,
        S_minus_F_AP_points={k:100*(ap["S"][k]-ap["F"][k]) for k in ap["F"]})
    with (PAIR/"ROOT_SAFE_FIXED_EVAL_IOU_DECOMPOSITION.json").open("x",encoding="utf-8",newline="\n") as f:
        json.dump(result,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()

