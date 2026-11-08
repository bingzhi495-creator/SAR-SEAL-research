"""Compare score-ranked access to high-IoU masks at the fixed COCO maxDet cutoffs."""
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
from pycocotools import mask
from scipy.optimize import linear_sum_assignment
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
    gt=json.loads(raw_ref(v["integer_GT_ref"],1024**2))
    images={r["id"]:(r["height"],r["width"]) for r in gt["images"]}
    targets={i:[] for i in images}
    for a in gt["annotations"]:
        i=a["image_id"]
        h,w=images[i]
        seg=a["segmentation"]
        if isinstance(seg,list):
            seg=mask.merge(mask.frPyObjects(seg,h,w))
        elif isinstance(seg["counts"],list):
            seg=mask.frPyObjects(seg,h,w)
        targets[i].append(seg)
    ranks={}
    for name in ("E24","F","S"):
        pred=json.loads(raw_ref(v["arms"][name]["predictions_ref"],256*1024**2))
        by_image={i:[] for i in images}
        for row in pred:
            by_image[row["image_id"]].append(row)
        counts={k:0 for k in (1,10,100)}
        for i,t in targets.items():
            ordered=sorted(by_image[i],key=lambda row:-row["score"])
            if not t or not ordered:
                continue
            binary=(mask.iou([x["segmentation"] for x in ordered[:100]],t,[0]*len(t))>=.90).astype(np.int32)
            for k in counts:
                m=binary[:k]
                p,q=linear_sum_assignment(m,maximize=True)
                counts[k]+=int(m[p,q].sum())
        need(counts[100]==safe["arms"][name]["retained_strict_coverage"],"Full cap agrees with original strict count")
        ranks[name]=counts
    result=dict(status="EXPLORATORY_FIXED_DEV_STRICT_COVERAGE_BY_COCO_SCORE_CUTOFF",
        protocol_commit=safe["protocol_commit"],images=187,GT_instances=380,TEST=False,new_training=False,
        cutoffs=[1,10,100],IoU=.9,maximum_injective_score_ranked_coverage=ranks)
    with (PAIR/"ROOT_SAFE_STRICT_COVERAGE_SCORE_CUTOFF.json").open("x",encoding="utf-8",newline="\n") as f:
        json.dump(result,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()

