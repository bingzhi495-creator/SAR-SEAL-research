"""Separate deep and boundary false negatives on fixed closed HRSID large matches."""
import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from io import StringIO
import json
import os
from pathlib import Path
from types import ModuleType

import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.ndimage import binary_erosion


P282_PATH = "HRSID_COMMON_DEV705_LARGE_FAILURE_20261111/closed_common_diagnostics.py"
P282_SHA = "15e924e60c8573a48c8ef523be8636a04588f922d685d4d74ca410271497fa7e"
BASELINE_SHA = "2b1286203dae3ffc2ce11153619bc4b387f83ac338c4ff2855d1275ba4bc7012"
GROUP_PATH = "HRSID_MASK_PHASE_REPEAT_DEV705_20261109/dev705_group_admission.py"
GROUP_SHA = "079a3a4d1d00ed8719a15f376c2aabb34056c4da8d1b1db1d85c621b23c25033"
MANIFEST_SHA = "39a40d089d4a54448fe6efca8ea5cc98ad1eb7c0587815f3dd9a3b097173e78f"
PRIMARY_SHA = "581a0ca8fc9966388dd6cb07a729f51726932c81bd5c037887b25c0b303dff56"
PAIR_SHA = "d080a27ade36369a8fc8d2cb003c37cc3670d75860e83b30e365bd47e77543c0"
TRAIN_PROTOCOL = "04308c52bd2b4730f1474335c6316f6b8a11762e"
ARMS = ("native","center")
FIELDS = ("fn","core_fn","band_fn","fp")


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def split_errors(truth, pred):
    need(truth.shape == pred.shape and truth.dtype == pred.dtype == bool and truth.any(),
         "Exact Boolean target and prediction geometry required")
    core = binary_erosion(truth,structure=np.ones((5,5),dtype=bool),border_value=0)
    band = truth & ~core
    missed = truth & ~pred
    counts = dict(gt_area=int(truth.sum()), core_area=int(core.sum()), band_area=int(band.sum()),
                  fn=int(missed.sum()), core_fn=int((missed & core).sum()),
                  band_fn=int((missed & band).sum()), fp=int((pred & ~truth).sum()))
    need(counts["fn"] == counts["core_fn"]+counts["band_fn"]
         and counts["gt_area"] == counts["core_area"]+counts["band_area"], "FN region partition failed")
    return counts


def best_shift(truth, pred):
    """Choose exact best overlap while retaining every original predicted pixel."""
    need(truth.shape == pred.shape and truth.dtype == pred.dtype == bool and pred.any(), "Nonempty fixed mask required")
    yy,xx = np.nonzero(pred)
    h,w = truth.shape
    moves = [(dy,dx) for dy in range(-2,3) for dx in range(-2,3)
             if 0 <= yy.min()+dy <= yy.max()+dy < h and 0 <= xx.min()+dx <= xx.max()+dx < w]
    dy,dx = min(moves,key=lambda p:(-int(truth[yy+p[0],xx+p[1]].sum()),p[0]**2+p[1]**2,*p))
    output = np.zeros_like(pred)
    output[yy+dy,xx+dx] = True
    need(int(output.sum()) == len(yy) and (output & truth).sum() >= (pred & truth).sum(),
         "Aligned mask loses pixels or overlap")
    return (dy,dx),output


def load(path,digest,name):
    raw=path.read_bytes()
    need(sha256(raw).hexdigest()==digest,"Exact closed source required")
    mod=ModuleType(name)
    mod.__file__=str(path)
    exec(compile(raw,str(path),"exec"),mod.__dict__)
    return mod


def fixed_matches(gt, selected, adapter, match):
    """Use stock COCO evaluation records only, without accumulating AP."""
    coco=COCO()
    coco.dataset=deepcopy(gt)
    ids=[x["id"] for x in gt["images"]]
    with redirect_stdout(StringIO()):
        coco.createIndex()
        result=coco.loadRes(adapter.segm_copy(selected))
        ev=COCOeval(coco,result,"segm")
        ev.params.imgIds=ids
        ev.params.catIds=[1]
        ev.params.maxDets=[1,10,100]
        ev.evaluate()
    need(not ev.eval and len(ev.params.iouThrs)==10 and ev.params.iouThrs[0]==.50,
         "Only original .50 match records may be used")
    return match(ev.evalImgs,selected,{x["id"]:x["image_id"] for x in gt["annotations"]})


def aggregate(rows, phase):
    output=dict(pairs=len(rows))
    gt=Counter()
    a,b=Counter(),Counter()
    for row in rows:
        aa,bb=(row[phase][key] for key in ARMS)
        need(all(aa[key]==bb[key] for key in ("gt_area","core_area","band_area")),"GT area changed between arms")
        gt.update({key:aa[key] for key in ("gt_area","core_area","band_area")})
        a.update({key:aa[key] for key in FIELDS})
        b.update({key:bb[key] for key in FIELDS})
    delta={key:b[key]-a[key] for key in FIELDS}
    output.update(native=dict(a),center=dict(b),center_minus_native=delta,area=dict(gt),
                  center_minus_native_core_rate=delta["core_fn"]/gt["core_area"] if gt["core_area"] else None,
                  center_minus_native_band_rate=delta["band_fn"]/gt["band_area"] if gt["band_area"] else None,
                  pairs_with_excess_core=sum(r[phase]["center"]["core_fn"]>r[phase]["native"]["core_fn"] for r in rows))
    return output


def run(args):
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="-1" and args.output.parent.resolve(strict=True)==args.output.parent,
         "Closed CPU diagnostic only")
    base_raw=args.prior_result.read_bytes()
    need(sha256(base_raw).hexdigest()==BASELINE_SHA,"Exact prior large-mask result required")
    base=json.loads(base_raw)
    need((args.pair_sha256,args.training_protocol,args.accepted_result_sha256)==(PAIR_SHA,TRAIN_PROTOCOL,PRIMARY_SHA)
         and base["accepted_result_sha256"]==PRIMARY_SHA,"Exact prior scientific lineage required")
    root=args.code_root
    closed=load(root/P282_PATH,P282_SHA,"_p360_closed_pair")
    group=load(root/GROUP_PATH,GROUP_SHA,"_p360_group_gate")
    gt,native,center,proof=closed.verified_inputs(root,args.pair,PAIR_SHA,native_arm_receipt=args.native_arm_receipt,
         center_arm_receipt=args.center_arm_receipt,training_protocol=TRAIN_PROTOCOL)
    need(proof==base["provenance"],"Closed prediction lineage differs")
    driver=group.load_driver()
    pair=driver.read_json(args.pair,PAIR_SHA)
    group_gt,_,groups=group.load_population(driver,dict(path=args.image_manifest.as_posix(),sha256=MANIFEST_SHA),pair["gt"])
    need(group_gt==gt,"Source group GT differs from fixed prediction cohort")
    modules=closed.load_diagnostics(root)
    failure=modules["dev705_paired_mask_failure_scope"]
    score=modules["dev705_score_template"]
    ordered,counts=score.prefix_rows(score.nonempty_by_image(native,gt["images"]),score.nonempty_by_image(center,gt["images"]))
    need(sum(counts)==41482,"Exact per-arm equal prefix count required")
    adapter=failure.load_official_adapter()
    matches={arm:fixed_matches(gt,ordered[arm],adapter,failure.match_official) for arm in ARMS}

    coco=COCO()
    coco.dataset=gt
    with redirect_stdout(StringIO()):
        coco.createIndex()
    rows=[]
    support=Counter()
    shifts=Counter()
    for ann in gt["annotations"]:
        if ann["area"]<9216 or ann.get("ignore",0) or ann.get("iscrowd",0):
            continue
        rle=coco.annToRLE(ann)
        if not mask.area(rle):
            continue
        support["eligible"]+=1
        aid=ann["id"]
        support.update({arm:int(aid in matches[arm]) for arm in ARMS})
        if not all(aid in matches[arm] for arm in ARMS):
            continue
        pred={arm:mask.decode(matches[arm][aid]).astype(bool,copy=False) for arm in ARMS}
        truth=mask.decode(rle).astype(bool,copy=False)
        stratum=failure.stratum(rle,ann["area"])
        row={"group":groups[ann["image_id"]],"stratum":stratum,
             "raw":{arm:split_errors(truth,pred[arm]) for arm in ARMS},"aligned":{}}
        for arm in ARMS:
            disp,shifted=best_shift(truth,pred[arm])
            row["aligned"][arm]=split_errors(truth,shifted)
            shifts[arm]+=disp!=(0,0)
        rows.append(row)
    strata={key:[row for row in rows if key=="all" or row["stratum"]==key]
            for key in ("all","large_edge","large_interior")}
    summary={key:{phase:aggregate(items,phase) for phase in ("raw","aligned")} for key,items in strata.items()}
    base_strata=base["diagnostics"]["paired_mask_failure_scope"]["strata"]
    need(support=={"eligible":16,"native":15,"center":16} and len(rows)==15
         and len(strata["large_edge"])==4 and len(strata["large_interior"])==11
         and all(summary[key]["raw"]["center_minus_native"]["fn"]==base_strata[key]["center_minus_native"]["fn_pixels"]
                 for key in ("large_edge","large_interior"))
         and summary["all"]["raw"]["center_minus_native"]["fn"]==21989,"Prior raw large-match evidence differs")
    group_names=sorted({row["group"] for row in rows})
    need(len(group_names)==3,"Fixed large-target proxy group count differs")
    leave=[aggregate([r for r in rows if r["group"]!=g],"aligned") for g in group_names]
    per_group=[aggregate([r for r in rows if r["group"]==g],"aligned") for g in group_names]
    aligned=summary["all"]["aligned"]
    delta=aligned["center_minus_native"]
    criteria=dict(excess_after_translation=delta["fn"]>0,
                  core_at_least_half_net_excess=delta["core_fn"]>0 and 2*delta["core_fn"]>=delta["fn"],
                  positive_after_omitting_any_group=all(r["center_minus_native"]["core_fn"]>0 for r in leave),
                  core_rate_excess_at_least_band_rate=aligned["center_minus_native_core_rate"] is not None and
                    aligned["center_minus_native_band_rate"] is not None and
                    aligned["center_minus_native_core_rate"]>=aligned["center_minus_native_band_rate"])
    return dict(status="HRSID_COMMON_DEV705_LARGE_CORE_FN_ACTUAL",support=dict(support,common=15),
                proxy_group_count=3, shifted_pairs_by_arm=dict(shifts), strata=summary, gate=criteria,
                aligned_groups=per_group,aligned_leave_one_group_out=leave,core_sampler_candidate_passes=all(criteria.values()),
                core_chebyshev_erosion_radius=2,per_arm_translation_limit_px=2,off_canvas_shifts_forbidden=True,
                fixed_original_matching=True, mask_edits_for_deployment=False, official_TEST_opened=False,
                COCO_AP=False, model_forward_calls=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("code-root","pair","native-arm-receipt","center-arm-receipt","accepted-result",
                 "prior-result","image-manifest","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    for name in ("pair-sha256","training-protocol","accepted-result-sha256"):
        parser.add_argument("--"+name,required=True)
    args=parser.parse_args()
    report=run(args)
    data=(json.dumps(report,sort_keys=True,allow_nan=False)+"\n").encode()
    with args.output.open("xb") as stream:
        stream.write(data)
    print(json.dumps(dict(status=report["status"],sha256=sha256(data).hexdigest())),flush=True)


if __name__=="__main__":
    main()
