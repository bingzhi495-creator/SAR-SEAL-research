"""Label-based geometric ceilings on the frozen PSeg TEST232 endpoint support."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask as codec
import torch


REPLAY_SHA = "f1b6e83e98851bf867bd72f3ecb082ca2997632bdd1aedaaf64110cf06b04fd5"
DIAGNOSTIC_SHA = "107f31864bbb7e807a4194e552c7a1778d410f0e564f80437c5717fe6b47098d"
CAPTURE_READBACK = "dc48e18c0b988fda3eada68dc06b1404fd5c3806502e32d455a08a2e96d0121c"
FIT_RECEIPT = "e99d0f2bbd9e1839e260a88075f6af187e05ffc2b86abaf6cc33b788372b7b71"
ARMS = ("raw", "natural", "fixed_pair_oracle", "balanced_oracle", "free_area_oracle")
EFFECTS = ((1,0),(2,1),(3,1),(3,2),(4,3))


def need(value, why):
    if not value:
        raise ValueError(why)


def pinned(path, expected):
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == expected, "Exact original input: " + path.name)
    return raw


def module(path, expected, name):
    m = ModuleType(name)
    m.__file__ = str(path)
    exec(compile(pinned(path, expected), str(path), "exec"), m.__dict__)
    return m


def iou(inter, predicted, target):
    slack = min(predicted,target)
    need(-1e-7 <= inter <= slack + 1e-7, "Valid polygon intersection and mask area")
    safe = min(max(0.,inter),slack)
    return 100*safe/(predicted + target - safe)


def attainable(q, pairs, predicted, target, intersection):
    """Maximize IoU on the same points using exactly the realized edit budget."""
    k = len(pairs)
    origin = iou(intersection, predicted, target)
    if not k:
        return np.full(len(ARMS),origin)
    actions = np.asarray(pairs,dtype=np.intp)
    gain = (q[actions[:,0],0] - q[actions[:,1],1]).sum()
    fixed_pair = np.sort(q[:,0] - q[:,1])[-k:].sum()
    add = np.r_[0., np.cumsum(np.sort(q[:,0])[::-1])]
    drop = np.r_[0., np.cumsum(np.sort(q[:,1]))]
    balanced = iou(intersection+add[k]-drop[k], predicted, target)
    free = max(iou(intersection+add[n]-drop[2*k-n], predicted+2*n-2*k,target)
               for n in range(max(0,2*k-len(q)),min(2*k,len(q))+1))
    measured = np.array([origin, iou(intersection+gain,predicted,target),
                         iou(intersection+fixed_pair,predicted,target),balanced,free])
    need(all(balanced+1e-9>=v for v in measured[1:3]) and free+1e-9>=balanced,
         "Feasible sets and continuous optimum bounds")
    return measured


def analyze(groups):
    image_names = sorted(set(row[0] for row in groups))
    sampled = {n:[] for n in image_names}
    for image,values,swaps in groups:
        sampled[image].append(values)
    means = np.asarray([np.mean(sampled[n],axis=0) for n in image_names])
    choices = np.random.Generator(np.random.PCG64(5217)).integers(len(means),size=(4096,len(means)))
    raw = np.asarray([v for _,v,_ in groups])
    effects = {}
    for a,b in EFFECTS:
        diff = means[:,a]-means[:,b]
        effects[ARMS[a]+" minus "+ARMS[b]] = dict(
            delta_points=float(diff.mean()),
            interval95_points=np.percentile(diff[choices].mean(axis=1),[2.5,97.5]).tolist(),
            paired_improved=int(np.sum(raw[:,a] > raw[:,b]+1e-10)),
            paired_degraded=int(np.sum(raw[:,a] < raw[:,b]-1e-10)))
    return dict(original_matches=len(groups), image_groups=len(means),
                affected_matches=sum(k>0 for _,_,k in groups), swaps=sum(k for _,_,k in groups),
                continuous_iou_points=dict(zip(ARMS,means.mean(axis=0).tolist())),
                strict90_counts=dict(zip(ARMS,(raw>=90).sum(axis=0).tolist())),
                paired_effects=effects)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("replay","diagnostic","manifest","capture","accepted","images","code","score","gt",
                 "native","audit","edits","replay-accepted","refit-result","refit-acceptance","weights","shapely-site","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--manifest-sha",required=True)
    args = parser.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1",
         "Exact CPU-only historical diagnostic")
    started = perf_counter()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.accepted_sha = CAPTURE_READBACK
    replay = module(args.replay,REPLAY_SHA,"_orig_pseg_replay")
    m = replay.modules(args.code,args)
    m.author.cpu_runtime()
    roster,receipt = replay.admit(m,args)
    diag = module(args.diagnostic,DIAGNOSTIC_SHA,"_orig_geometry")
    qualification=json.loads(pinned(args.replay_accepted,FIT_RECEIPT))
    need(qualification["status"]==replay.VERIFIED and qualification["exact_native_support"]
         and qualification["images"]==232,"Verified actual previously closed edit receipt")
    score,gt,records = diag.read_data(args)
    need(qualification["edits_sha256"]==score["edits_sha256"]
         and qualification["refit_accepted_sha256"]==m.old.REFIT_ACCEPTED,"Exact adjudicated candidate edits")
    matches,_ = diag.evaluate_raw(gt,records,sorted(records),score["views"]["official"]["metrics_fraction"]["raw"])
    sys.path.append(str(args.shapely_site.resolve(strict=True)))
    import shapely
    need(shapely.__version__=="2.0.2" and shapely.geos_version_string=="3.11.2","Exact geometry engine")
    engine = diag.load(args.code/"MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py",diag.GEOM)
    geometry = engine.geometry_modules(args.code)
    truth,neutral,_ = geometry.geometry.polygon_support(gt,geometry.vector)
    need(len(truth)==546 and not neutral,"Original float polygons")
    pair,rank,free,learner,head,center,scale = m.old.restore(args)
    pair.raw = m.legacy.pin(args.pair_math,m.native.PAIR)
    matched_by_name = defaultdict(list)
    for gid,(image_id,native_rank) in matches.items():
        rec=records[image_id]
        if 32<=rec["audit"][native_rank-1]["mask_area"]<1024:
            matched_by_name[rec["name"]].append((gid,rec,native_rank))
    samples = []
    filtered = []
    for item,rows,details,planes in m.legacy.blocks(args,m.author,roster,receipt,True):
        name=item["image_ref"]["path"]
        population=matched_by_name.get(name,[])
        if not population:
            continue
        image=m.legacy.rgb(args,item)
        adapted,probs=m.native.make_native(pair,rank,*item["original_hw"])
        features=pair.image_features(image)
        for gid,rec,j in population:
            row,detail,plane=rows[j-1],details[j-1],planes[j-1]
            polygon=truth[gid][0]
            area=detail["mask_area"]
            ref_area=polygon.area
            rectangles=geometry.geometry.boxes_from_counts(geometry.shift.runs(row["segmentation"]["counts"]),*row["segmentation"]["size"])
            need(np.prod(rectangles[:,2:]-rectangles[:,:2],axis=1).sum()==area,"Original exact native area")
            inter=geometry.geometry.overlap_area(rectangles,polygon)
            data=adapted.paired_points(codec.decode(row["segmentation"]).astype(bool),detail["box_xyxy"],
                                       area,row["score"],plane,features,probs)
            saved=rec["edit"].get(str(j),{}).get("free_natural",{})
            natural=[]
            if data is not None:
                coords, feature = data[:2]
                p=learner.probabilities(head,feature,feature,center,scale)
                natural=free.choose_free(p,min(8,area//10,len(p)))
                y,x=np.divmod(coords,item["original_hw"][1])
                q=np.asarray(shapely.area(shapely.intersection(shapely.box(x,y,x+1,y+1),polygon)))
            else:
                q=np.empty((0,2))
            need(saved.get("swaps",0)==len(natural),"Exact original per-candidate realized natural budget")
            if natural:
                stock=codec.decode(row["segmentation"]).astype(bool)
                exact=m.old.edit(stock,coords,natural)
                need(exact==saved["rle"],"Unmodified natural edit payload")
            metrics=attainable(q,natural,area,ref_area,inter)
            record=(rec["rgb"],metrics,len(natural))
            samples.append(record)
            if int(Path(name).stem) not in diag.OMIT:
                filtered.append(record)
    need(len(samples)==293 and len(filtered)==279,"Full eligible fixed-match populations")
    result=dict(status="PSEG_TEST232_FROZEN_SUPPORT_ORACLE_GEOMETRY_COMPLETE",official=analyze(samples),
                exclude_exact_fit_overlap=analyze(filtered),historical_scoring_scope=True,
                source_score_sha256=diag.SCORE,refit_accepted_sha256=m.old.REFIT_ACCEPTED,
                original_replay_acceptance_sha256=FIT_RECEIPT,source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                actual_model_forwards=0,optimizer_updates=0,cohort_AP_scored=False,ceilings_use_target_truth=True,
                actual_raw_coco_recalculated_for_pair_identity=True,seconds=perf_counter()-started)
    content=(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    with args.output.open("xb") as f:
        f.write(content)
    print(json.dumps(dict(status=result["status"],result_sha256=sha256(content).hexdigest())),flush=True)


if __name__=="__main__":
    main()
