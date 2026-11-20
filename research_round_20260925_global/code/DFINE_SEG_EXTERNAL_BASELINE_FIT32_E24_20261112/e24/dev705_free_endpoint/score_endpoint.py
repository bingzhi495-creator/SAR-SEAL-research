"""Score sealed DEV705 endpoint edits under the prospectively frozen protocol."""
import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path, PurePosixPath
import re
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask as codec
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

REPLAY_SOURCE="6681d3903199214cd109efd9a8bf92681d838901e6c9c8780e4f19f7e16703b7"
HISTORICAL="1d76e8d55be1da022e0962917e1490769994caf3636aaa4a2c1bbe36ab424588"
HISTORICAL_INDEPENDENT="7e179942859dddbaaa0c3ed576964a6664423d6a90ce5debf1a3fa3bc192950d"
GEOM="7cad8b51a9230101e2233db8455f84c28163abfb2ee8ddd8cbe8718a2e719bd0"
MATCH="12fd1684b262a3c9b6732b90a0c1e624034c523a7d130919a079e17728f80075"
MODES=("raw","fixed","free_matched_count","free_natural","native_matched_count")


def need(ok,reason):
    if not ok:
        raise ValueError(reason)


def load_worker(path):
    raw=path.read_bytes()
    need(sha256(raw).hexdigest()==REPLAY_SOURCE,"Exact locked label-free worker")
    out=ModuleType("_frozen_dev_prelabel")
    out.__file__=str(path)
    exec(compile(raw,str(path),"exec"),out.__dict__)
    return out


def candidate_rows(coco,changes,mode):
    for i,row in enumerate(coco,1):
        v=changes.get(str(i),{}).get(mode)
        yield row if v is None else dict(row,segmentation=v["rle"])


def official(gt,rows,adapter):
    g=COCO()
    g.dataset=deepcopy(gt)
    captured=[]
    def make(*args):
        model=COCOeval(*args)
        captured.append(model)
        return model
    with redirect_stdout(StringIO()):
        g.createIndex()
        metrics=adapter.evaluate_population(dict(COCO=COCO,COCOeval=make),g,deepcopy(rows),[im["id"] for im in gt["images"]])
    model,=captured
    i,=np.flatnonzero(np.isclose(model.params.iouThrs,.90))
    values=model.eval["precision"][i,:,:,0,-1]
    valid=values[values>-1]
    return dict(AP=metrics[0],AP75=metrics[2],AP90=float(valid.mean()) if valid.size else -1.,
                APs=metrics[3],AR100=metrics[8],all12=metrics)


def as_group(item):
    name=PurePosixPath(item["file_name"]).stem
    group,sep,_=name.partition("_")
    need(sep and re.fullmatch(r"P[0-9]+",group) is not None,"Exact source-scene prefix")
    return group


def continuous(rle,area,truth,modules):
    rectangles=modules.geometry.boxes_from_counts(modules.shift.runs(rle["counts"]),*rle["size"])
    need(np.prod(rectangles[:,2:]-rectangles[:,:2],axis=1).sum()==area,"Exact unit-cell native mask area")
    overlap=modules.geometry.overlap_area(rectangles,truth)
    cap=min(area,truth.area)
    need(-1e-8<=overlap<=cap+1e-8,"Exact continuous intersection bounds")
    overlap=float(np.clip(overlap,0,cap))
    return (100*overlap/(area+truth.area-overlap),100*(area-overlap)/truth.area,100*(truth.area-overlap)/truth.area)


def transition(before,after,k):
    if before and after:
        return "edited_retained" if k else "clean_retained"
    if before:
        return "lost"
    if after:
        return "new"
    return "stable_miss"


def geometry(gt,blocks,stock,modules):
    truth,neutral,poly=modules.geometry.polygon_support(gt,modules.vector)
    annotations=defaultdict(list)
    for ann in gt["annotations"]:
        annotations[ann["image_id"]].append(ann)
    counts=Counter(stock_matches=0,neutral_or_invalid=0,area_outside=0,eligible=0)
    edits={arm:Counter() for arm in MODES}
    rows=[]
    witnesses=[]
    transitions={arm:Counter() for arm in MODES}
    for im,(item,coco,audit,changes) in zip(gt["images"],blocks,strict=True):
        identity=im["id"]
        need(identity==item["image_id"],"Exact stock match source identity")
        pool=[dict(rank=i,score=row["score"],rle=row["segmentation"],area=c["mask_area"])
              for i,(row,c) in enumerate(zip(coco,audit,strict=True),1)]
        with redirect_stdout(StringIO()):
            matches,_,_=stock.freeze_coco(im,annotations[identity],pool)
        roles={}
        counts["stock_matches"]+=len(matches)
        for gt_id,rank in matches.items():
            if gt_id in neutral or gt_id not in truth:
                counts["neutral_or_invalid"]+=1
                roles[rank]="neutral"
                continue
            roles[rank]="valid"
            native=pool[rank-1]
            area=native["area"]
            if not 32<=area<1024:
                counts["area_outside"]+=1
                continue
            counts["eligible"]+=1
            gt_shape=truth[gt_id][0]
            raw=continuous(native["rle"],area,gt_shape,modules)
            raw_strict=raw[0]>=90.
            measures=[]
            states={}
            for arm in MODES:
                edited=changes.get(str(rank),{}).get(arm)
                values=raw if edited is None else continuous(edited["rle"],area,gt_shape,modules)
                measures.append(values)
                k=0 if edited is None else edited["swaps"]
                state=transition(raw_strict,values[0]>=90.,k)
                transitions[arm][state]+=1
                states[arm]=dict(state=state,swaps=k)
            rows.append((as_group(item),np.asarray(measures)))
            witnesses.append(dict(image_id=identity,native_rank=rank,original_gt_id=gt_id,raw_strict90=raw_strict,arms=states))
        for rank,arms in changes.items():
            role=roles.get(int(rank),"unmatched")
            for arm,value in arms.items():
                edits[arm].update(edited=1,swaps=value["swaps"])
                edits[arm]["edited_"+role]+=1
    need(counts["stock_matches"]==1652 and counts["stock_matches"]==
         counts["neutral_or_invalid"]+counts["area_outside"]+counts["eligible"] and len(rows)==counts["eligible"],
         "Exact original stock match denominator and fixed geometry partition")
    grouped=defaultdict(list)
    for group,m in rows:
        grouped[group].append(m)
    pairs=np.asarray([x[1] for x in rows])
    groups=np.asarray([np.mean(grouped[g],axis=0) for g in sorted(grouped)])
    draws=np.random.Generator(np.random.PCG64(5217)).integers(len(groups),size=(4096,len(groups)))
    comparisons={}
    for a,b in (("free_natural","raw"),("free_natural","fixed"),("free_natural","free_matched_count"),
                ("free_natural","native_matched_count"),("free_matched_count","fixed")):
        delta=groups[:,MODES.index(a),0]-groups[:,MODES.index(b),0]
        comparisons[a+"_minus_"+b]=dict(delta_points=float(delta.mean()),
                                        interval95_points=np.percentile(delta[draws].mean(axis=1),[2.5,97.5]).tolist())
    metrics={}
    strict={}
    names=("clean_retained","edited_retained","lost","new","stable_miss")
    for i,arm in enumerate(MODES):
        t=transitions[arm]
        hit=int((pairs[:,i,0]>=90).sum())
        need(sum(t.values())==len(rows) and t["clean_retained"]+t["edited_retained"]+t["new"]==hit,
             "Exact independent original strict identity partition")
        metrics[arm]=dict(group_equal_iou_points=float(groups[:,i,0].mean()),pair_equal_iou_points=float(pairs[:,i,0].mean()),
                          fp_per_truth_points=float(groups[:,i,1].mean()),fn_per_truth_points=float(groups[:,i,2].mean()),
                          strict75=int((pairs[:,i,0]>=75).sum()),strict90=hit)
        strict[arm]={k:t[k] for k in names}
        edits[arm]["no_op"]=70500-edits[arm]["edited"]
    safe=dict(counts=dict(counts),source_groups=len(groups),bootstrap_draws=4096,bootstrap_seed=5217,metrics=metrics,
              strict90_transitions=strict,comparisons=comparisons,budgets={a:dict(v) for a,v in edits.items()},polygon_support=poly)
    return safe,witnesses


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("protocol","replay-source","dev-source","dev-root","sidecar","sidecar-accepted","replay",
                "replay-accepted","code","score-adapter","shapely-site","historical","historical-independent","output"):
        p.add_argument("--"+key,type=Path,required=True)
    for key in ("sidecar-accepted-sha","replay-sha","replay-accepted-sha"):
        p.add_argument("--"+key,required=True)
    a=p.parse_args()
    replay=load_worker(a.replay_source)
    replay.pin(a.protocol,replay.PROTOCOL)
    dev=replay.load(a.dev_source,replay.DEV_SOURCE,"_dev_score_author")
    dev.cpu_runtime()
    sidecar=replay.probability_receipt(a,dev)
    manifest=dev.read_manifest(a.dev_root)
    prepared=replay.require_prelabel(a)
    accepted=json.loads(replay.pin(a.replay_accepted,a.replay_accepted_sha))
    need(accepted["status"]==replay.VERIFIED and accepted["protocol_sha256"]==replay.PROTOCOL and
         accepted["replay_sha256"]==a.replay_sha and accepted["sidecar_accepted_sha256"]==a.sidecar_accepted_sha and
         accepted["edits_sha256"]==prepared["edits_sha256"] and accepted["original_exit"]==0 and
         accepted["images"]==705 and accepted["candidates"]==70500 and
         accepted["DEV"] is True and accepted["GT_opened"] is accepted["AP"] is accepted["TEST"] is accepted["FIT"] is False,
         "Exact label-free replay independent acceptance")
    prior=json.loads(replay.pin(a.historical,HISTORICAL))
    check=json.loads(replay.pin(a.historical_independent,HISTORICAL_INDEPENDENT))
    need(check["exact_26_metrics"] is True and check["score_sha256"]==HISTORICAL and
         prior["predictions_sha256"]==replay.PREDICTIONS and prior["candidate_audit_sha256"]==replay.AUDIT and
         prior["gt_sha256"]==dev.FILES["DEV_ORIGINAL_GT_PRIVATE.json"] and prior["primary"]["AP"]==.4816040264713373,
         "Exact historical official DEV endpoint")
    adapter=replay.load(a.score_adapter,dev.SCORE_ADAPTER_SHA,"_dev_score_adapter")
    source=replay.load(a.code/"MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py",GEOM,"_dev_continuous")
    stock=replay.load(a.code/"HRSID_FIT_HULL_CDF_20261112/stock_coco_match.py",MATCH,"_dev_stock_match")
    sys.path.append(str(a.shapely_site.resolve(strict=True)))
    import shapely
    need(shapely.__version__=="2.0.2" and shapely.geos_version_string=="3.11.2","Exact continuous geometry runtime")
    modules=source.geometry_modules(a.code)
    blocks=list(replay.replay_rows(a,dev,manifest,sidecar,prepared))
    need(len(blocks)==705 and sum(len(x[1]) for x in blocks)==70500,"All original top100 predictions")
    a.output.mkdir()
    receipt=dev.write_new(a.output/"PRELABEL_SAFE.json",dict(status="DFINE_DEV705_FREE_ENDPOINT_SCORE_PRELABEL",
                         protocol_sha256=replay.PROTOCOL,replay_sha256=a.replay_sha,replay_accepted_sha256=a.replay_accepted_sha,
                         edits_sha256=prepared["edits_sha256"],history_sha256=HISTORICAL,DEV=True,FIT=False,TEST=False,
                         GT_opened=False,AP=False))
    gt=dev.dev_json(a.dev_root,"DEV_ORIGINAL_GT_PRIVATE.json")
    dev.verify_labels(gt,dev.dev_json(a.dev_root,"DEV_EMPTY_RASTERS_PRIVATE.json"),manifest)
    started=perf_counter()
    coco={m:official(gt,[r for _,rows,_,changes in blocks for r in candidate_rows(rows,changes,m)],adapter)
          for m in MODES}
    need(all(coco["raw"][k]==prior["primary"][k] for k in ("AP","AP75","AP90","APs","AR100","all12")),
         "Exact raw historical COCO parity")
    fixed,witness=geometry(gt,blocks,stock,modules)
    out=a.output/"STRICT_TRANSITIONS_PRIVATE.jsonl"
    with out.open("xb") as file:
        for row in witness:
            file.write(dev.encoded(row))
    loss=fixed["strict90_transitions"]["free_natural"]["lost"]
    gates=dict(AP_vs_raw=100*(coco["free_natural"]["AP"]-coco["raw"]["AP"])>=1.,
               AP_vs_fixed=100*(coco["free_natural"]["AP"]-coco["fixed"]["AP"])>=.50,
               AP75_retained=coco["free_natural"]["AP75"]>=coco["raw"]["AP75"]-1e-12,
               zero_original_strict90_loss=loss==0)
    for arm in ("raw","fixed"):
        v=fixed["comparisons"]["free_natural_minus_"+arm]
        gates["geometry_vs_"+arm]=v["delta_points"]>=.20 and v["interval95_points"][0]>0.
    result=dict(status="DFINE_DEV705_FREE_ENDPOINT_ACTUAL",protocol_sha256=replay.PROTOCOL,
                historical_score_sha256=HISTORICAL,prelabel_sha256=receipt,replay_accepted_sha256=a.replay_accepted_sha,
                transition_witness_sha256=replay.file_hash(out),images=705,candidates=70500,raw_exact_parity=True,
                coco_fraction=coco,geometry_points=fixed,prelabel_counts=prepared["counts"],
                replay_seconds=prepared["replay_seconds"],score_seconds=perf_counter()-started,
                DEV=True,FIT=False,TEST=False,GT_opened=True,AP=True,official_TEST_AP=False,detector_forwards=0,
                gates=gates,advance=all(gates.values()))
    digest=dev.write_new(a.output/"RESULT_SAFE.json",result)
    print(json.dumps(dict(status=result["status"],result_sha256=digest,gates=gates,advance=result["advance"]),
                     sort_keys=True),flush=True)


if __name__=="__main__":
    main()
