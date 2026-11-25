"""Read closed scores and decompose net false foreground on original matches."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import runpy
import sys
from time import perf_counter
from zipfile import ZipFile

HERE=Path(__file__).resolve().parent
LOCK="a991c38c5e6fb652d187e9f5e69c16560bbf6b9692a78bd2f6de76af85559779"
GEOM="3d6582f550edf6b7fbecea62c2d20360b36868a3ec6a0621ae3dcb6653b88203"
CLOSURE="dd1b56f42d4358e356f7127f0c46ad8b4f4489502ccb54a2fbe3992361b9c4b6"
ARMS=("fixed_iou","variable_iou")


def require(ok,reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def verify(path,sha):
    require(digest(path.read_bytes())==sha,"Verified input: "+path.name)


def summarize(rows,np):
    fields=sorted({key for items in rows.values() for row in items for key in row})
    matrix=np.array([[np.mean([row.get(key,0.) for row in rows[group]]) for key in fields] for group in sorted(rows)])
    draws=np.random.Generator(np.random.PCG64(5217)).integers(len(rows),size=(4096,len(rows)))
    intervals=np.percentile(matrix[draws].mean(axis=1),[2.5,97.5],axis=0).T
    return {key:dict(points=float(mean),interval95=interval.tolist())
            for key,mean,interval in zip(fields,matrix.mean(axis=0),intervals,strict=True)}


def run(a):
    started=perf_counter()
    verify(a.closure,CLOSURE)
    prior_audit=runpy.run_path(str(a.closure))
    bound,prior_score,prior=prior_audit["closed"](a.transfer)
    sys.path[:0]=[str(prior.OLD/"capture/CODEC_PRIVATE"),str(prior.OLD/"numpy_1_26_4_only"),str(HERE)]
    import numpy as np
    from pycocotools import mask as codec
    from geometry import attachment

    old,m,args,roster,capture,check,_,scoring,policy=prior.setup()
    stream=scoring.read_stream(a.transfer/"render/RENDER_PRIVATE.jsonl",bound["raw_sha256"])
    previous=scoring.read_stream(prior.OLD/"prepare/EDITS_PRIVATE.jsonl",check["edits_sha256"])
    predictions={}
    ledger={}
    for j,(item,native,_,_) in enumerate(m.legacy.blocks(args,m.author,roster,capture,False)):
        iid=item["image_id"]
        masks=stream[j]
        historical=previous[j]
        require(iid==masks["image_id"]==historical["image_id"] and len(native)==100
             and masks["rgb_sha256"]==historical["rgb_sha256"]==item["rgb_sha256"],
             "Exact admitted original 100 slots and historical rank identity")
        require(set(masks["edits"])=={k for k,v in historical["changes"].items()
                                       if "attraction_static_at_dynamic_count" in v},
                "Exact archived positive-budget subset")
        predictions[iid]=[dict(native=x["segmentation"],score=x["score"],masks={}) for x in native]
        ledger[iid]=masks["edits"]
    require(len(predictions)==len(stream)==len(previous)==232 and sum(len(row) for row in ledger.values())==10393,
           "Original candidate census parity")
    require(old.file_hash(prior.ARCHIVE)==scoring.ARCHIVE_SHA,"Exact original SAR-SSP author archive")
    names={row["image_id"]:(Path(row["image_ref"]["path"]).name,row["image_ref"]["sha256"]) for row in roster}
    with ZipFile(prior.ARCHIVE) as archive:
        truth,groups,edges=scoring.exact_ground_truth(archive,names)
    evaluate=scoring.load_evaluator(prior.CODE/scoring.EVALUATOR_PATH)
    raw_ap=evaluate(truth,scoring.coco_rows(predictions,"native"))["AP"]
    observed=prior_score["scoring"]
    require(100*raw_ap==observed["coco_mask_metric_points"]["native"]["AP"],"Exact native AP")
    matches=scoring.native_matches(truth,predictions,raw_ap)
    require(len(matches)==88 and len(truth["annotations"])==139,"Exact native unmatched census")
    groups_out={n:{sub:defaultdict(list) for sub in ("all","nonedge")} for n in (8,4)}
    raw={n:Counter() for n in (8,4)}
    rosters=Counter()
    changed_group=set()
    for ref in truth["annotations"]:
        tid,iid=ref["id"],ref["image_id"]
        gt=ref["segmentation"]
        gt_area=int(codec.area(gt))
        values=dict(fixed_matching_iou=0.,variable_matching_iou=0.)
        entry={}
        if tid in matches:
            matched_iid,rank=matches[tid]
            require(matched_iid==iid,"Correct original image match")
            native=predictions[iid][rank-1]["native"]
            edits=ledger[iid].get(str(rank))
            fixed,variable=[edits["arms"][arm]["rle"] if edits else native for arm in ARMS]
            measured=[scoring.hull_measures(rle,gt,gt_area) for rle in (fixed,variable)]
            values=dict(fixed_matching_iou=measured[0][0],variable_matching_iou=measured[1][0],
                        net_iou=measured[1][0]-measured[0][0])
            if fixed!=variable:
                require(bool(edits),"Changed fixed/variable masks have a locked positive quota")
                for rle,arm in zip((fixed,variable),ARMS,strict=True):
                    action=edits["arms"][arm]["action"]
                    signed=scoring.signed_counts(native,rle,int(codec.area(native)))
                    require(tuple(edits["arms"][arm]["counts"])==signed==tuple(map(len,action))
                            and sum(signed)==2*edits["k"],"Exact original 2k edits and both signed quotas")
                arrays=[codec.decode(rle).astype(bool) for rle in (native,fixed,variable,gt)]
                for n in (8,4):
                    entry[n]=attachment(*arrays,n)
                    raw[n].update(entry[n])
                    require(abs(100*entry[n]["net_fp"]/gt_area-(measured[1][1]-measured[0][1]))<1e-10
                         and abs(100*entry[n]["net_fn"]/gt_area-(measured[1][2]-measured[0][2]))<1e-10,
                         "Exact pixel sums to previous normalized false positives and misses")
                rosters.update(changed_matches=1,changed_nonedge=int(tid not in edges))
                changed_group.add(groups[iid])
        for n in (8,4):
            row=values|{k:100*v/gt_area for k,v in entry.get(n,{}).items()}
            groups_out[n]["all"][groups[iid]].append(row)
            if tid not in edges:
                groups_out[n]["nonedge"][groups[iid]].append(row)
    require(rosters["changed_matches"]==48 and len(changed_group)==20,"Original changed match and parent counts")
    results={str(n):{sub:summarize(items,np) for sub,items in groups_out[n].items()} for n in (8,4)}
    for n in ("8","4"):
        for sub,key in (("all","fixed_native_match_all_truth"),("nonedge","fixed_native_match_nonedge_sensitivity")):
            geom=observed[key]
            arm=geom["arms"]
            for lhs,rhs in (("fixed",ARMS[0]),("variable",ARMS[1])):
                require(abs(results[n][sub][lhs+"_matching_iou"]["points"]-arm[rhs]["iou_points"])<1e-10,
                        "Historical fixed-IoU and non-edge sensitivity")
            expected=geom["comparisons"]["variable_iou__vs__fixed_iou"]
            for short,long in (("net_fp","false_positive_area_over_truth_points"),("net_fn","false_negative_area_over_truth_points"),
                               ("net_iou","iou_points")):
                require(np.allclose(results[n][sub][short]["interval95"],expected[long]["percentile95"],rtol=0,atol=1e-10)
                     and abs(results[n][sub][short]["points"]-expected[long]["delta"])<1e-10,
                     "Exact prior pooled geometry and group interval parity")
    both=lambda key:all(results[n]["all"][key]["interval95"][0]>0 for n in ("8","4"))
    return dict(status="SAR_SSP_FALSE_FOREGROUND_ATTACHMENT_COMPLETE",protocol_sha256=LOCK,worker_sha256=digest(Path(__file__).read_bytes()),
                geometry_sha256=GEOM,author_label_assisted=True,historical_exposure=True,method_tested=False,
                no_training_or_new_detector_forwards=True,
                prior_render_sha256=prior_audit["RENDER"],prior_score_sha256=prior_audit["SCORE"],exact_historical_parity=True,
                n_images=232,n_hulls=139,n_parents=23,n_matched=88,changed_parent_count=len(changed_group),**rosters,
                raw_pixel_counts={str(n):dict(v) for n,v in raw.items()},geometry_points=results,
                maternal_bootstrap_count=4096,main_gain_dominant=both("net_fp") and both("main_vs_other"),
                main_gain_explains_net_after_all_removals=both("main_less_all_removal"),
                elapsed_seconds=perf_counter()-started)


def main():
    p=ArgumentParser(description=__doc__)
    for key in ("closure","transfer","output"):
        p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args()
    require(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1"
            and not a.output.exists(),"Fresh isolated CPU audit")
    verify(HERE/"PRELOCK.md",LOCK)
    verify(HERE/"geometry.py",GEOM)
    def offline(event,_):
        if event.startswith("socket."):
            raise PermissionError("This audit is offline")
    sys.addaudithook(offline)
    doc=run(a)
    raw=(json.dumps(doc,sort_keys=True,allow_nan=False,indent=2)+"\n").encode()
    with a.output.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=doc["status"],result_sha256=digest(raw))),flush=True)

if __name__=="__main__":
    main()

