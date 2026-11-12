"""Prepare label-free equal-budget masks and score fixed D-FINE FIT109 probability ranking."""
import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
from pycocotools import mask as codec

PROTOCOL = "cb6b500a82174cc0f07090a8fc56d89c90df752f74029a21580b0a5235fac699"
MATH = "cb438f8937f8e29dbdcbc441abdf5a788ac9c442fd63f1b3ad504b8ae0fcca0b"
CAPTURE = "7d9b19175bc445fef6fb1387b287b1c32002d7c74054cc16ae4431963e25f3ad"
ACCEPTED = "cd4c2a6a7859319429546f8b9a2ab31f6cfbf7076569ad44611c51f77a5f1d35"
CORE = "c1d4d0dcdebbe47b083e24ef96be5db57d4655caea919703d183ea5544e0c0fa"
GT = "e95cf26cc60a2c6fb651e907541380a256f2df7b1e153244a6c0f803b0b19f8f"
GEOM = "7cad8b51a9230101e2233db8455f84c28163abfb2ee8ddd8cbe8718a2e719bd0"
MATCH = "12fd1684b262a3c9b6732b90a0c1e624034c523a7d130919a079e17728f80075"
HULL = "dedd47d5453b2442ff02585c9b4d216917e5ed175e01eae07a354a22710f3a94"
PRELABEL = "DFINE_FIT109_BOUNDARY_PROBABILITY_PRELABEL"
MODES = ("raw","geometry","probability","permuted","oracle")
MIB = 2**20


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def exact(path, digest, cap=4*MIB):
    need(path.is_absolute() and path.resolve(strict=True)==path and path.is_file()
         and 0 < path.stat().st_size <= cap,"Exact canonical bounded file")
    raw=path.read_bytes()
    need(sha256(raw).hexdigest()==digest,"Exact closed SHA "+path.name)
    return raw


def encoded(value):
    return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def seal(path,raw):
    with path.open("xb") as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())
    return sha256(raw).hexdigest()


def load(path,digest,name):
    module=ModuleType(name)
    module.__file__=str(path)
    exec(compile(exact(path,digest),str(path),"exec"),module.__dict__)
    return module


def read_core(a):
    capture=json.loads(exact(a.capture/"RESULT_SAFE.json",CAPTURE))
    check=json.loads(exact(a.accepted,ACCEPTED))
    need(check["raw_probability_arrays_verified"] and check["result_sha256"]==CAPTURE
         and check["status"]=="DFINE_E24_FIT114_IMAGE_CAPTURE_INDEPENDENT_ACCEPTED"
         and capture["split_reports"]["core"]==check["split_reports"]["core"]
         and all(r["FIT"] and not any(r[k] for k in ("DEV","TEST","GT_opened","AP")) for r in (check,capture)),
         "Exact independently closed FIT capture")
    core=a.capture/"core"
    blocks=[json.loads(line) for line in exact(core/"CANDIDATES_PRIVATE.jsonl",CORE,32*MIB).splitlines()]
    need(len(blocks)==len({b["group"] for b in blocks})==len({b["image_id"] for b in blocks})==
         len({b["rgb_sha256"] for b in blocks})==109 and [b["ordinal"] for b in blocks]==list(range(1,110))
         and sum(len(b["coco"]) for b in blocks)==27735
         and all(b["probabilities"]["rows"]==100 and len(b["coco"])>=100 for b in blocks),
         "Exact ordered distinct 109 FIT source groups")
    return core,blocks


def apply(original, action):
    mask=codec.decode(original).astype(bool,order="C")
    add,drop=action
    mask.ravel()[add]=True
    mask.ravel()[drop]=False
    rle=codec.encode(np.asfortranarray(mask,dtype=np.uint8))
    return dict(size=rle["size"],counts=rle["counts"].decode("ascii"))


def prepare(a,core,blocks,math):
    a.output.mkdir()
    stream_sha=sha256()
    totals=Counter()
    with (a.output/"ACTIONS_PRIVATE.jsonl").open("xb") as out:
        for block in blocks:
            ref=block["probabilities"]
            raw=exact(core/ref["filename"],ref["sha256"],11*MIB)
            probabilities=np.load(BytesIO(raw),allow_pickle=False)
            need(len(raw)==ref["bytes"] and probabilities.shape==(100,160,160) and probabilities.dtype==np.float32,
                 "Exact original captured plane")
            actions={}
            for index,(candidate,row) in enumerate(zip(block["candidates"][:100],block["coco"][:100],strict=True)):
                area=candidate["mask_area"]
                rank=index+1
                need(candidate["native_rank"]==rank and candidate["probability_row"]==index and
                     row["image_id"]==block["image_id"] and row["category_id"]==1 and row["score"]==candidate["score"],
                     "Exact ordered post-NMS candidate identity")
                totals.update(top100=1,empty=int(area==0))
                if not 32<=area<1024:
                    continue
                totals["eligible_predicted_area"]+=1
                hard=codec.decode(row["segmentation"]).astype(bool)
                pools,quotas=math.editable_pools(hard,candidate["box_xyxy"])
                if not sum(quotas.values()):
                    totals["zero_capacity"]+=1
                    continue
                scores={key:math.sample_probability(probabilities[index],pool[0],800) for key,pool in pools.items()}
                for key,p in scores.items():
                    disagreement=int(np.count_nonzero((p>=.5)!=(key.startswith("drop"))))
                    totals.update(queried_pixels=len(p),threshold_discordant_pixels=disagreement,
                                  exact_half_pixels=int((p==.5).sum()))
                result=math.choose(pools,quotas,scores,block["group"],rank)
                math.verify_actions(result["geometry"],pools,quotas)
                math.verify_actions(result["probability"],pools,quotas)
                for edit in result["permuted"]:
                    need(sum(map(len,edit))==sum(quotas.values()),"Exact permutation edit size")
                totals.update(edited=1,added=quotas["add1"]+quotas["add2"],deleted=quotas["drop1"]+quotas["drop2"])
                actions[str(rank)]=dict(quotas=quotas,**result)
            line=encoded(dict(image_id=block["image_id"],group=block["group"],actions=actions))
            out.write(line)
            stream_sha.update(line)
            if block["ordinal"]%25==0 or block["ordinal"]==109:
                print(json.dumps(dict(event="DFINE_FIT109_PROBABILITY_PREPARE_PROGRESS",images=block["ordinal"])),flush=True)
        out.flush()
        os.fsync(out.fileno())
    need(totals["top100"]==10900 and totals["edited"]+totals["zero_capacity"]==totals["eligible_predicted_area"]
         and totals["added"]==totals["deleted"],"Exact balanced core support")
    receipt=dict(status=PRELABEL,protocol_sha256=PROTOCOL,math_sha256=MATH,capture_sha256=CAPTURE,
                 core_sha256=CORE,images=109,FIT=True,DEV=False,TEST=False,AP=False,GT_opened=False,
                 model_forwards=0,optimizer_updates=0,permutation_draws=math.SHAMS,totals=dict(totals),
                 actions_sha256=stream_sha.hexdigest())
    digest=seal(a.output/"PRELABEL_SAFE.json",encoded(receipt))
    print(json.dumps(dict(status=PRELABEL,prelabel_sha256=digest),sort_keys=True),flush=True)


def endpoint(masks, area, truth, base_inter, action, fractions):
    add,drop=action
    intersection=base_inter+sum(fractions[p] for p in add)-sum(fractions[p] for p in drop)
    need(-1e-8<=intersection<=min(area,truth.area)+1e-8,"Exact area-preserving continuous intersection")
    i=max(0.,min(min(area,truth.area),intersection))
    binary_add=int(np.count_nonzero(masks[add]))
    binary_drop=int(len(drop)-np.count_nonzero(masks[drop]))
    return np.array([100*i/(area+truth.area-i),100*(area-i)/truth.area,100*(truth.area-i)/truth.area,
                     binary_add,binary_drop,len(add),len(drop),int(i/(area+truth.area-i)>=.75),
                     int(i/(area+truth.area-i)>=.90)])


def mask_metrics(hard,cand,source_rle,truth,gt_rle,act,math,geom):
    import shapely
    area=float(cand["mask_area"])
    boxes=geom.geometry.boxes_from_counts(geom.shift.runs(source_rle["counts"]),*source_rle["size"])
    need(np.prod(boxes[:,2:]-boxes[:,:2],axis=1).sum()==area,"Exact disjoint native footprint")
    base_inter=geom.geometry.overlap_area(boxes,truth)
    truth_mask=codec.decode(gt_rle).astype(bool).ravel(order="C")
    raw=endpoint(truth_mask,area,truth,base_inter,([],[]),{})
    if act is None:
        return np.stack([raw]*len(MODES))
    pools,quotas=math.editable_pools(hard,cand["box_xyxy"])
    need(quotas==act["quotas"],"Exact locked prediction-only quota")
    by_stratum={}
    values={}
    for key,(coords,_) in pools.items():
        y,x=np.divmod(coords,800)
        fractions=np.asarray(shapely.area(shapely.intersection(shapely.box(x,y,x+1,y+1),truth)),dtype=float)
        need(np.isfinite(fractions).all() and np.all(fractions>=0) and np.all(fractions<=1+1e-12),
             "Exact continuous GT cell fraction")
        by_stratum[key]=fractions
        values.update(zip(coords.tolist(),fractions.tolist(),strict=True))
    oracle=math.oracle(pools,quotas,by_stratum)

    def read(selection):
        math.verify_actions(selection,pools,quotas)
        return endpoint(truth_mask,area,truth,base_inter,selection,values)
    result=np.stack([raw,read(act["geometry"]),read(act["probability"]),
                     np.mean([read(i) for i in act["permuted"]],axis=0),read(oracle)])
    need(result[4,0]>=max(result[1:4,0])-1e-10,"Exact within-quota truth-optimal location")
    return result


def summarize(rows):
    pairs=np.asarray([r[1] for r in rows])
    groups=defaultdict(list)
    for group,metrics in rows:
        groups[group].append(metrics)
    keys=sorted(groups)
    data=np.asarray([np.mean(groups[k],axis=0) for k in keys])
    draws=np.random.Generator(np.random.PCG64(5217)).integers(len(data),size=(4096,len(data)))
    folds=np.asarray([int.from_bytes(sha256(k.encode()).digest()[:8],"little")%3 for k in keys])
    metrics={}
    for i,mode in enumerate(MODES):
        t=pairs[:,i].sum(axis=0)
        metrics[mode]=dict(pair_equal_iou_points=float(pairs[:,i,0].mean()),
                           group_equal_iou_points=float(data[:,i,0].mean()),
                           fp_per_gt_points=float(data[:,i,1].mean()),fn_per_gt_points=float(data[:,i,2].mean()),
                           add_precision=float(t[3]/t[5]) if t[5] else None,
                           drop_precision=float(t[4]/t[6]) if t[6] else None,
                           strict_75=float(t[7]),strict_90=float(t[8]))
    comparisons={}
    for i,j in ((2,1),(2,3),(2,0),(4,2),(4,1)):
        delta=data[:,i,0]-data[:,j,0]
        comparisons[MODES[i]+"_minus_"+MODES[j]]=dict(
            group_equal_delta_points=float(delta.mean()),interval95_points=np.percentile(delta[draws].mean(axis=1),[2.5,97.5]).tolist(),
            fold_points=[float(delta[folds==f].mean()) if (folds==f).any() else None for f in range(3)])
    return dict(pairs=len(rows),source_groups=len(keys),metrics=metrics,comparisons=comparisons)


def stock_ap(gt, rows):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with redirect_stdout(StringIO()):
        coco=COCO()
        coco.dataset=deepcopy(gt)
        coco.createIndex()
        ev=COCOeval(coco,coco.loadRes(deepcopy(rows)),"segm")
        ev.params.imgIds=list(coco.imgs)
        ev.params.catIds=[1]
        ev.params.maxDets=[1,10,100]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return dict(mask_AP=float(100*ev.stats[0]),AP75=float(100*ev.stats[2]))


def score(a,blocks,math):
    receipt=json.loads(exact(a.prelabel/"PRELABEL_SAFE.json",a.prelabel_sha))
    wait=json.loads(exact(a.prelabel_wait,a.prelabel_wait_sha))
    term=wait["terminal"]
    tail=json.loads(term["output"].strip().splitlines()[-1])
    need(term["exit_code"]==0 and tail==dict(status=PRELABEL,prelabel_sha256=a.prelabel_sha)
         and receipt["status"]==PRELABEL and receipt["protocol_sha256"]==PROTOCOL and receipt["math_sha256"]==MATH
         and receipt["GT_opened"] is False and receipt["AP"] is False,"Exact natural label-free prep closure")
    planned=[json.loads(l) for l in exact(a.prelabel/"ACTIONS_PRIVATE.jsonl",receipt["actions_sha256"],96*MIB).splitlines()]
    need(len(planned)==len(blocks)==109 and sum(len(r["actions"]) for r in planned)==receipt["totals"]["edited"],
         "Exact immutable prediction-only action stream")
    sys.path.append(str(a.shapely_site.resolve(strict=True)))
    import shapely
    need(shapely.__version__=="2.0.2" and shapely.geos_version_string=="3.11.2","Exact continuous raster geometry")
    source=load(a.code/"MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py",GEOM,"_df_fit_geom")
    geom=source.geometry_modules(a.code)
    match=load(a.code/"HRSID_FIT_HULL_CDF_20261112/stock_coco_match.py",MATCH,"_df_fit_match")
    hull=load(a.code/"MASK_RTDETR_DEV705_FIXED_MATCH_GEOMETRY_20261112/fixed_match_geometry.py",HULL,"_df_fit_hull")
    a.output.mkdir()
    pregt=seal(a.output/"PRELABEL_SCORE_SAFE.json",encoded(dict(status="DFINE_FIT109_PROBABILITY_SCORE_PRELABEL",
          FIT=True,DEV=False,TEST=False,GT_opened=False,AP=False,prepared_sha256=a.prelabel_sha,
          prepared_original_wait_sha256=a.prelabel_wait_sha,actions_sha256=receipt["actions_sha256"],images=109)))
    gt=json.loads(exact(a.fit_gt,GT,64*MIB))
    need(len(gt["images"])==2937 and len(gt["annotations"])==9260,"Exact original author FIT roster")
    originals={r["id"]:r for r in gt["images"]}
    ids={r["image_id"] for r in blocks}
    images=[]
    anns=defaultdict(list)
    for b in blocks:
        image=originals[b["image_id"]]
        need(image["file_name"]==b["basename"] and (image["height"],image["width"])==(800,800),
             "Exact P192/native original FIT identity join")
        images.append(image)
    for r in gt["annotations"]:
        if r["image_id"] in ids:
            anns[r["image_id"]].append(r)
    subset=dict(images=images,categories=gt["categories"],annotations=[r for rs in anns.values() for r in rs])
    truth,neutral,_=geom.geometry.polygon_support(subset,geom.vector)
    output=defaultdict(list)
    ap={m:[] for m in ("raw","geometry","probability")}
    scope=Counter()
    for image,b,p in zip(images,blocks,planned,strict=True):
        image_id=image["id"]
        need((b["image_id"],b["group"])==(p["image_id"],p["group"]),"Exact prepared image alignment")
        edits=p["actions"]
        pool=[dict(rank=i,score=r["score"],rle=r["segmentation"],area=c["mask_area"])
              for i,(r,c) in enumerate(zip(b["coco"][:100],b["candidates"][:100],strict=True),1)]
        with redirect_stdout(StringIO()):
            matches,rasters,_=match.freeze_coco(image,anns[image_id],pool)
        by_rank={rank:ann for ann,rank in matches.items()}
        by_ann={r["id"]:r for r in anns[image_id]}
        scope["stock_matches"]+=len(matches)
        for i,(row,cand) in enumerate(zip(b["coco"][:100],b["candidates"][:100],strict=True),1):
            raw=row["segmentation"]
            act=edits.get(str(i))
            for mode in ap:
                rle=apply(raw,act[mode]) if act and mode in ("geometry","probability") else raw
                ap[mode].append(dict(row,segmentation=rle))
            identity=by_rank.get(i)
            valid=identity in truth and identity not in neutral
            if act:
                scope["edited_valid" if valid else "edited_unmatched" if identity is None else "edited_neutral"]+=1
            area=cand["mask_area"]
            if not valid or not 32<=area<1024:
                continue
            hard=codec.decode(raw).astype(bool)
            vector=truth[identity][0]
            row_values=mask_metrics(hard,cand,raw,vector,rasters[identity],act,math,geom)
            item=(b["group"],row_values)
            output["all"].append(item)
            if act:
                output["edited"].append(item)
            small=by_ann[identity]["area"]<1024
            output["gt_small" if small else "gt_non_small"].append(item)
            if small and hull.hull(vector)[0]>=2:
                output["small_slender"].append(item)
            output["predicted_32_128" if area<128 else "predicted_128_512" if area<512 else "predicted_512_1024"].append(item)
    need(all(len(v)==10900 for v in ap.values()) and scope["edited_valid"]+scope["edited_unmatched"]+
         scope["edited_neutral"]==receipt["totals"]["edited"] and output["all"],"Exact fixed match and all-candidate accounting")
    summaries={k:summarize(v) for k,v in output.items() if v}
    primary=summaries["all"]
    cc,mm=primary["comparisons"],primary["metrics"]
    gates={}
    for rival in ("geometry","permuted"):
        v=cc["probability_minus_"+rival]
        gates["beats_"+rival]=v["group_equal_delta_points"]>=.20 and v["interval95_points"][0]>0.
    gates.update(beats_raw=cc["probability_minus_raw"]["group_equal_delta_points"]>=.20,
                 preserves_strict90=mm["probability"]["strict_90"]>=mm["raw"]["strict_90"])
    result=dict(status="DFINE_FIT109_BOUNDARY_PROBABILITY_ACTUAL",FIT=True,DEV=False,TEST=False,GT_opened=True,
                AP=True,AP_role="descriptive_full_candidate_secondary",model_forwards=0,optimizer_updates=0,
                protocol_sha256=PROTOCOL,prelabel_sha256=a.prelabel_sha,prelabel_score_sha256=pregt,
                counts=receipt["totals"],match_scope=dict(scope),cohorts=summaries,gates=gates,advance=all(gates.values()),
                descriptive_coco={m:stock_ap(subset,rows) for m,rows in ap.items()},original_annotations=len(subset["annotations"]))
    digest=seal(a.output/"RESULT_SAFE.json",encoded(result))
    print(json.dumps(dict(status=result["status"],result_sha256=digest,advance=result["advance"],gates=gates),
                     sort_keys=True),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="mode",required=True)
    common=("protocol","capture","accepted","math","output")
    for mode,extra in (("prepare",()),("score",("prelabel","prelabel-wait","fit-gt","code","shapely-site"))):
        q=sub.add_parser(mode)
        for field in common+extra:
            q.add_argument("--"+field,type=Path,required=True)
        if mode=="score":
            q.add_argument("--prelabel-sha",required=True)
            q.add_argument("--prelabel-wait-sha",required=True)
    a=p.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="-1","Exact CPU-only stage")
    exact(a.protocol,PROTOCOL)
    math=load(a.math,MATH,"_rank_math")
    core,blocks=read_core(a)
    if a.mode=="prepare":
        prepare(a,core,blocks,math)
    else:
        score(a,blocks,math)


if __name__=="__main__":
    main()