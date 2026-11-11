"""Evaluate closed HRSID FIT association witnesses without editing or AP scoring."""

from collections import Counter
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.ndimage import binary_dilation


SEEDS=("5201","5202","5203")
THRESHOLDS=("90","85")
METHOD_KEYS=("triplets","all_known_same","known_different","unresolved","unique_known_same_GT")
SUPPRESSION_KEYS=("suppressed","same_known","distinct_known","unresolved","distinct_known_near_2px",
                  "stock_distinct_matched","stock_distinct_near_2px","distinct_known_stock_distinct")


def need(ok, message):
    if not ok:
        raise ValueError(message)


def gt_rle(segmentation, height, width):
    if isinstance(segmentation,list):
        if not segmentation:
            return mask.encode(np.zeros((height,width),dtype=np.uint8,order="F"))
        return mask.merge(mask.frPyObjects(segmentation,height,width))
    if isinstance(segmentation["counts"],list):
        return mask.frPyObjects(segmentation,height,width)
    return segmentation


def assign_by_margin(pool, eligible, gt):
    known={}
    if not pool or not eligible:
        return known
    iou=mask.iou([r["rle"] for r in pool],[gt[i]["rle"] for i in eligible],[0]*len(eligible))
    for candidate, values in zip(pool,iou,strict=True):
        ordered=np.argsort(-values,kind="stable")
        best=float(values[ordered[0]])
        second=float(values[ordered[1]]) if len(ordered)>1 else 0.
        if best>=.50 and best-second>=.10-1e-12:
            known[candidate["rank"]]=eligible[ordered[0]]
    return known


def stock_co50_matches(pool, gt, eligible, shape):
    """Use original stable score ordering and full-area native COCO .50 matching, without AP."""
    if not pool or not gt:
        return {}
    height,width=shape
    coco=COCO()
    coco.dataset={
        "info":{},"images":[dict(id=1,height=height,width=width)],"categories":[dict(id=1,name="ship")],
        "annotations":[dict(id=i,image_id=1,category_id=1,segmentation=r["rle"],area=r["area"],
                            bbox=mask.toBbox(r["rle"]).tolist(),iscrowd=r["iscrowd"],ignore=r["ignore"])
                       for i,r in gt.items()],
    }
    pred=[dict(image_id=1,category_id=1,score=r["score"],segmentation=r["rle"]) for r in pool]
    with redirect_stdout(StringIO()):
        coco.createIndex()
        evaluator=COCOeval(coco,coco.loadRes(pred),"segm")
        evaluator.params.imgIds=[1]
        evaluator.params.catIds=[1]
        evaluator.params.iouThrs=np.asarray([.5])
        evaluator.params.areaRng=[evaluator.params.areaRng[0]]
        evaluator.params.areaRngLbl=["all"]
        evaluator.params.maxDets=[100]
        evaluator.evaluate()
    record,=evaluator.evalImgs
    admitted=set(eligible)
    result={}
    for index,value,ignored in zip(record["dtIds"],record["dtMatches"][0],record["dtIgnore"][0],strict=True):
        if value and not ignored and int(value) in admitted:
            result[pool[index-1]["rank"]]=int(value)
    return result


def risk_for_image(pools, structural, ground_truth, shape):
    height,width=shape
    gt={}
    for row in ground_truth:
        binary=gt_rle(row["segmentation"],height,width)
        gt[row["id"]]=dict(rle=binary,iscrowd=int(row.get("iscrowd",0)),ignore=int(row.get("ignore",0)),
                           area=row["area"])
    eligible=[i for i,row in gt.items() if not row["iscrowd"] and not row["ignore"] and mask.area(row["rle"])>0]
    known={seed:assign_by_margin(pools[seed],eligible,gt) for seed in SEEDS}
    stock={seed:stock_co50_matches(pools[seed],gt,eligible,shape) for seed in SEEDS}
    dilated={}
    decoded={}
    neighbor={}
    def near(left,right):
        key=tuple(sorted((left,right)))
        if key not in neighbor:
            a,b=key
            if a not in dilated:
                if a not in decoded:
                    decoded[a]=mask.decode(gt[a]["rle"]).astype(bool)
                dilated[a]=binary_dilation(decoded[a],structure=np.ones((5,5),bool))
            if b not in decoded:
                decoded[b]=mask.decode(gt[b]["rle"]).astype(bool)
            neighbor[key]=bool((dilated[a]&decoded[b]).any())
        return neighbor[key]

    suppression={}
    for threshold in THRESHOLDS:
        by_seed={}
        for seed in SEEDS:
            counts=Counter(dict.fromkeys(SUPPRESSION_KEYS,0))
            for native,representative in structural["reps"][threshold][seed].items():
                native=int(native)
                if native==representative:
                    continue
                counts["suppressed"]+=1
                a,b=known[seed].get(native),known[seed].get(representative)
                distinct=a is not None and b is not None and a!=b
                if a is None or b is None:
                    counts["unresolved"]+=1
                elif distinct:
                    counts["distinct_known"]+=1
                    counts["distinct_known_near_2px"]+=near(a,b)
                else:
                    counts["same_known"]+=1
                sa,sb=stock[seed].get(native),stock[seed].get(representative)
                if sa is not None and sb is not None and sa!=sb:
                    counts["stock_distinct_matched"]+=1
                    counts["stock_distinct_near_2px"]+=near(sa,sb)
                    counts["distinct_known_stock_distinct"]+=distinct
            by_seed[seed]=dict(counts)
        suppression[threshold]=by_seed

    methods={}
    for name,method in structural["methods"].items():
        counts=Counter(dict.fromkeys(METHOD_KEYS,0))
        same_truth_ids=set()
        for anchor,(donor1,donor3) in method["pairs"].items():
            ids=(known["5201"].get(donor1),known["5202"].get(int(anchor)),known["5203"].get(donor3))
            seen={i for i in ids if i is not None}
            counts["triplets"]+=1
            if len(seen)>1:
                counts["known_different"]+=1
            elif None in ids:
                counts["unresolved"]+=1
            else:
                counts["all_known_same"]+=1
                same_truth_ids.add(ids[0])
        counts["unique_known_same_GT"]=len(same_truth_ids)
        methods[name]=dict(counts)
    return dict(suppression=suppression,methods=methods)




def unique(pairs):
    value=dict(pairs)
    need(len(value)==len(pairs),"Duplicate JSON keys are not allowed")
    return value


def invalid_constant(value):
    raise ValueError("Nonfinite JSON constant: "+value)


def document(raw):
    import json
    return json.loads(raw,object_pairs_hook=unique,parse_constant=invalid_constant)


def pinned(path,sha,limit):
    from hashlib import sha256
    import re
    need(type(sha) is str and re.fullmatch(r"[0-9a-f]{64}",sha) is not None,"Exact input SHA-256 required")
    need(path.is_absolute() and path.resolve(strict=True)==path and path.is_file()
         and 0<path.stat().st_size<=limit,"Bounded canonical closed input required")
    raw=path.read_bytes()
    need(sha256(raw).hexdigest()==sha,"Input digest mismatch: "+path.name)
    return raw


def jsonl(raw):
    need(raw.endswith(b"\n") and all(raw.splitlines()),"Complete nonblank JSONL rows required")
    return [document(line) for line in raw.splitlines()]


def write(path,value):
    from hashlib import sha256
    import json
    import os
    raw=(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256(raw).hexdigest()


def stock_scope(value):
    return value["FIT"] is True and value["DEV"] is value["TEST"] is False


def load_closed(capture,structural,structural_sha):
    m=1024**2
    result=document(pinned(structural/"RESULT_SAFE.json",structural_sha,m))
    need(result["status"]=="HRSID_FIT_ASSOC_STRUCTURE_ACTUAL" and stock_scope(result)
         and result["GT_opened"] is False and result["AP"] is False,"Exact GT-free structural result")
    report=document(pinned(capture/"RESULT_SAFE.json",result["capture_sha256"],m))
    need(stock_scope(report) and report["AP"] is False,"Exact FIT-only closed capture result")
    arms={str(row["seed"]):row for row in report["arms"]}
    need(set(arms)==set(SEEDS) and len(report["arms"])==3,"Exact native donor and anchor seeds")
    pools={seed:jsonl(pinned(capture/seed/"CANDIDATES_PRIVATE.jsonl",arm["prediction_sha256"],64*m))
           for seed,arm in arms.items()}
    witness=jsonl(pinned(structural/"WITNESS_PRIVATE.jsonl",result["witness_sha256"],64*m))
    n=result["images"]
    need(len(witness)==n and all(len(pools[seed])==arms[seed]["images"]==n for seed in SEEDS),
         "Original structural and seed streams must cover the same complete images")
    return result,pools,witness


def rank_key(key):
    need(type(key) is str and key.isascii() and key.isdigit() and str(int(key))==key and int(key)>0,
         "Exact decimal native rank key")
    return int(key)


def validated_rows(pools,witness):
    import math
    rows=[]
    names=None
    seen=set()
    for item,*carrier in zip(witness,*(pools[seed] for seed in SEEDS),strict=True):
        ordinal=item["ordinal"]
        basename=item["basename"]
        need(type(ordinal) is int and ordinal not in seen and type(basename) is str and basename,
             "Unique original structural image ordinal")
        seen.add(ordinal)
        group=carrier[0]["group"]
        need(type(group) is str and group and all((r["ordinal"],r["basename"],r["group"])==(ordinal,basename,group)
                                               for r in carrier),"Exact cross-seed image and source group join")
        by_seed={}
        ranks={}
        for seed,row in zip(SEEDS,carrier,strict=True):
            candidates=row["candidates"]
            need(type(candidates) is list and len(candidates)<=100,"Bounded native mapped candidates")
            native=[cand["rank"] for cand in candidates]
            need(all(type(rank) is int and 1<=rank<=100 for rank in native)
                 and all(a<b for a,b in zip(native,native[1:])),"Exact increasing original candidate ranks")
            scores=[cand["score"] for cand in candidates]
            need(all(type(score) in (float,int) and math.isfinite(score) and 0<=score<=1 for score in scores)
                 and all(a>=b for a,b in zip(scores,scores[1:])),"Exact stable original score order")
            for cand in candidates:
                need(type(cand["area"]) in (int,float) and cand["area"]>=0 and int(mask.area(cand["rle"]))==cand["area"],
                     "Exact original candidate RLE area")
            by_seed[seed]=candidates
            ranks[seed]=set(native)
        methods=item["methods"]
        need(type(methods) is dict and bool(methods),"Exact fixed method roster")
        if names is None:
            names=set(methods)
        need(set(methods)==names,"Structural method roster changed between images")
        for method in methods.values():
            pairs=method["pairs"]
            need(type(pairs) is dict,"Exact retained rank triplet dictionary")
            for rank,donors in pairs.items():
                need(rank_key(rank) in ranks["5202"] and type(donors) is list and len(donors)==2
                     and all(type(donor) is int and donor in ranks[seed]
                             for donor,seed in zip(donors,("5201","5203"),strict=True)),
                     "Every retained triplet must bind three original candidate ranks")
        reps=item["reps"]
        need(set(reps)==set(THRESHOLDS),"Exact representative overlap thresholds")
        for threshold in THRESHOLDS:
            need(set(reps[threshold])==set(SEEDS),"Exact three representative maps")
            for seed,representatives in reps[threshold].items():
                need(type(representatives) is dict and {rank_key(key) for key in representatives}==ranks[seed],
                     "Full original-rank representative map required")
                need(all(type(rep) is int and rep in ranks[seed] and representatives[str(rep)]==rep
                         for rep in representatives.values()),"Every original rank maps directly to a retained representative")
        rows.append((basename,group,by_seed,item))
    return rows


def aggregate(rows, index):
    from collections import defaultdict
    from pathlib import PurePosixPath
    ship=[cat["id"] for cat in index["categories"] if cat["name"].casefold()=="ship"]
    need(len(ship)==1,"Exact author ship category required")
    by_name={}
    for image in index["images"]:
        name=PurePosixPath(image["file_name"].replace(chr(92),"/")).name
        need(name not in by_name,"Author FIT label basename is not unique")
        by_name[name]=image
    annotations=defaultdict(list)
    for row in index["annotations"]:
        if row["category_id"]==ship[0]:
            annotations[row["image_id"]].append(row)
    method_totals=defaultdict(Counter)
    same_groups=defaultdict(set)
    different_groups=defaultdict(set)
    suppression={t:{seed:Counter(dict.fromkeys(SUPPRESSION_KEYS,0)) for seed in SEEDS} for t in THRESHOLDS}
    for basename,group,pools,witness in rows:
        need(basename in by_name,"Capture basename missing from author FIT index")
        image=by_name[basename]
        shape=(image["height"],image["width"])
        need(all(tuple(p["rle"]["size"])==shape for pool in pools.values() for p in pool),
             "Predictions and original FIT labels have different native grids")
        score=risk_for_image(pools,witness,annotations[image["id"]],shape)
        for name,values in score["methods"].items():
            method_totals[name].update(values)
            if values["all_known_same"]:
                same_groups[name].add(group)
            if values["known_different"]:
                different_groups[name].add(group)
        for threshold in THRESHOLDS:
            for seed in SEEDS:
                suppression[threshold][seed].update(score["suppression"][threshold][seed])
    for name,total in method_totals.items():
        total["source_groups_all_known_same"]=len(same_groups[name])
        total["source_groups_known_different"]=len(different_groups[name])
        need(total["triplets"]==total["all_known_same"]+total["known_different"]+total["unresolved"],
             "Method identity risk categories must partition retained triplets")
    for threshold in THRESHOLDS:
        for total in suppression[threshold].values():
            need(total["suppressed"]==total["same_known"]+total["distinct_known"]+total["unresolved"],
                 "Suppression identity risk categories must partition suppressed ranks")
    return dict(methods=method_totals,suppression=suppression,
                images=len(rows),source_groups=len({row[1] for row in rows}))


def main():
    from argparse import ArgumentParser
    from hashlib import sha256
    from importlib.metadata import version
    import json
    import os
    from pathlib import Path
    import sys
    parser=ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("capture","structural","fit-index","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    for name in ("structural-sha","fit-index-sha"):
        parser.add_argument("--"+name,required=True)
    args=parser.parse_args()
    need(version("pycocotools")=="2.0.11" and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1",
         "Pinned COCO CPU runtime required")
    need(args.output.is_absolute() and args.output.parent.resolve(strict=True)==args.output.parent
         and not args.output.exists(),"Fresh exact output directory")
    def offline(event,_):
        if event.startswith("socket."):
            raise PermissionError("FIT association risk is offline")
    sys.addaudithook(offline)
    structure,pools,witness=load_closed(args.capture,args.structural,args.structural_sha)
    rows=validated_rows(pools,witness)
    args.output.mkdir(exist_ok=False)
    custody=dict(status="HRSID_FIT_ASSOC_RISK_STRUCTURAL_BEFORE_LABELS_V1",structural_sha256=args.structural_sha,
                 witness_sha256=structure["witness_sha256"],capture_sha256=structure["capture_sha256"],
                 images=len(rows),source_groups=len({row[1] for row in rows}),FIT=True,DEV=False,TEST=False,
                 GT_opened=False,AP=False,source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    custody_sha=write(args.output/"STRUCTURAL_RECEIPT_SAFE.json",custody)
    print(json.dumps(dict(status=custody["status"],receipt_sha256=custody_sha),sort_keys=True),flush=True)
    index=document(pinned(args.fit_index,args.fit_index_sha,32*1024**2))
    outcome=aggregate(rows,index)
    result=dict(status="HRSID_FIT_ASSOC_RISK_ACTUAL_V1",FIT=True,DEV=False,TEST=False,GT_opened=True,AP=False,
                optimizer_updates=0,model_forward_calls=0,edits=False,pre_label_receipt_sha256=custody_sha,
                structural_sha256=args.structural_sha,fit_index_sha256=args.fit_index_sha,**outcome)
    sha=write(args.output/"RESULT_SAFE.json",result)
    print(json.dumps(dict(status=result["status"],result_sha256=sha),sort_keys=True),flush=True)


if __name__=="__main__":
    main()

