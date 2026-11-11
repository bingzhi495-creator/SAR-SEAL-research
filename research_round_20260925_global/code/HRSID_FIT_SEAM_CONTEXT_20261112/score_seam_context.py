"""Score fixed paired seam/interior HRSID FIT hulls on closed stock predictions."""
from argparse import ArgumentParser
from collections import defaultdict
from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
import json
import os
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter
from types import ModuleType

PROTOCOL_SHA="6b0955af80a701175f2a5b1ebdb4f9d82d239df2e18586e5037dc414c850eba2"
ROSTER_SHA="e8568f25f3a1c04dc8ec5fc99546d502c69c25ed2008ec2a28f484047f20cd00"
FIT_SHA="e95cf26cc60a2c6fb651e907541380a256f2df7b1e153244a6c0f803b0b19f8f"
MATCH_SHA="8741bebb448120e37e0c10b58467a83d724f2294d20fddbc93720301285084aa"
GEOMETRY_SHA="7cad8b51a9230101e2233db8455f84c28163abfb2ee8ddd8cbe8718a2e719bd0"


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def exact(path, digest, limit=64*2**20):
    need(path.resolve(strict=True)==path and 0<path.stat().st_size<=limit,"Exact bounded path")
    data=path.read_bytes()
    need(sha256(data).hexdigest()==digest,"Pinned source: "+path.name)
    return data


def module(path, digest, name):
    m=ModuleType(name)
    m.__file__=str(path)
    sys.modules[name]=m
    exec(compile(exact(path,digest,2**20),str(path),"exec"),m.__dict__)
    return m


def summarize(rows):
    both=[r for r in rows if r["near"]>0 and r["far"]>0]
    near=mean(r["near"] for r in rows)*100
    far=mean(r["far"] for r in rows)*100
    nf_abs=mean(abs(r["far"]-r["near"]) for r in rows)*100
    interior_abs=mean(abs(r["inside_a"]-r["inside_b"]) for r in rows)*100
    strict={k:sum(r[k]>=.75 for r in rows) for k in ("near","far","inside_a","inside_b")}
    return dict(groups=len(rows),near_iou_points=near,far_iou_points=far,far_minus_near_points=far-near,
                near_far_absolute_change_points=nf_abs,interior_absolute_change_points=interior_abs,
                disagreement_excess_points=nf_abs-interior_abs,strict_hits=strict,
                strict_far_minus_near=strict["far"]-strict["near"],
                far_gains=sum(r["far"]>r["near"] for r in rows),
                near_gains=sum(r["near"]>r["far"] for r in rows),
                matched_both=len(both),far_only_matches=sum(r["far"]>0 and r["near"]==0 for r in rows),
                near_only_matches=sum(r["near"]>0 and r["far"]==0 for r in rows),
                neither_matches=sum(r["near"]==0 and r["far"]==0 for r in rows),
                interior_matched_both=sum(r["inside_a"]>0 and r["inside_b"]>0 for r in rows),
                paired_matched_only_delta_points=100*mean(r["far"]-r["near"] for r in both) if both else None)


def geometry_score(rle,target,geom):
    boxes=geom.geometry.boxes_from_counts(geom.shift.runs(rle["counts"]),*rle["size"])
    inter=geom.geometry.overlap_area(boxes,target)
    from pycocotools import mask as codec
    area=float(codec.area(rle))
    need(-1e-8<=inter<=min(area,target.area)+1e-8,"Exact finite overlap")
    inter=max(0.,min(inter,area,target.area))
    return inter/(area+target.area-inter)


def read_capture(args):
    report=json.loads(exact(args.capture/"RESULT_SAFE.json",args.capture_sha))
    independent=json.loads(exact(args.witness,args.witness_sha))
    need(report["FIT"] is True and report["DEV"] is report["TEST"] is report["GT_opened"] is report["AP"] is False
         and report["protocol_sha256"]==PROTOCOL_SHA and report["images"]==report["model_forwards"]==144
         and report["source_groups"]==40 and report["optimizer_updates"]==0,"Exact closed capture")
    need(independent["status"]=="HRSID_FIT_SEAM_CONTEXT_CAPTURE_INDEPENDENT_ACCEPTED"
         and independent["result_sha256"]==args.capture_sha and independent["predictions_verified"] is True
         and independent["FIT"] is True and independent["DEV"] is independent["TEST"] is independent["GT_opened"] is False,
         "Exact independent no-label acceptance")
    p=json.loads(exact(args.selected,ROSTER_SHA))
    records=[r for r in p["records"] if r["matched_interior"] is not None]
    blocks=[json.loads(line) for line in exact(args.capture/"CANDIDATES_PRIVATE.jsonl",report["prediction_sha256"]).splitlines()]
    names={b["basename"]:b for b in blocks}
    need(len(records)==len({r["group"] for r in records})==40 and len(blocks)==len(names)==144
         and [b["ordinal"] for b in blocks]==list(range(1,145)),"Exact locked 40-source/144-view roster")
    want={c["native_basename"] for r in records for k in ("near_far","matched_interior") for c in r[k]["pair"]}
    need(want==set(names),"Identical target and forward images")
    for r in records:
        for k in ("near_far","matched_interior"):
            for c in r[k]["pair"]:
                b=names[c["native_basename"]]
                need(b["group"]==r["group"] and b["rgb_sha256"]==c["decoded_rgb_sha256"],"Exact source and RGB join")
    return names,records


def run(args):
    started=perf_counter()
    exact(args.protocol,PROTOCOL_SHA,65536)
    predictions,records=read_capture(args)
    matching=module(args.match_source,MATCH_SHA,"_seam_stock_match")
    admission=module(args.code/"MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py",GEOMETRY_SHA,"_seam_geometry")
    sys.path.append(str(args.shapely_site.resolve(strict=True)))
    import shapely
    need(shapely.__version__=="2.0.2" and shapely.geos_version_string=="3.11.2","Exact geometry version")
    geom=admission.geometry_modules(args.code)
    receipt=dict(status="HRSID_FIT_SEAM_CONTEXT_PRELABEL_SCOPE",FIT=True,DEV=False,TEST=False,GT_opened=False,
                 capture_sha256=args.capture_sha,independent_sha256=args.witness_sha,protocol_sha256=PROTOCOL_SHA,
                 scored_groups=40,scored_target_pairs=80)
    args.output.mkdir()
    with (args.output/"PRELABEL_SAFE.json").open("xb") as f:
        f.write((json.dumps(receipt,sort_keys=True)+"\n").encode())
    labels=json.loads(exact(args.fit_index,FIT_SHA))
    need(len(labels["images"])==2937 and len(labels["annotations"])==9260,"Exact native FIT label population")
    images={im["file_name"]:im for im in labels["images"] if im["file_name"] in predictions}
    ids={im["id"] for im in images.values()}
    need(set(images)==set(predictions) and len(ids)==144,"Exact image/label join")
    annotations=defaultdict(list)
    for a in labels["annotations"]:
        if a["image_id"] in ids:
            annotations[a["image_id"]].append(a)
    subset=dict(images=list(images.values()),annotations=[a for aa in annotations.values() for a in aa],
                categories=labels["categories"])
    targets,neutral,_=geom.geometry.polygon_support(subset,geom.vector)
    mappings={}
    with redirect_stdout(StringIO()):
        for name,im in images.items():
            pool=predictions[name]["candidates"]
            rows={a["id"]:a for a in annotations[im["id"]]}
            match,_,_=matching.freeze_coco(im,list(rows.values()),pool)
            mappings[name]=({p["rank"]:p for p in pool},match,rows)

    def pair_values(w):
        score=[]
        for im,truth_id in zip(w["pair"],w["annotation_ids"],strict=True):
            name=im["native_basename"]
            pool,matched,rows=mappings[name]
            need(truth_id in rows and truth_id not in neutral and truth_id in targets,"Exact valid selected GT identity")
            rank=matched.get(truth_id)
            score.append(0. if rank is None else geometry_score(pool[rank]["rle"],targets[truth_id][0],geom))
        return score

    out_rows=[]
    for record in records:
        seam,interior=record["near_far"],record["matched_interior"]
        offset=tuple(abs(i) for i in seam["shift_yx"])
        need(offset==tuple(abs(i) for i in interior["shift_yx"]) and min(seam["margins"])<=16
             and max(seam["margins"])>=64 and min(interior["margins"])>=64,"Original matched offset and margins")
        ab=pair_values(seam)
        near_index=0 if seam["margins"][0]<seam["margins"][1] else 1
        c,d=pair_values(interior)
        out_rows.append(dict(offset=offset,near=ab[near_index],far=ab[1-near_index],inside_a=c,inside_b=d))
    summary=summarize(out_rows)
    offsets=sorted({r["offset"] for r in out_rows})
    strata={",".join(map(str,s)):summarize([r for r in out_rows if r["offset"]==s]) for s in offsets}
    lead=(summary["far_minus_near_points"]>=1.5 and summary["strict_far_minus_near"]>=5 and
          summary["disagreement_excess_points"]>=.5)
    result=dict(status="HRSID_FIT_SEAM_CONTEXT_SCORE_ACTUAL",FIT=True,DEV=False,TEST=False,GT_opened=True,AP=False,
                model_forwards=0,optimizer_updates=0,mask_edits=0,target_pairs=80,
                gate="SEAM_CONTEXT_LEAD" if lead else "NO_SEAM_CONTEXT_LEAD",summary=summary,strata=strata,
                capture_sha256=args.capture_sha,witness_sha256=args.witness_sha,protocol_sha256=PROTOCOL_SHA,
                selected_sha256=ROSTER_SHA,fit_index_sha256=FIT_SHA,worker_seconds=perf_counter()-started)
    data=(json.dumps(result,sort_keys=True,allow_nan=False,indent=2)+"\n").encode()
    with (args.output/"RESULT_SAFE.json").open("xb") as f:
        f.write(data)
    print(json.dumps(dict(status=result["status"],gate=result["gate"],sha256=sha256(data).hexdigest())))


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    for name in ("capture","witness","protocol","selected","fit-index","match-source","code","shapely-site","output"):
        p.add_argument("--"+name,type=Path,required=True)
    for name in ("capture-sha","witness-sha"):
        p.add_argument("--"+name,required=True)
    args=p.parse_args()
    need(sys.flags.isolated and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1","Isolated CPU only")
    def offline(event,_args):
        if event.startswith("socket."):
            raise PermissionError("No network")
    sys.addaudithook(offline)
    run(args)
