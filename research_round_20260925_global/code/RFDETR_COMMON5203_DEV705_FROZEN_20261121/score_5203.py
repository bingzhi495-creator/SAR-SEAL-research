"""Post-capture CPU readback and DEV705 scoring for the frozen common-5203 replication."""
import argparse
from collections import defaultdict
from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
import json
import os
from pathlib import Path
import re
import sys
from types import ModuleType

import numpy as np
from pycocotools import mask as codec


def optional_directory(variable):
    value=os.environ.get(variable)
    return Path(value).expanduser().resolve() if value else None


RELEASE = optional_directory("SAR_SEAL_RELEASE_ROOT") or Path(__file__).resolve().parents[2]
CODE = RELEASE / "code"
DEV = optional_directory("SAR_SEAL_DEV705_INPUTS")
SHAPELY = optional_directory("SAR_SEAL_SHAPELY_SITE")
SEEDS = ("5217", "5218", "5219")
ARMS = ("SF", "SU=F", "SU", "LW=S")
CONTRASTS = (("SU=F", "SF"), ("SU", "SF"), ("SU", "N"), ("SU", "LW=S"), ("SU", "SU=F"))
OLD_ENDPOINT = "011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11"
OLD_STOCK = "da55f2d3f9bbf08b26d5fae76606b2496a5b482c5fb61f4fab53c23172e5c06e"
HEADS = "476b0f3fa748d1d3fd2393bdeb1af2d83e1532995d58b12625d57222e3f2638a"
PRIMARY = "RFDETR_COMMON5203_DEV705_FROZEN_READOUT_INDEPENDENT_ACCEPTED_V1"
EARLIER_SCORE = "c0be05b24ca1bb23283ac19043b2ad96bbf2db39bc0ef525e1ea41a3dc9316e9"
LOCK = "ed3238e963cc89a3f8a284bcd3e004c5e34d16953214c9a8081f6efa98372fd5"
PUB = {
    "reader": ("HRSID_OFFICIAL_FROZEN_REFINEMENT_CAPTURE_READBACK_20261121/readback.py","5503e284690b6e0d36c09b6e427ea0cd9ca6acbc26e0ce1a20e8e84e3f58cc30"),
    "stream": ("HRSID_OFFICIAL_FROZEN_REFINEMENT_SCORE_20261121/score_refinement.py","f047034705e6048230764d7f5712419546a387cd1d3c3a92ec00c57561dea385"),
    "parser": ("HRSID_OFFICIAL_BASELINE_SCORE_20261121/score_native.py","c3322233db5234bf89510ea370c88a255dc4c995d80fc483d9903ef43ef01628"),
    "dev": ("HRSID_COCO_E24_DEV705_NATIVE_EVAL_20261109/hrsid_native_dev705.py","1ce0eaa2106fe52452129fad534d0fc11bb8ad5fc4611f70858dbdf6f87a5ec3"),
    "metrics": ("RFDETR_MATCHED_POINT_FORMAL_20261120/score_dev.py","271554ea01f5e5e24e48ff2c53c3229755615aa514624d7d89e5c9ec559fb79c"),
    "matcher": ("HRSID_FIT_HULL_CDF_20261112/stock_coco_match.py","12fd1684b262a3c9b6732b90a0c1e624034c523a7d130919a079e17728f80075"),
    "geometry": ("MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py","7cad8b51a9230101e2233db8455f84c28163abfb2ee8ddd8cbe8718a2e719bd0"),
}


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def valid_sha(s):
    return isinstance(s,str) and re.fullmatch("[a-f0-9]{64}",s) is not None


def pinned(path, h):
    data=path.read_bytes()
    need(valid_sha(h) and sha256(data).hexdigest()==h, "Invalid pin: "+path.name)
    return data


def load(name):
    rel,h=PUB[name]
    path=CODE/rel
    obj=ModuleType("_replicate_"+name)
    obj.__file__=str(path)
    exec(compile(pinned(path,h),str(path),"exec"),obj.__dict__)
    return obj


def write(path, value):
    data=(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
    with path.open("xb") as out:
        out.write(data)
        out.flush()
        os.fsync(out.fileno())
    return sha256(data).hexdigest()


def validate_capture(terminal, training, native, refine):
    need(terminal["status"]=="RFDETR_COMMON5203_DEV705_PRELABEL_TWO_STAGES_CLOSED_V1" and
         all(type(terminal[k]) is int and terminal[k]==0 for k in ("original_exit","native_exit","refinement_exit")),
         "Both original subprocesses must have exited successfully")
    need(terminal["DEV"] is True and all(terminal[k] is False for k in ("FIT","TEST","GT_opened","AP")) and
         terminal["images"]==705 and terminal["fit_accepted_sha256"]==HEADS, "Frozen DEV-only capture")
    endpoint=terminal["endpoint_sha256"]
    need(training["status"]=="COMMON5203_SINGLE_CENTER_E24_ACCEPTED" and training["FIT"] is True and
         training["DEV"] is training["TEST"] is training["AP"] is False and valid_sha(endpoint) and endpoint!=OLD_ENDPOINT
         and training["arm"]["checkpoint_sha256"]==endpoint, "Newly trained, accepted common 5203 E24 checkpoint")
    need(native["status"]=="RFDETR_COMMON5203_DEV705_NATIVE_CAPTURED_V1" and
         refine["status"]=="RFDETR_COMMON5203_DEV705_FROZEN_CAPTURED_V1" and native["images"]==705 and
         native["mapped_rows"]==terminal["mapped_rows"] and 0<=native["mapped_rows"]<=70500 and
         valid_sha(native["predictions_sha256"]) and native["predictions_sha256"]!=OLD_STOCK and
         valid_sha(native["raw_hashes_sha256"]) and native["raw_hashes_sha256"]==refine["raw_hashes_sha256"],
         "Own 5203 raw/native outputs and independent raw tensor agreement")


def validate_raw_roster(raw, ids):
    lines=raw.splitlines(keepends=True)
    keys={"image_id","pred_logits_sha256","pred_boxes_sha256","pred_masks_sha256"}
    need(len(lines)==len(ids)==len(set(ids)), "Original DEV-only ordered image roster")
    for line, image_id in zip(lines,ids,strict=True):
        row=json.loads(line)
        need(line.endswith(b"\n") and set(row)==keys and type(row["image_id"]) is int and row["image_id"]==image_id
             and all(valid_sha(row[k]) for k in keys-{"image_id"}), "Exact per-image Q164 tensor identity hashes")


def readback(a):
    need(DEV is not None,"Set SAR_SEAL_DEV705_INPUTS to the closed DEV705 input directory")
    need(not a.output.exists(), "Fresh independent destination")
    train=json.loads(pinned(a.training,a.training_sha256))
    native_dir=a.capture/"native"
    refine_dir=a.capture/"refine"
    receipt=json.loads(pinned(a.capture/"TERMINAL_SAFE.json",a.terminal_sha256))
    need(receipt["training_acceptance_sha256"]==a.training_sha256 and receipt["admission_sha256"]==a.admission_sha256,
         "Bind the independently supplied training and admission digests")
    admission=json.loads(pinned(a.capture/"ADMISSION.json",a.admission_sha256))
    native=json.loads(pinned(native_dir/"RESULT_SAFE.json",receipt["native_result_sha256"]))
    refine=json.loads(pinned(refine_dir/"RESULT_SAFE.json",receipt["refinement_result_sha256"]))
    validate_capture(receipt,train,native,refine)
    dev=load("dev")
    dev.DEV=DEV
    manifest=dev.qualified_manifest()
    need(admission["endpoint_sha256"]==receipt["endpoint_sha256"] and
         admission["training_acceptance_sha256"]==a.training_sha256 and admission["fit_accepted_sha256"]==HEADS and
         admission["manifest_sha256"]==dev.FILES["DEV_IMAGE_ONLY_MANIFEST_PRIVATE.json"] and admission["images"]==705,
         "Accepted detector and three frozen-head identities in original admission")
    raw=pinned(native_dir/"RAW_HASHES_PRIVATE.jsonl",native["raw_hashes_sha256"])
    need(pinned(refine_dir/"RAW_HASHES_PRIVATE.jsonl",refine["raw_hashes_sha256"])==raw and len(raw.splitlines())==705,
         "Independent raw Q164 logits / boxes / masks stream parity")
    validate_raw_roster(raw,[im["image_id"] for im in manifest])
    reader=load("reader")
    original=native_dir/"DEV_NATIVE_PREDICTIONS_PRIVATE.jsonl"
    a.output.mkdir()
    streams,names,tally,features=reader.replay(refine_dir/"IMAGE_ROWS_PRIVATE",original,refine,native,a.output,
                                               native_sha=native["predictions_sha256"])
    result=dict(status=PRIMARY,original_exit=0,detector_seed=5203,DEV=True,FIT=False,TEST=False,GT_opened=False,AP=False,
                optimizer_updates=0,images=705,mapped_rows=tally["mapped"],native_predictions_sha256=native["predictions_sha256"],
                native_bytes=native["prediction_bytes"],endpoint_sha256=receipt["endpoint_sha256"], fit_accepted_sha256=HEADS,
                training_acceptance_sha256=a.training_sha256,admission_sha256=a.admission_sha256,terminal_sha256=a.terminal_sha256,
                exact_candidate_identity=True,exact_native_per_image=True,exact_per_candidate_budgets=True,
                actual_XOR_area_RLE_verified=True,point_decisions_replayed=True,raw_forward_parity_sha256=native["raw_hashes_sha256"],
                descriptor_identity_ledger_sha256=features,streams=streams,files=names,
                reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    return write(a.output/"ACCEPTED_SAFE.json",result)


def validate_readback(item, endpoint, training_sha):
    need(item["status"]==PRIMARY and item["original_exit"]==0 and type(item["original_exit"]) is int and
         item["detector_seed"]==5203 and item["fit_accepted_sha256"]==HEADS and
         item["training_acceptance_sha256"]==training_sha and item["endpoint_sha256"]==endpoint!=OLD_ENDPOINT
         and valid_sha(endpoint) and item["DEV"] is True and all(item[k] is False for k in ("FIT","TEST","GT_opened","AP"))
         and item["optimizer_updates"]==0 and item["images"]==705 and 0<=item["mapped_rows"]<=70500 and
         valid_sha(item["native_predictions_sha256"]) and item["native_predictions_sha256"]!=OLD_STOCK
         and item["reader_sha256"]==sha256(Path(__file__).read_bytes()).hexdigest(), "Accepted 5203 readback")
    need(all(item[k] is True for k in ("exact_candidate_identity","exact_native_per_image","exact_per_candidate_budgets",
                                       "actual_XOR_area_RLE_verified","point_decisions_replayed")) and
         all(valid_sha(item[k]) for k in ("terminal_sha256","admission_sha256","raw_forward_parity_sha256")) and
         set(item["streams"])==set(SEEDS) and all(set(item["streams"][s])==set(ARMS) for s in SEEDS),
         "Complete native and treatment closure")
    for s in SEEDS:
        for arm in ARMS:
            ref=item["streams"][s][arm]
            need(item["files"][s][arm]==s+"_"+arm+".jsonl" and valid_sha(ref["sha256"]) and
                 type(ref["bytes"]) is int and ref["bytes"]>0, "Closed per-seed standard streams")


def native_ranks(path, expected):
    digest=sha256()
    ranks={}
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            block=json.loads(line)
            im=block["image_id"]
            need(im not in ranks,"Unique 5203 native rank roster")
            ranks[im]=block["native_ranks"]
    need(digest.hexdigest()==expected,"Exact original ranks on native geometry reread")
    return ranks


def fixed_matches(gt, manifest, native_rows, ranks, matcher, poly):
    truths,neutral,_=poly.geometry.polygon_support(gt,poly.vector)
    anns=defaultdict(list)
    for r in gt["annotations"]:
        anns[r["image_id"]].append(r)
    records=[]
    total=0
    for image, m in zip(gt["images"],manifest,strict=True):
        im=image["id"]
        rows=native_rows[im]
        pool=[dict(rank=rank,score=r["score"],rle=r["segmentation"],area=int(codec.area(r["segmentation"])))
              for rank,r in zip(ranks[im],rows,strict=True)]
        positions={rank:j for j,rank in enumerate(ranks[im])}
        if not pool or not anns[im]:
            continue
        with redirect_stdout(StringIO()):
            pairs,_,_=matcher.freeze_coco(image,anns[im],pool)
        total+=len(pairs)
        for aid, rank in pairs.items():
            pos=positions[rank]
            if aid not in neutral and aid in truths and 32<=pool[pos]["area"]<1024:
                records.append((im,pos,aid,Path(m["file_name"]).stem.split("_")[0],truths[aid][0]))
    need(bool(records),"Original 5203 fixed-match geometry population")
    return records,total


def continuous(rows,records,poly):
    values=[]
    for im,pos,_,_,truth in records:
        rle=rows[im][pos]["segmentation"]
        rect=poly.geometry.boxes_from_counts(poly.shift.runs(rle["counts"]),*rle["size"])
        area=int(codec.area(rle))
        need(np.prod(rect[:,2:]-rect[:,:2],axis=1).sum()==area,"Exact foreground pixel area")
        overlap=poly.geometry.overlap_area(rect,truth) if area else 0.
        values.append(overlap/(area+truth.area-overlap))
    return np.asarray(values)


def collapse(values, groups):
    return np.asarray([np.mean(values[ix]) for ix in groups])


def evaluate_contrasts(views, values, groups, past):
    draws=np.random.Generator(np.random.PCG64(5217)).integers(len(groups),size=(4096,len(groups)))
    result={}
    for left,right in CONTRASTS:
        per_seed={}
        group_delta=[]
        old_ious=[]
        for s in SEEDS:
            g=collapse(values[s][left]-values[s][right],groups)*100
            group_delta.append(g)
            old=past["geometry_summary"]["per_seed"][s]["arms"]
            old_iou=old[left]["group_equal_iou_points"]-old[right]["group_equal_iou_points"]
            old_ious.append(old_iou)
            per_seed[s]={"5203_group_equal_iou_points":float(np.mean(g)),"5202_group_equal_iou_points":old_iou}
            for name in ("primary","sensitivity"):
                new=views[s]
                previous=past["coco"][s]
                deltas={}
                for metric in ("AP","AP50","AP75","APs","AP90","AR100"):
                    v=[new[left][name][metric],new[right][name][metric],previous[left][name][metric],previous[right][name][metric]]
                    if min(v)<0:
                        deltas[metric]=None
                    else:
                        d0,d1=100*(v[2]-v[3]),100*(v[0]-v[1])
                        deltas[metric]={"5202":d0,"5203":d1,"5203_minus_5202":d1-d0}
                per_seed[s][name]=deltas
        change=np.mean(group_delta,axis=0)
        new_iou=float(change.mean())
        old_iou=float(np.mean(old_ious))
        averages={}
        for name in ("primary","sensitivity"):
            averages[name]={}
            for m in per_seed[SEEDS[0]][name]:
                metrics=[per_seed[s][name][m] for s in SEEDS]
                averages[name][m]=None if any(v is None for v in metrics) else {
                    k:float(np.mean([v[k] for v in metrics])) for k in ("5202","5203","5203_minus_5202")}
        result[left+" minus "+right]=dict(per_head_seed=per_seed,mean_COCO_points=averages,
                mean_fixed_iou_points={"5202":old_iou,"5203":new_iou,"5203_minus_5202":new_iou-old_iou,
                                       "5203_prefix_interval95":np.percentile(change[draws].mean(axis=1),[2.5,97.5]).tolist()})
    return result


def score(a):
    need(DEV is not None,"Set SAR_SEAL_DEV705_INPUTS to the closed DEV705 input directory")
    need(not a.output.exists() and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1","Fresh CPU score output")
    certified=json.loads(pinned(a.accepted/"ACCEPTED_SAFE.json",a.accepted_sha256))
    train=json.loads(pinned(a.training,a.training_sha256))
    need(train["status"]=="COMMON5203_SINGLE_CENTER_E24_ACCEPTED","Qualified actual training endpoint")
    validate_readback(certified, train["arm"]["checkpoint_sha256"], a.training_sha256)
    dev,scorer,stream,parser=load("dev"),load("metrics"),load("stream"),load("parser")
    need(parser.runtime()==dict(python="3.10.13",numpy="1.26.4",pycocotools="2.0.11"),"Original RF DEV CPU runtime")
    dev.DEV,dev.OFFICIAL=DEV,(CODE/"DIFFSAR_NATIVE_CPU_COCO_ADAPTER_DRAFT_20261107_01/worker.py")
    manifest=dev.qualified_manifest()
    ids=[m["image_id"] for m in manifest]
    native_ref=dict(bytes=certified["native_bytes"],sha256=certified["native_predictions_sha256"])
    native,_,signatures=stream.load_arm(a.native,native_ref,ids,(800,800),certified["mapped_rows"],parser)
    gt=dev.dev_file("DEV_ORIGINAL_GT_PRIVATE.json")
    empty=dev.dev_file("DEV_EMPTY_RASTERS_PRIVATE.json")
    exclude={r["annotation_id"] for r in empty if r["kind"]=="EMPTY_COCO_RASTER"}
    need([r["id"] for r in gt["images"]]==ids and len(gt["annotations"])==1782 and len(exclude)==len(empty)==4,
         "Original DEV populations")
    if SHAPELY is not None:
        sys.path.append(str(SHAPELY.resolve(strict=True)))
    import shapely
    need(shapely.__version__=="2.0.2" and shapely.geos_version_string=="3.11.2","Original geometry convention")
    poly=load("geometry").geometry_modules(CODE)
    original_ranks=native_ranks(a.native,certified["native_predictions_sha256"])
    records,total=fixed_matches(gt,manifest,native,original_ranks,load("matcher"),poly)
    common=scorer.score_official(dev,gt,[r for im in ids for r in native[im]],exclude)
    native_iou=continuous(native,records,poly)
    grouped=defaultdict(list)
    for i,row in enumerate(records):
        grouped[row[3]].append(i)
    groups=[grouped[g] for g in sorted(grouped)]
    views={s:{"N":common} for s in SEEDS}
    values={s:{"N":native_iou} for s in SEEDS}
    del native
    for s in SEEDS:
        for arm in ARMS:
            rows,_,_=stream.load_arm(a.accepted/certified["files"][s][arm],certified["streams"][s][arm],ids,
                                  (800,800),certified["mapped_rows"],parser,signatures)
            views[s][arm]=scorer.score_official(dev,gt,[r for im in ids for r in rows[im]],exclude)
            values[s][arm]=continuous(rows,records,poly)
            del rows
            print(json.dumps(dict(event="DEV705_5203_ARM_SCORED",seed=s,arm=arm)),flush=True)
    previous=json.loads(pinned(RELEASE/"results/RFDETR_CENTER5202_DEV705_MATCHED_SCORE_PUBLIC_SAFE_20261120.json",EARLIER_SCORE))
    contrasts=evaluate_contrasts(views,values,groups,previous)
    witness={s:{arm:dict(group_equal_iou_points=float(100*collapse(v,groups).mean()),
                           strict90_retained=int(np.sum((native_iou>.90)&(v>.90))),
                           strict90_lost=int(np.sum((native_iou>.90)&(v<=.90))),
                           strict90_gained=int(np.sum((native_iou<=.90)&(v>.90)))) for arm,v in values[s].items()} for s in SEEDS}
    result=dict(status="RFDETR_COMMON5203_DEV705_FROZEN_SCORE_COMPLETE_V1",DEV=True,FIT=False,TEST=False,
                optimizer_updates=0,model_forwards=0,training_seed=5203,images=705,annotations=1782,empty_sensitivity=1778,
                mapped_rows=certified["mapped_rows"],stock_matches=total,eligible_valid_matches=len(records),
                source_prefixes=len(groups),score_units="COCO fractions; paired effects in percentage points",
                per_seed_COCO=views,fixed_geometry=witness,paired_contrasts=contrasts,primary="SU=F minus SF",
                endpoint_sha256=certified["endpoint_sha256"],training_acceptance_sha256=a.training_sha256,
                readback_sha256=a.accepted_sha256,native_sha256=certified["native_predictions_sha256"],
                frozen_fit_sha256=HEADS,previous_5202_score_sha256=EARLIER_SCORE,dev_gt_sha256=dev.FILES["DEV_ORIGINAL_GT_PRIVATE.json"],
                prelock_sha256=LOCK,source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    return write(a.output,result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    subs=p.add_subparsers(dest="mode",required=True)
    for verb in ("readback","score"):
        options=subs.add_parser(verb)
        for name in (("capture","output","training") if verb=="readback" else ("native","accepted","output","training")):
            options.add_argument("--"+name,type=Path,required=True)
        for name in (("terminal-sha256","training-sha256","admission-sha256") if verb=="readback" else
                     ("accepted-sha256","training-sha256")):
            options.add_argument("--"+name,required=True)
    a=p.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode,"Isolated CPU runtime required")
    pinned(CODE/"RFDETR_COMMON5203_E24_REPLICATION_20261121/PRELOCK.json",LOCK)
    digest=readback(a) if a.mode=="readback" else score(a)
    print(json.dumps(dict(phase=a.mode,output_sha256=digest)),flush=True)


if __name__=="__main__":
    main()
