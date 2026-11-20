"""Frozen HRSID point head, native PSeg-SSDD label-free mask replay."""
import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
from time import perf_counter
from types import ModuleType

import numpy as np
from PIL import Image, ImageOps

PROTOCOL="23240cfd35212ffc9931ec914ea8af3a0db6ce7cf7bf2a639a755c5771740ed5"
OLD="6681d3903199214cd109efd9a8bf92681d838901e6c9c8780e4f19f7e16703b7"
NATIVE="475ecbdd932a2129f160e48bd1d838653b971df6cd69b9edda103c037ee3fcdc"
DEV="5be622cdec5c889ca8520aab5547f6f15f01a54568a0263423a816dca7f96626"
MANIFEST="18df007e15e3e838e607f178f14e7a26f2b37b992ee39bf091e0b74cf084fb13"
ACCEPTED="DFINE_HRSID_E24_SSDD_DEV187_PROBABILITY_INDEPENDENTLY_ACCEPTED_V1"
PRELABEL="DFINE_HRSID_TO_SSDD_DEV187_POINT_PRELABEL_COMPLETE"
VERIFIED="DFINE_HRSID_TO_SSDD_DEV187_POINT_PRELABEL_INDEPENDENT_ACCEPTED"
MODES=("raw","fixed","free_matched_count","free_natural","native_matched_count")
EDITS="EDITS_PRIVATE.jsonl"


def need(ok,why):
    if not ok:
        raise ValueError(why)


def pin(path,digest,limit=32*1024**2):
    need(path.is_absolute() and path.resolve(strict=True)==path and path.is_file() and
         0<path.stat().st_size<=limit,"Exact bounded input")
    raw=path.read_bytes()
    need(sha256(raw).hexdigest()==digest,"Exact immutable input "+path.name)
    return raw


def load(path,digest,name):
    raw=pin(path,digest)
    obj=ModuleType(name)
    obj.__file__=str(path)
    exec(compile(raw,str(path),"exec"),obj.__dict__)
    return obj


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def accepted(args,dev):
    r=json.loads(pin(args.accepted,args.accepted_sha))
    need(r["status"]==ACCEPTED and r["manifest_sha256"]==MANIFEST and r["images"]==187 and
         r["probability_files"]==187 and r["original_os_exit"]==0 and r["job_drained"] is True and
         r["ema_sha256"]=="82ca80e2cb9a0b9673a136fad3d7de3097b7327f4f76e3a2d3e7184ab07e7fd8" and
         r["exact_top100_binary_replay"]==r["probability_rows"] and r["original_author_postprocess_parity"] is True and
         r["DEV"] is True and all(r[k] is False for k in ("FIT","TEST","GT_opened","AP")) and
         type(r["exact_author_binary_replay"]) is int and r["exact_author_binary_replay"]==r["post_nms_rows"] and
         0<=r["probability_rows"]<=18700 and all(dev.digest(r[k]) for k in
             ("predictions_sha256","candidate_audit_sha256","probability_index_sha256",
              "export_result_sha256","export_terminal_sha256","original_tool_wait_sha256")),
         "Exact naturally closed independently accepted target inference")
    return r


def manifest(args):
    m=json.loads(pin(args.manifest,MANIFEST))
    need(m["TEST"] is m["AP"] is m["model_execution"] is False and len(m["items"])==187 and
         len(set(m["expected_ids"]))==187 and [i["image_id"] for i in m["items"]]==m["expected_ids"],
         "Exact target label-free manifest")
    return m["items"]


def blocks(args,dev,roster,receipt,planes):
    digests=[sha256(),sha256(),sha256()]
    counts=Counter()
    with (args.capture/dev.PREDICTIONS).open("rb") as pf,(args.capture/dev.AUDIT).open("rb") as af, (
            args.capture/"PROBABILITIES_PRIVATE.jsonl").open("rb") as ix:
        for ordinal,item in enumerate(roster,1):
            lines=[file.readline(4*1024**2) for file in (pf,af,ix)]
            need(all(v.endswith(b"\n") for v in lines),"Complete native streams")
            for digest,raw in zip(digests,lines,strict=True):
                digest.update(raw)
            pred,audit,index=map(json.loads,lines)
            h,w=item["original_hw"]
            n,_=dev.validate_rows(item["image_id"],(h,w),pred,audit)
            k=min(100,n)
            need(index["image_id"]==item["image_id"] and len(index["query_ids"])==len(set(index["query_ids"]))==n and
                 all(type(v) is int and 0<=v<300 for v in index["query_ids"]),"Native query rank identity")
            ref=index["probability"]
            need(ref["file"]==f"PROBABILITY_PRIVATE/{ordinal:04d}.npy" and ref["rows"]==k,
                 "Original post-NMS probability mapping")
            p=None
            if planes:
                raw=pin(args.capture/ref["file"],ref["sha256"],11*1024**2)
                need(len(raw)==ref["bytes"],"Exact plane file bytes")
                p=np.load(BytesIO(raw),allow_pickle=False)
                need(p.shape==(k,160,160) and p.dtype==np.dtype("<f4") and np.isfinite(p).all() and
                     (p.size==0 or 0<=p.min()<=p.max()<=1),"Exact native author probability")
            counts.update(post_nms_rows=n,probability_rows=k)
            yield item,pred["coco"][:k],audit["candidates"][:k],p
        need(not any(file.read(1) for file in (pf,af,ix)) and
             [v.hexdigest() for v in digests]==[receipt[k] for k in
                 ("predictions_sha256","candidate_audit_sha256","probability_index_sha256")] and
             all(receipt[k]==v for k,v in counts.items()),"Exact closed target stream hashes and counts")


def rgb(args,item):
    name=PurePosixPath(item["image_ref"]["path"]).name
    need(name.endswith(".jpg") and name==Path(name).name,"Fixed native JPEG name")
    raw=pin(args.images/name,item["image_ref"]["sha256"],4*1024**2)
    need(len(raw)==item["image_ref"]["bytes"],"Exact source JPG byte count")
    with Image.open(BytesIO(raw)) as src:
        need(src.format=="JPEG","Original JPEG input")
        result=np.asarray(ImageOps.exif_transpose(src).convert("RGB"),dtype=np.uint8)
    need(result.shape==(*item["original_hw"],3),"Original EXIF-transposed native canvas")
    return result


def prepare(args,dev,old,roster,receipt):
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    native=load(args.native,NATIVE,"_native_target_point_math")
    original,rank,free,learner,head,center,scale=old.restore(args)
    original.raw=pin(args.pair_math,native.PAIR)
    started=perf_counter()
    counts=Counter()
    groups=set()
    d=sha256()
    args.output.mkdir()
    with (args.output/EDITS).open("xb") as out:
        for j,(item,coco,audit,p) in enumerate(blocks(args,dev,roster,receipt,True),1):
            image=rgb(args,item)
            digest=sha256(image.tobytes(order="C")).hexdigest()
            groups.add(digest)
            adapted,prob=native.make_native(original,rank,*item["original_hw"])
            models=(adapted,prob,free,learner,head,center,scale)
            change=old.image_edits(coco,audit,p,original.image_features(image),models,counts)
            counts["candidates"]+=len(coco)
            raw=dev.encoded(dict(image_id=item["image_id"],rgb_sha256=digest,changes=change))
            out.write(raw)
            d.update(raw)
            if j%25==0 or j==187:
                print(json.dumps(dict(event="SSDD_POINT_PRELABEL_PROGRESS",images=j)),flush=True)
        out.flush()
        os.fsync(out.fileno())
    need(counts["candidates"]==receipt["probability_rows"] and
         counts["area_eligible"]==counts["zero_support"]+counts["supported"] and len(groups)==182 and
         counts["fixed_swaps"]==counts["free_matched_count_swaps"] and
         counts["free_natural_swaps"]==counts["native_matched_count_swaps"],"Exact target support and common budgets")
    result=dict(status=PRELABEL,protocol_sha256=PROTOCOL,worker_sha256=file_hash(Path(__file__)),
                acceptance_sha256=args.accepted_sha,manifest_sha256=MANIFEST,edits_sha256=d.hexdigest(),
                refit_accepted_sha256=old.REFIT_ACCEPTED,images=187,rgb_groups=len(groups),counts=dict(counts),
                input_predictions_sha256=receipt["predictions_sha256"],candidate_audit_sha256=receipt["candidate_audit_sha256"],
                DEV=True,FIT=False,TEST=False,GT_opened=False,AP=False,detector_forwards=0,seconds=perf_counter()-started)
    h=dev.write_new(args.output/"RESULT_SAFE.json",result)
    print(json.dumps(dict(status=PRELABEL,result_sha256=h)),flush=True)


def require_prepare(args):
    r=json.loads(pin(args.replay/"RESULT_SAFE.json",args.replay_sha))
    need(r["status"]==PRELABEL and r["protocol_sha256"]==PROTOCOL and r["worker_sha256"]==file_hash(Path(__file__)) and
         r["acceptance_sha256"]==args.accepted_sha and r["images"]==187 and r["rgb_groups"]==182 and
         r["DEV"] is True and all(r[k] is False for k in ("FIT","TEST","GT_opened","AP")),
         "Exact complete label-free replay")
    return r


def edit_blocks(args,dev,old,roster,receipt,expected):
    d=sha256()
    with (args.replay/EDITS).open("rb") as out:
        for item,coco,audit,_ in blocks(args,dev,roster,receipt,False):
            raw=out.readline(4*1024**2)
            need(raw.endswith(b"\n"),"Exact complete edit stream")
            d.update(raw)
            row=json.loads(raw)
            need(set(row)=={"image_id","rgb_sha256","changes"} and row["image_id"]==item["image_id"] and
                 dev.digest(row["rgb_sha256"]),"Exact edited image identity")
            for key,arms in row["changes"].items():
                k=int(key)
                need(str(k)==key and 1<=k<=len(coco) and arms and set(arms)<=set(MODES[1:]),
                     "Original candidate rank and mode set")
                area=audit[k-1]["mask_area"]
                need(32<=area<1024,"Original source-derived native predicted-area eligibility")
                for arm,v in arms.items():
                    need(set(v)=={"swaps","rle"} and type(v["swaps"]) is int and 1<=v["swaps"]<=min(8,area//10),
                         "Exact swap cap")
                    old.validate_edit(coco[k-1]["segmentation"],v["rle"],area,v["swaps"])
                need(arms.get("fixed",{}).get("swaps",0)==arms.get("free_matched_count",{}).get("swaps",0) and
                     arms.get("free_natural",{}).get("swaps",0)==arms.get("native_matched_count",{}).get("swaps",0),
                     "Exact per-candidate matched counts")
            yield item,coco,audit,row
        need(not out.read(1) and d.hexdigest()==expected["edits_sha256"],"Exact completed edit hash")


def verify(args,dev,old,roster,receipt):
    r=require_prepare(args)
    terminal=json.loads(pin(args.wait,args.wait_sha))
    need(terminal["terminal"]["exit_code"]==0 and args.replay_sha in terminal["terminal"]["output"],
         "Exact original natural tool terminal")
    counts=Counter()
    groups=set()
    for _,_,_,row in edit_blocks(args,dev,old,roster,receipt,r):
        groups.add(row["rgb_sha256"])
        for arms in row["changes"].values():
            for arm,v in arms.items():
                counts.update({arm+"_edited":1,arm+"_swaps":v["swaps"]})
    need(len(groups)==182 and all(counts[m+"_"+k]==r["counts"].get(m+"_"+k,0) for m in MODES[1:]
                                 for k in ("edited","swaps")),"Independently reconstructed edit budgets")
    args.output.mkdir()
    h=dev.write_new(args.output/"RESULT_SAFE.json",dict(status=VERIFIED,protocol_sha256=PROTOCOL,replay_sha256=args.replay_sha,
                    acceptance_sha256=args.accepted_sha,wait_sha256=args.wait_sha,edits_sha256=r["edits_sha256"],
                    images=187,rgb_groups=182,counts=dict(counts),DEV=True,FIT=False,TEST=False,GT_opened=False,AP=False))
    print(json.dumps(dict(status=VERIFIED,result_sha256=h)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("prepare","verify"))
    for key in ("protocol","old","native","dev-source","manifest","capture","accepted","output"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--accepted-sha",required=True)
    for key in ("images","refit","refit-result","refit-acceptance","weights","common","gray","pair-math","rank-math",
                "free","training","replay","wait"):
        p.add_argument("--"+key,type=Path)
    p.add_argument("--replay-sha")
    p.add_argument("--wait-sha")
    a=p.parse_args()
    pin(a.protocol,PROTOCOL)
    old=load(a.old,OLD,"_historical_endpoint_replay")
    dev=load(a.dev_source,DEV,"_author_mask_contract")
    dev.cpu_runtime()
    roster=manifest(a)
    witness=accepted(a,dev)
    if a.mode=="prepare":
        prepare(a,dev,old,roster,witness)
    else:
        verify(a,dev,old,roster,witness)


if __name__=="__main__":
    main()
