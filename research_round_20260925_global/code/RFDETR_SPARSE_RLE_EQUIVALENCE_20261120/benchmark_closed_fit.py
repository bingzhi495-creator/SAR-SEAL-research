"""Bounded encoding cost check on independently closed RF FIT109 RLE only."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import statistics
from time import perf_counter

import numpy as np
from pycocotools import mask as codec

from sparse_rle import decode_runs, xor_rle

BINS=((1,16),(17,64),(65,256),(257,None))
CKPT="011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11"


def need(ok,why):
    if not ok:
        raise ValueError(why)


def encoded(value):
    return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def pin(path,digest,limit=64*1024**2):
    need(path.is_file() and not path.is_symlink() and 0<path.stat().st_size<=limit,"Bounded closed FIT input")
    raw=path.read_bytes()
    need(sha256(raw).hexdigest()==digest,"Exact closed input "+path.name)
    return raw


def write(path,value):
    raw=encoded(value)
    with path.open("xb") as f:
        f.write(raw)
    return sha256(raw).hexdigest()


def admit(args):
    public=json.loads(args.public_summary.read_bytes())
    need(public["status"]=="ACTUAL_FULL_FIT109_CAPTURE_INDEPENDENTLY_ACCEPTED" and public["FIT"] is True and
         all(public[k] is False for k in ("DEV","TEST","GT_opened","AP")) and public["images"]==109 and
         public["checkpoint_sha256"]==CKPT,"Published exact source FIT109 capture")
    independent=json.loads(pin(args.acceptance,public["independent_result_sha256"]))
    need(independent["status"]=="RF_CENTER5202_FULL_CAPTURE_INDEPENDENTLY_ACCEPTED_V1" and
         independent["FIT"] is True and independent["mode"]=="fit" and
         all(independent[k] is False for k in ("DEV","TEST","GT_opened","AP")) and
         independent["original_os_exit"]==0 and independent["job_drained"] is True and
         independent["independent_CPU_raster_disagree_rows"]==independent["independent_CPU_raster_disagree_pixels"]==0 and
         independent["source_checkpoint_sha256"]==CKPT and independent["images"]==109 and
         all(independent[k]==public[k] for k in ("capture_result_sha256","stock_predictions_sha256",
                                                "queries_sha256","image_manifest_sha256")),
         "Independent closed source lineage")
    wait=json.loads(pin(args.wait,independent["original_tool_wait_sha256"]))
    closure=json.loads(pin(args.control/"OWNER_CLOSURE_PRIVATE.json",independent["original_owner_closure_sha256"]))
    need(closure["status"]=="RF_FULL_CAPTURE_OWNER_NATURAL_CLOSE_V1" and closure["mode"]=="fit" and
         closure["original_exit"]==0 and closure["resource_closed"] is True and wait["terminal"]["exit_code"]==0 and
         json.loads(wait["terminal"]["output"].strip().splitlines()[-1])==
             dict(status=closure["status"],closure_sha256=independent["original_owner_closure_sha256"],
                  result_sha256=public["capture_result_sha256"]),"Exact original FIT natural terminal")
    child=json.loads(pin(args.control/"CHILD_TERMINAL_PRIVATE.json",closure["child_terminal_sha256"]))
    need(child["original_os_wait"] is True and child["original_os_exit"]==0 and child["job_assigned"] is True and
         child["job_active_after_wait"]==0 and child["stop_reason"] is None,"Closed original contained worker")
    report=json.loads(pin(args.control/"capture/RESULT_SAFE.json",public["capture_result_sha256"]))
    need(report["checkpoint_sha256"]==CKPT and report["mode"]=="fit" and report["FIT"] is True and
         all(report[k] is False for k in ("DEV","TEST","GT_opened","AP")) and
         report["native_mapped_rows"]==public["native_mapped_rows"]==10488 and
         report["stock_predictions_sha256"]==public["stock_predictions_sha256"],"Exact native author FIT masks")
    return public,independent


def candidates(args,public):
    raw=pin(args.control/"capture/STOCK_NATIVE_PREDICTIONS_PRIVATE.jsonl",public["stock_predictions_sha256"])
    lines=[json.loads(line) for line in raw.splitlines()]
    need(raw.endswith(b"\n") and len(lines)==109 and len({r["image_id"] for r in lines})==109,
         "Complete original source mask population")
    total=nonempty=0
    unique={}
    for ordinal,block in enumerate(lines,1):
        ranks=block["native_ranks"]
        need(block["native_count"]==100 and len(ranks)==len(block["coco"]) and
             ranks==sorted(set(ranks)) and all(1<=r<=100 for r in ranks) and
             block["unmapped_slot1_count"]+len(ranks)==100,"Original mapped joint ranks")
        for rank,row in zip(ranks,block["coco"],strict=True):
            total+=1
            rle=row["segmentation"]
            need(row["image_id"]==block["image_id"] and rle["size"]==[800,800],"Original native FIT raster")
            runs=decode_runs(rle["counts"])
            need(sum(runs)==640000,"Exact run coverage")
            area=sum(runs[1::2])
            if area==0:
                continue
            nonempty+=1
            digest=sha256(encoded(rle)).hexdigest()
            unique.setdefault(digest,dict(rle=rle,hash=digest,runs=runs,run_count=len(runs),area=area,
                                          ordinal=ordinal,image_id=block["image_id"],rank=rank))
    need(total==10488,"Full original mapped population")
    selected=[]
    populations=[]
    for low,high in BINS:
        pool=[c for c in unique.values() if c["run_count"]>=low and (high is None or c["run_count"]<=high)]
        pool.sort(key=lambda c:c["hash"])
        picks=pool[:8]
        if len(pool)>8:
            worst=min(pool,key=lambda c:(-c["run_count"],c["hash"]))
            picks=pool[:7]+([worst] if worst not in pool[:7] else pool[7:8])
        label=f"{low}-{high if high is not None else 'max'}"
        for candidate in picks:
            selected.append(dict(candidate,stratum=label))
        populations.append(dict(stratum=label,unique_masks=len(pool),selected=len(picks)))
    return selected,dict(native_rows=total,nonempty_rows=nonempty,unique_nonempty_RLE=len(unique),
                         run_count_min=min(c["run_count"] for c in unique.values()),
                         run_count_max=max(c["run_count"] for c in unique.values()),strata=populations)


def boundary_flips(runs,height=800,width=800):
    total=height*width
    pixels=[]
    seen=set()
    offset=0
    for run in runs:
        offset+=run
        for position in (offset-1,offset):
            if 0<=position<total:
                native=(position%height)*width+position//height
                if native not in seen:
                    pixels.append(native)
                    seen.add(native)
                    if len(pixels)==16:
                        return pixels,16
    boundary_count=len(pixels)
    for native in [i*(total-1)//15 for i in range(16)]+list(range(16)):
        if native not in seen:
            pixels.append(native)
            seen.add(native)
            if len(pixels)==16:
                return pixels,boundary_count
    raise ValueError("Exactly sixteen unique native pixels")


def dense(source,pixels):
    mask=codec.decode(source).copy(order="C")
    mask.reshape(-1)[pixels]^=1
    result=codec.encode(np.asfortranarray(mask,dtype=np.uint8))
    return dict(size=result["size"],counts=result["counts"].decode("ascii"))


def summary(values):
    return dict(median_us=float(np.median(values)),p90_us=float(np.quantile(values,.90)),max_us=max(values))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("public-summary","acceptance","wait","control","output"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--repeats",type=int,default=16)
    a=p.parse_args()
    need(2<=a.repeats<=32,"Bounded interleaved repeat count")
    public,independent=admit(a)
    selected,population=candidates(a,public)
    need(0<len(selected)<=32,"At most32 deterministic real FIT masks")
    reports=[]
    private=[]
    for i,c in enumerate(selected):
        flips,near=boundary_flips(c["runs"])
        original=c["rle"]
        expected=dense(original,flips)
        need(xor_rle(original,flips)==expected,"Real FIT exact compressed-byte XOR")
        sparse_times=[]
        dense_times=[]
        for repeat in range(a.repeats):
            order=("sparse","dense") if (i+repeat)%2==0 else ("dense","sparse")
            for kind in order:
                before=perf_counter()
                result=xor_rle(original,flips) if kind=="sparse" else dense(original,flips)
                duration=1e6*(perf_counter()-before)
                need(result==expected,"Every timed call byte-identical")
                (sparse_times if kind=="sparse" else dense_times).append(duration)
        s,t=summary(sparse_times),summary(dense_times)
        reports.append(dict(sample=i+1,stratum=c["stratum"],source_runs=c["run_count"],
                            source_compressed_bytes=len(original["counts"]),area=c["area"],
                            boundary_neighborhood_flips=near,total_unique_flips=16,all_calls_byte_exact=True,
                            sparse=s,dense=t,median_wall_ratio_dense_over_sparse=t["median_us"]/s["median_us"]))
        private.append(dict(sample=i+1,source_ordinal=c["ordinal"],image_id=c["image_id"],native_rank=c["rank"],
                            source_rle_sha256=c["hash"],row_major_pixels=flips))
    a.output.mkdir()
    manifest=write(a.output/"SELECTION_PRIVATE.json",private)
    output=dict(status="RF_FIT109_REAL_MASK_ENCODING_EQUIVALENCE_AND_COST",FIT=True,DEV=False,TEST=False,
                GT_opened=False,AP=False,model_calls=0,image_IO=False,annotation_IO=False,
                published_capture_result_sha256=public["capture_result_sha256"],
                independent_capture_sha256=public["independent_result_sha256"],
                original_wait_sha256=independent["original_tool_wait_sha256"],
                no_live_binding="coordinating root confirmed closed FIT109 payload unbound before execution",
                checkpoint_sha256=CKPT,source_stock_sha256=public["stock_predictions_sha256"],
                population=population,selected_masks=len(selected),selection_sha256=manifest,
                selection_rule="unique nonempty RLEs; each run stratum first7 canonical RLE hashes plus maximum run-count mask; up to8",
                flip_rule="sixteen unique source-transition-neighbor pixels in Fortran order mapped to row-major; evenly spaced fallback",
                repeats_per_mask=a.repeats,timing_order="alternates sparse/dense first per mask/repeat",units="wall microseconds",
                all_calls_byte_identical=True,results=reports,
                median_of_mask_medians=dict(sparse_us=statistics.median(r["sparse"]["median_us"] for r in reports),
                                            dense_us=statistics.median(r["dense"]["median_us"] for r in reports)),
                extrapolation=False,original_outputs_modified=False)
    digest=write(a.output/"RESULT_SAFE.json",output)
    print(json.dumps(dict(result_sha256=digest,selected=len(selected),population=population,results=reports)),flush=True)


if __name__=="__main__":
    main()
