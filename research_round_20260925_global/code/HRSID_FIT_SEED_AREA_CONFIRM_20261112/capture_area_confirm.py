"""Capture a prospectively fixed FIT source-group area-signal cohort without labels."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
import json
from math import log2, sqrt
import os
from pathlib import Path
from statistics import mean, median
import sys
from time import perf_counter
from types import ModuleType

import numpy as np

PROTOCOL_SHA = "4ac12dcc198bfafc424084f613e8ca6a69fdb4ecef3ee9b62e3d8bc0ab64b185"
MANIFEST_SHA = "0b87d626884550f5617f450ebc6ca217c4bbc331a1ac41e3d975289e403e3629"
PRIOR_CAPTURE_SHA = "17a81104ab91a1570aea21e9caccc763855aa9242690f96ddec141ffeb42f680"
STRUCTURE_SHA = "30ace2a0fa00f4353a66dd7b10b271f95c6521d94b6aa6699bd7a8e1b2ecee4c"
SEEDS = (5201, 5202, 5203)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def pinned(path, digest, limit=2**20):
    require(path.resolve(strict=True) == path and 0 < path.stat().st_size <= limit, "Exact bounded path")
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == digest, "Pinned digest: " + path.name)
    return raw


def module(name, path, digest):
    m = ModuleType(name)
    m.__file__ = str(path)
    sys.modules[name] = m
    exec(compile(pinned(path, digest),str(path),"exec"),m.__dict__)
    return m


def native_soft(logit, hard):
    import torch
    from scipy.ndimage import binary_dilation, binary_erosion
    from scipy.special import expit
    from torch.nn.functional import interpolate

    raw = torch.from_numpy(logit).reshape(1,1,*logit.shape)
    interpolated = interpolate(raw, size=hard.shape, mode="bilinear", align_corners=False)[0,0].numpy()
    require(np.array_equal(interpolated > 0., hard), "Exact stock native mask reconstruction")
    yy,xx = np.nonzero(hard)
    require(yy.size > 0, "Nonempty fixed native mask")
    box = np.s_[max(0,int(yy.min())-2):int(yy.max())+3, max(0,int(xx.min())-2):int(xx.max())+3]
    crop = hard[box]
    connect = np.ones((3,3),dtype=bool)
    band = binary_dilation(crop, structure=connect, iterations=2) & ~binary_erosion(crop, structure=connect, iterations=2)
    return float(np.sum(expit(interpolated[box][band].astype(np.float64))) - crop[band].sum())


def image_signals(blocks, structure, codec, output):
    anchor = blocks[5202]
    require(all((b["ordinal"],b["group"],b["basename"],b["rgb_sha256"]) ==
                (anchor["ordinal"],anchor["group"],anchor["basename"],anchor["rgb_sha256"]) for b in blocks.values()),
            "Identical source image and order across seeds")
    data = {seed:blocks[seed]["candidates"] for seed in SEEDS}
    reps = {seed:structure.greedy_representatives(structure.iou(p,p,codec),.90)[0] for seed,p in data.items()}
    eligible = sum(32<=x["area"]<1024 for x in data[5202])
    anchor_ids = [i for i in reps[5202] if 32<=data[5202][i]["area"]<1024]
    anchor_rows = [data[5202][i] for i in anchor_ids]
    donors = {seed:[data[seed][i] for i in reps[seed]] for seed in (5201,5203)}
    left = structure.iou(anchor_rows,donors[5201],codec)
    right = structure.iou(anchor_rows,donors[5203],codec)
    pairs = structure.first_exclusive(left,right)
    logit = anchor["logit"]
    archive = pinned(output/"5202"/logit["filename"],logit["sha256"],8*2**20)
    require(len(archive)==logit["bytes"],"Exact native anchor logit size")
    raw = np.load(BytesIO(archive), allow_pickle=False)
    require(raw.shape==(logit["rows"],144,144) and raw.dtype==np.float32 and np.isfinite(raw).all(),
            "Exact native anchor FP32 planes")
    signals=[]
    for i,(l,r) in pairs.items():
        point=anchor_rows[i]
        dp=(donors[5201][l],donors[5203][r])
        require(point["logit_row"] is not None and all(d["logit_row"] is not None for d in dp),
                "All linked raw mask planes remain admitted")
        area=point["area"]
        other=[x["area"] for x in dp]
        hard=codec.decode(point["rle"]).astype(bool)
        require(int(hard.sum())==area,"Native area identity")
        cap=2*min(8,area//10)
        signals.append(dict(rank=point["rank"],donor_ranks={s:x["rank"] for s,x in zip((5201,5203),dp,strict=True)},
                            mean_delta=mean([area,*other])-area,median_delta=median([area,*other])-area,
                            soft_delta=native_soft(raw[point["logit_row"]],hard),
                            size_delta=cap*log2(area/sqrt(32*1024))/2.5,cap=cap))
    return dict(ordinal=anchor["ordinal"],group=anchor["group"],basename=anchor["basename"],signals=signals),eligible,len(anchor_ids)


def run(args):
    import torch
    from pycocotools import mask as codec

    begin=perf_counter()
    here=Path(__file__).resolve().parent
    pinned(here/"PROTOCOL.md",PROTOCOL_SHA)
    require(sys.flags.isolated and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1" and not torch.cuda.is_initialized(),
            "Isolated CPU only")
    require(torch.__version__=="2.10.0+cu130" and version("rfdetr")=="1.12.0" and version("Pillow")=="12.3.0"
            and version("pycocotools")=="2.0.11" and np.__version__=="1.26.4","Exact earlier native runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    require(args.output.parent.resolve(strict=True)==args.output.parent and not args.output.exists(),
            "Fresh canonical capture output")
    prior=module("_area_prior_capture",args.code/"HRSID_FIT_SEED_CONSENSUS_20261112/capture_consensus.py",PRIOR_CAPTURE_SHA)
    structure=module("_area_prior_structure",args.code/"HRSID_FIT_ASSOC_STRUCTURE_20261112/score_structure.py",STRUCTURE_SHA)
    admission=prior.module(args.code/"HRSID_TO_SSDD_FROZEN_DEV187_20261110/export_frozen_hrsid_ssdd.py",prior.ADMISSION_SHA,
                          "_area_admission")
    post=prior.module(args.code/"HRSID_FIT_NATIVE_QUERY_RETENTION_20261109/fit_query_export.py",prior.POST_SHA,"_area_post")
    plan=json.loads(pinned(args.source_plan,prior.SOURCE_PLAN_SHA))
    manifest=json.loads(pinned(args.manifest,MANIFEST_SHA,65536))
    require(manifest["status"]=="FIT_SEED_CONSENSUS_IMAGE_MANIFEST_V1" and
            manifest["variant"]=="P316_COORDINATE_DISJOINT_ORACLE36_AND_CONSENSUS36_PREFIX_EXCLUDED","Locked fresh FIT cohort")
    rows=prior.manifest_rows(manifest)
    refs=prior.source_refs(args.code,admission)
    args.output.mkdir()
    data,arms={},[]
    for seed in SEEDS:
        data[seed],result=prior.arm_run(args,plan["pairs"][str(seed)],seed,refs,rows,admission,post)
        arms.append(result)
    counts=Counter()
    eligible=representative=groups=dir_groups=0
    lines=[]
    for blocks in zip(*(data[s] for s in SEEDS),strict=True):
        row,e,r=image_signals(dict(zip(SEEDS,blocks,strict=True)),structure,codec,args.output)
        eligible+=e
        representative+=r
        shifts=[x["median_delta"] for x in row["signals"]]
        groups+=bool(shifts)
        dir_groups+=any(shifts)
        counts.update("add" if x>0 else "drop" if x<0 else "zero" for x in shifts)
        lines.append(prior.encode(row))
    digest=prior.write(args.output/"SIGNALS_PRIVATE.jsonl",b"".join(lines))
    covered=sum(counts.values())
    nonzero=counts["add"]+counts["drop"]
    gate=covered>=60 and groups>=20 and nonzero>=30 and dir_groups>=15
    coverage=dict(eligible=eligible,representative_eligible=representative,covered_both=covered,covered_groups=groups,
                  nonzero_median_area_offset=nonzero,nonzero_groups=dir_groups,
                  offsets={key:counts[key] for key in ("add","drop","zero")},signals_sha256=digest,
                  gate="COVERAGE_PASS" if gate else "COVERAGE_STOP")
    result=dict(status="HRSID_FIT_SEED_AREA_CONFIRM_CAPTURE_ACTUAL",FIT=True,DEV=False,TEST=False,GT_opened=False,AP=False,
                optimizer_updates=0,images=36,model_forwards=108,arms=arms,coverage=coverage,
                manifest_sha256=MANIFEST_SHA,protocol_sha256=PROTOCOL_SHA,worker_seconds=perf_counter()-begin)
    result_digest=prior.write(args.output/"RESULT_SAFE.json",prior.encode(result))
    print(json.dumps(dict(status=result["status"],result_sha256=result_digest,gate=coverage["gate"])),flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    for field in ("code","source-plan","manifest","fit-images","output"):
        p.add_argument("--"+field,type=Path,required=True)
    args=p.parse_args()
    def offline(event,_args):
        if event.startswith("socket."):
            raise PermissionError("No network")
    sys.addaudithook(offline)
    run(args)
