"""Apply the unchanged D-FINE point writer to closed center candidates, without labels."""

import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from time import perf_counter

import numpy as np
from PIL import Image
from pycocotools import mask as codec
import torch


REPLAY = "f1b6e83e98851bf867bd72f3ecb082ca2997632bdd1aedaaf64110cf06b04fd5"
ARM_KEYS = ("fixed","free_matched_count","free_natural","native_matched_count")
STRIDE=160*160*4


def need(value,reason):
    if not value:
        raise ValueError(reason)


def read(path,digest):
    raw=path.read_bytes()
    need(sha256(raw).hexdigest()==digest,"Exact closed " + path.name)
    return raw


def center_rank(candidates):
    for i,c in enumerate(candidates,1):
        a,b,x,y=c["box_xyxy"]
        if 40 <= (a+x)/2 < 61 and 40 <= (b+y)/2 < 61:
            return i
    return None


def emit(obj):
    return (json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for v in ("capture","capture-terminal","manifest","images","replay","code","refit-result","refit-acceptance","weights","output"):
        p.add_argument("--"+v,type=Path,required=True)
    for v in ("capture-terminal-sha","manifest-sha","capture-result-sha"):
        p.add_argument("--"+v,required=True)
    a=p.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1"
         and not a.output.exists(),"Fresh isolated CPU writer")
    term=json.loads(read(a.capture_terminal,a.capture_terminal_sha))
    need(term["status"]=="SCHWEGMANN_DFINE_FROZEN_CAPTURE_TERMINAL" and term["actual_process_terminal"] is True
         and type(term["actual_exit"]) is int and term["actual_exit"]==0 and term["result_sha256"]==a.capture_result_sha
         and term["tool_session_handle"]>0,"Actual frozen GPU process closure")
    result=json.loads(read(a.capture/"RESULT_SAFE.json",a.capture_result_sha))
    need(result["status"]=="SCHWEGMANN_DFINE_FROZEN_CENTER_CAPTURE_COMPLETE"
         and result["labels_opened"] is False and result["optimizer_updates"]==0
         and result["manifest_sha256"]==a.manifest_sha,"Frozen image-only inference")
    roster=json.loads(read(a.manifest,a.manifest_sha))
    original=read(a.capture/"CENTER_CANDIDATES_PRIVATE.jsonl",result["native_rows_sha256"])
    planes=read(a.capture/"SELECTED_PLANES_PRIVATE.bin",result["selected_planes_sha256"])
    capture=[json.loads(line) for line in original.splitlines()]
    need(len(capture)==result["counts"]["images"]==result["model_forwards"]==len(roster["items"])
         and original.endswith(b"\n"),"Complete unchanged image population")
    module=ModuleType("_writer_readout")
    module.__file__=str(a.replay)
    exec(compile(read(a.replay,REPLAY),str(a.replay),"exec"),module.__dict__)
    m=module.modules(a.code,a)
    m.author.cpu_runtime()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    pair,rank,free,learner,head,center,scale=m.old.restore(a)
    pair.raw=m.legacy.pin(a.pair_math,m.native.PAIR)
    adapted,probability=m.native.make_native(pair,rank,101,101)
    models=(adapted,probability,free,learner,head,center,scale)
    a.output.mkdir()
    total=Counter()
    digest=sha256()
    n_bytes=0
    used=0
    start=perf_counter()
    with (a.output/"ARMS_PRIVATE.jsonl").open("xb") as f:
        for item,row in zip(roster["items"],capture,strict=True):
            need(row["image_id"]==item["image_id"] and row["group"]==item["acquisition_group"]
                 and row["rank"]==center_rank(row["candidates"]),"Exact independently selected rank and group")
            rank_number=row["rank"]
            raw=row["prediction"]["segmentation"] if rank_number is not None else None
            score=row["prediction"]["score"] if rank_number is not None else None
            arms=dict(raw=raw)
            swaps={key:0 for key in ARM_KEYS}
            arms.update({key:raw for key in ARM_KEYS})
            if rank_number is not None:
                candidate=row["candidates"][rank_number-1]
                area=int(codec.area(raw))
                need(area==candidate["mask_area"] and score==candidate["score"] and raw["size"]==[101,101],
                     "Exact captured native mask, area and confidence")
                eligible=32<=area<1024
                total.update(selected=1, nonempty=int(area>0), eligible=int(eligible))
                need(eligible==(row["plane_offset"] is not None),"Original point observation eligibility")
                if eligible:
                    need(row["plane_offset"]==used,"Exact sequential selected probability block")
                    payload=planes[used:used+STRIDE]
                    need(len(payload)==STRIDE and sha256(payload).hexdigest()==row["plane_sha256"],
                         "Exact selected original probability bytes")
                    used+=STRIDE
                    plane=np.frombuffer(payload,dtype="<f4").reshape(160,160)
                    need(np.isfinite(plane).all() and 0<=plane.min()<=plane.max()<=1,"Original signed finite range")
                    ref=item["image_ref"]
                    body=read(a.images/ref["path"],ref["sha256"])
                    need(len(body)==ref["bytes"],"Exact decoded image read length")
                    with Image.open(BytesIO(body)) as img:
                        rgb=np.asarray(img,dtype=np.uint8)
                    need(rgb.shape==(101,101,3) and sha256(rgb.tobytes(order="C")).hexdigest()==item["rgb_sha256"],
                         "Exact model-image pixel fingerprint")
                    trials=m.old.image_edits([row["prediction"]],[candidate],[plane],pair.image_features(rgb),models,total)
                    for key,edit in trials.get("1",{}).items():
                        need(key in ARM_KEYS and int(codec.area(edit["rle"]))==area,"Exact balanced edit")
                        arms[key]=edit["rle"]
                        swaps[key]=edit["swaps"]
            else:
                need(row["prediction"] is None and row["plane_offset"] is None,"Missing selection stays absent")
                total["missing"]+=1
            need(swaps["fixed"]==swaps["free_matched_count"] and swaps["free_natural"]==swaps["native_matched_count"],
                 "Each candidate keeps the two exact matched edit counts")
            val=emit(dict(image_id=row["image_id"],group=row["group"],rank=rank_number,score=score,arms=arms,swaps=swaps))
            digest.update(val)
            n_bytes+=len(val)
            f.write(val)
        f.flush()
        os.fsync(f.fileno())
    need(used==len(planes) and all(total[x]==result["counts"].get(x,0) for x in ("selected","nonempty","eligible","missing")),
         "Full capture and native point census")
    record=dict(status="SCHWEGMANN_DFINE_FROZEN_CENTER_EDITS_COMPLETE",labels_opened=False,manifest_sha256=a.manifest_sha,
                capture_result_sha256=a.capture_result_sha,edits_sha256=digest.hexdigest(),edits_bytes=n_bytes,
                images=len(capture),model_forwards=0,optimizer_updates=0,counts=dict(total),
                head_acceptance_sha256=m.old.REFIT_ACCEPTED,source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                elapsed_seconds=perf_counter()-start)
    data=emit(record)
    with (a.output/"RESULT_SAFE.json").open("xb") as f:
        f.write(data)
    print(json.dumps(dict(status=record["status"],result_sha256=sha256(data).hexdigest())),flush=True)


if __name__=="__main__":
    main()
