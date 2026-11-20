"""Independently verify a naturally closed frozen RF full capture, without labels."""
from argparse import ArgumentParser
from hashlib import sha256
from importlib.util import module_from_spec,spec_from_file_location
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from time import perf_counter


def need(condition,message):
    if not condition:
        raise ValueError(message)


def pin(path,digest,limit=4*1024**2):
    need(path.is_file() and not path.is_symlink() and 0<path.stat().st_size<=limit,"Exact closed bounded input")
    raw=path.read_bytes()
    need(sha256(raw).hexdigest()==digest,"Exact closed digest "+path.name)
    return raw


def main():
    p=ArgumentParser(allow_abbrev=False)
    for name in ("root","worker","capture","closure","wait","output"):
        p.add_argument("--"+name,type=Path,required=True)
    for name in ("worker-sha","protocol-sha","closure-sha","wait-sha","result-sha"):
        p.add_argument("--"+name,required=True)
    a=p.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and os.environ["CUDA_VISIBLE_DEVICES"]=="-1",
         "Independent isolated CPU readback")
    before=perf_counter()
    raw=pin(a.worker,a.worker_sha,1024**2)
    scope={"__name__":"_accepted_capture","__file__":str(a.worker)}
    exec(compile(raw,str(a.worker),"exec"),scope)
    closure=json.loads(pin(a.closure,a.closure_sha))
    wait=json.loads(pin(a.wait,a.wait_sha))
    terminal=wait["terminal"]
    footer=json.loads(terminal["output"].strip().splitlines()[-1])
    need(closure["status"]=="RF_FULL_CAPTURE_OWNER_NATURAL_CLOSE_V1" and closure["resource_closed"] is True
         and closure["original_exit"]==0 and terminal["exit_code"]==0
         and type(wait["launch"]["session_id"]) is int
         and footer==dict(status=closure["status"],closure_sha256=a.closure_sha,result_sha256=a.result_sha)
         and closure["result_sha256"]==a.result_sha and closure["worker_sha256"]==a.worker_sha
         and closure["protocol_sha256"]==a.protocol_sha,"Exact original natural tool/owner closure")
    child=json.loads(pin(a.closure.parent/"CHILD_TERMINAL_PRIVATE.json",closure["child_terminal_sha256"]))
    need(child["original_os_wait"] is True and child["original_os_exit"]==0 and child["job_assigned"] is True
         and child["job_active_after_wait"]==0 and child["stop_reason"] is None,
         "Original contained child natural exit and drained job")
    report=json.loads(pin(a.capture/"RESULT_SAFE.json",a.result_sha))
    need(report["status"]=="RFDETR_CENTER5202_FULL_CAPTURE_COMPLETE_V1"
         and report["worker_sha256"]==a.worker_sha and report["protocol_sha256"]==a.protocol_sha
         and report["GT_opened"] is report["AP"] is report["TEST"] is False
         and report["raw_logit_raster_RLE_exact"]==report["raw_logit_rows"],"Exact complete worker scope")
    from types import SimpleNamespace
    args=SimpleNamespace(root=a.root,mode=report["mode"])
    core=scope["load_core"](a.root)
    rows=scope["metadata"](args,core)
    base,_=scope["setup"](args,core)
    import numpy as np
    import torch
    from pycocotools import mask as codec
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    qraw=pin(a.capture/"QUERIES_PRIVATE.jsonl",report["queries_sha256"],128*1024**2)
    blocks=[json.loads(line) for line in qraw.splitlines()]
    sraw=pin(a.capture/"STOCK_NATIVE_PREDICTIONS_PRIVATE.jsonl",report["stock_predictions_sha256"],128*1024**2)
    stock_lines=sraw.splitlines(keepends=True)
    manifest=json.loads(pin(a.capture/"IMAGE_MANIFEST_PRIVATE.json",report["image_manifest_sha256"]))
    need(len(rows)==len(blocks)==len(stock_lines)==len(manifest)==report["images"],"Complete population")
    planes=files=nbytes=mapped=0
    names=set()
    cpu_disagree=cpu_pixels=0
    for ordinal,(row,block,stock_line,item) in enumerate(zip(rows,blocks,stock_lines,manifest,strict=True),1):
        need(row["image_id"]==block["image_id"]==item["image_id"]
             and row["encoded_sha256"]==block["encoded_sha256"]==item["encoded_sha256"]
             and row["decoded_rgb_sha256"]==block["image_content_sha256"]==item["decoded_rgb_sha256"],
             "Exact complete native image identity join")
        selected=scope["eligible_candidates"](block,codec)
        ref=block["logit_capture"]
        expected=[dict(logit_row=i,query_id=q,native_rank=rank,score=block["ship_scores"][q],predicted_area=area)
                  for i,(q,rank,area) in enumerate(selected)]
        need(ref["candidates"]==expected and ref["relative_path"]==f"RAW_LOGITS/{ordinal:03d}.npy",
             "Exact raw-plane ordered native candidate linkage")
        data=pin(a.capture/ref["relative_path"],ref["sha256"],9*1024**2)
        stream=BytesIO(data)
        values=np.load(stream,allow_pickle=False)
        need(not stream.read(1) and values.dtype==np.dtype("<f4") and values.shape==(len(selected),144,144)
             and list(values.shape)==ref["shape"] and len(data)==ref["bytes"] and bool(np.isfinite(values).all()),
             "Exact finite FP32 native logit data")
        need(base.encoded(scope["native_block"](row["image_id"],block))==stock_line,
             "Exact original stock score/rank/class-zero RLE bytes")
        for start in range(0,len(selected),8):
            raster=(torch.nn.functional.interpolate(torch.from_numpy(values[start:start+8,None]),
                    size=(800,800),mode="bilinear",align_corners=False)>0)[:,0].numpy()
            for binary,(qid,_,_) in zip(raster,selected[start:start+8],strict=True):
                old=codec.decode(block["all_rles"][qid]).astype(bool)
                different=np.count_nonzero(binary!=old)
                cpu_disagree+=int(different>0)
                cpu_pixels+=int(different)
        mapped+=len(block["native_ids"])
        planes+=len(selected)
        files+=1
        nbytes+=len(data)
        names.add(Path(ref["relative_path"]).name)
    need({p.name for p in (a.capture/"RAW_LOGITS").iterdir()}==names and files==report["raw_logit_files"]
         and planes==report["raw_logit_rows"] and nbytes==report["raw_logit_bytes"]
         and mapped==report["native_mapped_rows"] and not torch.cuda.is_initialized(),"Exact closed census and CPU-only readback")
    if report["mode"]=="fit":
        need(cpu_disagree==cpu_pixels==0 and report["cache_hits"]==36 and report["model_forwards"]==73,
             "Same original CPU native masks across cache and new images")
    else:
        need(report["stock_predictions_sha256"]==scope["DEV_STOCK"] and mapped==67386
             and report["historical_CUDA_stock_exact"] is True,"Exact historical CUDA stock")
    result=dict(status="RF_CENTER5202_FULL_CAPTURE_INDEPENDENTLY_ACCEPTED_V1",mode=report["mode"],
        images=len(rows),native_mapped_rows=mapped,raw_logit_rows=planes,raw_logit_files=files,
        raw_logit_bytes=nbytes,source_checkpoint_sha256=scope["CKPT"],
        worker_sha256=a.worker_sha,protocol_sha256=a.protocol_sha,capture_result_sha256=a.result_sha,
        original_owner_closure_sha256=a.closure_sha,original_tool_wait_sha256=a.wait_sha,
        original_os_exit=0,job_drained=True,stock_predictions_sha256=report["stock_predictions_sha256"],
        queries_sha256=report["queries_sha256"],image_manifest_sha256=report["image_manifest_sha256"],
        same_device_raster_exact=report["raw_logit_raster_RLE_exact"],
        independent_CPU_raster_disagree_rows=cpu_disagree,independent_CPU_raster_disagree_pixels=cpu_pixels,
        cache_hits=report["cache_hits"],model_forwards=0,GT_opened=False,AP=False,
        FIT=report["mode"]=="fit",DEV=report["mode"]=="dev",TEST=False,elapsed_seconds=perf_counter()-before)
    out=scope["encoded"](result)
    with a.output.open("xb") as stream:
        need(stream.write(out)==len(out),"Complete independent receipt")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=result["status"],result_sha256=sha256(out).hexdigest())),flush=True)


if __name__=="__main__":
    main()

