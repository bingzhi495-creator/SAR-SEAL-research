"""Independently bind one naturally closed native CUDA probability patch."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sys


def need(ok,why):
    if not ok:
        raise ValueError(why)


def encoded(value):
    return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def pin(path,digest,limit=16*1024**2):
    need(path.is_absolute() and path.resolve(strict=True)==path and 0<path.stat().st_size<=limit,"Canonical closed patch source")
    raw=path.read_bytes()
    need(sha256(raw).hexdigest()==digest,"Exact original closed patch digest")
    return raw


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for name in ("control","closure","wait","output"):
        p.add_argument("--"+name,type=Path,required=True)
    for name in ("closure-sha","wait-sha"):
        p.add_argument("--"+name,required=True)
    a=p.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and os.environ["CUDA_VISIBLE_DEVICES"]=="-1",
         "Independent original CPU patch readback")
    closure=json.loads(pin(a.closure,a.closure_sha,1024**2))
    wait=json.loads(pin(a.wait,a.wait_sha,1024**2))
    terminal=wait["terminal"]
    footer=json.loads(terminal["output"].strip().splitlines()[-1])
    need(closure["status"]=="RF_SINGLE_PLANE_OWNER_NATURAL_CLOSE_V1" and closure["resource_closed"] is True
         and closure["original_os_exit"]==0 and terminal["exit_code"]==0
         and (type(wait["launch"].get("session_id")) is int or
              (wait["launch"].get("exit_code")==0 and wait["launch"]==terminal))
         and footer==dict(status=closure["status"],closure_sha256=a.closure_sha,result_sha256=closure["result_sha256"]),
         "Exact natural original CUDA owner wait and release")
    child=json.loads(pin(a.control/"CHILD_TERMINAL_PRIVATE.json",closure["child_terminal_sha256"],1024**2))
    need(child["original_os_wait"] is True and child["original_os_exit"]==0 and child["job_assigned"] is True
         and child["job_active_after_wait"]==0 and child["stop_reason"] is None,"Original contained worker natural termination")
    report=json.loads(pin(a.control/"render/RENDER_RESULT_PRIVATE.json",closure["result_sha256"],1024**2))
    need(report["status"]=="RF_SINGLE_NATIVE_CUDA_PROBABILITY_RENDER_COMPLETE_V1"
         and report["capture_acceptance_sha256"]=="77ef95b694f74324229507ee79dea015323acff3b716052c3aefa4438105e844"
         and report["strict_cuda_logit_sign_RLE_exact"] is True and report["raw_planes_on_cuda"]==1
         and report["model_forward_calls"]==0 and report["DEV"] is True
         and report["FIT"] is report["TEST"] is report["GT_opened"] is report["AP"] is False,
         "Exact one-plane author-native label-free scope")
    import numpy as np
    from pycocotools import mask as codec
    ref=report["native_probability_ref"]
    data=pin(Path(ref["path"]),ref["sha256"],3*1024**2)
    need(len(data)==ref["bytes"] and ref["shape"]==[800,800] and ref["dtype"]=="<f4",
         "Exact bounded native probability reference")
    stream=BytesIO(data)
    array=np.load(stream,allow_pickle=False)
    need(not stream.read(1) and array.shape==(800,800) and array.dtype==np.dtype("<f4")
         and np.isfinite(array).all() and array.min()>=0 and array.max()<=1
         and sha256(array.tobytes(order="C")).hexdigest()==report["native_probability_tensor_sha256"],
         "Exact CUDA-proven FP32 probability bytes and shape")
    need(sha256(encoded(report["stock_rle"])).hexdigest()==report["stock_rle_sha256"]
         and sha256(codec.decode(report["stock_rle"]).tobytes(order="C")).hexdigest()==
             report["strict_cuda_logit_sign_tensor_sha256"],"Independent native sign tensor linkage to original RLE")
    result=dict(report,status="RF_CENTER5202_DEV705_SINGLE_NATIVE_CUDA_PROBABILITY_PATCH_ACCEPTED_V1",
        original_os_exit=0,original_os_wait=True,job_drained=True,resource_closed=True,
        original_tool_wait_sha256=a.wait_sha,original_owner_closure_sha256=a.closure_sha,
        cuda_render_result_sha256=closure["result_sha256"],independent_probability_readback_exact=True)
    raw=encoded(result)
    with a.output.open("xb") as target:
        need(target.write(raw)==len(raw),"Complete private exact native patch")
        target.flush()
        os.fsync(target.fileno())
    print(json.dumps(dict(status=result["status"],patch_sha256=sha256(raw).hexdigest())),flush=True)


if __name__=="__main__":
    main()

