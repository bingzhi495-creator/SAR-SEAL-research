"""Locate the single independently observed render discrepancy without any labels."""
from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from time import monotonic

ACCEPT = "77ef95b694f74324229507ee79dea015323acff3b716052c3aefa4438105e844"
QUERY = "3533d397589a0e615d93ccf92a4bc8f6beea711dfcf08912ec654968c756ed70"
CAPTURE = "ccfab7913513ece3c186349f12d720b0fa09b4a7412b0754294b5470fcf9b0d7"


def need(ok, why):
    if not ok:
        raise ValueError(why)


def encoded(value):
    return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def pin(path,digest,limit):
    need(path.is_file() and not path.is_symlink() and 0<path.stat().st_size<=limit,"Bounded closed source")
    data=path.read_bytes()
    need(sha256(data).hexdigest()==digest,"Exact closed source digest")
    return data


def write(path,data):
    with path.open("xb") as stream:
        need(stream.write(data)==len(data),"Complete original private artifact")
        stream.flush()
        os.fsync(stream.fileno())
    return sha256(data).hexdigest()


def main():
    p=ArgumentParser(allow_abbrev=False)
    for name in ("acceptance","capture","output"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and os.environ["CUDA_VISIBLE_DEVICES"]=="-1",
         "Isolated bounded CPU locator")
    accepted=json.loads(pin(a.acceptance,ACCEPT,1024**2))
    need(accepted["status"]=="RF_CENTER5202_FULL_CAPTURE_INDEPENDENTLY_ACCEPTED_V1"
         and accepted["capture_result_sha256"]==CAPTURE and accepted["queries_sha256"]==QUERY
         and accepted["independent_CPU_raster_disagree_rows"]==accepted["independent_CPU_raster_disagree_pixels"]==1,
         "Exact closed single-discrepancy population")
    report=json.loads(pin(a.capture/"RESULT_SAFE.json",CAPTURE,1024**2))
    rows=[json.loads(line) for line in pin(a.capture/"QUERIES_PRIVATE.jsonl",QUERY,128*1024**2).splitlines()]
    import torch
    import numpy as np
    from pycocotools import mask as codec
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    need(not torch.cuda.is_initialized() and len(rows)==report["images"]==705,"CPU exact full row source")
    start=monotonic()
    checked=0
    a.output.mkdir()
    for ordinal,block in enumerate(rows,1):
        ref=block["logit_capture"]
        raw=pin(a.capture/ref["relative_path"],ref["sha256"],9*1024**2)
        values=np.load(BytesIO(raw),allow_pickle=False)
        candidates=ref["candidates"]
        need(values.dtype==np.dtype("<f4") and values.shape==(len(candidates),144,144),"Exact raw plane carrier")
        for offset in range(0,len(candidates),8):
            need(monotonic()-start<900,"Bounded locator wall")
            native=torch.nn.functional.interpolate(torch.from_numpy(values[offset:offset+8,None]),
                size=(800,800),mode="bilinear",align_corners=False)[:,0].numpy()
            for j,(logit,candidate) in enumerate(zip(native,candidates[offset:offset+8],strict=True),offset):
                q=candidate["query_id"]
                old=codec.decode(block["all_rles"][q]).astype(bool)
                different=np.flatnonzero((logit>0)!=old)
                checked+=1
                if not len(different):
                    continue
                need(len(different)==1,"Exact single-pixel discrepancy before narrow repair")
                plane=np.ascontiguousarray(values[j],dtype="<f4")
                single=BytesIO()
                np.save(single,plane,allow_pickle=False)
                plane_path=a.output/"ONE_RAW_PLANE_PRIVATE.npy"
                plane_file_sha=write(plane_path,single.getvalue())
                request=dict(status="RF_CENTER5202_DEV705_SINGLE_RENDER_ROW_LOCATED_V1",
                    capture_acceptance_sha256=ACCEPT,capture_result_sha256=CAPTURE,queries_sha256=QUERY,
                    ordinal=ordinal,image_id=block["image_id"],query_id=q,native_rank=candidate["native_rank"],
                    logit_row=j,source_logit_ref=dict(path=str((a.capture/ref["relative_path"]).resolve()),
                        relative_path=ref["relative_path"],bytes=ref["bytes"],sha256=ref["sha256"]),
                    source_raw_plane_sha256=sha256(plane.tobytes(order="C")).hexdigest(),
                    single_raw_plane_ref=dict(path=str(plane_path.resolve()),bytes=len(single.getvalue()),sha256=plane_file_sha),
                    stock_rle=block["all_rles"][q],stock_rle_sha256=sha256(encoded(block["all_rles"][q])).hexdigest(),
                    original_hw=[800,800],cpu_disagreement_rows=1,cpu_disagreement_pixels=1,
                    discrepancy_flat_index=int(different[0]),cpu_native_logit=float(logit.flat[different[0]]),
                    original_binary_value=int(old.flat[different[0]]),
                    source_acceptance_ref=dict(path=str(a.acceptance.resolve()),sha256=ACCEPT),
                    checked_planes=checked,GT_opened=False,FIT=False,DEV=True,TEST=False,AP=False,
                    model_forward_calls=0,elapsed_seconds=monotonic()-start)
                request_sha=write(a.output/"REQUEST_PRIVATE.json",encoded(request))
                print(json.dumps(dict(status=request["status"],request_sha256=request_sha,
                    discrepancy_rows=1,discrepancy_pixels=1,model_forward_calls=0)),flush=True)
                return
    raise ValueError("Accepted discrepancy did not reproduce")


if __name__=="__main__":
    main()

