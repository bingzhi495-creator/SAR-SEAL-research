"""Render one accepted raw RF mask plane with the exact native CUDA operators."""
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
    need(path.is_absolute() and path.resolve(strict=True)==path and 0<path.stat().st_size<=limit,"Exact closed input")
    raw=path.read_bytes()
    need(sha256(raw).hexdigest()==digest,"Exact frozen source digest")
    return raw


def write(path,data):
    with path.open("xb") as stream:
        need(stream.write(data)==len(data),"Complete private CUDA artifact")
        stream.flush()
        os.fsync(stream.fileno())
    return sha256(data).hexdigest()


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for name in ("request","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--request-sha",required=True)
    a=p.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and not a.output.exists(),"Original isolated fresh renderer")
    request=json.loads(pin(a.request,a.request_sha,1024**2))
    need(request["status"]=="RF_CENTER5202_DEV705_SINGLE_RENDER_ROW_LOCATED_V1"
         and request["capture_acceptance_sha256"]=="77ef95b694f74324229507ee79dea015323acff3b716052c3aefa4438105e844"
         and request["cpu_disagreement_rows"]==request["cpu_disagreement_pixels"]==1
         and request["original_hw"]==[800,800]
         and request["GT_opened"] is request["FIT"] is request["TEST"] is request["AP"] is False,
         "Exact accepted single native render discrepancy")
    accepted=json.loads(pin(Path(request["source_acceptance_ref"]["path"]),request["capture_acceptance_sha256"],1024**2))
    need(accepted["queries_sha256"]==request["queries_sha256"]
         and accepted["capture_result_sha256"]==request["capture_result_sha256"]
         and accepted["independent_CPU_raster_disagree_rows"]==accepted["independent_CPU_raster_disagree_pixels"]==1,
         "Independent complete carrier acceptance")
    import numpy as np
    import torch
    from pycocotools import mask as codec
    need(sys.version_info[:3]==(3,10,13) and torch.__version__=="2.10.0+cu130" and np.__version__=="1.26.4"
         and torch.cuda.is_available() and torch.cuda.device_count()==1,"Original pinned native CUDA runtime")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(min(.35,2*1024**3/torch.cuda.mem_get_info()[1]))
    ref=request["single_raw_plane_ref"]
    plane=np.load(BytesIO(pin(Path(ref["path"]),ref["sha256"],1024**2)),allow_pickle=False)
    source=request["source_logit_ref"]
    original=np.load(BytesIO(pin(Path(source["path"]),source["sha256"],9*1024**2)),allow_pickle=False)
    need(plane.dtype==np.dtype("<f4") and plane.shape==(144,144) and np.isfinite(plane).all()
         and plane.tobytes(order="C")==original[request["logit_row"]].tobytes(order="C")
         and sha256(plane.tobytes(order="C")).hexdigest()==request["source_raw_plane_sha256"],
         "Only the exact single original raw plane enters CUDA")
    need(sha256(encoded(request["stock_rle"])).hexdigest()==request["stock_rle_sha256"],
         "Exact original compressed native mask")
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        tensor=torch.from_numpy(plane[None,None]).to("cuda")
        native=torch.nn.functional.interpolate(tensor,size=(800,800),mode="bilinear",align_corners=False)[0,0]
        sign=(native>0).to(torch.uint8).cpu().numpy()
        old=codec.decode(request["stock_rle"])
        need(np.array_equal(sign,old),"Exact CUDA strict raw-logit sign and original stock RLE")
        probability=native.sigmoid().cpu().numpy().astype("<f4",copy=False)
        native_cpu=native.cpu().numpy()
    need(probability.shape==(800,800) and bool(np.isfinite(probability).all())
         and probability.min()>=0 and probability.max()<=1,"Native FP32 sigmoid probabilities")
    a.output.mkdir()
    data=BytesIO()
    np.save(data,probability,allow_pickle=False)
    path=a.output/"NATIVE_PROBABILITY_PRIVATE.npy"
    file_sha=write(path,data.getvalue())
    result=dict(request,status="RF_SINGLE_NATIVE_CUDA_PROBABILITY_RENDER_COMPLETE_V1",
        request_sha256=a.request_sha,native_probability_ref=dict(path=str(path.resolve()),bytes=len(data.getvalue()),
            sha256=file_sha,shape=[800,800],dtype="<f4"),
        native_probability_tensor_sha256=sha256(probability.tobytes(order="C")).hexdigest(),
        strict_cuda_logit_sign_tensor_sha256=sha256(sign.tobytes(order="C")).hexdigest(),
        strict_cuda_logit_sign_RLE_exact=True,
        sigmoid_half_ties_positive_logit_pixels=int(((probability==.5)&(native_cpu>0)).sum()),
        sigmoid_half_ties_negative_logit_pixels=int(((probability==.5)&(native_cpu<0)).sum()),
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        renderer_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),raw_planes_on_cuda=1,
        model_forward_calls=0,GT_opened=False,FIT=False,DEV=True,TEST=False,AP=False)
    result_sha=write(a.output/"RENDER_RESULT_PRIVATE.json",encoded(result))
    print(json.dumps(dict(status=result["status"],result_sha256=result_sha)),flush=True)


if __name__=="__main__":
    main()

