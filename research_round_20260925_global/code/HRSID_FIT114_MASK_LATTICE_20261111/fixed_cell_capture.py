"""Read one frozen 144x144 logit plane for each preselected FIT query, without GT."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from zipfile import ZipFile

import numpy as np


P326_SHA = "193927ae8b3684f557bb2d1dbebfd4c7635680e26cd97c673b48ed531be41945"
P327_SHA = "b835b8d6634ba9cd9a4e6327645e76277944d814072593bb0822ce1ed7b9a8e9"
P326_SAFE = "71664d11f32a2e861bd98fe2cf885b44280e867f015aa682e459af9a0ef7e01a"
P326_WAIT = "b8febbda660d0f731c3182ae17c2d2d329022e00dc7adb5d4ef461770e514840"
CELLS = 8
STATUS = "HRSID_FIXED_QUERY_CELL_DENSE_FIT_CAPTURE_ACTUAL"


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def route_cells(plane,limit):
    need(plane.ndim == 2 and min(plane.shape)>=2 and np.isfinite(plane).all(),"Finite coarse logit grid required")
    corners=np.stack((plane[:-1,:-1],plane[1:,:-1],plane[:-1,1:],plane[1:,1:])).astype(np.float64)
    yy,xx=np.nonzero((corners.min(axis=0)<=0)&(corners.max(axis=0)>0))
    risk=np.abs(corners.mean(axis=0)[yy,xx])
    ids=np.lexsort((xx,yy,risk))[:limit]
    return [[int(yy[i]),int(xx[i])] for i in ids]


def pinned(path,pin,name):
    raw=path.read_bytes()
    need(sha256(raw).hexdigest()==pin,"Exact previously closed source or receipt required")
    m=ModuleType(name)
    m.__file__=str(path)
    sys.modules[name]=m
    exec(compile(raw,str(path),"exec"),m.__dict__)
    return m


def run(a):
    import torch
    from pycocotools import mask as codec
    from rfdetr.datasets.coco import make_coco_transforms_square_div_64
    from rfdetr.datasets.io_utils import decode_pil_image_bytes
    from rfdetr.models.postprocess import PostProcess
    from rfdetr.utilities.tensors import nested_tensor_from_tensor_list
    from torch.nn.functional import interpolate

    p326=pinned(a.code_root/"HRSID_FIT114_TILED_BAND_20261111/tiled_band.py",P326_SHA,"_p362_p326")
    p327=pinned(a.code_root/"HRSID_FIT114_TILED_BAND_20261111/tile_score.py",P327_SHA,"_p362_p327")
    p316,p318,p314=(p326.load(a.code_root,key) for key in ("p316","p318","p314"))
    a.p314=p314
    p192,cohort_bytes,_,_=p314.read_original(a.code_root,a.original)
    reader=p318.load_scope(a.code_root)
    closed=p318.read_closed_capture(SimpleNamespace(p316_report_sha256=p318.P316_REPORT_SHA,
        p316_terminal_sha256=p318.P316_TERMINAL_SHA,p316_root=a.p316_root,p316_terminal=a.p316_terminal),reader)
    terminal=a.p326_terminal.read_bytes()
    need(sha256(terminal).hexdigest()==P326_WAIT,"Exact closed original P326 wait required")
    wait=json.loads(terminal)
    need(wait["original_os_wait"] is True and wait["original_os_exit"]==0 and wait["worker_closed"] is True,
         "Previous tile capture did not close")
    _,prior_tiles=p327.read_tile_inputs(a.p326_root/"CAPTURE_SAFE.json",P326_SAFE,
        a.p326_root/"TILED_BAND_PRIVATE.jsonl",closed["candidate_stream_sha256"])
    roster=json.loads(cohort_bytes)["images"]
    stream_path=a.p316_root/"CANDIDATES_PRIVATE.jsonl"
    with stream_path.open("rb") as stream:
        identity=p318.verify_stream(stream_path,stream,closed["candidate_stream_bytes"],closed["candidate_stream_sha256"])
        chosen,_=p326.choose(roster,stream,codec)
        p318.require_unchanged(stream_path,stream,identity)
    selection=[[r["image_id"],c["query_id"],c["native_rank"],crop[0],crop[1]] for r,_,c,_,crop,_ in chosen]
    raw_selection=json.dumps(selection,separators=(",",":")).encode()
    need(p326.EXPECTED_SELECTION_SHA in (sha256(raw_selection).hexdigest(),sha256(raw_selection+b"\n").hexdigest()),
         "Exact prior labels-free selection required")

    base,filtered,shift=p316.source_model(a)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    model,config,train,_=base.load_model(a.checkpoint,p314.CHECKPOINT_SHA,torch,a.output.parent,filtered.TRAINING_PROTOCOL)
    model=model.cpu().eval()
    need(not model.training and model.num_queries==164 and model.group_detr==13,"Exact frozen Q164 carrier required")
    transform=make_coco_transforms_square_div_64("val",576,multi_scale=False,expanded_scales=False,
        skip_random_resize=True,patch_size=config.patch_size,num_windows=config.num_windows,aug_config=None,
        scale_jitter=train.scale_jitter,gpu_postprocess=False)
    manifest=[]
    dense_hash=sha256()
    n_rows=n_band=n_cells=0
    binary=a.output.parent/"DENSE_LOGITS_PRIVATE.bin"
    with a.bundle.open("rb") as input_stream:
        identity=p318.verify_stream(a.bundle,input_stream,151_320_868,reader.BUNDLE_SHA)
        with ZipFile(input_stream) as archive,binary.open("xb") as output,torch.inference_mode():
            need(reader.member(archive,"COHORT_PRIVATE.json",reader.COHORT_SHA,p326.MIB)==cohort_bytes,
                 "Exact original image-only roster")
            for (row,original,candidate,pred,crop,yx),prior in zip(chosen,prior_tiles,strict=True):
                image=p316.image_rows(archive,row,shift,np,decode_pil_image_bytes)
                tensor,target=transform(image,None)
                need(target is None and tensor.shape==(3,576,576) and tensor.dtype==torch.float32,"Exact stock val576 transform")
                sample=nested_tensor_from_tensor_list([tensor],block_size=config.patch_size*config.num_windows)
                need(sample.mask is not None and not sample.mask.any(),"No padding on fixed stock image")
                raw=model(sample)
                observed,_=p316.observed_rows(raw,image_id=row["image_id"],encoded_sha=row["encoded_sha256"],
                    torch=torch,np=np,codec=codec,base=base,postprocess=PostProcess)
                need([{k:v for k,v in r.items() if k!="points"} for r in observed]==original,
                     "Full native scores, query order and masks must exactly replay P316")
                n_rows+=len(observed)
                plane=raw["pred_masks"][0,candidate["query_id"]]
                need(plane.shape==(144,144) and plane.dtype==torch.float32 and bool(torch.isfinite(plane).all()),
                     "Exact raw finite 144 logit plane required")
                dense=interpolate(plane[None,None],size=(800,800),mode="bilinear",align_corners=False)[0,0].numpy()
                need(np.array_equal(dense>0,pred) and prior["candidate"]==candidate
                     and prior["image_id"]==row["image_id"] and prior["crop_xyxy"]==list(crop),
                     "Native rendered mask or original tile identity differs")
                y,x=yx.T
                need(np.array_equal(np.asarray([(i["y"],i["x"]) for i in prior["points"]]),yx)
                     and dense[y,x].tolist()==[i["full"] for i in prior["points"]],
                     "Original archived tile-band logit values differ")
                n_band+=len(yx)
                values=np.ascontiguousarray(plane.numpy(),dtype="<f4")
                cells=route_cells(values,CELLS)
                payload=values.tobytes()
                need(output.write(payload)==len(payload),"Incomplete dense plane write")
                dense_hash.update(payload)
                manifest.append(dict(image_id=row["image_id"],candidate=candidate,cells=cells,
                                     plane_sha256=sha256(payload).hexdigest()))
                n_cells+=len(cells)
                if len(manifest)%4==0:
                    print(json.dumps(dict(event="DENSE_FIXED_FIT_IMAGES_DONE",images=len(manifest))),flush=True)
            output.flush()
            os.fsync(output.fileno())
        p318.require_unchanged(a.bundle,input_stream,identity)
    need(len(manifest)==16 and n_rows==1391 and n_band==9429 and binary.stat().st_size==1327104
         and not torch.cuda.is_initialized(),"Expected exact 16 forward and native identity scope")
    rows=p192.encoded(manifest)
    with (a.output.parent/"ROUTE_PRIVATE.json").open("xb") as f:
        f.write(rows)
    report=dict(status=STATUS,FIT=True,DEV=False,TEST=False,GT_opened=False,AP=False,
                model_forward_calls=16,optimizer_updates=0,images=16,candidate_rows_reproduced=n_rows,
                prior_fp32_band_points_reproduced=n_band,selected_native_masks_reproduced=16,
                cells_per_candidate_limit=CELLS,total_routed_cells=n_cells,routing="mixed-sign corners then smallest absolute corner mean",
                dense_grid_sha256=dense_hash.hexdigest(),dense_grid_bytes=binary.stat().st_size,
                route_sha256=sha256(rows).hexdigest(),p326_selection_sha256=p326.EXPECTED_SELECTION_SHA,
                checkpoint_sha256=p314.CHECKPOINT_SHA)
    out=p192.encoded(report)
    with a.output.open("xb") as f:
        f.write(out)
    print(json.dumps(dict(event=STATUS,sha256=sha256(out).hexdigest())),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("code-root","original","bundle","acceptance","checkpoint","p316-root","p316-terminal",
                 "p326-root","p326-terminal","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="-1" and os.environ.get("OMP_NUM_THREADS")=="1",
         "Fixed CPU-only FIT capture required")
    def offline(event,_):
        if event in ("socket.connect","socket.getaddrinfo"):
            raise PermissionError("Offline frozen capture only")
    sys.addaudithook(offline)
    run(args)


if __name__=="__main__":
    main()
