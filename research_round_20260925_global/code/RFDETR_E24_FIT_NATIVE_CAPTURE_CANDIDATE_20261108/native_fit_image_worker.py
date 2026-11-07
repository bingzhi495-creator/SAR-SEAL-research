"""Image-only real FIT capture for a pinned closed RF-DETR Nano E24 archive."""
from dataclasses import replace
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time
import types


def need(ok, message):
    if not ok:
        raise ValueError(message)


def exact_file(ref, maximum):
    path=Path(ref["path"])
    need(path.is_absolute() and path.is_file() and not path.is_symlink()
         and path.resolve(strict=True)==path and 0<ref["bytes"]<=maximum, "Exact original input path")
    return path


def exact_bytes(ref, maximum):
    path=exact_file(ref,maximum)
    with path.open("rb") as stream:
        before=os.fstat(stream.fileno())
        raw=stream.read(ref["bytes"]+1)
        need(len(raw)==before.st_size==ref["bytes"] and hashlib.sha256(raw).hexdigest()==ref["sha256"],
             "Exact input bytes")
        unchanged(stream,path,before)
    return raw


def unchanged(stream,path,before):
    key=lambda st:(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
    need(key(before)==key(os.fstat(stream.fileno()))==key(path.stat()),"Input changed during use")


def module(ref,alias):
    mod=types.ModuleType(alias)
    mod.__file__=ref["path"]
    sys.modules[alias]=mod
    exec(compile(exact_bytes(ref,1024**2),mod.__file__,"exec"),mod.__dict__)
    return mod


def json_out(path,value):
    raw=(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
    with path.open("xb") as stream:
        need(stream.write(raw)==len(raw),"Complete exclusive output")
        stream.flush()
        os.fsync(stream.fileno())


def run(spec_path,expected_sha):
    started=time.monotonic()
    raw=spec_path.read_bytes()
    need(hashlib.sha256(raw).hexdigest()==expected_sha,"Exact original capture spec")
    spec=json.loads(raw)
    need(spec["schema"]=="Root-RFDETR-E24-one-image-FIT-capture-v1" and spec["GT"] is spec["DEV"] is spec["TEST"] is False
         and os.environ.get("HF_HUB_OFFLINE")=="1" and os.environ.get("TRANSFORMERS_OFFLINE")=="1",
         "FIT-only offline image scope")
    def offline(event,args):
        if event in ("socket.connect","socket.getaddrinfo"):
            raise RuntimeError("Offline RF-DETR FIT capture")
    sys.addaudithook(offline)

    import numpy as np
    import torch
    from pycocotools import mask as codec
    from rfdetr import config
    from rfdetr.datasets.coco import make_coco_transforms_square_div_64
    from rfdetr.datasets.io_utils import decode_pil_image_bytes
    from rfdetr.models._defaults import MODEL_DEFAULTS
    from rfdetr.models.lwdetr import build_model_from_config
    from rfdetr.models.postprocess import PostProcess, _MASK_CHUNK
    from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

    names=("core","sparse","localization","pilot","archive","boundary","configuration","e24_export")
    need(set(spec["sources"])==set(names),"Exact reusable source roles")
    mods={name:module(spec["sources"][name],"_fit_native_"+name) for name in names}
    pilot=mods["pilot"]
    need(all(version(name)==expected for name,expected in pilot.VERSIONS.items()),
         "Exact RF-DETR and frozen numerical library versions")
    device=torch.device(spec["device"])
    need(device.type in ("cpu","cuda") and not torch.is_autocast_enabled("cpu")
         and not torch.is_autocast_enabled("cuda"),"Admitted eager FP32 device")
    if device.type=="cpu":
        need(not torch.cuda.is_available() and not torch.cuda.is_initialized(),"GPU-hidden CPU qualification")
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
    else:
        need(torch.cuda.is_available() and torch.cuda.device_count()==1
             and torch.cuda.mem_get_info(device)[0]>=4*1024**3,"Guarded isolated CUDA capture")
    q=spec["archive_ref"]
    path=exact_file(q,1024**3)
    need(path.name=="checkpoint_23.ckpt","Original accepted E24 endpoint name")
    with path.open("rb") as stream:
        before=os.fstat(stream.fileno())
        digest=hashlib.sha256()
        for block in iter(lambda:stream.read(1024**2),b""):
            digest.update(block)
        need(before.st_size==q["bytes"] and digest.hexdigest()==q["sha256"],
             "Full frozen E24 original archive bytes")
        stream.seek(0)
        archive=torch.load(stream,map_location="cpu",weights_only=True)
        unchanged(stream,path,before)
    mods["archive"].validate_archive_progress(archive,24,boundary_api=mods["boundary"],
                                              recipe_sha256=spec["recipe_sha256"])
    recipe=exact_file(spec["recipe"],1024**2)
    exact_bytes(spec["recipe"],1024**2)
    m,t=mods["configuration"]._load_configs(recipe,spec_path.parent)
    pilot.match_model_config(archive,m,t,config)
    detector=build_model_from_config(m,t,defaults=replace(MODEL_DEFAULTS,force_no_pretrain=True))
    ema=mods["e24_export"].strict_ema_load(detector,archive,torch)
    del archive
    detector=detector.to(device).eval()
    need(detector.num_queries==100 and detector.group_detr==13 and not detector.training,"Frozen native Nano eval carrier")
    image_raw=exact_bytes(spec["image_ref"],16*1024**2)
    hw=tuple(spec["original_hw"])
    image,scale=decode_pil_image_bytes(image_raw,draft_size=None)
    need(image.mode=="RGB" and image.size==(hw[1],hw[0]) and scale==(1.,1.),"Exact original-size native JPEG decode")
    gray=np.ascontiguousarray(image.convert("L"),dtype=np.uint8)
    transform=make_coco_transforms_square_div_64(
        "val",576,multi_scale=False,expanded_scales=False,skip_random_resize=True,
        patch_size=m.patch_size,num_windows=m.num_windows,aug_config=None,
        scale_jitter=t.scale_jitter,gpu_postprocess=False)
    tensor,target=transform(image,None)
    need(target is None and tuple(tensor.shape)==(3,576,576) and tensor.dtype==torch.float32
         and bool(torch.isfinite(tensor).all().item()),"Exact 576 image-only native tensor")
    processor=mods["core"].make_native_top100(PostProcess)
    with torch.inference_mode():
        samples=nested_tensor_from_tensor_list([tensor],block_size=m.patch_size*m.num_windows).to(device)
        need(tuple(samples.tensors.shape)==(1,3,576,576) and samples.mask is not None
             and not bool(samples.mask.any().item()),"No unexpected padding")
        raw=detector(samples)
        rows=mods["core"].iter_native_candidates(raw,hw,processor=processor,torch_api=torch,
            interpolate=torch.nn.functional.interpolate,mask_chunk_size=_MASK_CHUNK)
        artifact=mods["sparse"].pack_candidates(rows,hw,np=np,codec=codec,dilate=mods["localization"].dilate_chebyshev)
    out=spec_path.parent
    with (out/"SPARSE.npz").open("xb") as f:
        np.savez(f,offsets=artifact["offsets"],positions=artifact["positions"],logits=artifact["logits"])
        f.flush()
        os.fsync(f.fileno())
    json_out(out/"ROWS.json",dict(image_id=spec["image_id"],image_sha256=spec["image_ref"]["sha256"],original_hw=list(hw),
             gray_sha256=hashlib.sha256(gray.tobytes()).hexdigest(),records=artifact["records"]))
    json_out(out/"RESULT.json",dict(status="ONE_ORIGINAL_FIT_IMAGE_CAPTURED_PENDING_ROOT",device=device.type,
             checkpoint_sha256=q["sha256"],images=1,retained=len(artifact["records"]),stock_native_rows=100,
             support_pixels=int(artifact["positions"].size),ema_tensors=ema["tensors"],EMA_updates=4464,
             mask_logit_sign_exact=True,raw_hw=[144,144],image_only=True,GT=False,AP=False,DEV=False,TEST=False,
             training=False,torch_CUDA_initialized=torch.cuda.is_initialized(),elapsed_seconds=time.monotonic()-started))


if __name__=="__main__":
    need(len(sys.argv)==3,"Exact capture spec and SHA")
    run(Path(sys.argv[1]),sys.argv[2])
