"""Freeze all 74 native SRNet probability maps and ordered stock masks without labels."""
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np


PILOT_CODE = "436a980f1428b2ed56797e5c1ac1f1b9edfddfaa5d453b5a24cae0111b64274f"
PILOT_RESULT = "d914fcb254a5dcf143c496ceeedfed43080cc3ad28903f2054eae3e4818be112"
PILOT_TERMINAL = "d17eb362c93613828a8f69291fab2009243005ad6b1802fdb488a87246354aa7"
BOX_TRANSFORMS = "899315d58f3004da58b639df9d369ce01de3e8ca33076100ed99a704ab6308d2"
STATE = None


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, expected):
    raw=path.read_bytes()
    require(digest(raw)==expected,"Bound SHA-256 mismatch: "+path.name)
    return raw


def load_pilot(source):
    path=source/"native_parity.py"
    raw=pin(path,PILOT_CODE)
    spec=spec_from_file_location("_srnet_pilot_native",str(path))
    module=module_from_spec(spec)
    sys.modules[spec.name]=module
    exec(compile(raw,str(path),"exec"),vars(module))
    return module


def candidate_record(rank, box, corners, native, p, codec):
    require(rank >= 1 and type(rank) is int and box.shape == (6,) and box.dtype == np.float32 and
            np.isfinite(box).all() and box[2]>0 and box[3]>0 and .3<=float(box[5])<=1 and
            corners.shape == (4,2) and np.isfinite(corners).all(),"Original centered RBox and score")
    require(native.shape == p.shape == (512,512) and native.dtype == np.bool_ and p.dtype == np.float32 and
            np.isfinite(p).all() and p.min()>=-1e-6 and p.max()<=1+1e-6 and np.array_equal(native,p>.5),
            "Same-call exact strict native threshold parity")
    clipped=np.clip(np.r_[corners.min(axis=0),corners.max(axis=0)],0,512)
    encoded=codec.encode(np.asfortranarray(native,dtype=np.uint8))
    require(encoded["size"]==[512,512] and np.array_equal(codec.decode(encoded).astype(bool),native),
            "Exact native COCO RLE"
    )
    return dict(rank=rank,category_id=1,score=float(box[5]),rbox_xywha=box[:5].tolist(),
                clipped_corner_xyxy=clipped.tolist(),
                rle=dict(size=encoded["size"],counts=encoded["counts"].decode("ascii")))


def initialize(args, image_index, image_root):
    import cv2
    import mmcv
    import torch

    global STATE
    pilot=load_pilot(args.pilot)
    sys.path.insert(0,str(args.source))
    import mmrotate
    from mmrotate.datasets import build_dataset
    from mmrotate.models import build_detector
    from pycocotools import mask as codec

    require(Path(mmrotate.__file__).resolve()==args.source/"mmrotate/__init__.py" and
            {k:version(k) for k in pilot.PACKAGES}==pilot.PACKAGES,
            "Original author import and qualified CPU runtime")
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(torch.get_num_threads()==torch.get_num_interop_threads()==1 and not torch.cuda.is_initialized(),
            "One CPU numerical thread per process")
    config=mmcv.Config.fromfile(str(args.source/pilot.CONFIG))
    require(dict(config.model.test_cfg)==pilot.STOCK_TEST and config.angle_version=="le90",
            "Exact SSDD trained deployment, score and NMS settings")
    settings=config.data.test.copy()
    settings.ann_file=str(image_index)
    settings.img_prefix=str(image_root)+os.sep
    settings.test_mode=True
    dataset=build_dataset(settings)
    require(dataset.CLASSES==("ship",) and dataset.cat_ids==[1] and len(dataset)==74,
            "Image-only original dataset with all reserve IDs")
    model=build_detector(config.model).cpu().eval()
    pilot.load_exact_checkpoint(model,args.fit/"RESULT/epoch_24.pth")
    head=import_module("mmrotate.models.dense_heads.SRNet_head")
    utils=import_module("mmrotate.models.dense_heads.SRNet_utils")
    require(head.get_mask is utils.get_mask,"Exact pinned original mask-paste entry")
    STATE=SimpleNamespace(pilot=pilot,torch=torch,codec=codec,dataset=dataset,model=model,head=head,utils=utils)


def observe_mask_projection(ctx, artifacts):
    original=ctx.head.get_mask
    torch=ctx.torch

    def tap(rois,labels,crops,angle_version,output_size,img_meta,thr,num_classes,rescale):
        original_masks=original(rois,labels,crops,angle_version,output_size,img_meta,thr,num_classes,rescale)
        require(angle_version=="le90" and output_size==28 and thr==.5 and num_classes==1 and rescale and
                crops.shape==(len(rois),1,28,28) and tuple(img_meta["pad_shape"][:2])==(608,608) and
                tuple(img_meta["ori_shape"][:2])==(512,512) and torch.all(labels==0) and
                torch.isfinite(crops).all() and torch.all((crops>=0)&(crops<=1)) and len(original_masks[0])==len(rois),
                "Same-call original cropped occupancy and native mask inputs")
        maps=np.empty((len(rois),512,512),dtype=np.float32)
        shape=tuple(img_meta["img_shape"][:2])
        for start in range(0,len(rois),16):
            end=min(start+16,len(rois))
            boxes=rois[start:end]
            native=ctx.pilot.project_native(boxes,crops[start:end],ctx.utils.obb2poly(boxes,angle_version)[:,:2],
                                           608,shape,(512,512))
            maps[start:end]=native.numpy()
        scaled=rois.detach().clone()
        scaled[:,:4]/=scaled.new_tensor(img_meta["scale_factor"])
        corners=ctx.utils.obb2poly(scaled,angle_version).reshape(len(rois),4,2).numpy()
        artifacts.append((scaled.numpy().copy(),corners,maps))
        return original_masks
    return tap


def capture_one(task):
    ordinal,row,output=task
    ctx=STATE
    torch=ctx.torch
    require(ctx.dataset.img_ids[ordinal-1]==row["image_id"],"Exact native image and order")
    prepared=ctx.dataset.prepare_test_img(ordinal-1)
    tensor,=prepared["img"]
    container,=prepared["img_metas"]
    if not torch.is_tensor(tensor):
        tensor=tensor.data
    metadata=container.data
    require(tuple(tensor.shape)==(3,608,608) and tensor.dtype==torch.float32 and tensor.device.type=="cpu" and
            tuple(metadata["ori_shape"][:2])==(512,512) and metadata["flip"] is False,
            "Original no-flip SSDD inference with native 512 canvas")
    captured=[]
    original=ctx.head.get_mask
    ctx.head.get_mask=observe_mask_projection(ctx,captured)
    try:
        with torch.inference_mode():
            per_image,=ctx.model.simple_test(tensor[None],[metadata],rescale=True,thr=.5)
    finally:
        ctx.head.get_mask=original
    boxes,native=per_image
    require(len(boxes)==len(native)==1 and len(boxes[0])==len(native[0]) and
            len(captured)==int(len(native[0])>0),"Original author order including the zero-candidate branch")
    if captured:
        scaled,corners,probabilities=captured[0]
        require(np.array_equal(scaled,boxes[0]),"Exact postprocessed score, order and native box coordinate parity")
        candidates=[candidate_record(rank,box,corner,mask,p,ctx.codec)
                    for rank,(box,corner,mask,p) in enumerate(zip(scaled,corners,native[0],probabilities,strict=True),1)]
    else:
        candidates=[]
        probabilities=np.empty((0,512,512),dtype=np.float32)
    packed=BytesIO()
    np.savez_compressed(packed,probability=probabilities)
    written=packed.getvalue()
    with np.load(BytesIO(written),allow_pickle=False) as artifact:
        require(artifact.files==["probability"],"Exactly one native probability array")
        unpacked=artifact["probability"]
        require(unpacked.dtype==np.float32 and
                np.array_equal(unpacked.view(np.uint32),probabilities.view(np.uint32)),
                "NPZ roundtrip retains every binary32 probability bit")
    relative=f"PROBABILITIES_PRIVATE/{ordinal:03d}.npz"
    with (output/relative).open("xb") as stream:
        stream.write(written)
    record=dict(ordinal=ordinal,image_id=row["image_id"],source_image_name=row["source_image_name"],
                acquisition_group=row["acquisition_group"],sensor=row["sensor"],original_hw=[512,512],
                image_sha256=row["image_ref"]["sha256"],rgb_sha256=row["rgb_sha256"],
                probability_file=dict(path=relative,bytes=len(written),sha256=digest(written)),
                candidates=candidates)
    nonempty=sum(bool(np.any(mask)) for mask in native[0])
    counts=dict(images=1,candidates=len(candidates),no_candidates=int(not candidates),empty_masks=len(candidates)-nonempty,
                nonempty_masks=nonempty,exact_projected_masks=len(candidates),native_probability_positions=probabilities.size)
    return record,counts


def run(args):
    started=perf_counter()
    require(sys.flags.isolated and sys.dont_write_bytecode and not args.output.exists() and
            sys.version_info[:3]==(3,10,13) and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1" and
            all(os.environ.get(k)=="1" for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")),
            "Fresh isolated single-thread-library CPU capture")
    pin(args.protocol,args.protocol_sha256)
    pilot=load_pilot(args.pilot)
    term=json.loads(pin(args.pilot/"observed_pilot/TERMINAL.json",PILOT_TERMINAL))
    receipt=json.loads(pin(args.pilot/"actual/RESULT_SAFE.json",PILOT_RESULT))
    require(term["actual_exit"]==0 and term["phase"]=="verify" and term["output_footer_verified"] is True and
            term["result_sha256"]==PILOT_RESULT and term["worker_sha256"]==receipt["worker_sha256"]==PILOT_CODE and
            receipt["status"]=="SRNET_SSDD_E24_RSDD74_NATIVE_PARITY_PILOT_COMPLETE" and
            receipt["intake_image_manifest_sha256"]==pilot.MANIFEST and receipt["weights_sha256"]==pilot.CHECKPOINT and
            receipt["all_author_order_scores_native_rle_exact"] is receipt["all_independent_projection_masks_exact"] is True and
            receipt["reference_files_opened"]==0 and receipt["stock_forwards"]==receipt["tapped_forwards"]==8,
            "Exact closed two-pass native-coordinate proof")
    pilot.admit(args)
    for name, expected in pilot.PINS.items():
        pin(args.source/name,expected)
    pin(args.source/"mmrotate/core/bbox/transforms.py",BOX_TRANSFORMS)
    manifest=json.loads(pin(args.intake/"intake/IMAGE_MANIFEST_PRIVATE.json",pilot.MANIFEST))
    selection=manifest["items"]
    image_root, whole_index=pilot.bind_images(args.intake,selection)
    from PIL import Image, ImageOps
    for item in selection:
        with Image.open(image_root/item["source_image_name"]) as image:
            rgb=ImageOps.exif_transpose(image).convert("RGB")
            require(rgb.size==(512,512) and digest(rgb.tobytes())==item["rgb_sha256"],
                    "Unchanged original JPEG and native decoded RGB identity")
    args.output.mkdir()
    (args.output/"PROBABILITIES_PRIVATE").mkdir()
    image_index=args.output/"IMAGE_ONLY_COCO_PRIVATE.json"
    index_hash=pilot.write_json(image_index,dict(images=whole_index,categories=[dict(id=1,name="ship")],annotations=[]))
    print(json.dumps(dict(stage="ADMITTED_CLOSED_74_IMAGE_ONLY_ROWS")),flush=True)
    index_sha=sha256()
    counts=Counter()
    sensors={name:Counter() for name in pilot.SENSORS}
    tasks=((i,row,args.output) for i,row in enumerate(selection,1))
    with (args.output/"INDEX_PRIVATE.jsonl").open("xb") as output, ProcessPoolExecutor(
            max_workers=args.workers,initializer=initialize,initargs=(args,image_index,image_root)) as pool:
        for row, stats in pool.map(capture_one,tasks,chunksize=1):
            require(row["ordinal"]==counts["images"]+1,"Preserved manifest ordinal")
            encoded=(json.dumps(row,sort_keys=True,ensure_ascii=True,separators=(",",":"),allow_nan=False)+"\n").encode()
            output.write(encoded)
            index_sha.update(encoded)
            counts.update(stats)
            sensors[row["sensor"]].update(stats)
            if counts["images"]%10==0:
                print(json.dumps(dict(stage="NATIVE_CAPTURED",images=counts["images"])),flush=True)
    require(counts["images"]==74 and [sensors[s]["images"] for s in pilot.SENSORS]==[42,32] and
            counts["candidates"]==counts["exact_projected_masks"] and
            counts["native_probability_positions"]==counts["candidates"]*512*512,
            "Complete unfiltered source matrix, native shape and probability parity")
    result=dict(status="SRNET_SSDD_E24_RSDD74_NATIVE_CAPTURE_COMPLETE",phase="capture",
        worker_sha256=digest(Path(__file__).read_bytes()),protocol_sha256=args.protocol_sha256,
        weights_sha256=pilot.CHECKPOINT,author_commit=pilot.AUTHOR_COMMIT,pilot_result_sha256=PILOT_RESULT,
        pilot_terminal_sha256=PILOT_TERMINAL,pilot_worker_sha256=PILOT_CODE,
        intake_image_manifest_sha256=pilot.MANIFEST,intake_result_sha256=pilot.INTAKE_RESULT,
        intake_terminal_sha256=pilot.INTAKE_TERMINAL,source_fit_result_sha256=pilot.FIT_RESULT,
        source_fit_terminal_sha256=pilot.FIT_TERMINAL,author_box_transforms_sha256=BOX_TRANSFORMS,
        index_sha256=index_sha.hexdigest(),private_image_only_index_sha256=index_hash,images=74,
        origin="project PSeg-SSDD FIT741 E24 refit",official_RSDD_checkpoint=False,
        native_raster_shape=[512,512],network_shape=[608,608],angle_version="le90",test_settings=pilot.STOCK_TEST,
        runtime_versions=pilot.PACKAGES,cpu_worker_processes=args.workers,torch_threads_per_process=1,
        whole_image_probability_dtype="float32",probability_npz_files=counts["images"],
        threshold_operator=">",mask_threshold=.5,all_native_rle_and_probability_masks_exact=True,
        all_original_rank_score_and_rotated_box_same_call_exact=True,all_archive_jpeg_and_rgb_exact=True,
        no_image_or_candidate_filtering=True,reference_files_opened=0,model_forwards=74,optimizer_updates=0,
        model_selected=False,threshold_selected=False,pixels_edited=0,AP=False,ground_truth_opened=False,
        counts=dict(counts),by_sensor=sensors,elapsed_seconds=perf_counter()-started)
    result_sha=pilot.write_json(args.output/"RESULT_SAFE.json",result)
    print(json.dumps(dict(status=result["status"],result_sha256=result_sha)),flush=True)


def main():
    parser=ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("source","fit","intake","pilot","protocol","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--protocol-sha256",required=True)
    parser.add_argument("--workers",type=int,choices=range(1,5),required=True)
    args=parser.parse_args()
    for name in ("source","fit","intake","pilot","protocol","output"):
        setattr(args,name,getattr(args,name).resolve())
    run(args)


if __name__=="__main__":
    main()
