"""Observe native-grid probability-to-mask parity on a frozen source-trained SRNet."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import torch
import torch.nn.functional as F


AUTHOR_COMMIT = "2e56c7208f217ff1824f1d4f3f3ea242a5f78bc6"
CONFIG = "configs/SRNet/SRNet_r50_fpn_3x_SSDD_le90.py"
PINS = {
    CONFIG: "222ba188f9476e822d1c9a1f4c5c15404bbd5a029bd599aeae41a9dafbbaa356",
    "configs/_base_/datasets/SSDD.py": "fd7ab145b360d64324f89b0cb10a8ad11ff3746a5a5f4521eb30a49d46b23c63",
    "configs/_base_/default_runtime.py": "c65c120d8543f4493a3e33aaf6f8fda5220c412ab20097224f11ab9391ee772c",
    "mmrotate/models/dense_heads/SRNet_utils.py": "7d9210f7619fa31ad4bfacfc30e380cb1b4fce5d77e7217de15c4c14b5632c7f",
    "mmrotate/models/dense_heads/SRNet_head.py": "144552cf27b5970c46ff4e3af1b4e0f59b1b273daa0aadf89cdcc513c8e2dba1",
    "mmrotate/models/detectors/SRNet.py": "7fad564a8c2fb6f34f9e7dc197c7cafc643fe72b16817846fe173e52fe74a40a",
}
PACKAGES = {"torch":"1.12.0+cpu", "torchvision":"0.13.0+cpu", "mmcv-full":"1.6.0",
            "mmdet":"2.28.2", "numpy":"1.26.4", "opencv-python":"4.11.0.86",
            "pycocotools":"2.0.11", "Pillow":"9.5.0", "e2cnn":"0.2.3"}
STOCK_TEST = dict(nms_pre=2000,min_bbox_size=0,score_thr=.3,nms=dict(iou_thr=.1),max_per_img=2000)
INTAKE_RESULT = "da9d92ea16e74f988d46d9855d26b530111b4c5e650fef513895c5d2e4441065"
INTAKE_TERMINAL = "b97ba047bf9422346f210b2a513be606cdc648cccdc0e8545affdd510f9ffaeb"
MANIFEST = "154d0d748b96df0d56ac249adcb8b6ae63275a09dfd7a0d7480839005e044338"
ROSTER = "2614e567e9ef1b2c791c1b357b03adfe7302ece53e911c1963c40e623ad405fc"
FIT_RESULT = "6f8551a15b28d0184e76199cec415b46bcb7a688eaff72640ba6347ccd9ea447"
FIT_TERMINAL = "4153fcf97ea9093ce0b657da0b18838262e202ac54f10d34dff793abeb071a91"
CHECKPOINT = "8998d0f6c11082250a46af48ec0c5455fb7606de8b6bb6a00ad2956046ea95ea"
CHECKPOINT_BYTES = 364691693
SENSORS = ("GF-3", "TerraSAR-X")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, expected):
    raw = path.read_bytes()
    require(digest(raw) == expected, "Pinned bytes differ: " + path.name)
    return raw


def write_json(path, data):
    raw = (json.dumps(data, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)+"\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(raw)
    return digest(raw)


def admit(args):
    terminal = json.loads(pin(args.intake/"observed_intake/TERMINAL.json", INTAKE_TERMINAL))
    intake = json.loads(pin(args.intake/"intake/RESULT_SAFE.json", INTAKE_RESULT))
    require(terminal["phase"] == "capture" and terminal["actual_exit"] == 0 and
            terminal["output_footer_verified"] is True and terminal["result_sha256"] == INTAKE_RESULT and
            terminal["worker_sha256"] == intake["worker_sha256"] and intake["images"] == 74 and
            intake["image_manifest_sha256"] == MANIFEST and intake["roster_sha256"] == ROSTER and
            intake["target_polygon_rows_opened"] == intake["optimizer_updates"] == intake["model_forwards"] == 0,
            "Exact original naturally closed label-free reserve intake")
    terminal = json.loads(pin(args.fit/"ORIGINAL_WAIT_PRIVATE.json", FIT_TERMINAL))
    fit = json.loads(pin(args.fit/"RESULT/RESULT_SAFE.json", FIT_RESULT))
    require(terminal["exit_code"] == 0 and fit["status"] == "P346_SRNET_AUTHOR_BATCH4_MATCHED_TRAIN_COMPLETE_V1" and
            fit["FIT"] is True and fit["DEV"] is fit["TEST"] is fit["AP"] is False and
            fit["author_commit"] == AUTHOR_COMMIT and fit["completed_epochs"] == 24 and
            fit["optimizer_updates"] == 4464 and fit["consumed_visits"] == 17856 and
            fit["checkpoint"]["bytes"] == CHECKPOINT_BYTES and fit["checkpoint"]["sha256"] == CHECKPOINT,
            "Original closed project SSDD FIT-only training")
    manifest = json.loads(pin(args.intake/"intake/IMAGE_MANIFEST_PRIVATE.json", MANIFEST))
    items = manifest["items"]
    require(len(items) == 74 and [r["image_id"] for r in items] == manifest["expected_ids"] and
            len(set(manifest["expected_ids"])) == 74 and
            Counter(r["sensor"] for r in items) == Counter({"GF-3":42, "TerraSAR-X":32}),
            "Exact historical source-only image census")
    selected = []
    for sensor in SENSORS:
        candidates = [r for r in items if r["sensor"] == sensor]
        groups = sorted({r["acquisition_group"] for r in candidates})[:4]
        selected.extend(next(r for r in candidates if r["acquisition_group"] == group) for group in groups)
    require(len(selected) == len({r["acquisition_group"] for r in selected}) == 8,
            "Prespecified distinct-parent balanced roster")
    return selected


def bind_images(intake, selected):
    image_root = (intake/"intake/images").resolve(strict=True)
    records = []
    for row in selected:
        reference = row["image_ref"]
        name = Path(row["source_image_name"])
        relative = Path(reference["path"])
        require(not name.is_absolute() and len(name.parts) == 1 and name.suffix.lower() == ".jpg" and
                relative == Path("images")/name and row["original_hw"] == [512,512],
                "Bound private JPEG-only row and native dimensions")
        path = (image_root/name).resolve(strict=True)
        require(path.parent == image_root and path.stat().st_size == reference["bytes"],
                "Local JPEG stays in the closed image directory")
        pin(path, reference["sha256"])
        records.append(dict(id=row["image_id"],file_name=name.name,height=512,width=512))
    return image_root, records


def load_exact_checkpoint(model, path):
    with path.open("rb") as stream:
        first = os.fstat(stream.fileno())
        h = sha256()
        for block in iter(lambda: stream.read(1<<20), b""):
            h.update(block)
        require(first.st_size == CHECKPOINT_BYTES and h.hexdigest() == CHECKPOINT,
                "Original trusted closed local E24 checkpoint")
        stream.seek(0)
        artifact = torch.load(stream, map_location="cpu")
        last = os.fstat(stream.fileno())
        require((first.st_size,first.st_mtime_ns) == (last.st_size,last.st_mtime_ns),
                "Original checkpoint changed during read")
    require(artifact["meta"]["epoch"] == 24 and artifact["meta"]["iter"] == 4464,
            "Exact deployed source-only checkpoint metadata")
    model.load_state_dict(artifact["state_dict"], strict=True)


def project_native(rois, crops, start_xy, pad_size, content_hw, original_hw):
    """Recreate the pinned author coordinate map through the final native resize."""
    n = crops.shape[-1]
    signed_angle = -rois[:,4]
    cosine, sine = torch.cos(signed_angle), torch.sin(signed_angle)
    indices = torch.arange(pad_size, device=rois.device, dtype=rois.dtype) + .5
    y, x = torch.meshgrid(indices, indices, indexing="ij")
    x = x[None]-start_xy[:,0,None,None]
    y = y[None]-start_xy[:,1,None,None]
    along = (x*cosine[:,None,None] - y*sine[:,None,None]) / (rois[:,2,None,None]/n)
    across = (y*cosine[:,None,None] + x*sine[:,None,None]) / (rois[:,3,None,None]/n)
    grid = torch.stack((along,across),dim=-1) / (n/2) - 1
    probability = F.grid_sample(crops,grid,align_corners=False)[:,0]
    probability = probability[:,:content_hw[0],:content_hw[1]]
    return F.interpolate(probability[None],original_hw,mode="bilinear",align_corners=False).squeeze(0)


def signatures(np, codec, result):
    boxes, masks = result
    require(len(boxes) == len(masks) == 1 and boxes[0].shape == (len(masks[0]),6) and
            boxes[0].dtype == np.float32 and np.isfinite(boxes[0]).all(),
            "Length-preserving one-class author RBox and mask rows")
    h = sha256(boxes[0].tobytes(order="C"))
    positives = 0
    for mask in masks[0]:
        require(mask.dtype == np.bool_ and mask.shape == (512,512), "Original author native boolean canvas")
        encoded = codec.encode(np.asfortranarray(mask,dtype=np.uint8))
        require(encoded["size"] == [512,512] and np.array_equal(codec.decode(encoded).astype(bool),mask),
                "Exact row-major native mask and COCO RLE round trip")
        h.update(len(encoded["counts"]).to_bytes(8,"little"))
        h.update(encoded["counts"])
        positives += bool(np.any(mask))
    return h.hexdigest(), len(boxes[0]), positives


def tap_mask(author, corners, np, records):
    def tap(rois,labels,crops,version,output_size,img_meta,thr,num_classes,rescale):
        masks = author(rois,labels,crops,version,output_size,img_meta,thr,num_classes,rescale)
        content_hw = tuple(img_meta["img_shape"][:2])
        native_hw = tuple(img_meta["ori_shape"][:2])
        require(version == "le90" and output_size == 28 and thr == .5 and num_classes == 1 and rescale and
                crops.shape == (len(rois),1,28,28) and tuple(img_meta["pad_shape"][:2]) == (608,608) and native_hw == (512,512) and
                torch.all(labels == 0) and torch.isfinite(rois).all() and torch.isfinite(crops).all() and
                torch.all((crops >= 0) & (crops <= 1)) and torch.all(rois[:,2:4] > 0) and len(masks[0]) == len(rois),
                "Author stock rotated projection input and no altered threshold")
        diag = Counter(candidates=len(rois),native_projected_pixels=len(rois)*512*512)
        for start in range(0,len(rois),16):
            end=min(start+16,len(rois))
            box=rois[start:end]
            p=project_native(box,crops[start:end],corners(box,version)[:,:2],608,content_hw,native_hw)
            require(p.shape == (end-start,512,512) and torch.isfinite(p).all() and
                    torch.all((p >= -1e-6) & (p <= 1+1e-6)),
                    "Finite native probabilities on every coordinate before strict threshold")
            reconstructed=(p > .5).numpy()
            for pred, expected in zip(reconstructed,masks[0][start:end],strict=True):
                require(np.array_equal(pred,expected),"Author and independently projected native mask differ")
            diag.update(exact_projected_masks=end-start, exact_half_probability_pixels=int((p == .5).sum()))
        scaled=rois.detach().clone()
        scaled[:,:4] /= scaled.new_tensor(img_meta["scale_factor"])
        records.append((diag,scaled.numpy().copy()))
        return masks
    return tap


def run(args):
    started=perf_counter()
    require(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
            and sys.version_info[:3] == (3,10,13) and not args.output.exists(),
            "Fresh isolated CPU-only native parity")
    pin(args.protocol,args.protocol_sha256)
    for name, expected in PINS.items():
        pin(args.source/name,expected)
    selected=admit(args)
    image_root, index=bind_images(args.intake,selected)
    args.output.mkdir()
    img_only=write_json(args.output/"IMAGE_ONLY_COCO_PRIVATE.json",
                        dict(images=index,categories=[dict(id=1,name="ship")],annotations=[]))
    print(json.dumps(dict(stage="ADMITTED_8_IMAGE_ONLY_CHIPS")),flush=True)

    sys.path.insert(0,str(args.source))
    import cv2
    import mmcv
    import mmrotate
    import numpy as np
    from mmrotate.datasets import build_dataset
    from mmrotate.models import build_detector
    from pycocotools import mask as codec

    require(Path(mmrotate.__file__).resolve() == args.source/"mmrotate/__init__.py" and
            {name:version(name) for name in PACKAGES} == PACKAGES,
            "Pinned original author import and CPU environment")
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config=mmcv.Config.fromfile(str(args.source/CONFIG))
    require(dict(config.model.test_cfg) == STOCK_TEST and config.angle_version == "le90",
            "Original unmodified SSDD test settings")
    settings=config.data.test.copy()
    settings.ann_file=str(args.output/"IMAGE_ONLY_COCO_PRIVATE.json")
    settings.img_prefix=str(image_root)+os.sep
    settings.test_mode=True
    dataset=build_dataset(settings)
    require(dataset.CLASSES == ("ship",) and dataset.cat_ids == [1] and
            list(dataset.img_ids) == [r["id"] for r in index],"No labels and no author dataset row filtering")
    model=build_detector(config.model).cpu().eval()
    load_exact_checkpoint(model,args.fit/"RESULT/epoch_24.pth")
    print(json.dumps(dict(stage="EXACT_FROZEN_SOURCE_WEIGHTS_LOADED")),flush=True)

    def forward(pos):
        image=dataset.prepare_test_img(pos)
        tensor,=image["img"]
        container,=image["img_metas"]
        if not torch.is_tensor(tensor):
            tensor=tensor.data
        meta=container.data
        require(tuple(tensor.shape) == (3,608,608) and tensor.dtype == torch.float32 and tensor.device.type == "cpu" and
                tuple(meta["ori_shape"][:2]) == (512,512) and meta["flip"] is False,
                "Stock author pipeline and native canvas")
        with torch.inference_mode():
            result,=model.simple_test(tensor[None],[meta],rescale=True,thr=.5)
        return result

    baseline=[]
    for pos in range(len(index)):
        baseline.append(signatures(np,codec,forward(pos)))
        print(json.dumps(dict(stage="AUTHOR_STOCK",completed=pos+1)),flush=True)

    head=import_module("mmrotate.models.dense_heads.SRNet_head")
    utils=import_module("mmrotate.models.dense_heads.SRNet_utils")
    original=head.get_mask
    require(original is utils.get_mask,"Only passively instrument the author paste entry")
    reports={s:Counter() for s in SENSORS}
    for pos,row in enumerate(selected):
        records=[]
        head.get_mask=tap_mask(original,utils.obb2poly,np,records)
        try:
            result=forward(pos)
        finally:
            head.get_mask=original
        signature,count,nonempty=signatures(np,codec,result)
        require((signature,count,nonempty) == baseline[pos],"Author ranks, scores or native RLE changed between passes")
        require(len(records) == int(count>0),"Every author nonempty candidate list passes through the tap once")
        if records:
            tally,scaled=records[0]
            require(np.array_equal(scaled,result[0][0]) and tally["exact_projected_masks"]==count,
                    "Tap preserves score, rotated box, rank and one mask per author candidate")
            reports[row["sensor"]].update(tally)
        reports[row["sensor"]].update(images=1,nonempty_masks=nonempty,
                                       images_with_nonempty_masks=int(nonempty>0),
                                       no_candidates=int(count==0),empty_masks=count-nonempty)
        print(json.dumps(dict(stage="PARITY_READBACK",completed=pos+1)),flush=True)
    combined=Counter()
    for tally in reports.values():
        combined.update(tally)
    require(not torch.cuda.is_initialized() and combined["images"] == 8 and
            combined["images_with_nonempty_masks"] >= 2 and
            combined["candidates"] == combined["exact_projected_masks"],
            "Coverage on at least two distinct prespecified chips and exact parity on every mask")

    result=dict(status="SRNET_SSDD_E24_RSDD74_NATIVE_PARITY_PILOT_COMPLETE",phase="verify",
        worker_sha256=digest(Path(__file__).read_bytes()),protocol_sha256=args.protocol_sha256,
        author_commit=AUTHOR_COMMIT,author_source_sha256=PINS,intake_result_sha256=INTAKE_RESULT,
        intake_terminal_sha256=INTAKE_TERMINAL,intake_image_manifest_sha256=MANIFEST,
        source_fit_result_sha256=FIT_RESULT,source_fit_terminal_sha256=FIT_TERMINAL,
        weights_sha256=CHECKPOINT,weight_training_split="PSeg-SSDD FIT741 project refit",
        official_RSDD_checkpoint=False,device="CPU float32",runtime_versions=PACKAGES,
        torch_intraop_threads=1,torch_interop_threads=1,test_settings=STOCK_TEST,
        source_angle_convention="le90",network_hw=[608,608],native_hw=[512,512],
        mask_threshold=.5,threshold_operator=">",selected_acquisition_groups=8,
        stock_forwards=8,tapped_forwards=8,all_author_order_scores_native_rle_exact=True,
        all_independent_projection_masks_exact=True,totals=dict(combined),by_sensor=reports,
        model_selected=False,threshold_selected=False,reference_files_opened=0,
        ground_truth_matched=False,AP=False,optimizer_updates=0,pixels_edited=0,
        private_image_only_index_sha256=img_only,elapsed_seconds=perf_counter()-started)
    sha=write_json(args.output/"RESULT_SAFE.json",result)
    print(json.dumps(dict(status=result["status"],result_sha256=sha)),flush=True)


def main():
    p=ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("source","fit","intake","protocol","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--protocol-sha256",required=True)
    a=p.parse_args()
    for name in ("source","fit","intake","protocol","output"):
        setattr(a,name,getattr(a,name).resolve())
    run(a)


if __name__ == "__main__":
    main()
