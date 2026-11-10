"""Frozen seed-5201 HRSID FIT-only input shift pilot on DGX CPU."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from statistics import fmean
import sys
from time import monotonic
from types import ModuleType
from zipfile import ZipFile


EXPORTER_SHA = "4acb5fbcaf35c5a32510f5f93ccf51f34e08641fd6d007c78c85becde7496727"
BUNDLE_SHA = "af12a4e86ea36e9d7fe6ac47b17c684a4812baba6aa9be019f889f833e0f577c"
COHORT_SHA = "70e9a0e8fb30f5ba70b48b9067c1b95d07bfec23d6b248d92c5983727acceb6e"
INDEX_SHA = "6f25f80ed0e0b9c3f0c57ab6cc3b25e947a102353757a33bca4d3823df3d547d"
PAIR_SHA = "3e8720e8fad83d6a4b313ee8b38e261cabb4f1a2a3eefe3a5e33e83d574ac1a1"
SCOPE_SALT = "P268_FIT114_SOURCE5201_REFLECT_Y1_V1"
OFFSETS = (-1, 0, 1)
MODES = ("native", "center")
SOURCE = {
    "native": ("56685edda809b9e95135e68e7b535039d93825279ddddf6bf9c3a172ec86c766",
               "0db537b95210672cab0465e99f00fa2c9f8bdf29e579670d3e9fbe32661f99f1",
               "198bc09cb93019e719342914965dffb31160bb284003a17b334f998c1f540c56"),
    "center": ("c0a83a2a1dcf21000eb14c1b2f255061a5005738ca8c0c98451f800538436b0c",
               "b57552cdb3a38202b77eac28b9af8ae278f3c22fccee175920173a85406ff4a5",
               "b95f1a41ab599b7eb27ed3f25029300838cf1b9da74d7bd0c3ccaaef449485db"),
}
SIZE, GROUPS, MARGIN = 800, 8, 8
STATUS = "P268_HRSID_FIT114_FROZEN_INPUT_SHIFT_COMPLETE_V1"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def reflect_shift(array, shift, np):
    need(shift in OFFSETS and array.ndim >= 2 and len(array) >= 3, "Exact ±1 reflected y shift required")
    padded = np.pad(array, ((1, 1), *((0, 0),) * (array.ndim - 1)), mode="reflect")
    return padded[1-shift:1-shift+len(array)]


def register_mask(mask, shift):
    return mask[1+shift:len(mask)-1+shift]


def box_iou(a, b):
    overlap = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    area = max(0., a[2]-a[0]) * max(0., a[3]-a[1]) + (b[2]-b[0]) * (b[3]-b[1])
    return overlap / (area - overlap) if overlap else 0.


def associate_boxes(predictions, targets, shift, size=SIZE):
    """Fixed native rank greedy matching, IoU≥.50, deterministic GT ID ties."""
    result = {}
    for rank, (x1, y1, x2, y2) in predictions:
        registered = x1, max(1., min(size-1., y1-shift)), x2, max(1., min(size-1., y2-shift))
        options = [(box_iou(registered, targets[i]), i) for i in sorted(targets) if i not in result]
        if options:
            score, iid = max(options, key=lambda row: (row[0], -row[1]))
            if score >= .50:
                result[iid] = rank
    return result


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def save(path, value):
    raw = canonical(value)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256(raw).hexdigest()


def pinned(path, expected, limit):
    need(path.resolve(strict=True) == path and path.is_file(), "Exact canonical historical reference")
    with path.open("rb") as stream:
        raw = stream.read(limit+1)
    need(0 < len(raw) <= limit and sha256(raw).hexdigest() == expected, "Historical reference digest differs")
    return raw


def member(zipfile, name, expected, limit):
    entry = zipfile.getinfo(name)
    need(0 < entry.file_size <= limit, "Bounded FIT member required")
    raw = zipfile.read(entry)
    need(len(raw) == entry.file_size and sha256(raw).hexdigest() == expected, "Exact closed FIT member required")
    return raw


def old_source(exporter, mode, closed_export_root):
    lock_sha, term_sha, ckpt_sha = SOURCE[mode]
    lock_path = closed_export_root / "SOURCE_CPU_LOCKS" / f"5201_{mode}_EXPORT_LOCK_PRIVATE.json"
    lock = exporter.read_lock(lock_path, lock_sha)
    original = closed_export_root / "RUN_ORIGINAL_CPU_01" / f"5201_{mode}"
    term = json.loads(pinned(original / "TERMINAL_PRIVATE.json", term_sha, 64*1024))
    need(term["original_os_wait"] is term["worker_closed"] is True and term["resource_stop"] is False
         and type(term["original_os_exit"]) is int and term["original_os_exit"] == 0
         and (term["seed"], term["mode"], term["export_lock_sha256"]) == (5201, mode, lock_sha)
         and (lock["seed"], lock["mode"], lock["device"], lock["pair_receipt"]["sha256"],
              lock["checkpoint"]["sha256"]) == (5201, mode, "cpu", PAIR_SHA, ckpt_sha),
         "Exact independently closed historical P250 source arm required")
    import psutil
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    for role in ("OWNER", "WORKER"):
        birth = json.loads((original / (role+"_BIRTH_PRIVATE.json")).read_bytes())
        need(set(birth) == {"pid", "created", "boot"}, "Exact historical birth shape required")
        if birth["boot"] == boot:
            try:
                process = psutil.Process(birth["pid"])
                need(process.create_time() != birth["created"], "Historical P250 original process still live")
            except psutil.NoSuchProcess:
                pass
    return lock


def select(zipfile, np, codec):
    cohort = json.loads(member(zipfile, "COHORT_PRIVATE.json", COHORT_SHA, 1024**2))
    index = json.loads(member(zipfile, "FIT114_COCO_PRIVATE.json", INDEX_SHA, 4*1024**2))
    images = {row["id"]:row for row in index["images"]}
    need(cohort["fit_only"] is True and len(images) == len(cohort["images"]) == 114, "Exact FIT114-only population")
    annotations = defaultdict(list)
    for row in index["annotations"]:
        need(row["image_id"] in images and row["category_id"] == 1, "Only original FIT ship labels permitted")
        annotations[row["image_id"]].append(row)
    ordered = sorted(cohort["images"], key=lambda row: sha256(canonical(
        [SCOPE_SALT,row["group"],row["encoded_sha256"]])).hexdigest())
    selected, seen_groups, seen_bytes = [], set(), set()
    for row in ordered:
        image_id = row["image_id"]
        if row["group"] in seen_groups or row["encoded_sha256"] in seen_bytes or not row["core"]:
            continue
        image = images[image_id]
        need((image["height"],image["width"],image["file_name"]) == (SIZE,SIZE,row["native_basename"]),
             "Exact original FIT geometry and basename")
        truth = {}
        for ann in annotations[image_id]:
            x,y,w,h = ann["bbox"]
            if (ann.get("iscrowd",0) or ann.get("ignore",0) or not (w>0 and h>0)
                    or min(x,y,SIZE-x-w,SIZE-y-h) < MARGIN):
                continue
            rle = codec.merge(codec.frPyObjects(ann["segmentation"],SIZE,SIZE))
            mask = codec.decode(rle).astype(bool, copy=False)
            if bool(mask[1:-1].any()):
                need(not mask[0].any() and not mask[-1].any() and ann["id"] not in truth,
                     "Eligible reference mask must remain inside common interior")
                truth[ann["id"]] = ((x,y,x+w,y+h),mask[1:-1])
        if truth:
            selected.append((row,truth))
            seen_groups.add(row["group"])
            seen_bytes.add(row["encoded_sha256"])
        if len(selected) == GROUPS:
            return selected
    raise ValueError("Fewer than eight predeclared eligible unique FIT source groups")


def iou(a,b,np):
    return int(np.count_nonzero(a & b)) / int(np.count_nonzero(a | b))


def centroid_y(a,np):
    return float(np.nonzero(a)[0].mean())


def score_group(views, truth, np):
    base = views[0]
    metrics = {}
    for shift, prediction in views.items():
        both = base.keys() & prediction.keys()
        metrics[str(shift)] = dict(
            eligible=len(truth), nonempty_matched=len(prediction), missed=len(truth)-len(prediction),
            baseline_nonempty_matched=len(base), both_nonempty_matched=len(both),
            baseline_only=len(base)-len(both), offset_only=len(prediction)-len(both),
            neither=len(truth)-len(base.keys() | prediction.keys()),
            mean_gt_iou_miss_zero=sum(iou(prediction[k],truth[k][1],np) for k in prediction)/len(truth),
            mean_baseline_signed_y_pixels=(fmean(centroid_y(base[k],np)-centroid_y(truth[k][1],np) for k in base)
                                           if base else None),
            mean_baseline_signed_y_on_both=(fmean(centroid_y(base[k],np)-centroid_y(truth[k][1],np) for k in both)
                                            if both else None),
            mean_registered_pair_iou=fmean(iou(prediction[k],base[k],np) for k in both) if both else None,
            mean_registered_signed_y_pixels=(fmean(centroid_y(prediction[k],np)-centroid_y(base[k],np) for k in both)
                                             if both else None),
        )
    return metrics


def main():
    begin = monotonic()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frozen-exporter", type=Path, required=True)
    parser.add_argument("--fit-bundle", type=Path, required=True)
    parser.add_argument("--closed-export-root", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    exporter_path = args.frozen_exporter.resolve(strict=True)
    bundle_path = args.fit_bundle.resolve(strict=True)
    closed_export_root = args.closed_export_root.resolve(strict=True)
    need(out.is_absolute() and out.parent.resolve(strict=True) == out.parent and not out.exists()
         and os.sched_getaffinity(0) == {2,3}, "Fresh canonical output and pinned CPU affinity")
    exporter = ModuleType("_p268_exact_p250")
    exporter.__file__ = str(exporter_path)
    exec(compile(pinned(exporter_path,EXPORTER_SHA,65536),str(exporter_path),"exec"),exporter.__dict__)
    sys.addaudithook(exporter.offline)
    locks = {mode:old_source(exporter, mode, closed_export_root) for mode in MODES}
    bases = {mode:exporter.admit_source(locks[mode]) for mode in MODES}
    exporter.check_runtime("cpu",bases["native"])
    import numpy as np
    import torch
    from PIL import Image
    from pycocotools import mask as codec
    from rfdetr.datasets.coco import make_coco_transforms_square_div_64
    from rfdetr.datasets.io_utils import decode_pil_image_bytes
    from rfdetr.models.postprocess import PostProcess
    from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

    device = exporter.device_admission(torch,"cpu")
    out.mkdir()
    with bundle_path.open("rb") as stream:
        original_stat = os.fstat(stream.fileno())
        fingerprint = exporter._identity(original_stat)
        digest = sha256()
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
        need(original_stat.st_size == 151320868 and digest.hexdigest() == BUNDLE_SHA, "Exact historical sealed FIT114 ZIP")
        stream.seek(0)
        with ZipFile(stream) as zipfile:
            chosen = select(zipfile,np,codec)
            roster_sha = save(out / "SELECTION_PRIVATE.json", [
                dict(image_id=row["image_id"],group=row["group"],source_png_sha256=row["encoded_sha256"],eligible_gt=sorted(gt))
                for row,gt in chosen])
            print(json.dumps(dict(event="P268_FIT_ROSTER_FROZEN",groups=len(chosen),roster_sha256=roster_sha)),flush=True)
            inputs = []
            for row,_ in chosen:
                raw = member(zipfile,"images/"+row["native_basename"],row["encoded_sha256"],4*1024**2)
                image, scales = decode_pil_image_bytes(raw,draft_size=None)
                array = np.asarray(image)
                need(image.size==(SIZE,SIZE) and image.mode=="RGB" and scales==(1.,1.)
                     and len(raw)==row["encoded_bytes"] and sha256(array.tobytes()).hexdigest()==row["decoded_rgb_sha256"],
                     "Exact closed original RGB source")
                inputs.append(array.copy())
        need(exporter._identity(os.fstat(stream.fileno()))==exporter._identity(bundle_path.stat())==fingerprint,
             "Sealed FIT source changed during readback")

    all_groups, calls = {}, 0
    for mode in MODES:
        base, lock = bases[mode], locks[mode]
        archive = exporter.ref_path(lock["checkpoint"],exporter.GIB,name="checkpoint_23.ckpt")
        model, config, train, _ = base.load_model(archive,SOURCE[mode][2],torch,out,exporter.TRAINING_PROTOCOLS[5201])
        model = model.to(device).eval()
        need(not model.training and model.num_queries==164 and model.group_detr==13,"Exact frozen source Q164 model")
        transform = make_coco_transforms_square_div_64(
            "val",576,multi_scale=False,expanded_scales=False,skip_random_resize=True,patch_size=config.patch_size,
            num_windows=config.num_windows,aug_config=None,scale_jitter=train.scale_jitter,gpu_postprocess=False)
        processor = PostProcess(num_select=100,upsample_masks_to_image_size=True)
        groups = []
        with torch.inference_mode():
            for ordinal, ((row,truth),rgb) in enumerate(zip(chosen,inputs,strict=True),1):
                views = {}
                for shift in OFFSETS:
                    tensor, target = transform(Image.fromarray(reflect_shift(rgb,shift,np)),None)
                    need(target is None and tuple(tensor.shape)==(3,576,576) and tensor.dtype==torch.float32
                         and bool(torch.isfinite(tensor).all()), "Exact image-only val576 input required")
                    sample = nested_tensor_from_tensor_list([tensor],block_size=config.patch_size*config.num_windows)
                    need(sample.mask is not None and not bool(sample.mask.any()),"Exact source no spatial padding required")
                    outputs = model(sample.to(device))
                    calls += 1
                    boxes = []

                    def capture(*args,**kwargs):
                        result = processor(*args,**kwargs)
                        boxes.extend(result[0]["boxes"].tolist())
                        return result

                    top = base.native_rows(outputs,image_id=row["image_id"],hw=(SIZE,SIZE),torch=torch,
                                           postprocess=capture,np=np,codec=codec)
                    need(len(boxes)==100 and all(len(box)==4 and all(math.isfinite(x) for x in box) for box in boxes),
                         "Exact source top100 box census required")
                    ranks = top["native_ranks"]
                    chosen_rank = associate_boxes([(rank,boxes[rank-1]) for rank in ranks],
                                                  {key:value[0] for key,value in truth.items()},shift)
                    by_rank = dict(zip(ranks,top["coco"],strict=True))
                    matched = {}
                    for key,rank in chosen_rank.items():
                        rle = by_rank[rank]["segmentation"]
                        mask = codec.decode(dict(rle,counts=rle["counts"].encode("ascii"))).astype(bool,copy=False)
                        registered = register_mask(mask,shift)
                        if registered.any():
                            matched[key]=registered
                    views[shift]=matched
                groups.append(dict(group_ordinal=ordinal,offsets=score_group(views,truth,np)))
                print(json.dumps(dict(event="P268_GROUP_FORWARD_COMPLETE",mode=mode,group=ordinal,calls=calls)),flush=True)
        all_groups[mode]=groups
        del model

    need(calls==48 and not torch.cuda.is_initialized(), "Exact 48 frozen CPU forwards required")
    result = dict(status=STATUS,source_seed=5201,source_training_protocol=exporter.TRAINING_PROTOCOLS[5201],
                  source_pair_receipt_sha256=PAIR_SHA,source_checkpoint_sha256={m:SOURCE[m][2] for m in MODES},
                  source_terminal_sha256={m:SOURCE[m][1] for m in MODES},fit_bundle_sha256=BUNDLE_SHA,
                  selection_sha256=roster_sha,selection_salt=SCOPE_SALT,source_groups=GROUPS,offsets_y=OFFSETS,
                  source_box_margin_pixels=MARGIN,reflection="no_wrap_reflect_then_stock_val576",
                  association="slot0_native_rank_greedy_box_iou_ge_.50_then_gt_id",
                  valid_registration_rows=[1,SIZE-1],by_arm=all_groups,model_forward_calls=calls,
                  CUDA_initialized=False,FIT=True,DEV=False,TEST=False,AP=False,training=False,
                  source_model_seen_FIT=True,worker_seconds=monotonic()-begin)
    result_sha = save(out / "RESULT_SAFE.json",result)
    print(json.dumps(dict(status=STATUS,result_sha256=result_sha,forward_calls=calls)),flush=True)


if __name__=="__main__":
    main()
