"""Exercise the pinned SRNet author SSDD loader on the fixed project FIT data."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import sys
from time import perf_counter
from zipfile import ZipFile

FIT_SHA = "1c8f9b75e79fbc02c983ccd283a70082fcd5ab0d6538c777e8b286dbfdb78100"
FIT_INDEX_SHA = "62966857c78cea1e3b060700c62a7b896942686c2e674d3eb09f516a92250bc1"
AUTHOR_COMMIT = "2e56c7208f217ff1824f1d4f3f3ea242a5f78bc6"
AUTHOR_CONFIG = "configs/SRNet/SRNet_r50_fpn_3x_SSDD_le90.py"
SOURCE_PINS = {
    "configs/_base_/datasets/SSDD.py": "fd7ab145b360d64324f89b0cb10a8ad11ff3746a5a5f4521eb30a49d46b23c63",
    AUTHOR_CONFIG: "222ba188f9476e822d1c9a1f4c5c15404bbd5a029bd599aeae41a9dafbbaa356",
    "mmrotate/datasets/SSDD.py": "fc89de2226c1428243e7a9fee4169aeef9a03a156254e8b52fe1800543efd3ec",
    "mmrotate/datasets/pipelines/RLoading.py": "8d63f66b10f0f768ebac9f70727c32cf1a1ae032f99e470fbc356c93cb28f04f",
    "mmrotate/datasets/pipelines/transforms.py": "62fd8ee3c695b78847f8cb420843ada5fbf7f8ee04a54600cc0c2b848f1a3040",
}
STATUS = "SRNET_AUTHOR_SSDD_FIT741_NATIVE_DATA_PREFLIGHT_COMPLETE"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def exact(path, digest, maximum):
    require(path.is_file() and 0<path.stat().st_size<=maximum,"Exact bounded regular file required")
    raw=path.read_bytes()
    require(sha256(raw).hexdigest()==digest,"Exact pinned source or dataset bytes differ")
    return raw


def key(value):
    return json.dumps(value,separators=(",",":"),allow_nan=False)


def choose(images):
    salt="SRNET-author-FIT741-native-loader-v1/"
    return sorted(images,key=lambda row:(sha256((salt+str(row["id"])).encode()).digest(),row["id"]))[:8]


def bitmap(polygon,h,w,codec):
    return codec.decode(codec.merge(codec.frPyObjects(polygon,h,w))).astype(bool)


def run(a):
    started=perf_counter()
    for name,digest in SOURCE_PINS.items():
        exact(a.source/name,digest,1024**2)
    index=a.fit/"_annotations.coco.json"
    raw=json.loads(exact(index,FIT_INDEX_SHA,2*1024**2))
    require(len(raw["images"])==741 and len(raw["annotations"])==1661 and raw["categories"]==[{"id":1,"name":"ship"}],
            "Exact project FIT741 COCO index expected")
    exact(a.archive,FIT_SHA,34*1024**2)
    with ZipFile(a.archive) as z:
        manifest=json.loads(z.read("manifest.json"))
    fit_items={row["image_id"]:row for row in manifest["rows"]}
    require(len(fit_items)==741,"Exact FIT manifest population")
    sys.path.insert(0,str(a.source))
    import cv2
    import mmcv
    import mmdet
    import mmrotate
    import numpy as np
    import torch
    from mmrotate.datasets import build_dataset
    from pycocotools import mask as codec
    from importlib.metadata import version
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    require(Path(mmrotate.__file__).resolve()==a.source/"mmrotate"/"__init__.py","Pinned author registry required")
    versions={k:version(k) for k in ("torch","torchvision","mmcv-full","mmdet","numpy","pycocotools","opencv-python")}
    require(versions=={"torch":"1.12.0+cpu","torchvision":"0.13.0+cpu","mmcv-full":"1.6.0","mmdet":"2.28.2",
                       "numpy":"1.26.4","pycocotools":"2.0.11","opencv-python":"4.11.0.86"},
            "Exact declared SRNet compatibility runtime required")
    cfg=mmcv.Config.fromfile(str(a.source/AUTHOR_CONFIG))
    settings=cfg.data.train.copy()
    require(settings.type=="RepeatDataset" and settings.times==3,"Exact author dataset repeat config")
    settings.dataset.ann_file=str(index)
    settings.dataset.img_prefix=str(a.fit)+os.sep
    dataset=build_dataset(settings)
    base=dataset.dataset
    require(type(base).__module__=="mmrotate.datasets.SSDD" and base.CLASSES==("ship",) and base.version=="le90",
            "Exact author SSDD registry")
    original={}
    for im in raw["images"]:
        original[im["id"]]=im
    rows=defaultdict(list)
    for ann in raw["annotations"]:
        rows[ann["image_id"]].append(ann)
    parse={}
    rejects=Counter()
    unknown=0
    for idx,im in enumerate(base.data_infos):
        image_id=im["id"]
        obs=base.get_ann_info(idx)
        require(obs["bboxes"].shape==(len(obs["masks"]),5) and len(obs["labels"])==len(obs["masks"])
                and bool(np.isfinite(obs["bboxes"]).all()),"Author output parsed geometry and mask identities")
        source_rows=Counter(key(x["segmentation"]) for x in rows[image_id])
        parsed_rows=Counter(key(x) for x in obs["masks"])
        require(not(parsed_rows-source_rows),"Author masks must preserve an original polygon payload")
        for ann in rows[image_id]:
            k=key(ann["segmentation"])
            if parsed_rows[k]>0:
                parsed_rows[k]-=1
                continue
            if ann.get("ignore",False) or ann.get("iscrowd",False):
                rejects["ignore_or_crowd"]+=1
            elif ann["area"]<=0:
                rejects["nonpositive_area"]+=1
            elif len(ann["segmentation"])>1:
                rejects["multipart_author_filtered"]+=1
            else:
                unknown+=1
        parse[image_id]=(idx,len(obs["masks"]))
    selected=choose(list(original.values()))
    ran, variants=0, Counter()
    sample_summaries=[]
    for im in selected:
        image_id=im["id"]
        row=fit_items[image_id]
        require(im["file_name"]==row["original_basename"] and image_id in parse,
                "Fixed prospective image and stock parsed index required")
        ref=row["original_image_ref"]
        exact(a.fit/im["file_name"],ref["sha256"],4*1024**2)
        idx,_=parse[image_id]
        random.seed(5201+ran)
        np.random.seed(5201+ran)
        torch.manual_seed(5201+ran)
        data=dict(img_info=base.data_infos[idx],ann_info=base.get_ann_info(idx))
        base.pre_pipeline(data)
        for step,tr in enumerate(base.pipeline.transforms):
            data=tr(data)
            if data is None:
                variants["returned_none"]+=1
                break
            if step==1:
                h,w=im["height"],im["width"]
                native=data["gt_masks"].masks
                target=[bitmap(p,h,w,codec) for p in base.get_ann_info(idx)["masks"]]
                require(native.shape==(len(target),h,w)
                        and all(np.array_equal(x,y) for x,y in zip(native,target)),
                        "Author pre-resize native mask must match the same COCO polygon raster")
                variants["native_masks_verified"]+=len(target)
            elif step==2:
                h,w=data["img"].shape[:2]
                variants["resized_masks"]+=len(data["gt_masks"])
                require(max(h,w)==608 and abs((h/w)/(im["height"]/im["width"])-1)<.01,
                        "Author native RResize should preserve aspect ratio within integer rounding")
        if data is not None:
            tensor=data["img"].data
            labels=data["gt_labels"].data
            masks=data["gt_masks"].data
            require(tuple(tensor.shape)==(3,608,608) and len(labels)==len(masks)
                    and masks.height==masks.width==608,"Author padded tensor and mask alignment")
            sample_summaries.append(dict(original_hw=[im["height"],im["width"]],
                                         native_targets=parse[image_id][1],pipeline_targets=len(labels)))
            variants["final_targets"]+=len(labels)
        ran+=1
    require(ran==8 and not torch.cuda.is_initialized(),"Eight actual FIT views on CPU only")
    kept=sum(x[1] for x in parse.values())
    filtered_images=set(original)-set(parse)
    filtered_image_targets=sum(len(rows[i]) for i in filtered_images)
    require(kept+filtered_image_targets+unknown+sum(rejects.values())==1661,
            "All original FIT annotation rows accounted for")
    return dict(status=STATUS,source_commit=AUTHOR_COMMIT,source_pins=SOURCE_PINS,package_versions=versions,
                FIT=True,DEV=False,TEST=False,AP=False,model_forward_calls=0,optimizer_updates=0,
                original_images=len(original),original_annotations=1661,author_parsed_images=len(parse),
                author_parsed_targets=kept,filtered_images=len(filtered_images),filtered_image_targets=filtered_image_targets,
                parser_drop_known=dict(rejects),parser_drop_other=unknown,author_repeat_times=settings.times,
                selected_view_count=ran,actual_pipeline_counts=dict(variants),sample_summaries=sample_summaries,
                source_annotation_sha256=FIT_INDEX_SHA,fit_archive_sha256=FIT_SHA,elapsed_seconds=perf_counter()-started)


def deny_network(event,_):
    if event.startswith("socket."):
        raise PermissionError("offline author FIT data preflight")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ("source","fit","archive","output"):
        p.add_argument("--"+flag,type=Path,required=True)
    a=p.parse_args()
    require(all(x.is_absolute() and x.resolve(strict=True)==x for x in (a.source,a.fit,a.archive))
            and a.output.is_absolute() and a.output.parent.resolve(strict=True)==a.output.parent
            and not a.output.exists() and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1","CPU and fresh canonical output")
    sys.addaudithook(deny_network)
    raw=(json.dumps(run(a),sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with a.output.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=STATUS,result_sha256=sha256(raw).hexdigest())),flush=True)


if __name__=="__main__":
    main()
