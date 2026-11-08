"""Observe the pinned native RF-DETR dataset converter on the five source-degenerate FIT targets."""
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
import json
import os
from pathlib import Path
from zipfile import ZipFile

from PIL import Image
from pycocotools import mask
from rfdetr.datasets.coco import ConvertCoco
import torch

HERE=Path(__file__).resolve().parent
AUDIT=Path("D:/SAR_SEAL_private_execution_20261021_01/HRSID_OFFICIAL_PNG_FIT_AUDIT_ACTUAL_PRIVATE_20261109_01")
ARCHIVE=Path("D:/SAR_DATASETS/HRSID_PNG_OFFICIAL_INTAKE_20261015/ARCHIVE_STANDARD_HTTP_TRANSFER_20261109_01/HRSID_PNG_ARCHIVE_COMPLETE_PRIVATE.zip")
ASSIGNMENT=Path("E:/SAR/seal/research_round_20260925_global/experiments/public_baseline_20260925/protocol/hrsid/image_group_assignment.json")


def need(ok,message):
    if not ok:
        raise ValueError(message)


def main():
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="-1" and not torch.cuda.is_available()
         and version("rfdetr")=="1.12.0" and version("pycocotools")=="2.0.11","Exact CPU native input runtime")
    report=(AUDIT/"REPORT_SAFE.json").read_bytes()
    need(sha256(report).hexdigest()=="d0e4deff1be184c3037990aed0e0df208f1503b20c079db2539bbd63a29e120d",
         "Exact original closed FIT PNG audit")
    private=json.loads((AUDIT/"FIT_FINDINGS_PRIVATE.json").read_bytes())
    need(len(private)==5 and all(x["kind"]=="EMPTY_COCO_RASTER" for x in private),"Only the five original FIT findings")
    raw=ASSIGNMENT.read_bytes()
    need(sha256(raw).hexdigest()=="c92b85dc278bc49cc8e34634fd7d354018844ae2a2692ea451b014a082bbaa5a",
         "Frozen FIT membership")
    assigned=json.loads(raw)
    members={r["image_id"] for r in assigned if r["split"]=="fit"}
    need(len(members)==2937 and all(r["image_id"] in members for r in private),"Only original FIT images")
    convert=ConvertCoco(include_masks=True,cat2label={1:0})
    stats=dict(selected_empty_retained_as_label=0,selected_empty_masks_zero=0,selected_empty_positive_boxes=0,
               same_image_nonempty_controls_retained=0,same_image_nonempty_control_masks_positive=0)
    with ZipFile(ARCHIVE) as z:
        need(ARCHIVE.stat().st_size==5698302315,"Exact complete official ZIP length")
        raw=z.read("annotations/train2017.json")
        need(sha256(raw).hexdigest()=="3e240c38014526e9b93206a946105fb41c808272238932b0d4c2422978d5116f",
             "Exact same official original annotations")
        source=json.loads(raw)
        images={x["id"]:x for x in source["images"] if x["id"] in members}
        selected={r["annotation_id"] for r in private}
        geometry={r["id"]:r for r in source["annotations"] if r["id"] in selected}
        need(set(geometry)==selected,"Every closed FIT diagnosis corresponds to original label")
        for rec in private:
            iid=rec["image_id"]
            a=geometry[rec["annotation_id"]]
            need(a["image_id"]==iid,"Original annotation pairing")
            control=next(row for row in source["annotations"] if row["image_id"]==iid and row["id"] not in selected
                         and mask.area(mask.merge(mask.frPyObjects(row["segmentation"],800,800)))>0)
            png="images/"+Path(images[iid]["file_name"]).stem+".png"
            with Image.open(BytesIO(z.read(png))) as im:
                im.load()
                need(im.mode=="RGB" and im.size==(800,800),"Exact real FIT PNG RGB grid")
                _,empty=convert(im,dict(image_id=iid,annotations=[a]))
                _,good=convert(im,dict(image_id=iid,annotations=[control]))
            if len(empty["labels"])==1:
                stats["selected_empty_retained_as_label"]+=1
                stats["selected_empty_masks_zero"]+=int(empty["masks"].sum().item()==0)
                box=empty["boxes"][0]
                stats["selected_empty_positive_boxes"]+=int(bool((box[2:]>box[:2]).all()))
            if len(good["labels"])==1:
                stats["same_image_nonempty_controls_retained"]+=1
                stats["same_image_nonempty_control_masks_positive"]+=int(good["masks"].sum().item()>0)
    result=dict(status="ACTUAL_NATIVE_CONVERTER_SELECTED_FIT_EMPTY_WITNESS",selected_source_annotations=5,
        distinct_fit_images=len({x["image_id"] for x in private}),training=False,AP=False,
        opened_official_TRAIN_member_only=True,DEV_or_TEST_image_opened=False,TEST=False,**stats)
    with (HERE/"NATIVE_CONVERTER_EMPTY_WITNESS_SAFE.json").open("x",encoding="utf-8",newline="\n") as f:
        json.dump(result,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()

