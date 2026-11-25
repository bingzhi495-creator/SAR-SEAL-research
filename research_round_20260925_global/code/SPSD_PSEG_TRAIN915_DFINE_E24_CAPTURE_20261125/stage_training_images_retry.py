"""Project a closed dual-reference TRAIN admission onto verified image-only bytes."""
from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
from struct import pack
from types import ModuleType
from zipfile import ZipFile

from PIL import Image, ImageOps

ADMISSION = "e40636ec2cfb4c42718673a553f069f5f4e2833e3e81264c9d44a43ece7b9927"
ROSTER = "f4e23e5eb7f58a881db48e4a89d2902a253aa945e7a92ab7d1674c8b02b3fb17"
ADMISSION_WORKER = "f18b292b9fc2db05e23195273ed8226936af39ef309c50425770876b228a283a"
ZIP = "23505cd2de8faa3afa9b4cb329397c486724d02cc24482fd1883ba70a01f77a5"
STATUS = "SPSD_PSEG_TRAIN915_IMAGE_ONLY_READY"
SCHEMA = "SPSD_PSEG_DUAL_REFERENCE_TRAIN915_IMAGE_ONLY_V1"


def need(test, why):
    if not test:
        raise ValueError(why)


def digest(raw):
    return sha256(raw).hexdigest()


def read_hash(path, wanted):
    value = path.read_bytes()
    need(digest(value) == wanted, "Closed input digest: " + path.name)
    return json.loads(value)


def hash_file(path):
    checksum = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024**2), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def write_json(path,value):
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode("utf-8")
    with path.open("xb") as file:
        file.write(raw)
        file.flush()
        os.fsync(file.fileno())
    return digest(raw)


def prepare(a):
    terminal=json.loads((a.admission/"ADMISSION_TERMINAL_PRIVATE.json").read_bytes())
    safe=read_hash(a.admission/"admission/RESULT_SAFE.json",ADMISSION)
    roster=read_hash(a.admission/"admission/TRAIN_ROSTER_PRIVATE.json",ROSTER)
    readback=json.loads((a.admission/"READBACK_SAFE.json").read_bytes())
    need(terminal.get("actual_exit")==0 and terminal.get("file_checksum_verified")==ADMISSION
         and terminal.get("phase")=="training_roster_admission" and terminal.get("tool_session_id")==7469
         and readback=={"partition_and_hashes_verified":True,"exact_cross_test_groups":10,"train":915,
                        "train_decoded_groups":901,"removed":13,"result_sha256":ADMISSION,
                        "worker_sha256":ADMISSION_WORKER}
         and safe["status"]=="SPSD_PSEG_DUAL_REFERENCE_TRAIN_ROSTER_ADMITTED"
         and safe["worker_sha256"]==ADMISSION_WORKER and safe["admitted_roster_private_sha256"]==ROSTER
         and safe["slgsea_zip_sha256"]==ZIP and safe["admitted_train"]["images"]==915
         and safe["removed_cross_test"]["images"]==12 and safe["removed_invalid_reference"]["images"]==1
         and safe["test_annotation_members_read"]==0 and safe["model_forwards"]==safe["optimizer_updates"]==0,
         "Closed original roster admission and independently verified partition")
    admitted = roster["admitted_train"]
    groups = roster["groups_by_first_filename"]
    names = sorted(admitted)
    need(len(names)==915 and len(groups)==901 and sorted(v for g in groups for v in g)==names,
         "Exact admitted image roster and grouping partition")
    need(hash_file(a.archive)==ZIP, "Pinned author archive")
    image_dir = a.output / "FIT_IMAGES"
    image_dir.mkdir(parents=False, exist_ok=False)
    output=[]
    groups_by_rgb = {}
    inventory=sha256()
    with ZipFile(a.archive) as z:
        for name in names:
            row=admitted[name]
            need(re.fullmatch(r"[0-9]{6}\.jpg",name) and int(name[:6])%10 not in (1,9)
                 and row["cross_test_decoded_overlap"] is row["cross_test_encoded_overlap"] is row["invalid_dual_reference"] is False,
                 "Only admitted official TRAIN rows with neither TEST match")
            member=f"datasets/SPSD/train2017/{name}"
            need(row["spsd_image_member"]==member, "Open only the admitted JPEG member")
            raw=z.read(member)
            need(0<len(raw)<=4*1024**2 and digest(raw)==row["jpeg_sha256"],"Exact admitted JPEG bytes")
            w,h=row["width"],row["height"]
            with Image.open(BytesIO(raw)) as source:
                need(source.format=="JPEG" and source.mode=="RGB","Original color JPEG")
                rgb=ImageOps.exif_transpose(source).convert("RGB")
                need(rgb.size==(w,h),"Exact admitted native dimensions")
                pixels=rgb.tobytes()
            size_rgb=digest(pack(">II",w,h)+pixels)
            need(size_rgb==row["rgb_size_prefixed_sha256"], "Exact EXIF-oriented admitted RGB identity")
            groups_by_rgb.setdefault(size_rgb,[]).append(name)
            with (image_dir/name).open("xb") as out:
                out.write(raw)
            inventory.update(f"{name}\t{len(raw)}\t{row['jpeg_sha256']}\n".encode("ascii"))
            output.append(dict(image_id=row["pseg_image_id"],original_hw=[h,w],
                  image_ref=dict(path=name,bytes=len(raw),sha256=row["jpeg_sha256"]),
                  rgb_sha256=digest(pixels),rgb_size_prefixed_sha256=size_rgb,
                  source_group=group_key(row["decoded_group_representative"]),
                  source_group_size=row["decoded_group_size"]))
    need(sorted(sorted(x) for x in groups_by_rgb.values())==sorted(sorted(x) for x in groups)
         and len({x["image_id"] for x in output})==915 and len({x["source_group"] for x in output})==901,
         "Original decoded RGB grouping and unique PSeg training IDs")
    digest_inventory=inventory.hexdigest()
    man=dict(schema=SCHEMA,population="clean_author_train_915",FIT=True,DEV=False,TEST=False,
             GT_opened=False,AP=False,model_execution=False,source_admission_sha256=ADMISSION,
             source_roster_sha256=ROSTER,source_archive_sha256=ZIP,
             image_inventory_sha256=digest_inventory,images=915,unique_rgb_groups=901,
             cross_test_encoded=0,cross_test_decoded=0,expected_ids=[i["image_id"] for i in output],items=output)
    man_sha=write_json(a.output/"TRAIN_IMAGE_ONLY_MANIFEST_PRIVATE.json",man)
    result=dict(status=STATUS,admission_sha256=ADMISSION,roster_sha256=ROSTER,
                image_manifest_sha256=man_sha,image_inventory_sha256=digest_inventory,images=915,
                source_groups=901,exact_cross_test_encoded=0,exact_cross_test_decoded=0,
                train_label_members_read=0,test_label_members_read=0,test_image_members_read=0,
                detector_forwards=0,optimizer_updates=0,source_archive_sha256=ZIP)
    out_sha=write_json(a.output/"RESULT_SAFE.json",result)
    print(json.dumps(dict(status=STATUS,result_sha256=out_sha)),flush=True)


def group_key(basename):
    return digest(basename.encode("ascii"))


if __name__ == "__main__":
    p=ArgumentParser(allow_abbrev=False)
    for field in ("admission","archive","output"):
        p.add_argument("--"+field,type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir()
    prepare(args)
