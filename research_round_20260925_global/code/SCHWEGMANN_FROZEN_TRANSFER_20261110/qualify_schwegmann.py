"""Qualify author Schwegmann positives and make isolated image and reference packets."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha1, sha256
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED
import json
import os
import re
import sys

import numpy as np


SOURCE_BYTES = 289068441
SOURCE_SHA256 = "78ff85f49eb5102716ae6d39bdbafaf27fc9c63a630be0d86a2c434ab09f053b"
SOURCE_BLOB = "0df2987ad896f2b0f4b8db644cb5ca3cff1ce986"
SOURCE_COMMIT = "9decec2a18b9d7d6fd2d057a435808ca67d2bad1"
SELECTION = b"schwegmann-cohort-source-group-254101\0"
SIZE = (101, 101)
POL = {"HH","HV","VH","VV"}


def require(ok, why):
    if not ok:
        raise ValueError(why)


def gray_levels(power):
    require(np.isfinite(power).all() and (power >= 0).all(),"Finite nonnegative calibrated linear input required")
    db = 10. * np.log10(np.maximum(power,1e-4))
    return np.rint((np.clip(db,-40.,10.)+40.)*(255./50.)).astype(np.uint8)


def acquisition(record):
    pol = record["polarisation"]
    short = record["id"].rsplit("_",2)
    require(pol in POL and len(short)==3 and short[1]==pol and re.fullmatch(r"[0-9]{4}",short[2]),
            "Exact author polarization and positive ordinal")
    product = record["fullname"]
    if record["sensor"]=="SEN1":
        m=re.fullmatch(r"(S1A)_EW_GRD[HM]_[A-Z0-9]{4}_([0-9]{8})T[0-9]{6}_[0-9]{8}T[0-9]{6}_"
                       r"([0-9]{6})_([A-F0-9]{6})_([A-F0-9]{4})\.SAFE",product)
        require(m is not None and short[0]=="S1A_"+m[5].zfill(6),"Exact Sentinel-1A author alias to product suffix binding")
        return m[1],m[2],m[3],m[4]
    require(record["sensor"]=="RSAT","Exact public sensor namespace")
    m=re.fullmatch(r"RS2_(OK[0-9]+)_(PK[0-9]+)_(DK[0-9]+)_SCNA_([0-9]{8})_([0-9]{6})_([A-Z]+)_SGF",product)
    require(m is not None and short[0]=="RS2_"+m[5] and pol in m[6],"Exact RADARSAT-2 acquisition binding")
    return "RS2",m[4],m[5],m[1],m[2],m[3]


def stable_archive(path):
    require(path.is_absolute() and path.resolve(strict=True)==path and path.is_file(),"Canonical fixed source required")
    with path.open("rb") as stream:
        before=os.fstat(stream.fileno())
        require(before.st_size==SOURCE_BYTES,"Exact original file size required")
        s256=sha256()
        blob=sha1(f"blob {SOURCE_BYTES}\0".encode())
        for chunk in iter(lambda:stream.read(1024**2),b""):
            s256.update(chunk)
            blob.update(chunk)
        require(s256.hexdigest()==SOURCE_SHA256 and blob.hexdigest()==SOURCE_BLOB,"Exact author content and Git blob required")
        stream.seek(0)
        payload=json.load(stream)
        after=os.fstat(stream.fileno())
        def identity(s):
            return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns
        require(identity(before)==identity(after)==identity(path.stat()),"Source file changed while qualifying")
    require(type(payload) is list and len(payload)==1 and type(payload[0]) is list
            and len(payload[0])==1596 and all(type(record) is dict for record in payload[0]),"Exact author record nesting")
    return payload[0]


def fixed_rows(records):
    groups=defaultdict(set)
    aliases={}
    candidates=defaultdict(list)
    pols=Counter()
    negative=Counter()
    nonfinite=Counter()
    ids=set()
    for record in records:
        key=acquisition(record)
        groups[key].add(record["fullname"])
        alias=record["id"].rsplit("_",2)[0]
        require(aliases.setdefault(alias,record["fullname"])==record["fullname"],"An author alias maps to multiple products")
        ident=record["id"]
        require(ident not in ids,"Duplicate author positive ID")
        ids.add(ident)
        pol=record["polarisation"]
        pols[pol]+=1
        power=np.asarray(record["patchfu"],dtype="<f8")
        mask=np.asarray(record["patchgt"])
        require(power.shape==mask.shape==SIZE and np.isin(mask,(0,1)).all() and mask.any()
                and type(record["row"]) is int and type(record["column"]) is int,"Author 101 binary nonempty reference required")
        if not np.isfinite(power).all():
            nonfinite[pol]+=1
            continue
        if (power<0).any():
            negative[pol]+=1
            continue
        gray=gray_levels(power)
        if pol not in ("HH","VV"):
            continue
        raw_sha=sha256(power.tobytes()).hexdigest()
        input_sha=sha256(gray.tobytes()).hexdigest()
        candidates[input_sha].append(dict(key=key,ident=ident,raw=raw_sha,gray=gray,mask=mask.astype(np.uint8),
                                          input=input_sha,polarisation=pol,
                                          low_clip=int(np.count_nonzero(power<1e-4)),
                                          high_clip=int(np.count_nonzero(power>10.))))
    products={product for rows in groups.values() for product in rows}
    require(len(groups)==16 and len(products)==len(aliases)==len(set(aliases.values()))==25
            and Counter(key[0] for key in groups)=={"S1A":13,"RS2":3}
            and Counter(map(len,groups.values()))=={1:7,2:9},"Exact conservative acquisition grouping")
    qualified=defaultdict(list)
    repeated=conflict=0
    for same in candidates.values():
        if len({row["key"] for row in same})!=1 or any(not np.array_equal(row["mask"],same[0]["mask"]) for row in same):
            conflict+=len(same)
            continue
        same.sort(key=lambda row:(row["raw"],row["ident"]))
        chosen=same[0]
        repeated+=len(same)-1
        chosen["rank"]=sha256(SELECTION+bytes.fromhex(chosen["input"])).hexdigest()
        qualified[chosen["key"]].append(chosen)
    require(set(qualified)==set(groups),"All author acquisition groups must retain a qualified co-polar record")
    selected=[]
    for group,key in enumerate(sorted(qualified),1):
        rows=sorted(qualified[key],key=lambda row:row["rank"])
        selected.extend((f"acq_{group:03}",row) for row in rows[:2])
    require(len({row["input"] for _,row in selected})==len(selected),"Unique actual model inputs required")
    return selected,dict(author_records=len(records),acquisition_groups=len(groups),author_products=len(products),
                         grouped_product_pairs=sum(len(items)==2 for items in groups.values()),original_polarisation_counts=dict(pols),
                         copol_candidates=sum(pols[p]-negative[p]-nonfinite[p] for p in ("HH","VV")),
                         excluded_negative_by_polarisation=dict(negative),excluded_nonfinite_by_polarisation=dict(nonfinite),exact_model_input_duplicates=repeated,
                         conflicting_or_cross_acquisition_duplicates=conflict,
                         sensor_acquisitions=dict(Counter(key[0] for key in groups)),
                         earliest_acquisition_date=min(key[1] for key in groups),latest_acquisition_date=max(key[1] for key in groups))


def write(path,value):
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with path.open("xb") as stream:
        require(stream.write(raw)==len(raw),"Complete manifest write")
        stream.flush()
        os.fsync(stream.fileno())
    return sha256(raw).hexdigest()


def qualify(args):
    from PIL import Image
    from pycocotools import mask as coco_mask
    require(version("Pillow")=="12.3.0" and version("pycocotools")=="2.0.11" and np.__version__=="1.26.4",
            "Pinned author cohort encode runtime")
    out=args.output
    require(out.is_absolute() and out.parent.resolve(strict=True)==out.parent and not out.exists() and not out.is_symlink(),
            "Fresh isolated cohort location")
    selected,census=fixed_rows(stable_archive(args.archive))
    out.mkdir()
    (out/"IMAGE_ONLY").mkdir()
    (out/"SCORER_ONLY").mkdir()
    images,references=[],[]
    zip_path=out/"IMAGE_ONLY/IMAGES_ONLY_PRIVATE.zip"
    with ZipFile(zip_path,"x",compression=ZIP_STORED) as bundle:
        for i,(group,row) in enumerate(selected,1):
            name=f"chip_{i:04}.png"
            data=BytesIO()
            Image.fromarray(row["gray"]).convert("RGB").save(data,format="PNG",compress_level=6)
            raw=data.getvalue()
            bundle.writestr(name,raw)
            images.append(dict(image_id=i,basename=name,bytes=len(raw),sha256=sha256(raw).hexdigest()))
            rle=coco_mask.encode(np.asfortranarray(row["mask"]))
            references.append(dict(image_id=i,acquisition=group,
                segmentation=dict(size=list(rle["size"]),counts=rle["counts"].decode("ascii"))))
    with ZipFile(zip_path) as bundle:
        require(bundle.namelist()==[r["basename"] for r in images],"Exact image-only ZIP member order")
        for reference,(_,row) in zip(images,selected,strict=True):
            data=bundle.read(reference["basename"])
            require(len(data)==reference["bytes"] and sha256(data).hexdigest()==reference["sha256"],"Exact image readback")
            with Image.open(BytesIO(data)) as image:
                require(image.mode=="RGB" and np.array_equal(np.asarray(image),np.repeat(row["gray"][:,:,None],3,axis=2)),
                        "Lossless repeated-channel PNG readback")
    mp=out/"IMAGE_ONLY/IMAGE_MANIFEST_PRIVATE.json"
    gp=out/"SCORER_ONLY/PATCHGT_PRIVATE.json"
    manifest_sha=write(mp,dict(status="P254_SCHWEGMANN_IMAGE_ONLY_V1",images=images))
    gt_sha=write(gp,dict(status="P254_SCHWEGMANN_SCORER_ONLY_GT_V1",images=references))
    for reference,(_,row) in zip(references,selected,strict=True):
        packed=dict(reference["segmentation"],counts=reference["segmentation"]["counts"].encode())
        require(np.array_equal(coco_mask.decode(packed),row["mask"]),"Lossless independent author mask encoding")
    pixels=len(images)*SIZE[0]*SIZE[1]
    summary=dict(status="P254_SCHWEGMANN_COHORT_QUALIFIED_BEFORE_MODELS",**census,images=len(images),
                 groups_with_two=sum(x==2 for x in Counter(group for group,_ in selected).values()),
                 source_commit=SOURCE_COMMIT,source_sha256=SOURCE_SHA256,source_git_blob=SOURCE_BLOB,
                 image_manifest_sha256=manifest_sha,image_manifest_bytes=mp.stat().st_size,
                 image_zip_sha256=sha256(zip_path.read_bytes()).hexdigest(),image_zip_bytes=zip_path.stat().st_size,
                 ground_truth_sha256=gt_sha,ground_truth_bytes=gp.stat().st_size,
                 gray_clip_low_fraction=sum(row["low_clip"] for _,row in selected)/pixels,
                 gray_clip_high_fraction=sum(row["high_clip"] for _,row in selected)/pixels,model_forward_calls=0)
    result_sha=write(out/"QUALIFICATION_SAFE.json",summary)
    print(json.dumps(dict(summary_sha256=result_sha,**summary),sort_keys=True),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("--archive",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    qualify(parser.parse_args())
