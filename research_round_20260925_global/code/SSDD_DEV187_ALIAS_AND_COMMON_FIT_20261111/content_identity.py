"""Check exact encoded/RGB split overlap and raster differences of DEV byte aliases."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import sys
from types import ModuleType
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageOps
from pycocotools import mask as codec
from scipy.optimize import linear_sum_assignment

FIT_SHA = "1c8f9b75e79fbc02c983ccd283a70082fcd5ab0d6538c777e8b286dbfdb78100"
DEV_SHA = "18df007e15e3e838e607f178f14e7a26f2b37b992ee39bf091e0b74cf084fb13"
ALIAS_SOURCE_SHA = "88086b3ff6a31d5788805ecb690472b01dbae2f1f92614fb4a946ae342b0f271"
RESULT = "SSDD_FIT_DEV_EXACT_CONTENT_AND_ALIAS_GT_COMPLETE"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(data):
    return sha256(data).hexdigest()


def exact(path, expected, maximum):
    require(path.is_absolute() and path.resolve(strict=True)==path and path.is_file()
            and 0<path.stat().st_size<=maximum, "Canonical bounded closed bytes required")
    raw = path.read_bytes()
    require(digest(raw)==expected,"Fixed input digest differs")
    return raw


def rgb_id(data, shape=None):
    with Image.open(BytesIO(data)) as original:
        require(original.format=="JPEG","Original JPEG required")
        picture = ImageOps.exif_transpose(original).convert("RGB")
        dims = (picture.height,picture.width)
        if shape is not None:
            require(dims==tuple(shape),"Fixed original dimensions differ")
        return digest(np.asarray(dims,dtype="<u4").tobytes()+picture.tobytes())


def fit_identities(archive_raw):
    encoded, pixels=set(),set()
    with ZipFile(BytesIO(archive_raw)) as z:
        names=z.namelist()
        require(len(names)==len(set(names))==732 and {"annotations/fit.json","manifest.json"}<=set(names),
                "Exact closed FIT-only archive shape")
        for name in names:
            if name in {"annotations/fit.json","manifest.json"}:
                continue
            m=re.fullmatch(r"images/([a-f0-9]{64})\.jpg",name)
            require(m is not None and 0<z.getinfo(name).file_size<=4*1024**2,"Bounded encoded FIT member")
            raw=z.read(name)
            require(digest(raw)==m[1],"FIT content-addressed encoded name mismatch")
            encoded.add(m[1])
            pixels.add(rgb_id(raw))
    require(len(encoded)==730,"Exact FIT unique encoded roster")
    return encoded,pixels


def dev_identities(items, root):
    pixels={}
    for item in items:
        ref=item["image_ref"]
        key=ref["sha256"]
        if key in pixels:
            continue
        name = Path(ref["path"]).name
        require(re.fullmatch(r"\d{6}\.jpg",name) is not None,"Fixed DEV basename required")
        file = root/name
        require(root in file.resolve(strict=True).parents,"Fixed DEV image root required")
        pixels[key]=rgb_id(exact(file,key,4*1024**2),item["original_hw"])
    require(len(pixels)==183,"Exact DEV unique encoded roster")
    return set(pixels),set(pixels.values())


def compare_alias_gt(gt, aliases):
    dims={row["id"]:(row["height"],row["width"]) for row in gt["images"]}
    masks={i:[] for pair in aliases for i in pair}
    for row in gt["annotations"]:
        i=row["image_id"]
        if i not in masks:
            continue
        shape=dims[i]
        poly=row["segmentation"]
        require(type(poly) is list and row["iscrowd"]==0,"Exact ordinary ship polygons required")
        masks[i].append(codec.merge(codec.frPyObjects(poly,*shape)))
    ious=[]
    pair_ranges=[]
    for a,b in aliases:
        ma,mb=masks[a],masks[b]
        require(len(ma)==len(mb) and len(ma)>0 and dims[a]==dims[b],"Paired alias label cardinality differs")
        matrix=codec.iou(ma,mb,[0]*len(mb))
        u,v=linear_sum_assignment(-matrix)
        values=matrix[u,v].tolist()
        ious.extend(values)
        pair_ranges.append(dict(ships=len(values),minimum=min(values),mean=float(np.mean(values)),
                                matched_exact=sum(x==1 for x in values),at_90=sum(x>=.90 for x in values)))
    return dict(matched=len(ious),exact=sum(x==1 for x in ious),at_50=sum(x>=.50 for x in ious),
                at_75=sum(x>=.75 for x in ious),at_90=sum(x>=.90 for x in ious),
                mean=float(np.mean(ious)),median=float(np.median(ious)),minimum=min(ious),
                sorted_alias_summaries=sorted(pair_ranges,key=lambda x:(x["ships"],x["minimum"])))


def run(a):
    source=exact(a.alias_source,ALIAS_SOURCE_SHA,1024**2)
    alias=ModuleType("_p331_alias_source")
    alias.__file__=str(a.alias_source)
    exec(compile(source,str(a.alias_source),"exec"),alias.__dict__)
    p299=alias.module_from(a.code/alias.P299[0],alias.P299[1],"_p331_p299")
    rec=p299.decode(p299.pinned_bytes(a.accepted,p299.ACCEPTED_SHA256,1024**2,"closed P295"),dict)
    require(rec["coordinator_closed"] is True and rec["coordinator_original_os_exit"]==0
            and rec["DEV"] is True and rec["TEST"] is False and rec["image_manifest_sha256"]==DEV_SHA,
            "Closed original DEV acceptance required")
    g=alias.module_from(a.code/p299.GEOMETRY_WRAPPER,p299.GEOMETRY_SHA256,"_p331_geometry")
    _, _, _, scoring, _, _ = g.load_helpers(a.code,a.code/p299.GEOMETRY_CORE)
    raw=exact(a.manifest,DEV_SHA,1024**2)
    _,ids,items=scoring.fixed_images(scoring.Artifact(dict(path=a.manifest.as_posix(),bytes=len(raw),sha256=DEV_SHA),raw))
    gt=p299.decode(p299.pinned_bytes(a.gt,rec["integer_gt_sha256"],1024**2,"fixed integer GT",
                                    exact_size=scoring.INTEGER_GT_BYTES),dict)
    scoring.validate_gt(gt,ids,items)
    _,pairs=alias.duplicate_groups(items)
    require([len(p) for p in pairs]==[2]*4,"Four fixed identical-byte pairs")
    fit,fit_rgb=fit_identities(exact(a.fit,FIT_SHA,34*1024**2))
    dev,dev_rgb=dev_identities(items,a.dev_root)
    return dict(status=RESULT,FIT_images_read=True,DEV_images_read=True,TEST=False,model_forward_calls=0,
                original_pair_sha256=p299.ACCEPTED_SHA256,fit_archive_sha256=FIT_SHA,dev_manifest_sha256=DEV_SHA,
                FIT=dict(logical_images=741,encoded_unique=len(fit),rgb_unique=len(fit_rgb)),
                DEV=dict(logical_images=187,encoded_unique=len(dev),rgb_unique=len(dev_rgb)),
                exact_encoded_intersection=len(fit&dev),exact_rgb_intersection=len(fit_rgb&dev_rgb),
                rgb_rule="EXIF-oriented RGB plus shape",alias_gt=compare_alias_gt(gt,pairs),
                near_duplicate_or_acquisition_identity_assessed=False,AP=False)


def deny_network(event,_):
    if event.startswith("socket."):
        raise PermissionError("closed-input CPU analysis")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("code","alias-source","accepted","manifest","gt","fit","dev-root","output"):
        p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args()
    require(a.dev_root.is_absolute() and a.dev_root.resolve(strict=True)==a.dev_root
            and a.output.is_absolute() and a.output.parent.resolve(strict=True)==a.output.parent
            and not a.output.exists() and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1","CPU and fresh output")
    sys.addaudithook(deny_network)
    raw=(json.dumps(run(a),sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with a.output.open("xb") as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())
    print(json.dumps(dict(status=RESULT,result_sha256=digest(raw))),flush=True)


if __name__=="__main__":
    main()
