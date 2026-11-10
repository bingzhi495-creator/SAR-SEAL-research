"""Exhaustive whole-frame screen for aligned image resize and re-encoding."""
import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from time import monotonic
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image
from scipy.stats import rankdata

sys.path.insert(0,str(Path(__file__).resolve(strict=True).parent))
import screen

SALT="HRSID_SSDD_CROSSSET_DENSE_WHOLE_FRAME_V1"
THRESHOLD=.95
SIZE=64
FIELDS=("coarse_rank","detail_rank")


def signature(rgb):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    grid=cv2.resize(gray,(SIZE,SIZE),interpolation=cv2.INTER_AREA).astype(np.float32)
    details=grid-cv2.GaussianBlur(grid,(0,0),1.75)
    def unit(values):
        ranks=rankdata(values.reshape(-1)).astype(np.float32)
        ranks-=ranks.mean()
        norm=np.linalg.norm(ranks)
        return ranks/norm if norm>0 else np.zeros_like(ranks)
    return np.stack([unit(grid),unit(details)])


def variants(rgb,ordinal):
    pic=Image.fromarray(rgb)
    jpeg=screen.jpeg_roundtrip(pic.resize((600,420),Image.Resampling.BICUBIC),75)
    gamma=.8 if ordinal%2 else 1.25
    table=np.rint(np.power(np.arange(256,dtype=float)/255,gamma)*255).astype(np.uint8)
    adjusted=Image.fromarray(table[rgb]).resize((420,600),Image.Resampling.BICUBIC)
    jpeg_gamma=screen.jpeg_roundtrip(adjusted,70)
    rng=np.random.default_rng(110250+ordinal)
    blocks=rgb.reshape(5,160,5,160,3).transpose(0,2,1,3,4).reshape(25,160,160,3)
    reordered=blocks[rng.permutation(25)].reshape(5,5,160,160,3).transpose(0,2,1,3,4)
    return {"resize_jpeg":jpeg,"resize_jpeg_gamma":jpeg_gamma,"shuffle_negative":reordered.reshape(800,800,3)}


def control_roster(rows):
    used={rows[i]["native_basename"].split("_",1)[0] for i in screen.choose_controls(rows)}
    selected=[]
    for i in sorted(range(len(rows)),key=lambda i:sha256((SALT+rows[i]["encoded_sha256"]).encode()).hexdigest()):
        group=rows[i]["native_basename"].split("_",1)[0]
        if group not in used:
            used.add(group)
            selected.append(i)
            if len(selected)==24:
                return selected
    raise ValueError("24 disjoint source-prefix controls required")


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for flag in ("fit-archive","fit-manifest","dev-archive","dev-manifest","output"):
        parser.add_argument("--"+flag,type=Path,required=True)
    args=parser.parse_args()
    screen.require(sys.flags.isolated==1 and sys.dont_write_bytecode
                   and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1","Isolated CPU only")
    screen.require(args.output.is_absolute() and not args.output.exists(),"Fresh private output")
    cv2.setNumThreads(1)
    t=monotonic()
    fit=json.loads(screen.checked(args.fit_manifest,screen.FIT_MANIFEST_SHA,2**21))
    rows=sorted(fit["images"],key=lambda x:x["zip_header_offset"])
    screen.require(len(rows)==2937 and len({x["member"] for x in rows})==2937,"Exact closed FIT whitelist")
    control=control_roster(rows)
    inputs={}
    bank=[]
    with args.fit_archive.open("rb") as fp:
        before=os.fstat(fp.fileno())
        screen.require(before.st_size==screen.HRSID_BYTES,"Exact official archive bytes")
        fp.seek(screen.CENTRAL_OFFSET)
        screen.require(sha256(fp.read(screen.CENTRAL_BYTES)).hexdigest()==screen.CENTRAL_SHA,"Exact central directory")
        with ZipFile(fp) as archive:
            for i,row in enumerate(rows):
                rgb=screen.read_fit_member(archive,row)
                bank.append(signature(rgb))
                if i in control:
                    inputs[i]=rgb.copy()
                if (i+1)%800==0:
                    print(json.dumps(dict(stage="FIT dense signatures",images=i+1)),flush=True)
        after=os.fstat(fp.fileno())
        screen.require((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),
                       "Original archive changed")
    a=np.stack(bank).transpose(1,0,2).copy()

    def search(rgb):
        query=signature(rgb)
        return np.minimum(a[0] @ query[0],a[1] @ query[1])

    ctr={k:Counter(attempts=0,source_match=0,any_match=0)
         for k in ("resize_jpeg","resize_jpeg_gamma","shuffle_negative")}
    for ordinal,i in enumerate(control):
        for name,rgb in variants(inputs[i],ordinal).items():
            s=search(rgb)
            ctr[name]["attempts"]+=1
            ctr[name]["source_match"]+=float(s[i])>=THRESHOLD
            ctr[name]["any_match"]+=float(s.max())>=THRESHOLD
    passed=(ctr["resize_jpeg"]["source_match"]>=22 and ctr["resize_jpeg_gamma"]["source_match"]>=22
            and ctr["shuffle_negative"]["any_match"]==0)
    print(json.dumps(dict(stage="dense controls locked",controls=ctr,passed=passed)),flush=True)

    m=json.loads(screen.checked(args.dev_manifest,screen.DEV_MANIFEST_SHA,2**20))
    dev=m["items"]
    screen.require(m["TEST"] is False and m["AP"] is False and m["model_execution"] is False
                   and m["expected_ids"]==[x["image_id"] for x in dev] and len(dev)==187,"Exact closed DEV only")
    g={}
    for x in dev:
        g.setdefault(x["image_ref"]["sha256"],[]).append(x)
    screen.require(len(g)==183,"Exact closed DEV JPEG count")
    scored=[]
    with ZipFile(BytesIO(screen.checked(args.dev_archive,screen.DEV_ZIP_SHA,8*2**20))) as archive:
        names={x["image_id"]:"valid/"+Path(x["image_ref"]["path"]).name for x in dev}
        screen.require(set(archive.namelist())==set(names.values()) and len(archive.namelist())==187,
                       "Exact original JPEG members")
        for ordinal,(digest,items) in enumerate(g.items(),1):
            rgb=None
            for row in items:
                raw=archive.read(names[row["image_id"]])
                screen.require(len(raw)==row["image_ref"]["bytes"] and sha256(raw).hexdigest()==digest,
                               "Exact DEV JPEG bytes")
                decoded=np.asarray(screen.image(raw,"JPEG"))
                screen.require(tuple(decoded.shape[:2])==tuple(row["original_hw"]),"Exact DEV JPEG shape")
                if rgb is None:
                    rgb=decoded
            s=search(rgb)
            hit=np.flatnonzero(s>=THRESHOLD)
            scored.append(dict(ordinal=ordinal,multiplicity=len(items),max_score=float(s.max()),
                               matched_source_indices=[int(i) for i in hit]))
    n=[row for row in scored if row["matched_source_indices"]]
    result=dict(status="HRSID_SSDD_DENSE_ALIGNED_WHOLE_FRAME_SCREEN_COMPLETE_V1",fit_images=len(rows),
                dev_images=len(dev),dev_unique=len(g),all_pairs=len(rows)*len(g),
                fit_manifest_sha256=screen.FIT_MANIFEST_SHA,dev_manifest_sha256=screen.DEV_MANIFEST_SHA,
                dev_zip_sha256=screen.DEV_ZIP_SHA,fit_central_sha256=screen.CENTRAL_SHA,
                scope="whole_frame_axis_aligned_only",screen_features=list(FIELDS),
                threshold=THRESHOLD,grid=SIZE,controls=ctr,controls_passed=passed,
                best_actual_similarity=max(x["max_score"] for x in scored),
                candidate_pairs=sum(len(x["matched_source_indices"]) for x in n),
                unique_dev_matches=len(n),logical_dev_matches=sum(x["multiplicity"] for x in n),
                elapsed_seconds=monotonic()-t,official_TEST_opened=False,annotation_reads=0,model_forward_calls=0,
                acquisition_identity_verified=False)
    args.output.mkdir()
    for name,value in (("RESULT_SAFE.json",result),("CANDIDATES_PRIVATE.json",scored)):
        with (args.output/name).open("x",encoding="utf-8",newline="\n") as f:
            json.dump(value,f,sort_keys=True,indent=2)
            f.write("\n")
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
