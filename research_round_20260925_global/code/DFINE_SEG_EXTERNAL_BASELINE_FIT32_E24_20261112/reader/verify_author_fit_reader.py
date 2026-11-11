"""Qualify stock D-FINE-seg COCO polygon reader for the closed full HRSID FIT source."""
from argparse import ArgumentParser
from collections import defaultdict
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys
import warnings

import numpy as np

SOURCE_SHA = "eee25d1bd1476bcbd98bb000c683bc5b5966d78265c6d8104dbb29a0bb7b1ba2"
FIT_SHA = "e95cf26cc60a2c6fb651e907541380a256f2df7b1e153244a6c0f803b0b19f8f"
LAYOUT_SHA = "a7e4d2a7ff9e8ff238adbf94a1d0397553d8746e2558c316db4e417f86f8fec9"


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def checked(path, digest, limit=16*2**20):
    require(path.resolve(strict=True)==path and 0<path.stat().st_size<=limit, "Exact bounded file")
    data=path.read_bytes()
    require(sha256(data).hexdigest()==digest, "Pinned source: "+path.name)
    return data


def partition(source):
    grouped=defaultdict(list)
    for row in source["annotations"]:
        grouped[row["image_id"]].append(row)
    return grouped


def compare(raw,layout,stock,category_map):
    ids=[r["id"] for r in raw["images"]]
    require(len(ids)==len(set(ids))==2937 and len(raw["annotations"])==9260,
            "Exact native FIT population")
    by_id={r["image_id"]:r for r in layout["images"]}
    require(set(ids)==set(by_id) and len(layout["images"])==len(by_id) and category_map=={1:0},
            "Exact single-class canonical mapping")
    grouped=partition(raw)
    require(set(grouped)<=set(ids), "All annotations linked to the FIT roster")
    seen=parts_total=multipart=empty=invalid=skipped=negative=ignored=0
    for im,e in zip(raw["images"],stock,strict=True):
        name=im["file_name"]
        require(Path(name).name==name and name.lower().endswith(".png") and name==e["file_name"]==by_id[im["id"]]["native_basename"]
                and im["height"]==im["width"]==800,"Exact native image identity and order")
        candidates=grouped[im["id"]]
        ignored+=sum(bool(x.get("ignore",0)) for x in candidates)
        accepted=[a for a in candidates if not a.get("iscrowd",0) and a["category_id"] in category_map]
        skipped+=len(candidates)-len(accepted)
        require(len(accepted)==len(e["targets"])==len(e["polys_abs"]),"Exact stock row inclusion")
        negative+=not accepted
        for a, box, parts in zip(accepted,e["targets"],e["polys_abs"],strict=True):
            x,y,w,h=a["bbox"]
            expected=np.array([0,x,y,x+w,y+h],dtype=np.float32)
            require(np.array_equal(box,expected) and np.isfinite(expected).all() and w>0 and h>0,
                    "Exact positive finite native box and class")
            vector=a["segmentation"]
            require(isinstance(vector,list) and len(parts)==sum(len(p)>=6 for p in vector),
                    "Exact polygon parts")
            empty+=not parts
            multipart+=len(parts)>1
            invalid+=sum(len(p)<6 for p in vector)
            parts_total+=len(parts)
            for p,s in zip((p for p in vector if len(p)>=6),parts,strict=True):
                p=np.asarray(p,dtype=np.float32).reshape(-1,2)
                require(np.array_equal(s,p) and np.isfinite(p).all(),"Exact native float32 vertices")
            seen+=1
    return dict(images=len(ids),source_annotations=len(raw["annotations"]),stock_targets=seen,source_skipped=skipped,
                flagged_ignore_source=ignored,empty_image_rows=negative,empty_polygon_targets=empty,
                polygon_parts=parts_total,multipart_targets=multipart,short_polygon_parts=invalid)


def run(args):
    from dfine_seg.dl import dataset
    require(version("dfine-seg")=="0.7.0","Exact pinned author package version")
    checked(Path(dataset.__file__).resolve(strict=True),SOURCE_SHA,2**20)
    require(args.fit_index.stat().st_size==30076663,"Exact native FIT index byte count")
    raw=json.loads(checked(args.fit_index,FIT_SHA,30076663))
    layout=json.loads(checked(args.layout_manifest,LAYOUT_SHA))
    require(layout["native_index_sha256"]==FIT_SHA,"Layout and FIT index join")
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        stock,mapping=dataset.load_coco_split(args.fit_index,use_one_class=True)
    observed=compare(raw,layout,stock,mapping)
    out=dict(status="DFINE_HRSID_FIT2937_READER_ACTUAL",FIT=True,DEV=False,TEST=False,AP=False,model_forwards=0,
             optimizer_updates=0,image_decodes=0,fit_index_sha256=FIT_SHA,native_manifest_sha256=LAYOUT_SHA,
             author_dataset_sha256=SOURCE_SHA,warning_count=len(records),
             warnings_bbox_escape=sum("bbox" in str(x.message).lower() for x in records),**observed)
    data=(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with args.output.open("xb") as f:
        f.write(data)
    print(json.dumps(dict(status=out["status"],sha256=sha256(data).hexdigest(),counts=observed)))


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    for name in ("fit-index","layout-manifest","output"):
        p.add_argument("--"+name,type=Path,required=True)
    def offline(event,_args):
        if event.startswith("socket."):
            raise PermissionError("Offline FIT reader qualification")
    sys.addaudithook(offline)
    run(p.parse_args())
