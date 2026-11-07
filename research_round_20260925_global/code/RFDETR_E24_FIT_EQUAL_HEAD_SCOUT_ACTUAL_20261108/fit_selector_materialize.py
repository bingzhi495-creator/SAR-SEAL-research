"""Materialize only the fixed FIT selector labels and equal-support features."""
from collections import Counter, defaultdict
from contextlib import redirect_stdout
import hashlib
from importlib.metadata import version
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import time
import types
from zipfile import ZipFile


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def checked(ref, maximum=64*1024**2):
    p=Path(ref["path"])
    need(p.is_absolute() and p.resolve(strict=True)==p and p.is_file() and not p.is_symlink()
         and 0<ref["bytes"]<=maximum,"Canonical bounded input")
    before=p.stat()
    b=p.read_bytes()
    after=p.stat()
    key=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
    need(len(b)==before.st_size==ref["bytes"] and digest(b)==ref["sha256"]
         and key(before)==key(after),"Exact stable input bytes")
    return b


def doc(ref):
    return json.loads(checked(ref))


def encode(value):
    return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def put(path,value):
    raw=encode(value)
    with path.open("xb") as f:
        need(f.write(raw)==len(raw),"Complete output file")
        f.flush()
        os.fsync(f.fileno())
    return file_ref(path)


def file_ref(path):
    b=path.read_bytes()
    return dict(path=path.as_posix(),bytes=len(b),sha256=digest(b))


def load(ref,name):
    api=types.ModuleType(name)
    api.__file__=ref["path"]
    sys.modules[name]=api
    exec(compile(checked(ref,1024**2),api.__file__,"exec"),api.__dict__)
    return api


def validate_gt_population(gt,items):
    images={r["id"]:[r["height"],r["width"]] for r in gt["images"]}
    expected={r["image_id"]:r["original_hw"] for r in items}
    need(len(images)==len(gt["images"])==len(expected)==len(items) and images==expected
         and [r["id"] for r in gt["categories"]]==[1],"Exact integer FIT annotation image and category population")
    annotations=gt["annotations"]
    need(len({r["id"] for r in annotations})==len(annotations)
         and all(r["image_id"] in images and r["category_id"]==1 for r in annotations),
         "All annotation IDs uniquely refer to admitted FIT")


def original_gt(spec,items):
    bundle=checked(spec["FIT_bundle"])
    with ZipFile(BytesIO(bundle)) as z:
        ann=z.read("annotations/fit.json")
    need((len(ann),digest(ann))==(498760,"62966857c78cea1e3b060700c62a7b896942686c2e674d3eb09f516a92250bc1"),
         "Original FIT integer label bytes")
    gt=json.loads(ann)
    validate_gt_population(gt,items)
    return gt


def captured_item(root,item,index):
    n=item["dataset_index"]
    dirname=f"image-{n:04d}"
    need(index["dataset_index"]==n and index["dirname"]==dirname
         and (index["image_id"],index["partition"])==(item["image_id"],item["partition"]),
         "Exact accepted CUDA order and FIT partition")
    out={}
    for role,name in (("records","ROWS.json"),("sparse","SPARSE.npz")):
        r=index[role]
        need(r["filename"]==name,"Exact native capture artifact role")
        out[role]=checked(dict(path=(root/dirname/name).as_posix(),bytes=r["bytes"],sha256=r["sha256"]),16*1024**2)
    row=json.loads(out["records"])
    need((row["image_id"],row["original_hw"],row["image_sha256"])
         ==(item["image_id"],item["original_hw"],item["image_ref"]["sha256"]),"Same accepted original CUDA image")
    return row,out["sparse"]


def runtime(spec):
    for r in spec["codec_refs"].values():
        checked(r,16*1024**2)
    sys.path.insert(0,spec["codec_overlay"])
    import numpy as np
    from pycocotools import coco,cocoeval,mask,_mask
    for role,m in (("coco",coco),("cocoeval",cocoeval),("mask",mask),("extension",_mask)):
        need(Path(m.__file__)==Path(spec["codec_refs"][role]["path"]),"Exact admitted COCO runtime role")
    need(version("pycocotools")=="2.0.11" and np.__version__=="1.26.4","Pinned CPU numerical and COCO versions")
    return np,coco,cocoeval,mask


def run(spec_path,expected_sha):
    start=time.monotonic()
    raw=spec_path.read_bytes()
    need(digest(raw)==expected_sha,"Frozen materialization spec")
    spec=json.loads(raw)
    need(spec["schema"]=="Root-FIT741-selector-materialization-v1" and spec["DEV"] is spec["TEST"] is False
         and os.environ.get("CUDA_VISIBLE_DEVICES")=="" and not any(n.split(".")[0]=="torch" for n in sys.modules),
         "CPU-only original FIT scope")
    np,coco,cocoeval,codec=runtime(spec)
    from PIL import Image
    modules={k:load(r,"_fit_materialize_"+k) for k,r in spec["sources"].items()}
    fit=doc(spec["image_manifest"])
    index=doc(spec["capture_index"])
    accepted=doc(spec["capture_acceptance"])
    items=fit["items"]
    need(accepted["status"]=="ACTUAL_FULL_FIT741_NATIVE_CUDA_CAPTURE_ACCEPTED"
         and accepted["original_index_ref"]==spec["capture_index"] and len(items)==len(index)==741
         and [x["dataset_index"] for x in items]==list(range(741))
         and fit["logical_counts"]=={"head_fit":516,"calibration":114,"head_holdout":111},
         "Same verified CUDA original and grouped scope")
    gt=original_gt(spec,items)
    tree=coco.COCO()
    tree.dataset=gt
    out=spec_path.parent
    res=[]
    counts=Counter()
    by=defaultdict(Counter)
    with (out/"OFFICIAL_MATCH.stdout").open("x",encoding="utf-8") as coco_log:
        with redirect_stdout(coco_log):
            tree.createIndex()
        for item,source in zip(items,index,strict=True):
            row,data=captured_item(Path(spec["capture_root"]),item,source)
            with Image.open(BytesIO(checked(item["image_ref"]))) as im:
                gray=np.ascontiguousarray(im.convert("RGB").convert("L"),dtype=np.uint8)
            with np.load(BytesIO(data),allow_pickle=False) as arrays,redirect_stdout(coco_log):
                material=modules["materializer"].materialize_image(
                    np,codec,cocoeval.COCOeval,integer_coco=tree,rows_document=row,arrays=arrays,gray=gray,
                    verify_capture=modules["capture_post"].verify_capture,
                    build_fixed_labels=modules["labels"].build_fixed_labels,
                    extract_features=modules["features"].extract_features)
            n=item["dataset_index"]
            directory=out/f"image-{n:04d}"
            directory.mkdir()
            ar=directory/"FEATURES.npz"
            with ar.open("xb") as f:
                np.savez(f,**material["arrays"])
                f.flush()
                os.fsync(f.fileno())
            m=put(directory/"MATCH.json",dict(rows=material["match_rows"],summary=material["summary"]))
            res.append(dict(dataset_index=n,image_id=item["image_id"],partition=item["partition"],match_ref=m,
                            arrays_ref=file_ref(ar)))
            s=material["summary"]
            counts.update(s)
            by[item["partition"]].update(s)
            if len(res)%25==0 or len(res)==741:
                print(json.dumps(dict(event="ACTUAL_FIT_MATERIALIZATION_PREFIX",images=len(res),
                           matched=counts["matched"],elapsed_seconds=time.monotonic()-start)),flush=True)
    idx=put(out/"MATERIALIZED_INDEX_PRIVATE.json",res)
    put(out/"RESULT.json",dict(status="FULL_FIT_FIXED_MATCH_FEATURES_COMPLETE_PENDING_ROOT",images=len(res),
           capture_acceptance_ref=spec["capture_acceptance"],index_ref=idx,original_FIT_annotation_count=len(gt["annotations"]),
           total=dict(counts),by_partition={k:dict(v) for k,v in by.items()},
           training=False,AP=False,DEV=False,TEST=False,elapsed_seconds=time.monotonic()-start))


if __name__=="__main__":
    need(len(sys.argv)==3,"Exact spec path and SHA")
    run(Path(sys.argv[1]),sys.argv[2])
