"""Capture one frozen D-FINE center candidate per Schwegmann image without labels."""

import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from types import ModuleType
import os
import sys
from time import perf_counter


CARRIER = "8a7dd193e1d4e14eae528cba453941af9bc7d646646f34888726ec9287e0ce12"
SOURCE = "78ff85f49eb5102716ae6d39bdbafaf27fc9c63a630be0d86a2c434ab09f053b"


def need(ok, why):
    if not ok:
        raise ValueError(why)


def pinned(p, sha):
    raw=p.read_bytes()
    need(sha256(raw).hexdigest()==sha, "Exact closed file " + p.name)
    return raw


def module(p, sha):
    obj=ModuleType(p.stem)
    obj.__file__=str(p)
    exec(compile(pinned(p,sha),str(p),"exec"),obj.__dict__)
    return obj


def center_rank(candidates):
    """First ranked box centered in the earlier 21-pixel window."""
    for i,row in enumerate(candidates[:100],1):
        x0,y0,x1,y1=row["box_xyxy"]
        if 40 <= (x0+x1)/2 <61 and 40<= (y0+y1)/2 <61:
            return i
    return None


def emit(obj):
    return (json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ("carrier","base","manifest","protocol","images","author-source","author-tree","acceptance",
              "ema","codec-wheel","output"):
        parser.add_argument("--"+n,type=Path,required=True)
    for n in ("manifest-sha","protocol-sha"):
        parser.add_argument("--"+n,required=True)
    a=parser.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES")=="0"
         and not a.output.exists(),"Fresh isolated CUDA capture")
    pinned(a.protocol,a.protocol_sha)
    manifest=json.loads(pinned(a.manifest,a.manifest_sha))
    rows=manifest["items"]
    ids=[row["image_id"] for row in rows]
    need(manifest["source_json_sha256"]==SOURCE and ids==manifest["expected_ids"]
         and len(ids)==len(set(ids)) and 200<len(ids)<1596
         and len({row["rgb_sha256"] for row in rows})==len(rows)
         and len({row["acquisition_group"] for row in rows})==16,"Original qualified unique native manifest")
    need({p.name for p in a.images.iterdir()}=={row["image_ref"]["path"] for row in rows},
         "Image-only directory matches roster")
    cap=module(a.carrier,CARRIER)
    def offline(event, _):
        if event.startswith("socket."):
            raise PermissionError("Frozen local evaluation")
    sys.addaudithook(offline)
    base,torch,model,author=cap.initialize(a)
    import numpy as np
    from PIL import Image
    forward,nms,hook,original_nms=cap.bind_observers(model,author,torch)
    a.output.mkdir()
    base.prepare_export_codec(a.codec_wheel,a.output/"CODEC_PRIVATE")
    digest, binary=sha256(),sha256()
    counts=Counter()
    start=perf_counter()
    torch.cuda.synchronize()
    target=a.output/"SELECTED_PLANES_PRIVATE.bin"
    with (a.output/"CENTER_CANDIDATES_PRIVATE.jsonl").open("xb") as target_rows, target.open("xb") as arrays, torch.inference_mode():
        for ordinal,item in enumerate(rows,1):
            key=item["image_ref"]
            name=key["path"]
            need(item["original_hw"]==[101,101] and name==f"{item['image_id']:04d}.png"
                 and item["image_id"]==item["record_ordinal"],"Original positive ordinal and native size")
            raw=pinned(a.images/name,key["sha256"])
            need(len(raw)==key["bytes"],"Encoded image length")
            with Image.open(BytesIO(raw)) as im:
                need(im.format=="PNG" and im.mode=="RGB" and im.size==(101,101),"Lossless original RGB101")
                rgb=np.asarray(im,dtype=np.uint8)
                need(sha256(rgb.tobytes(order="C")).hexdigest()==item["rgb_sha256"],"Exact decoded image identity")
                prediction,audit,planes=cap.capture_row(rgb,item,model,author,torch,np,base,forward,nms)
            bare=dict(audit,candidates=[{k:v for k,v in q.items() if k not in ("query_id","probability_row")}
                                       for q in audit["candidates"]])
            total,_=base.validate_rows(item["image_id"],(101,101),prediction,bare)
            candidates=audit["candidates"][:100]
            rank=center_rank(candidates)
            record=dict(image_id=item["image_id"], group=item["acquisition_group"],rank=rank,
                        candidates=candidates,prediction=None,plane_offset=None,plane_sha256=None)
            if rank is None:
                counts["missing"]+=1
            else:
                selected=prediction["coco"][rank-1]
                detail=candidates[rank-1]
                record["prediction"]=selected
                counts["selected"]+=1
                counts["nonempty"]+=int(detail["mask_area"]>0)
                if 32<=detail["mask_area"]<1024:
                    plane=planes[rank-1]
                    need(plane.shape==(160,160) and plane.dtype==np.dtype("<f4"),"Original selected FP32 plane")
                    raw=plane.tobytes(order="C")
                    record["plane_offset"]=arrays.tell()
                    record["plane_sha256"]=sha256(raw).hexdigest()
                    arrays.write(raw)
                    binary.update(raw)
                    counts["eligible"]+=1
            content=emit(record)
            target_rows.write(content)
            digest.update(content)
            counts.update(images=1,post_nms=total)
            if ordinal%50==0:
                print(json.dumps(dict(event="SCHWEGMANN_FROZEN_GPU_PROGRESS",count=ordinal,selected=counts["selected"])),flush=True)
        for f in (target_rows,arrays):
            f.flush()
            os.fsync(f.fileno())
    torch.cuda.synchronize()
    hook.remove()
    author.nms=original_nms
    result=dict(status="SCHWEGMANN_DFINE_FROZEN_CENTER_CAPTURE_COMPLETE",manifest_sha256=a.manifest_sha,
                protocol_sha256=a.protocol_sha,source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                native_rows_sha256=digest.hexdigest(),selected_planes_sha256=binary.hexdigest(),
                counts=dict(counts),wall_seconds=perf_counter()-start,
                author_E24_sha256=cap.EMA,labels_opened=False,optimizer_updates=0,model_forwards=counts["images"])
    data=emit(result)
    with (a.output/"RESULT_SAFE.json").open("xb") as f:
        f.write(data)
    print(json.dumps(dict(status=result["status"],result_sha256=sha256(data).hexdigest())),flush=True)


if __name__=="__main__":
    main()
