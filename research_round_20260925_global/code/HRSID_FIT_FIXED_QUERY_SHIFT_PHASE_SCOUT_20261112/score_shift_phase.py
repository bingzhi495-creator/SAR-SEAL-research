"""Score a closed HRSID registered-shift capture with the prelocked grouped readout."""
from argparse import ArgumentParser
from collections import defaultdict
from contextlib import redirect_stdout
from hashlib import sha256
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask

SOURCE_DIGESTS={
    "PROTOCOL.md":"4298aa741ba7075ef428b4bba12c31dff66fc29af9e56f2a539df6ba8b20834d",
    "phase_probe_math.py":"8498fea25a00d06db5262c36caaf111b065457fbe1492f9deb39abd1fdcb1b70",
    "phase_readout.py":"0c6ca02aa531e03d2290a723c28b17ccd1c7aa0ce2e848b695a68915ee33dd3e",
}
MATCH_SHA="8741bebb448120e37e0c10b58467a83d724f2294d20fddbc93720301285084aa"
GEOMETRY_SHA="7cad8b51a9230101e2233db8455f84c28163abfb2ee8ddd8cbe8718a2e719bd0"
LABEL_SHA="e95cf26cc60a2c6fb651e907541380a256f2df7b1e153244a6c0f803b0b19f8f"
PREDICTIONS_SHA="2ed2f67671705a8b5b502a3661190c6c16b354c8a4d940d28a8c67943786f9df"
SUPPORT_SHA="64aa340b7acdf4db4030ac82f8c9b8507bc9c1d9e26692234ee16c6883a0c379"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def exact(path, expected, cap=2**20):
    require(path.resolve(strict=True)==path and 0<path.stat().st_size<=cap,"Canonical bounded input")
    raw=path.read_bytes()
    require(sha256(raw).hexdigest()==expected,"Exact input digest: "+path.name)
    return raw


def load(path, expected, name):
    obj=ModuleType(name)
    obj.__file__=str(path)
    sys.modules[name]=obj
    exec(compile(exact(path,expected),str(path),"exec"),obj.__dict__)
    return obj


def reconstruct_support(views, blocks, match):
    routed=[]
    for entry,block in zip(views,blocks,strict=True):
        require(entry["image_id"]==block["image_id"],"Original fixed image order")
        pool,_,_=match.ranked_candidates(block)
        queries={p["query"]:p for p in pool}
        logits=np.load(BytesIO(exact(Path(entry["array_path"]),entry["array_sha"],4*2**20)),allow_pickle=False)
        require(logits.dtype==np.float32 and logits.ndim==2 and logits.shape[1]==5 and np.isfinite(logits).all(),
                "Original closed registered logit format")
        points={}
        at=0
        expected={q for q,p in queries.items() if 32<=p["area"]<1024}
        require({c["query"] for c in entry["candidates"]}==expected and len(entry["candidates"])==len(expected),
                "Exact original area-routed query set")
        for candidate in entry["candidates"]:
            p=queries[candidate["query"]]
            require(candidate["rank"]==p["rank"] and candidate["score"]==p["score"],"Original rank and score remain fixed")
            ip,op=candidate["inside"],candidate["outside"]
            n=len(ip)+len(op)
            require(len(ip)<=64 and len(op)<=64 and at+n<=len(logits),"Frozen per-side point cap and range")
            if ip and op:
                coordinates=np.asarray(ip+op,dtype=np.int64)
                sides=np.arange(n)<len(ip)
                require(coordinates.shape==(n,2) and ((1<=coordinates)&(coordinates<799)).all(),"Valid registered interior points")
                points[p["rank"]]=(coordinates,sides,logits[at:at+n])
            at+=n
        require(at==len(logits),"Every recorded point row consumed once")
        routed.append((pool,points,at))
    return routed


def run(args):
    started=perf_counter()
    source=Path(__file__).resolve().parent
    exact(source/"PROTOCOL.md",SOURCE_DIGESTS["PROTOCOL.md"])
    math=load(source/"phase_probe_math.py",SOURCE_DIGESTS["phase_probe_math.py"],"_phase_math")
    readout=load(source/"phase_readout.py",SOURCE_DIGESTS["phase_readout.py"],"_phase_readout")
    anchor=load(args.match_source,MATCH_SHA,"_fit_phase_match")
    admission=load(args.code_root/"MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py",
                   GEOMETRY_SHA,"_fit_phase_geometry")
    sys.path.append(str(args.shapely_site.resolve(strict=True)))
    import shapely
    require(shapely.__version__=="2.0.2" and shapely.geos_version_string=="3.11.2","Exact CPU geometry")
    geometry=admission.geometry_modules(args.code_root)
    request=json.loads(exact(args.request,args.request_sha,8*2**20))
    require(request["FIT"] is True and request["DEV"] is request["TEST"] is False
            and request["original_capture_accepted"] and request["forward_calls"]==180
            and request["original_support_sha"]==SUPPORT_SHA and len(request["images"])==36,
            "Exact already accepted original FIT capture")
    views=request["images"]
    stream=exact(Path(request["prediction_path"]),PREDICTIONS_SHA,32*2**20)
    blocks=[json.loads(line) for line in stream.splitlines()]
    ids=[v["image_id"] for v in views]
    require(len(blocks)==len(set(ids))==len({v["group"] for v in views})==36
            and [v["ordinal"] for v in views]==list(range(1,37)),"Exact distinct source core36 roster")
    routed=reconstruct_support(views,blocks,anchor)
    require(sum(row[2] for row in routed)==267782,"Exact closed registered support count")
    labels=json.loads(exact(args.fit_index,LABEL_SHA,32*2**20))
    require(len(labels["images"])==2937 and len(labels["annotations"])==9260,"Exact original HRSID FIT")
    images={v["id"]:v for v in labels["images"] if v["id"] in ids}
    by_image=defaultdict(list)
    for a in labels["annotations"]:
        if a["image_id"] in images:
            by_image[a["image_id"]].append(a)
    subset=dict(images=list(images.values()),annotations=[a for rows in by_image.values() for a in rows],
                categories=labels["categories"])
    targets,neutral,_=geometry.geometry.polygon_support(subset,geometry.vector)
    examples=[]
    original_matches=0
    with redirect_stdout(StringIO()):
        for entry,(pool,points,_) in zip(views,routed,strict=True):
            iid=entry["image_id"]
            matches,_,_=anchor.freeze_coco(images[iid],by_image[iid],pool)
            original_matches+=len(matches)
            for i,candidate in enumerate(pool):
                rank=candidate["rank"]
                if rank not in points:
                    continue
                source_id=next((key for key,value in matches.items() if value==rank),None)
                if source_id is None or source_id in neutral:
                    continue
                target,_=targets[source_id]
                rle=candidate["rle"]
                predicted=mask.decode(rle).astype(bool)
                rectangles=geometry.geometry.boxes_from_counts(geometry.shift.runs(rle["counts"]),*rle["size"])
                intersection=geometry.geometry.overlap_area(rectangles,target)
                xy,inside,logits=points[rank]
                example=readout.example(entry["ordinal"],predicted,xy,inside,logits,target,intersection,
                                        [p["rle"] for p in pool[:i]],math)
                examples.append(example)
    out=readout.crossfit(examples,math)
    out.update(status="HRSID_FIT_FIXED_QUERY_SHIFT_PHASE_SCOUT_ACTUAL",FIT=True,DEV=False,TEST=False,AP=False,
               optimizer_updates=0,model_forward_calls=0,archived_capture_forwards=180,
               images=36,original_matches=original_matches,
               original_capture_accepted=True,protocol_sha256=SOURCE_DIGESTS["PROTOCOL.md"],
               request_sha256=args.request_sha,worker_seconds=perf_counter()-started)
    data=(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with args.output.open("xb") as handle:
        handle.write(data)
    print(json.dumps(dict(status=out["status"],result_sha256=sha256(data).hexdigest(),gate=out["gate"])),flush=True)


def main():
    parser=ArgumentParser()
    for field in ("request","fit-index","code-root","match-source","shapely-site","output"):
        parser.add_argument("--"+field,type=Path,required=True)
    parser.add_argument("--request-sha",required=True)
    args=parser.parse_args()
    require(sys.flags.isolated and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1","Offline isolated CPU FIT")
    def offline(event,_):
        if event.startswith("socket."):
            raise PermissionError("No network")
    sys.addaudithook(offline)
    run(args)


if __name__=="__main__":
    main()

