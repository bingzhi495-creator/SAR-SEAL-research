"""Render the exact learned posterior k-matching under the realized stock cap."""
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from bound_common import (B5, REPLAY_CODE, CAPTURE_CODE, CAPTURE, CAP_TERMINAL, REPLAY, REP_TERMINAL,
                          MATCH, STOCK, OLD_EDITS, SEEDS, TOTAL, RENDER_STATUS, digest, pin, require, load,
                          closed, verified_lines, write_result)


CTX=None


def old_arms(coordinates, q, cap):
    order=np.arange(len(q),dtype=np.intp)
    gain=q[:,0]-q[:,1]
    paired=np.lexsort((order,-gain))
    fixed=paired[gain[paired]>0][:cap]
    k=len(fixed)
    ia=np.lexsort((order,-q[:,0]))[:k]
    idrop=np.lexsort((order,q[:,1]))[:k]
    return fixed,np.column_stack((coordinates[ia,0],coordinates[idrop,1])),ia,idrop


def verify_old(old, pixels, q, ai, di):
    k=len(ai)
    if not k:
        require(not old,"Original zero count leaves the native mask")
        return
    margin=np.column_stack((q[ai,0]-np.float32(.5),np.float32(.5)-q[di,1])).reshape(-1)
    require(old["flipped"]==pixels.reshape(-1).tolist() and
            old["adds"]==old["drops"]==old["actions"]==k and old["edits"]==2*k and old["area_delta"]==0
            and old["predicted_gain_sum"]==float(margin.sum(dtype=np.float64))
            and old["nonpositive_actions"]==int(np.count_nonzero(q[ai,0]-q[di,1]<=0))
            and old["nonpositive_flips"]==int(np.count_nonzero(margin<=0)),
            "Original edit positions, finite posterior action sums and exact signed count")


def initialize(args, repair):
    global CTX
    source=args.code/B5
    pin(source/"portable_dev705_capture.py",CAPTURE_CODE)
    replay=load(source/"portable_dev705_replay.py",REPLAY_CODE,"_dev705_bound_baseline")
    replay.initialize(args,repair)
    CTX=SimpleNamespace(dev=replay.CTX, solver=load(args.matching,MATCH,"_dev705_bound_solver"))


def render(task):
    ordinal,item,query_line,stock_line,sparse_line=task
    c,solver=CTX.dev,CTX.solver
    p,r=c.policy,c.reader
    block,hist=json.loads(query_line),json.loads(sparse_line)
    image=item["image_id"]
    require(image==block["image_id"]==hist["image_id"] and hist["ordinal"]==ordinal and
            hist["group"]==block["source_prefix"] and hist["original_hw"]==[800,800] and
            block["image_content_sha256"]==item["decoded_rgb_sha256"] and block["encoded_sha256"]==item["encoded_sha256"],
            "Original native image and sparse row identity")
    pool=p.spot.native_candidate_rows(block)
    require(r.legacy_stock_bytes(image,pool)==stock_line,"Published native score, candidate, rank and mask parity")
    for cand in pool:
        cand["area"]=int(c.codec.area(cand["rle"]))
    eligible=[cand for cand in pool if 32<=cand["area"]<1024]
    require(hist["mapped_identity_sha256"]==r.mapped_identity_sha256(pool) and
            [rec["rank"] for rec in hist["points"]]==[cand["rank"] for cand in eligible],
            "Every original area-eligible point identity retained")
    ref=block["logit_capture"]
    require(ref["relative_path"]==f"RAW_LOGITS/{ordinal:03d}.npy","Original raw 144-plane companion")
    path=c.root/ref["relative_path"]
    raw=r.pin(path,ref["sha256"],9*r.MIB)
    buf=BytesIO(raw)
    planes=np.load(buf,allow_pickle=False)
    require(not buf.read(1) and len(raw)==ref["bytes"] and planes.dtype==np.dtype("<f4") and
            list(planes.shape)==ref["shape"]==[len(eligible),144,144] and np.isfinite(planes).all(),
            "Exact prior immutable native logit planes")
    src=dict(relative_path=ref["relative_path"],path=str(path),bytes=len(raw),sha256=ref["sha256"])
    rgb,hw=c.png.read_png(c.archive,item)
    require(hw==(800,800),"Original native canvas")
    appearance=p.pair.image_features(rgb)
    boxes=p.spot.author_boxes(block["raw_boxes_cxcywh"],c.torch)
    by_seed={str(s):{} for s in SEEDS}
    stats={s:Counter() for s in SEEDS}
    post=sha256()
    before=c.repair["_consumed"]
    supported=0
    for j,(cand,proof,plane,point) in enumerate(zip(eligible,ref["candidates"],planes,hist["points"],strict=True)):
        qid,rank,area=cand["query_id"],cand["rank"],cand["area"]
        require(proof==dict(logit_row=j,query_id=qid,native_rank=rank,score=cand["score"],predicted_area=area)
                and point["rank"]==rank and point["query_id"]==qid,"Original native candidate and point rank")
        stock=c.codec.decode(cand["rle"]).astype(bool)
        identity=dict(ordinal=ordinal,image_id=image,query_id=qid,native_rank=rank,logit_row=j)
        probability=r.probability_with_repair(np,c.torch,p.features,p.posterior,plane,stock,cand["rle"],identity,src,c.repair)
        record=(p.features.extract_candidate(np,p.pair,p.rank,stock,boxes[qid],cand["score"],probability,appearance)
                if p.spot.has_author_support(boxes[qid]) else None)
        if record is None:
            require(point["coordinates"]==[] and all(not hist["seeds"][str(s)].get(str(rank),{}).get("SF") and
                    not hist["seeds"][str(s)].get(str(rank),{}).get("SU=F") for s in SEEDS),
                    "Exact unsupported and zero-budget native fallback")
            continue
        xy=record["coordinates"]
        require(np.array_equal(xy,np.asarray(point["coordinates"],dtype=np.int32)) and
                not stock.ravel()[xy[:,0]].any() and stock.ravel()[xy[:,1]].all(),
                "Exact archived signed coordinates and native signs")
        supported+=1
        cap=min(8,area//10,len(xy))
        for seed in SEEDS:
            model,center,scale=c.heads[seed]
            q=p.fit_step.predict(model["S"],record["features"],center,scale)
            require(q.dtype==np.float32 and np.isfinite(q).all() and np.all((0<=q)&(q<=1)),"Original binary32 small head")
            post.update(q.astype("<f4",copy=False).tobytes())
            f,ind,ia,idrop=old_arms(xy,q,cap)
            row=hist["seeds"][str(seed)].get(str(rank),{})
            verify_old(row.get("SF",{}),xy[f],q,f,f)
            verify_old(row.get("SU=F",{}),ind,q,ia,idrop)
            k=len(f)
            if not k:
                continue
            ai,di,counts=solver.bounded_endpoints(xy,q,k,800,f,f,ia,idrop)
            pairs=np.column_stack((ai,di))
            raw=xy[f]
            bounded=np.column_stack((xy[ai,0],xy[di,1]))
            both=lambda x:sorted(x.reshape(-1).tolist())
            if counts["independent_feasible"]:
                require(both(bounded)==both(ind),"Feasible independent optimum retained exactly")
            cap2=int(np.square(np.column_stack(np.divmod(raw[:,0],800))-
                              np.column_stack(np.divmod(raw[:,1],800))).sum(axis=1).max())
            require(cap2==counts["cap_squared_sum"],"Inherited cap is the max of realized original SF moves")
            by_seed[str(seed)][str(rank)]=dict(pairs=pairs.tolist(),max_fixed_distance_squared=cap2)
            stats[seed].update(positive=1,swaps_per_arm=k,differs_fixed=int(both(bounded)!=both(raw)),
                               differs_independent=int(both(bounded)!=both(ind)),**counts)
    row=dict(image_id=image,ordinal=ordinal,seeds=by_seed)
    return ((json.dumps(row,sort_keys=True,separators=(",",":"))+"\n").encode(),len(pool),len(eligible),supported,
            c.repair["_consumed"]-before,post.hexdigest(),stats)


def main():
    parser=ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("code","source","archive","heads","matching","protocol","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--workers",type=int,choices=range(1,9),required=True)
    parser.add_argument("--protocol-sha256",required=True)
    a=parser.parse_args()
    start=perf_counter()
    require(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1"
            and not a.output.exists(),"Fresh isolated CPU label-free replay")
    pin(a.protocol,a.protocol_sha256)
    pin(a.matching,MATCH)
    capture=closed(a.source,"capture",CAPTURE,CAP_TERMINAL,CAPTURE_CODE)
    history=closed(a.source,"replay",REPLAY,REP_TERMINAL,REPLAY_CODE)
    require(history["capture_result_sha256"]==CAPTURE and history["exact_historical_sparse_bytes"] is True
            and history["edits_sha256"]==OLD_EDITS and history["GT_opened"] is history["TEST"] is False,
            "Exact previous three-head frozen dev archive")
    a.intake,a.capture=a.source/"intake",a.source/"capture"
    source=a.code/B5
    replay=load(source/"portable_dev705_replay.py",REPLAY_CODE,"_dev705_bound_manifest")
    cap_sha,repair_sha,repair=replay.capture_inputs(a)
    require(cap_sha==CAPTURE and repair_sha==history["probability_repair_sha256"] and capture["images"]==705,
            "Reference native manifests and single qualified repair")
    manifest=json.loads(pin(a.intake/"DEV_IMAGE_ONLY_MANIFEST_PRIVATE.json",replay.cap.INPUT_MANIFEST))
    a.output.mkdir()
    stream=a.output/"BOUNDED_PRIVATE.jsonl"
    out_hash,post=sha256(),sha256()
    totals=Counter()
    reports={s:Counter() for s in SEEDS}
    tasks=((j,item,x,y,z) for j,(item,x,y,z) in enumerate(zip(manifest,
           verified_lines(a.capture/replay.cap.QUERIES,replay.cap.QUERIES_SHA),
           verified_lines(a.capture/replay.cap.STOCK,STOCK),
           verified_lines(a.source/"replay/SPARSE_EDITS_PRIVATE.jsonl",OLD_EDITS),strict=True),1))
    with stream.open("xb") as f, ProcessPoolExecutor(max_workers=a.workers,initializer=initialize,initargs=(a,repair)) as pool:
        for line,mapped,eligible,supported,repair_count,p,result in pool.map(render,tasks,chunksize=1):
            f.write(line)
            out_hash.update(line)
            post.update(bytes.fromhex(p))
            totals.update(images=1,mapped=mapped,eligible=eligible,supported=supported,cuda_repair=repair_count)
            for seed,stats in result.items():
                reports[seed].update(stats)
            if totals["images"]%50==0:
                print(json.dumps(dict(images=totals["images"])),flush=True)
    require(dict(totals)==dict(images=705,mapped=67386,eligible=42325,supported=40897,cuda_repair=1)
            and all((reports[s]["positive"],reports[s]["swaps_per_arm"])==TOTAL[s] for s in SEEDS),
            "Whole original DEV705 frozen candidate and arm realized-budget census")
    write_result(a.output/"RESULT_SAFE.json",dict(status=RENDER_STATUS,phase="replay",worker_sha256=digest(Path(__file__).read_bytes()),
        protocol_sha256=a.protocol_sha256,prior_capture_sha256=CAPTURE,prior_replay_sha256=REPLAY,
        matching_worker_sha256=MATCH,original_edits_sha256=OLD_EDITS,totals=dict(totals),census_by_seed=reports,
        bounded_stream_sha256=out_hash.hexdigest(),binary32_posterior_sha256=post.hexdigest(),
        exact_native_coordinates=True,exact_old_selected_mask_indices_and_probability_gains=True,
        labels_opened=False,detector_forwards=0,optimizer_updates=0,elapsed_seconds=perf_counter()-start))


if __name__=="__main__":
    main()
