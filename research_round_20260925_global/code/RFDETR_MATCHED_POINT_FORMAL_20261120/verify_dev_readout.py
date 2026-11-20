"""Independent closed DEV sparse-readout verification without target labels."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

STATUS="RFDETR_CENTER5202_DEV705_MATCHED_READOUT_INDEPENDENT_ACCEPTED"
READOUT_STATUS="RFDETR_MATCHED_POINTS_DEV705_READOUT_COMPLETE"


def need(ok,why):
    if not ok:
        raise ValueError(why)


def verify_terminal(wait,digest):
    need(wait["terminal"]["exit_code"]==0 and
         (type(wait["launch"].get("session_id")) is int or wait["launch"].get("exit_code")==0),
         "Exact original completed tool wait")
    footer=json.loads(wait["terminal"]["output"].strip().splitlines()[-1])
    need(footer==dict(status=READOUT_STATUS,result_sha256=digest),"Original natural terminal result and status")


def verify_points(np,stock,box,coords,pair,rank):
    need(stock.shape==(800,800) and stock.dtype==np.bool_ and coords.ndim==2 and coords.shape[1]==2,
         "Original native stock and point shape")
    if not (box[2]>box[0] and box[3]>box[1]):
        need(not len(coords),"No point support in degenerate author box")
        return True
    pools,_=rank.editable_pools(stock,box)
    add=np.concatenate([pools[k][0] for k in ("add1","add2")])
    drop=np.concatenate([pools[k][0] for k in ("drop1","drop2")])
    n=min(32,len(add),len(drop))
    need(len(coords)==n and np.isin(coords[:,0],add).all() and np.isin(coords[:,1],drop).all(),
         "All original available point counts and native two-pixel author-box support")
    if n:
        need(np.array_equal(pair.pair_assignment(coords[:,0],coords[:,1],800),coords),
             "Original deterministic minimum-distance point pairing")
    return False


def verify_masks(np,codec,stock,source_rle,coords,arms,get_edit,feature,seed,rank,cache=None):
    # The caller supplies one cache per original candidate, shared across seed/arm views.
    if cache is None:
        cache={}
    area=int(codec.area(source_rle))
    cap=min(8,area//10,len(coords))
    counts={}
    for arm in arms:
        edit=get_edit(seed,rank,arm)
        indices=np.asarray(edit["flipped"])
        if indices.size==0:
            indices=np.empty(0,dtype=np.int32)
        need(indices.ndim==1 and indices.dtype.kind in "iu" and len(np.unique(indices))==len(indices) and
             np.isin(indices,coords.reshape(-1)).all(),"Unique edited pixels on sealed support")
        y,x=np.divmod(indices,stock.shape[1])
        drops=int(np.count_nonzero(stock[y,x]))
        adds=len(indices)-drops
        expected_kind="flip" if arm=="LW=S" else "swap"
        expected_actions=len(indices) if expected_kind=="flip" else adds
        need(all(type(edit[k]) is int for k in ("edits","adds","drops","area_delta","actions",
                                               "nonpositive_actions","nonpositive_flips")) and
             edit["edits"]==len(indices)==adds+drops<=2*cap and edit["adds"]==adds and edit["drops"]==drops and
             edit["area_delta"]==adds-drops and (arm=="LW=S" or adds==drops) and
             edit["action_kind"]==expected_kind and edit["actions"]==expected_actions and
             0<=edit["nonpositive_actions"]<=expected_actions and 0<=edit["nonpositive_flips"]<=len(indices) and
             np.isfinite(edit["predicted_gain_sum"]) and (arm!="N" or len(indices)==0),
             "Per-arm declarations and cap agree with every original sparse set")
        key=tuple(sorted(indices.tolist()))
        if key not in cache:
            changed=feature.apply_sparse(np,stock,dict(edit,flipped=indices))
            xor=int(np.count_nonzero(changed!=stock))
            add=int(np.count_nonzero(changed & ~stock))
            drop=int(np.count_nonzero(stock & ~changed))
            delta=int(changed.sum())-area
            packed=codec.encode(np.asfortranarray(changed,dtype=np.uint8))
            rle=dict(size=packed["size"],counts=packed["counts"].decode())
            need(xor==add+drop and delta==add-drop and int(codec.area(rle))==area+delta and
                 (xor or rle==source_rle),"Actual independent XOR, sign, area, RLE and zero-edit stock identity")
            cache[key]=(xor,add,drop,delta)
        xor,add,drop,delta=cache[key]
        need((xor,add,drop,delta)==(edit["edits"],edit["adds"],edit["drops"],edit["area_delta"]),
             "Cached actual mask witness agrees with this arm's declarations")
        counts[arm]=dict(edited=int(xor>0),edits=xor,adds=add,drops=drop,area_delta=delta,
                        nonpositive_actions=edit["nonpositive_actions"],nonpositive_flips=edit["nonpositive_flips"])
    return counts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("readout","readout-source","score-source","capture","capture-accepted","image-manifest",
                "spot-source","code","wait","output"):
        p.add_argument("--"+key,type=Path,required=True)
    for key in ("readout-result-sha","readout-source-sha","score-source-sha","capture-accepted-sha","wait-sha"):
        p.add_argument("--"+key,required=True)
    p.add_argument("--probability-repair",type=Path)
    p.add_argument("--probability-repair-sha")
    a=p.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1",
         "Independent isolated CPU readback")
    before=perf_counter()
    from types import ModuleType
    source=a.readout_source.read_bytes()
    need(sha256(source).hexdigest()==a.readout_source_sha,"Exact closed readout source")
    reader=ModuleType("_closed_readout")
    reader.__file__=str(a.readout_source)
    exec(compile(source,str(a.readout_source),"exec"),reader.__dict__)
    result=json.loads(reader.pin(a.readout/"RESULT_SAFE.json",a.readout_result_sha,8*reader.MIB))
    wait=json.loads(reader.pin(a.wait,a.wait_sha))
    verify_terminal(wait,a.readout_result_sha)
    need(result["status"]==READOUT_STATUS and result["worker_sha256"]==a.readout_source_sha and
         result["capture_accepted_sha256"]==a.capture_accepted_sha and result["stock_predictions_sha256"]==reader.STOCK and
         result["seeds"]==list(reader.SEEDS) and result["arms"]==list(reader.ARMS) and
         result["model_forward_calls"]==result["point_optimizer_updates"]==0 and result["DEV"] is True and
         all(result[k] is False for k in ("FIT","TEST","GT_opened","AP")),"Complete original label-free readout scope")
    spot=reader.load(a.spot_source/"run_spot.py",reader.PINS["run_spot.py"],"_original_native_slots")
    score=reader.load(a.score_source,a.score_source_sha,"_independent_sparse_contract")
    feature=reader.load(a.spot_source/"feature_adapter.py",reader.PINS["feature_adapter.py"],"_original_sparse_apply")
    fit=reader.load(a.spot_source/"fit_step.py",reader.PINS["fit_step.py"],"_original_points")
    args=SimpleNamespace(capture_accepted=a.capture_accepted,capture_accepted_sha=a.capture_accepted_sha,
                         capture=a.capture,image_manifest=a.image_manifest,
                         probability_repair=a.probability_repair,probability_repair_sha=a.probability_repair_sha)
    images,queries,accepted=reader.original_population(args,spot)
    raw=reader.pin(a.readout/"SPARSE_EDITS_PRIVATE.jsonl",result["edits_sha256"],512*reader.MIB)
    sparse=[json.loads(line) for line in raw.splitlines()]
    need(raw.endswith(b"\n") and len(sparse)==705,"All completed native sparse image rows")
    import numpy as np
    import torch
    from pycocotools import mask as codec
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    need(not torch.cuda.is_initialized(),"CPU-only independent readout")
    capture_rows=[]
    for item,block in zip(images,queries,strict=True):
        pool=spot.native_candidate_rows(block)
        for cand in pool:
            cand["area"]=int(codec.area(cand["rle"]))
        capture_rows.append(dict(native_block=reader.stock_block(item["image_id"],pool),pool=pool,
                                 source_prefix=block["source_prefix"]))

    repair=reader.read_probability_repair(a)
    if repair is not None:
        ordinal=repair["ordinal"]
        need(type(ordinal) is int and 1<=ordinal<=705,"Exact repair image ordinal")
        block=queries[ordinal-1]
        pool=capture_rows[ordinal-1]["pool"]
        cand=next((r for r in pool if r["rank"]==repair["native_rank"]),None)
        need(cand is not None and block["image_id"]==repair["image_id"] and cand["query_id"]==repair["query_id"] and
             32<=cand["area"]<1024,"Repair is an original eligible mapped candidate")
        ref=block["logit_capture"]
        i=repair["logit_row"]
        proof=ref["candidates"][i]
        need((proof["logit_row"],proof["query_id"],proof["native_rank"])==
             (i,cand["query_id"],cand["rank"]),"Independent raw plane repair identity")
        payload=reader.pin(a.capture/ref["relative_path"],ref["sha256"],9*reader.MIB)
        from io import BytesIO
        planes=np.load(BytesIO(payload),allow_pickle=False)
        need(planes.dtype==np.dtype("<f4") and list(planes.shape)==ref["shape"] and len(payload)==ref["bytes"],
             "Original exact raw candidate sidecar")
        source_ref=dict(relative_path=ref["relative_path"],path=str((a.capture/ref["relative_path"]).resolve()),
                        bytes=ref["bytes"],sha256=ref["sha256"])
        identity={key:repair[key] for key in ("ordinal","image_id","query_id","native_rank","logit_row")}
        stock=codec.decode(cand["rle"]).astype(bool)
        reader.repaired_probability(np,torch,planes[i],stock,cand["rle"],identity,source_ref,repair)
    correction=reader.repair_summary(accepted,repair)
    need(all(result[key]==value for key,value in correction.items()),
         "Accepted same-CUDA repair consumed once and original differences retained")

    prepared=score.prepare_reconstruction(np,codec,capture_rows,sparse)
    base=a.code/"DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24"
    pair=fit.load(base/"fit109_paired_point/pair_math.py",spot.PAIR_SHA,"_independent_point_pairs")
    rank=fit.load(base/"fit109_probability_rank/rank_math.py",spot.RANK_SHA,"_independent_point_support")
    totals=Counter(images=0,mapped=0,eligible=0,empty=0,supported=0,unsupported=0,point_pairs=0,zero_area_author_boxes=0)
    budgets={str(seed):{arm:Counter() for arm in reader.ARMS} for seed in reader.SEEDS}
    for block,canonical,image in zip(queries,capture_rows,prepared,strict=True):
        original=image["sparse"]
        need(original["group"]==block["source_prefix"],"Original source group retained")
        boxes=spot.author_boxes(block["raw_boxes_cxcywh"],torch)
        by_rank={cand["rank"]:cand for cand in canonical["pool"]}
        totals.update(images=1,mapped=len(by_rank),eligible=len(image["points"]),
                      empty=sum(c["area"]==0 for c in by_rank.values()))
        for native_rank,coords in image["points"].items():
            cand=by_rank[native_rank]
            stock=codec.decode(cand["rle"]).astype(bool)
            degenerate=verify_points(np,stock,boxes[cand["query_id"]],coords,pair,rank)
            totals["zero_area_author_boxes"]+=int(degenerate)
            totals["point_pairs"]+=len(coords)
            totals["supported" if len(coords) else "unsupported"]+=1
            if not len(coords):
                continue
            mask_cache={}
            for seed in reader.SEEDS:
                read=lambda s,r,arm:score.get_edit(original,s,r,arm)
                observed=verify_masks(np,codec,stock,cand["rle"],coords,reader.ARMS,read,feature,seed,native_rank,cache=mask_cache)
                for arm,values in observed.items():
                    budgets[str(seed)][arm].update(verified=1,**values)
    totals["exact_original_masks"]=totals["eligible"]
    need(all(totals[k]==result["counts"].get(k,0) for k in set(totals)|set(result["counts"])) and
         totals["images"]==705 and totals["mapped"]==67386,
         "Complete independent source support and unchanged native population counts")
    for seed,arms in budgets.items():
        for arm,values in arms.items():
            values["unchanged"]=67386-values["edited"]
            need(dict(values)==result["budgets"][seed][arm],"All independently reconstructed realized arm budgets")
    a.output.mkdir()
    witness=dict(status=STATUS,**correction,readout_result_sha256=a.readout_result_sha,readout_source_sha256=a.readout_source_sha,
                 capture_accepted_sha256=a.capture_accepted_sha,original_tool_wait_sha256=a.wait_sha,
                 original_exit=0,images=705,native_mapped_rows=67386,edits_sha256=result["edits_sha256"],
                 stock_predictions_sha256=reader.STOCK,exact_candidate_identity=True,exact_native_support=True,
                 exact_per_candidate_budgets=True,actual_XOR_area_RLE_verified=True,counts=dict(totals),
                 elapsed_seconds=perf_counter()-before,model_forward_calls=0,point_optimizer_updates=0,
                 FIT=False,DEV=True,TEST=False,GT_opened=False,AP=False)
    digest=reader.write(a.output/"RESULT_SAFE.json",witness)
    print(json.dumps(dict(status=STATUS,result_sha256=digest)),flush=True)


if __name__=="__main__":
    main()
