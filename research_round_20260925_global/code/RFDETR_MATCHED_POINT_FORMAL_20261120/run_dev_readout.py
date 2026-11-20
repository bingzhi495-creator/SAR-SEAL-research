"""Three frozen FIT seed heads, label-free original DEV masks and sparse readouts."""
import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType
from zipfile import ZipFile

SEEDS=(5217,5218,5219)
ARMS=("N","SF","SU=F","SU","LF=S","LU=S","LW=S","NP=S")
CHECKPOINT="011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11"
STOCK="da55f2d3f9bbf08b26d5fae76606b2496a5b482c5fb61f4fab53c23172e5c06e"
IMAGE_MANIFEST="39a40d089d4a54448fe6efca8ea5cc98ad1eb7c0587815f3dd9a3b097173e78f"
PINS={"run_spot.py":"75ff72a3177e99f06d212a878c8d8921de4db6da8f4249722fb762718083d0ee",
      "fit_step.py":"94e60f5b5557787687d1f228c33c249a1d7889172a225373b97bd789ec8b72b1",
      "feature_adapter.py":"747c4c4b43561f8d073e01854adf241f1cb66c4efa1d29f304ae8837ecd67b73",
      "fit_plan.py":"1fd0fe8f815b5ad4e01d0d8eeca0aa0e1480433292fc48f94fee17e5837e5c35"}
PNG_READER="5be622cdec5c889ca8520aab5547f6f15f01a54568a0263423a816dca7f96626"
STATUS="RFDETR_MATCHED_POINTS_DEV705_READOUT_COMPLETE"
MIB=1024**2


def need(ok,why):
    if not ok:
        raise ValueError(why)


def encoded(value):
    return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def pin(path,digest,limit=4*MIB):
    need(path.is_file() and not path.is_symlink() and 0<path.stat().st_size<=limit,"Bounded closed input")
    raw=path.read_bytes()
    need(sha256(raw).hexdigest()==digest,"Exact source or input "+path.name)
    return raw


def load(path,digest,name):
    obj=ModuleType(name)
    obj.__file__=str(path)
    exec(compile(pin(path,digest),str(path),"exec"),obj.__dict__)
    return obj


def write(path,value):
    raw=encoded(value)
    with path.open("xb") as f:
        need(f.write(raw)==len(raw),"Complete durable JSON")
        f.flush()
        os.fsync(f.fileno())
    return sha256(raw).hexdigest()


def mapped_identity_sha256(pool):
    identity=[{k:row[k] for k in ("rank","query_id","score","area")} for row in pool]
    return sha256(json.dumps(identity,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def validate_capture_receipt(r):
    need(r["status"]=="RF_CENTER5202_FULL_CAPTURE_INDEPENDENTLY_ACCEPTED_V1" and r["mode"]=="dev" and
         r["images"]==r["raw_logit_files"]==705 and r["native_mapped_rows"]==67386 and
         r["source_checkpoint_sha256"]==CHECKPOINT and r["stock_predictions_sha256"]==STOCK and
         r["same_device_raster_exact"]==r["raw_logit_rows"] and r["model_forwards"]==0 and
         r["original_os_exit"]==0 and r["job_drained"] is True and r["DEV"] is True and
         all(r[k] is False for k in ("FIT","TEST","GT_opened","AP")),"Exact closed center DEV capture")
    need(r["independent_CPU_raster_disagree_rows"]==r["independent_CPU_raster_disagree_pixels"]==0,
         "CPU native raster differs from original stock; separate renderer required")


def validated_refits(result,accepted,result_sha):
    need(accepted["status"]=="RFDETR_MATCHED_POINTS_FIT109_INDEPENDENT_ACCEPTED" and
         accepted["fit_result_sha256"]==result_sha and accepted["run_natural_exit"]==0 and
         accepted["exact_schedule"] is True and accepted["checkpoint_bundles_verified"]==18 and
         accepted["head_updates_verified"]==7200 and accepted["model_forward_calls"]==0 and
         accepted["FIT"] is True and accepted["DEV"] is accepted["TEST"] is False,"Accepted original FIT schedule")
    need(result["status"]=="RFDETR_MATCHED_POINTS_FIT109_RUN_COMPLETE" and result["images"]==109 and
         result["source_seed_values"]==list(SEEDS) and result["head_updates"]==7200 and
         result["detector_checkpoint_sha256"]==CHECKPOINT and result["detector_forward_calls"]==0 and
         result["FIT"] is True and all(result[k] is False for k in ("DEV","TEST","AP")) and
         len(result["all_initializations"])==18,"Complete source-only FIT fits")
    rows=[r for r in result["all_initializations"] if r["fold"] is None]
    need(len(rows)==3 and {r["seed"] for r in rows}==set(SEEDS),"Three all-FIT seed heads")
    for r in rows:
        need(r["endpoint"]=="all_fit" and r["relative_weights"]==f"{r['seed']}_all_fit.npz" and
             r["steps"]==200 and r["batch"]==64 and set(r["heads"])=={"S","L"} and
             all(r["heads"][k]["optimizer_steps"]==[200] and r["heads"][k]["parameters"]==n
                 for k,n in (("S",1321),("L",139265))),"Original all-FIT actual budget")
    return {r["seed"]:r for r in rows}


def stock_block(image_id,candidates):
    return dict(image_id=image_id,raw_query_count=164,native_count=100,unmapped_slot1_count=100-len(candidates),
                native_ranks=[r["rank"] for r in candidates],
                coco=[dict(image_id=image_id,category_id=1,score=r["score"],segmentation=r["rle"]) for r in candidates])




def legacy_stock_bytes(image_id,candidates):
    block=stock_block(image_id,candidates)
    for row in block["coco"]:
        original=row["segmentation"]
        row["segmentation"]=dict(size=original["size"],counts=original["counts"])
    return (json.dumps(block,separators=(",",":"),ensure_ascii=False,allow_nan=False)+"\n").encode()

def original_population(a,spot):
    accepted=json.loads(pin(a.capture_accepted,a.capture_accepted_sha))
    validate_capture_receipt(accepted)
    report=json.loads(pin(a.capture/"RESULT_SAFE.json",accepted["capture_result_sha256"]))
    need(report["status"]=="RFDETR_CENTER5202_FULL_CAPTURE_COMPLETE_V1" and report["mode"]=="dev" and
         report["historical_CUDA_stock_exact"] is True and report["stock_predictions_sha256"]==STOCK and
         report["worker_sha256"]==accepted["worker_sha256"] and report["protocol_sha256"]==accepted["protocol_sha256"] and
         report["GT_opened"] is report["AP"] is report["TEST"] is False and report["DEV"] is True,
         "Complete original label-free CUDA capture")
    images=json.loads(pin(a.image_manifest,IMAGE_MANIFEST))
    manifest=json.loads(pin(a.capture/"IMAGE_MANIFEST_PRIVATE.json",accepted["image_manifest_sha256"]))
    need(len(images)==len(manifest)==705 and len({r["image_id"] for r in images})==705 and
         all((x["image_id"],x["encoded_sha256"],x["decoded_rgb_sha256"])==
             (y["image_id"],y["encoded_sha256"],y["decoded_rgb_sha256"]) for x,y in zip(images,manifest,strict=True)),
         "Exact full image-only roster")
    raw=pin(a.capture/"QUERIES_PRIVATE.jsonl",accepted["queries_sha256"],128*MIB)
    query=[json.loads(line) for line in raw.splitlines()]
    stock=pin(a.capture/"STOCK_NATIVE_PREDICTIONS_PRIVATE.jsonl",STOCK,128*MIB).splitlines(keepends=True)
    need(raw.endswith(b"\n") and len(query)==len(stock)==705,"Complete original streams")
    mapped=planes=0
    for item,block,line in zip(images,query,stock,strict=True):
        need(item["image_id"]==block["image_id"] and item["encoded_sha256"]==block["encoded_sha256"] and
             item["decoded_rgb_sha256"]==block["image_content_sha256"],"Original native candidate-image join")
        pool=spot.native_candidate_rows(block)
        need(legacy_stock_bytes(item["image_id"],pool)==line,"Legacy stock score, rank, category and RLE exact")
        mapped+=len(pool)
        planes+=len(block["logit_capture"]["candidates"])
    need(mapped==67386 and planes==accepted["raw_logit_rows"],"Complete candidate/plane counts")
    return images,query,accepted


def sparse_edit(edit):
    keys=("flipped","edits","adds","drops","area_delta","actions","action_kind","nonpositive_actions",
          "nonpositive_flips","predicted_gain_sum")
    return {k:edit[k].tolist() if k=="flipped" else edit[k] for k in keys}


def validate_sparse(np,stock,edit,feat,codec,original):
    mask=feat.apply_sparse(np,stock,edit)
    xor=int(np.count_nonzero(mask!=stock))
    delta=int(mask.sum())-int(stock.sum())
    rle=codec.encode(np.asfortranarray(mask,dtype=np.uint8))
    rle["counts"]=rle["counts"].decode()
    need(xor==edit["edits"] and delta==edit["area_delta"] and int(codec.area(rle))==int(stock.sum())+delta and
         (xor or rle==original),"Exact sparse XOR, area, RLE and no-op")


def emit_candidate(np,stock,record,refits,fit_step,feat,codec,budgets):
    result={str(s):{} for s in SEEDS}
    cap=min(8,int(stock.sum())//10,len(record["coordinates"]))
    for seed,(heads,center,scale) in refits.items():
        small=fit_step.predict(heads["S"],record["features"],center,scale)
        large=fit_step.predict(heads["L"],record["features"],center,scale)
        edits=feat.readouts_sparse(np,stock,record["coordinates"],small,large,record["features"][:,:,0],cap)
        need(set(edits)==set(ARMS) and edits["SF"]["edits"]==edits["SU=F"]["edits"] and
             all(edits[k]["edits"]==edits["SU"]["edits"] for k in ("LF=S","LU=S","LW=S","NP=S")),
             "Exact per-candidate common edit budgets")
        for arm,edit in edits.items():
            validate_sparse(np,stock,edit,feat,codec,record["rle"])
            need(arm=="LW=S" or edit["area_delta"]==0,"Balanced arm native area")
            budgets[seed][arm].update(verified=1,edited=int(edit["edits"]>0))
            for k in ("edits","adds","drops","area_delta","nonpositive_actions","nonpositive_flips"):
                budgets[seed][arm][k]+=edit[k]
            if edit["edits"]:
                result[str(seed)][arm]=sparse_edit(edit)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("capture","capture-accepted","image-manifest","image-archive","fit-result","fit-accepted","head-root",
                "spot-source","fit-plan","code","posterior","png-reader","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    for key in ("capture-accepted-sha","fit-result-sha","fit-accepted-sha"):
        parser.add_argument("--"+key,required=True)
    a=parser.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="-1" and sys.flags.isolated and sys.dont_write_bytecode,
         "Isolated CPU-only fixed readout")
    started=perf_counter()
    spot=load(a.spot_source/"run_spot.py",PINS["run_spot.py"],"_original_spot")
    images,blocks,capture=original_population(a,spot)
    fit=json.loads(pin(a.fit_result,a.fit_result_sha,8*MIB))
    witness=json.loads(pin(a.fit_accepted,a.fit_accepted_sha))
    bundles=validated_refits(fit,witness,a.fit_result_sha)
    fit_step=load(a.spot_source/"fit_step.py",PINS["fit_step.py"],"_original_trainer")
    feat=load(a.spot_source/"feature_adapter.py",PINS["feature_adapter.py"],"_original_features")
    plan=load(a.fit_plan,PINS["fit_plan.py"],"_original_fit_plan")
    import numpy as np
    import torch
    from pycocotools import mask as codec
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    need(not torch.cuda.is_initialized(),"No GPU context")
    base=a.code/"DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24"
    trainer=fit_step.load_trainer(base/"fit109_paired_point/train_point.py")
    refits={seed:plan.restore_bundle(a.head_root/r["relative_weights"],r["weights_sha256"],trainer,fit_step)
            for seed,r in bundles.items()}
    posterior=fit_step.load(a.posterior,spot.ADAPTER_SHA,"_original_RF_probability")
    png=load(a.png_reader,PNG_READER,"_original_PNG_reader")
    pair=fit_step.load(base/"fit109_paired_point/pair_math.py",spot.PAIR_SHA,"_original_pair_math")
    rank=fit_step.load(base/"fit109_probability_rank/rank_math.py",spot.RANK_SHA,"_original_rank_math")
    a.output.mkdir()
    pre=write(a.output/"PRELABEL_SAFE.json",dict(status="RFDETR_MATCHED_POINTS_DEV705_READOUT_STARTED",
              capture_accepted_sha256=a.capture_accepted_sha,fit_result_sha256=a.fit_result_sha,
              fit_accepted_sha256=a.fit_accepted_sha,stock_predictions_sha256=STOCK,images=705,
              FIT=False,DEV=True,TEST=False,GT_opened=False,AP=False))
    counts=Counter()
    budgets={seed:{arm:Counter() for arm in ARMS} for seed in SEEDS}
    digest=sha256()
    with a.image_archive.open("rb") as file,(a.output/"SPARSE_EDITS_PRIVATE.jsonl").open("xb") as out:
        need(a.image_archive.stat().st_size==png.ARCHIVE_BYTES,"Exact original PNG archive size")
        file.seek(png.CENTRAL_OFFSET)
        need(sha256(file.read(png.CENTRAL_BYTES)).hexdigest()==png.CENTRAL_SHA,"Exact archive central directory")
        with ZipFile(file) as archive:
            for ordinal,(item,block) in enumerate(zip(images,blocks,strict=True),1):
                rgb,hw=png.read_png(archive,item)
                need(hw==(800,800),"Original native canvas")
                image_features=pair.image_features(rgb)
                pool=spot.native_candidate_rows(block)
                for cand in pool:
                    cand["area"]=int(codec.area(cand["rle"]))
                eligible=[cand for cand in pool if 32<=cand["area"]<1024]
                ref=block["logit_capture"]
                need(ref["relative_path"]==f"RAW_LOGITS/{ordinal:03d}.npy","Original raw plane path")
                raw=pin(a.capture/ref["relative_path"],ref["sha256"],9*MIB)
                values=np.load(BytesIO(raw),allow_pickle=False)
                need(len(raw)==ref["bytes"] and values.dtype==np.dtype("<f4") and
                     list(values.shape)==ref["shape"]==[len(eligible),144,144] and np.isfinite(values).all(),
                     "Exact retained original raw planes")
                boxes=spot.author_boxes(block["raw_boxes_cxcywh"],torch)
                points=[]
                selected={str(seed):{} for seed in SEEDS}
                for i,(cand,proof,plane) in enumerate(zip(eligible,ref["candidates"],values,strict=True)):
                    query=cand["query_id"]
                    need((proof["logit_row"],proof["query_id"],proof["native_rank"],proof["score"],proof["predicted_area"])==
                         (i,query,cand["rank"],cand["score"],cand["area"]),"Original query/rank plane identity")
                    stock=codec.decode(cand["rle"]).astype(bool)
                    probability=feat.render_posterior(torch,np,posterior,plane,stock)
                    counts["exact_original_masks"]+=1
                    record=None
                    if spot.has_author_support(boxes[query]):
                        record=feat.extract_candidate(np,pair,rank,stock,boxes[query],cand["score"],probability,image_features)
                    else:
                        counts["zero_area_author_boxes"]+=1
                    coords=[] if record is None else record["coordinates"].tolist()
                    points.append(dict(rank=cand["rank"],query_id=query,coordinates=coords))
                    if record is None:
                        counts["unsupported"]+=1
                        continue
                    record["rle"]=cand["rle"]
                    counts["supported"]+=1
                    counts["point_pairs"]+=len(record["coordinates"])
                    chosen=emit_candidate(np,stock,record,refits,fit_step,feat,codec,budgets)
                    for seed,arms in chosen.items():
                        if arms:
                            selected[seed][str(cand["rank"])]=arms
                counts.update(images=1,mapped=len(pool),eligible=len(eligible),empty=sum(c["area"]==0 for c in pool))
                row=dict(ordinal=ordinal,image_id=item["image_id"],group=block["source_prefix"],original_hw=[800,800],
                         mapped_identity_sha256=mapped_identity_sha256(pool),points=points,seeds=selected)
                line=encoded(row)
                need(out.write(line)==len(line),"Complete sparse edit row")
                digest.update(line)
                if ordinal%25==0 or ordinal==705:
                    print(json.dumps(dict(event="RF_DEV_READOUT_PROGRESS",images=ordinal)),flush=True)
        out.flush()
        os.fsync(out.fileno())
    need(counts["images"]==705 and counts["mapped"]==67386 and
         counts["eligible"]==capture["raw_logit_rows"]==counts["exact_original_masks"]==
         counts["supported"]+counts["unsupported"] and not torch.cuda.is_initialized(),"Full fixed population readout")
    result=dict(status=STATUS,capture_accepted_sha256=a.capture_accepted_sha,fit_result_sha256=a.fit_result_sha,
                fit_accepted_sha256=a.fit_accepted_sha,prelabel_sha256=pre,
                worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),edits_sha256=digest.hexdigest(),
                stock_predictions_sha256=STOCK,counts=dict(counts),seeds=list(SEEDS),arms=list(ARMS),
                budgets={str(seed):{arm:dict(v,unchanged=67386-v["edited"]) for arm,v in rows.items()}
                         for seed,rows in budgets.items()},wall_seconds=perf_counter()-started,
                model_forward_calls=0,point_optimizer_updates=0,FIT=False,DEV=True,TEST=False,GT_opened=False,AP=False)
    h=write(a.output/"RESULT_SAFE.json",result)
    print(json.dumps(dict(status=STATUS,result_sha256=h)),flush=True)


if __name__=="__main__":
    main()
