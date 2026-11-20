"""Label-free HRSID DEV705 frozen endpoint masks and independent readback."""
import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
from time import perf_counter
from types import ModuleType
from zipfile import ZipFile

import numpy as np
from pycocotools import mask as codec

PROTOCOL="842cb787b83882e67d0dce8b8105274113dabfe0cbea0dcfe075249e74562131"
DEV_SOURCE="5be622cdec5c889ca8520aab5547f6f15f01a54568a0263423a816dca7f96626"
RANK_MATH="cb438f8937f8e29dbdcbc441abdf5a788ac9c442fd63f1b3ad504b8ae0fcca0b"
REFIT_SOURCE="36562cf56c94a2eeeab194075afe3bb954086b0bbde52dcd684f74bb30ad313b"
REFIT_RESULT="a3a54c03ef6544f717fdbdf99d6d1571ff608e1c79adf810542921475529d2b0"
REFIT_ACCEPTED="dca00310ba7398790a874e2f32c7fa26324fed1b8e071372831a85e7a83ecd3a"
FREE_SOURCE="d24b7050be95786fffdee386e1e8a6f87ddcb8df8b69c2171eb752e6b6a35fef"
COMMON="f26da339c97bb4e22044d85e1cc4abb78e117a6bc3fde69500c08e2fca155ae5"
PREDICTIONS="80382f6f7318ac7a033e4d8169cf278322a9d51731f2d86e11b5f426d318d0ed"
AUDIT="29258878c41ec3d0cb2b173066d846a440197e2a0f45cd1277455dba0f786389"
SIDECAR_ACCEPTED="HRSID_DFINE_E24_DEV705_PROBABILITY_INDEPENDENTLY_ACCEPTED_V1"
PRELABEL="DFINE_DEV705_FREE_ENDPOINT_PRELABEL_COMPLETE"
VERIFIED="DFINE_DEV705_FREE_ENDPOINT_PRELABEL_INDEPENDENT_ACCEPTED"
MODES=("raw","fixed","free_matched_count","free_natural","native_matched_count")
EDITS="EDITS_PRIVATE.jsonl"


def need(ok,reason):
    if not ok:
        raise ValueError(reason)


def pin(path,digest,limit=32*1024**2):
    need(path.resolve(strict=True)==path and path.is_file() and 0<path.stat().st_size<=limit,"Exact bounded input")
    data=path.read_bytes()
    need(sha256(data).hexdigest()==digest,"Exact immutable input")
    return data


def load(path,digest,name):
    out=ModuleType(name)
    out.__file__=str(path)
    exec(compile(pin(path,digest),str(path),"exec"),out.__dict__)
    return out


def file_hash(path):
    with path.open("rb") as stream:
        return sha256(stream.read()).hexdigest()


def probability_receipt(args,dev):
    a=json.loads(pin(args.sidecar_accepted,args.sidecar_accepted_sha))
    need(a["status"]==SIDECAR_ACCEPTED and a["predictions_sha256"]==PREDICTIONS and
         a["candidate_audit_sha256"]==AUDIT and a["manifest_sha256"]==dev.FILES[dev.DEV_MANIFEST] and
         a["probability_files"]==705 and a["probability_rows"]==70500 and type(a["exact_author_binary_replay"]) is int and
         a["exact_author_binary_replay"]==70500 and a["historical_stream_byte_parity"] is True and
         a["original_os_exit"]==0 and a["job_drained"] is True and
         a["GT_opened"] is a["FIT"] is a["TEST"] is a["AP"] is False and a["DEV"] is True and
         all(dev.digest(a[k]) for k in ("probability_index_sha256","export_result_sha256",
                                      "export_terminal_sha256","original_tool_wait_sha256")),
         "Exact independently accepted label-free probability sidecar with historical raw parity")
    return a


def source_blocks(args,dev,manifest,receipt,with_probabilities):
    ph,ah,ih=sha256(),sha256(),sha256()
    with (args.sidecar/dev.PREDICTIONS).open("rb") as ps,(args.sidecar/dev.AUDIT).open("rb") as audit, (
            args.sidecar/"PROBABILITIES_PRIVATE.jsonl").open("rb") as index:
        for ordinal,item in enumerate(manifest,1):
            p,a,i=(f.readline(4*1024**2) for f in (ps,audit,index))
            need(all(x.endswith(b"\n") for x in (p,a,i)),"Exact three complete DEV streams")
            ph.update(p)
            ah.update(a)
            ih.update(i)
            pred,detail,row=(json.loads(x) for x in (p,a,i))
            identity=item["image_id"]
            n,_=dev.validate_rows(identity,(800,800),pred,detail)
            need(n>=100 and row["image_id"]==identity and len(row["query_ids"])==n and
                 len(set(row["query_ids"]))==n and all(type(x) is int and 0<=x<300 for x in row["query_ids"]),
                 "Original author top100 query identity")
            ref=row["probability"]
            need(ref["file"]==f"PROBABILITY_PRIVATE/{ordinal:04d}.npy" and ref["rows"]==100,
                 "Exact probability file coordinate")
            probabilities=None
            if with_probabilities:
                raw=pin(args.sidecar/ref["file"],ref["sha256"],11*1024**2)
                need(len(raw)==ref["bytes"],"Exact indexed plane bytes")
                probabilities=np.load(BytesIO(raw),allow_pickle=False)
                need(probabilities.shape==(100,160,160) and probabilities.dtype==np.dtype("<f4") and
                     np.isfinite(probabilities).all() and probabilities.min()>=0 and probabilities.max()<=1,
                     "Exact author FP32 sigmoid probability plane")
            yield item,pred["coco"][:100],detail["candidates"][:100],probabilities
        need(not any(f.read(1) for f in (ps,audit,index)) and (ph.hexdigest(),ah.hexdigest(),ih.hexdigest())==
             (PREDICTIONS,AUDIT,receipt["probability_index_sha256"]),"Exact historical and sidecar stream digests")


def edit(stock,coordinates,actions):
    picks=np.asarray(actions,dtype=np.intp)
    adds=coordinates[picks[:,0],0]
    drops=coordinates[picks[:,1],1]
    rle=stock.copy(order="C")
    flat=rle.ravel()
    need(len(set(adds.tolist()))==len(adds)==len(drops)==len(set(drops.tolist())) and
         not flat[adds].any() and flat[drops].all(),"Exact distinct opposite signs on immutable stock")
    flat[adds],flat[drops]=True,False
    packed=codec.encode(np.asfortranarray(rle,dtype=np.uint8))
    return dict(size=packed["size"],counts=packed["counts"].decode("ascii"))


def validate_edit(original,modified,area,k):
    need(set(modified)=={"size","counts"} and original["size"]==modified["size"] and type(modified["counts"]) is str and
         int(codec.area(modified))==area and int(codec.area(codec.merge([original,modified],intersect=True)))==area-k,
         "Exact native area and number of added and dropped cells")


def restore(args):
    refit=load(args.refit,REFIT_SOURCE,"_dev_refit")
    outcome=json.loads(pin(args.refit_result,REFIT_RESULT))
    verified=json.loads(pin(args.refit_acceptance,REFIT_ACCEPTED))
    need(verified["status"]=="DFINE_FIT109_BCE_FINAL_REFIT_INDEPENDENT_ACCEPTED" and verified["exact_tensor_replay"] and
         verified["result_sha256"]==REFIT_RESULT and outcome["weight_sha256"]==verified["weight_sha256"] and
         outcome["weight_bytes"]==verified["weight_bytes"],"Exact independently accepted final all-FIT head")
    weight=pin(args.weights,outcome["weight_sha256"],64*1024)
    need(len(weight)==outcome["weight_bytes"],"Exact fixed FIT-only head bytes")
    common=load(args.common,COMMON,"_dev_common")
    gray=common.load(args.gray,common.GRAY,"_dev_gray")
    pair=common.load(args.pair_math,gray.PAIR_MATH,"_dev_pairs")
    rank=load(args.rank_math,RANK_MATH,"_dev_ranking")
    free=load(args.free,FREE_SOURCE,"_dev_free")
    learner=gray.ablation_training(args.training)
    free.adapt_function(args.training,gray.TRAIN,learner,"gains",None,
                       (('def gains(','def probabilities('),('return points[:,0]-points[:,1]','return points')))
    head,center,scale=refit.restore_head(learner,weight)
    return pair,rank,free,learner,head,center,scale


def image_edits(coco,details,probabilities,image,models,counts):
    pair,rank,free,learner,head,center,scale=models
    result={}
    for i,(row,candidate,plane) in enumerate(zip(coco,details,probabilities,strict=True),1):
        area=candidate["mask_area"]
        if not 32<=area<1024:
            continue
        counts["area_eligible"]+=1
        old=row["segmentation"]
        stock=codec.decode(old).astype(bool,order="C")
        data=pair.paired_points(stock,candidate["box_xyxy"],area,row["score"],plane,image,rank)
        if data is None:
            counts["zero_support"]+=1
            continue
        coordinates,features,_,discord,half=data
        counts.update(supported=1,pairs=len(features),threshold_discordant=discord,threshold_half=half)
        p=learner.probabilities(head,features,features,center,scale)
        cap=min(8,area//10,len(features))
        fixed=[(j,j) for j in pair.choose_pairs(p[:,0]-p[:,1],cap)]
        natural=free.choose_free(p,cap)
        matched=free.choose_free(p,cap,len(fixed))
        native=free.choose_free(features[:,:,0],cap,len(natural))
        actions=dict(fixed=fixed,free_matched_count=matched,free_natural=natural,native_matched_count=native)
        changes={}
        for mode,picks in actions.items():
            if not picks:
                continue
            new=edit(stock,coordinates,picks)
            validate_edit(old,new,area,len(picks))
            changes[mode]=dict(swaps=len(picks),rle=new)
            counts.update({mode+"_edited":1,mode+"_swaps":len(picks)})
        if matched:
            counts["free_matched_count_nonpositive"]+=int((free.endpoint_deltas(p,matched)<=0).sum())
        if native:
            counts["native_matched_count_nonpositive"]+=int((free.endpoint_deltas(features[:,:,0],native)<=0).sum())
        if changes:
            result[str(i)]=changes
    return result


def prelabel(args,dev,manifest,receipt):
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    models=restore(args)
    pair=models[0]
    started=perf_counter()
    counts=Counter()
    d=sha256()
    args.output.mkdir()
    with args.archive.open("rb") as file,(args.output/EDITS).open("xb") as out:
        need(args.archive.stat().st_size==dev.ARCHIVE_BYTES,"Exact original DEV PNG archive")
        file.seek(dev.CENTRAL_OFFSET)
        need(sha256(file.read(dev.CENTRAL_BYTES)).hexdigest()==dev.CENTRAL_SHA,"Exact original archive directory")
        with ZipFile(file) as archive:
            for j,(item,coco,audit,probs) in enumerate(source_blocks(args,dev,manifest,receipt,True),1):
                rgb,hw=dev.read_png(archive,item)
                need(hw==(800,800),"Exact author native canvas")
                features=pair.image_features(rgb)
                changes=image_edits(coco,audit,probs,features,models,counts)
                line=dev.encoded(dict(image_id=item["image_id"],changes=changes))
                d.update(line)
                out.write(line)
                if j%25==0 or j==705:
                    print(json.dumps(dict(event="DFINE_DEV705_FREE_ENDPOINT_PRELABEL_PROGRESS",images=j)),flush=True)
        out.flush()
        os.fsync(out.fileno())
    need(counts["area_eligible"]==counts["zero_support"]+counts["supported"] and
         counts["fixed_swaps"]==counts["free_matched_count_swaps"] and
         counts["free_natural_swaps"]==counts["native_matched_count_swaps"],"Exact support and budget partition")
    result=dict(status=PRELABEL,protocol_sha256=PROTOCOL,worker_sha256=file_hash(Path(__file__)),
                sidecar_accepted_sha256=args.sidecar_accepted_sha,probability_index_sha256=receipt["probability_index_sha256"],
                predictions_sha256=PREDICTIONS,candidate_audit_sha256=AUDIT,refit_accepted_sha256=REFIT_ACCEPTED,
                edits_sha256=d.hexdigest(),images=705,candidates=70500,counts=dict(counts),
                DEV=True,FIT=False,TEST=False,GT_opened=False,AP=False,detector_forwards=0,replay_seconds=perf_counter()-started)
    digest=dev.write_new(args.output/"RESULT_SAFE.json",result)
    print(json.dumps(dict(status=PRELABEL,result_sha256=digest),sort_keys=True),flush=True)


def replay_rows(args,dev,manifest,receipt,expected):
    d=sha256()
    with (args.replay/EDITS).open("rb") as edits:
        for item,coco,audit,_ in source_blocks(args,dev,manifest,receipt,False):
            line=edits.readline(4*1024**2)
            need(line.endswith(b"\n"),"Exact complete target edit stream")
            d.update(line)
            row=json.loads(line)
            need(set(row)=={"image_id","changes"} and row["image_id"]==item["image_id"],"Exact target edit identity")
            for key,arms in row["changes"].items():
                k=int(key)
                need(str(k)==key and 1<=k<=100 and set(arms)<=set(MODES[1:]) and arms,"Exact original target rank")
                old=coco[k-1]["segmentation"]
                area=audit[k-1]["mask_area"]
                need(32<=area<1024,"Original target eligibility")
                counts={}
                for arm,v in arms.items():
                    n=v["swaps"]
                    need(set(v)=={"swaps","rle"} and type(n) is int and 1<=n<=min(8,area//10),
                         "Exact fixed swap cap")
                    validate_edit(old,v["rle"],area,n)
                    counts[arm]=n
                need(counts.get("fixed",0)==counts.get("free_matched_count",0) and
                     counts.get("free_natural",0)==counts.get("native_matched_count",0),"Exact per-candidate equal counts")
            yield item,coco,audit,row["changes"]
        need(not edits.read(1) and d.hexdigest()==expected["edits_sha256"],"Exact sealed edit stream")


def require_prelabel(args):
    summary=json.loads(pin(args.replay/"RESULT_SAFE.json",args.replay_sha))
    need(summary["status"]==PRELABEL and summary["protocol_sha256"]==PROTOCOL and
         summary["worker_sha256"]==file_hash(Path(__file__)) and
         summary["sidecar_accepted_sha256"]==args.sidecar_accepted_sha and summary["refit_accepted_sha256"]==REFIT_ACCEPTED and
         summary["predictions_sha256"]==PREDICTIONS and summary["candidate_audit_sha256"]==AUDIT and
         summary["images"]==705 and summary["candidates"]==70500 and summary["detector_forwards"]==0 and
         summary["DEV"] is True and summary["GT_opened"] is summary["AP"] is summary["TEST"] is summary["FIT"] is False,
         "Exact target label-free replay")
    return summary


def verify(args,dev,manifest,receipt):
    expected=require_prelabel(args)
    terminal=json.loads(pin(args.original_wait,args.original_wait_sha))
    need(terminal["terminal"]["exit_code"]==0 and expected["edits_sha256"] and
         args.replay_sha in terminal["terminal"]["output"],"Original CPU replay terminal and result witnessed")
    totals=Counter()
    for _,_,_,changes in replay_rows(args,dev,manifest,receipt,expected):
        for arms in changes.values():
            for arm,v in arms.items():
                totals.update({arm+"_edited":1,arm+"_swaps":v["swaps"]})
    need(all(totals[m+"_"+x]==expected["counts"].get(m+"_"+x,0) for m in MODES[1:] for x in ("edited","swaps")),
         "Independently reproduced exact budgets")
    args.output.mkdir()
    digest=dev.write_new(args.output/"RESULT_SAFE.json",dict(status=VERIFIED,protocol_sha256=PROTOCOL,
                         replay_sha256=args.replay_sha,sidecar_accepted_sha256=args.sidecar_accepted_sha,
                         original_wait_sha256=args.original_wait_sha,original_exit=0,images=705,candidates=70500,
                         edits_sha256=expected["edits_sha256"],counts=dict(totals),DEV=True,FIT=False,TEST=False,
                         GT_opened=False,AP=False))
    print(json.dumps(dict(status=VERIFIED,result_sha256=digest),sort_keys=True),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("prepare","verify"))
    for key in ("protocol","dev-source","dev-root","sidecar","sidecar-accepted","output"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--sidecar-accepted-sha",required=True)
    for key in ("archive","refit","refit-result","refit-acceptance","weights","common","gray","pair-math",
                "rank-math","free","training","replay","original-wait"):
        p.add_argument("--"+key,type=Path)
    p.add_argument("--replay-sha")
    p.add_argument("--original-wait-sha")
    a=p.parse_args()
    pin(a.protocol,PROTOCOL)
    dev=load(a.dev_source,DEV_SOURCE,"_author_dev")
    dev.cpu_runtime()
    receipt=probability_receipt(a,dev)
    manifest=dev.read_manifest(a.dev_root)
    if a.mode=="prepare":
        prelabel(a,dev,manifest,receipt)
    else:
        verify(a,dev,manifest,receipt)


if __name__=="__main__":
    main()
