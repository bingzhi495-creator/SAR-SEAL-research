"""Write four support/count policies using the fifth native Phoenix posterior."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from importlib.metadata import version
import json
from math import fsum
import os
from pathlib import Path
import random
from time import perf_counter
from types import ModuleType

import numpy as np


ARMS = ("nc", "nt", "wc", "wt")
BASE = "ea98a849ae181f06f3ffc0e41b29d2a91a7f867b4a4bcba6909c957f27db2b99"
REQUEST = "a40c0b8a3b6473a4c4f1f6c58c04f9cf6070f34c3ef7331783134a1dd77447ac"
ENDPOINT = "127ae10d68af63bab6aa2d7b0364a3d235f5f1105724596d2dea66a46e739ec6"
AREA = "62edaffa2d41da227b0f4f33ba67118a4346d3010cd244aafe23d807737f2363"
WHOLE = "17eb0e07d6788e3ad13d6c77b3b714739f2d4c4c1ed9c89d338bc2e6c697c42b"
LOCK = "f9ad7f1bd128fea19b88ef6e97efa2679ad942565691da2bd408dab8b711ee9f"
RUNTIME = {"torch":"2.10.0+cu130","numpy":"1.26.4","FastGeodis":"1.0.5","Pillow":"12.3.0","pycocotools":"2.0.11"}
STATUS = "RSDD74_PHOENIX_SUPPORT_BUDGET_INFERENCE_COMPLETE"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def top_sites(sites, flat_prob, addition, limit):
    """Sites arrive in the required tie order. Sort at most 'limit' selected indices."""
    value = flat_prob[sites]
    if not addition:
        value = -value
    limit = min(len(sites),limit)
    if limit == 0:
        return sites[:0]
    if len(sites) > limit:
        cutoff = np.partition(value,len(sites)-limit)[len(sites)-limit]
        strictly_better = np.flatnonzero(value > cutoff)
        n_ties = limit-len(strictly_better)
        chosen = np.concatenate((strictly_better,np.flatnonzero(value == cutoff)[:n_ties]))
    else:
        chosen = np.arange(len(sites))
    return sites[chosen[np.argsort(-value[chosen],kind="stable")]]


def fixed_count(native, prob, add, drop, k):
    """Enumerate every feasible signed allocation of the same global number of flips."""
    flat = prob.ravel()
    add = top_sites(add,flat,True,2*k)
    drop = top_sites(drop,flat,False,2*k)
    au = flat[add].astype(np.float64)-.5
    du = .5-flat[drop].astype(np.float64)
    options = []
    for a in range(max(0,2*k-len(drop)),min(2*k,len(add))+1):
        d=2*k-a
        value=fsum([*au[:a],*du[:d]])
        options.append((-value,abs(a-d),a))
    minus_utility,_,a=min(options)
    output=native.copy(order="C")
    output.flat[add[:a]]=True
    output.flat[drop[:2*k-a]]=False
    return output,-minus_utility


def policies(native, coords, prob, author_binary, k):
    require(native.shape == prob.shape == author_binary.shape and native.dtype == author_binary.dtype == np.bool_
            and prob.dtype == np.float32 and np.isfinite(prob).all() and np.all((prob>=0)&(prob<=1)),
            "Native 2-D Boolean masks and finite original f32 probabilities required")
    require(coords.ndim==2 and coords.shape[1]==2 and coords.dtype.kind in "iu" and np.all((0<=coords)&(coords<native.size))
            and np.unique(coords).size == coords.size and not native.flat[coords[:,0]].any() and native.flat[coords[:,1]].all()
            and 0<k<=len(coords), "Invalid starting signed support")
    half=prob==.5
    disagreement=(author_binary != (prob>.5)) & ~half
    require(not disagreement.any(), "Author binary and sigmoid disagree away from the representable half")
    narrow, nuc=fixed_count(native,prob,coords[:,0],coords[:,1],k)
    # All archived sites precede every other site; remaining native coordinates are in C row-major order.
    excluded=np.zeros_like(native)
    excluded.flat[coords]=True
    add=np.concatenate((coords[:,0],np.flatnonzero(~native & ~excluded)))
    drop=np.concatenate((coords[:,1],np.flatnonzero(native & ~excluded)))
    wide, wuc=fixed_count(native,prob,add,drop,k)
    require(wuc>=nuc, "Larger signed feasible set lowered optimum")
    chosen = wide!=native
    def extremum(sign,selected,get,empty):
        values = prob[(native==sign) & (chosen==selected)]
        if len(values)==0:
            return empty
        if sign:
            return .5-float((values.max if get=="min" else values.min)())
        return float((values.min if get=="min" else values.max)())-.5
    lo=min(extremum(False,True,"min",np.inf),extremum(True,True,"min",np.inf))
    hi=max(extremum(False,False,"max",-np.inf),extremum(True,False,"max",-np.inf))
    require(np.count_nonzero(chosen)==2*k and lo>=hi, "Whole fixed-count solution failed independent rank certificate")
    threshold=native.copy(order="C")
    threshold.flat[coords]=author_binary.flat[coords]
    return dict(nc=narrow,nt=threshold,wc=wide,wt=author_binary), dict(
        threshold_half_cells=int(half.sum()),threshold_half_original_foreground=int(np.count_nonzero(half & native)),
        threshold_zero_utility_flipped_cells=int(np.count_nonzero(half & (native!=author_binary))),
        threshold_half_strict_binary_differences=int(np.count_nonzero(half & author_binary)),
        threshold_half_narrow_endpoint_cells=int(np.count_nonzero(half.flat[coords])),
        threshold_disagreements_away_from_half=0)


def baseline(path):
    data=path.read_bytes()
    require(sha256(data).hexdigest()==BASE, "Original qualified author worker is required")
    module=ModuleType("_verified_original_phoenix")
    module.__file__=str(path)
    exec(compile(data,str(path),"exec"),module.__dict__)
    return module


def initialize(args):
    global author
    author=baseline(args.baseline_worker)
    author.init_worker(args)


def encode(mask, codec):
    result=codec.encode(np.asfortranarray(mask,dtype=np.uint8))
    require(np.array_equal(codec.decode(result),mask),"Written RLE differs from native mask")
    return dict(size=list(mask.shape),counts=result["counts"].decode("ascii"))


def image_row(task):
    ordinal,image,old,old_area,old_whole=task
    api=author
    image_id=image["image_id"]
    require(image_id==old["image_id"]==old_area["image_id"] and ordinal==old_whole["ordinal"],
            "Different closed input order")
    whole={r["position"]:r["segmentation"] for r in old_whole["lvis"]}
    items=image["candidates"]
    positions=[x["position"] for x in items]
    require(positions == [x["position"] for x in old["candidates"]] == sorted(map(int,old_area["changes"]))
            and set(positions)<=set(whole), "Different historical candidate positions")
    path=api.request_dir/image["image_file"]
    require(path.resolve(strict=True)==path and api.file_sha(path)==image["encoded_sha256"],"Wrong original image bytes")
    with api.Image.open(path) as img:
        require(img.format=="PNG" and img.mode=="RGB" and img.size==(512,512),"Expected native RGB")
        rgb=np.array(img,dtype=np.uint8)
    require(api.digest(rgb.tobytes())==image["decoded_rgb_sha256"],"Original RGB changed")
    random.seed(api.SEED)
    np.random.seed(api.SEED)
    api.torch.random.default_generator.manual_seed(api.SEED)
    started=perf_counter()
    with api.torch.inference_mode():
        api.refiner.set_image(rgb)
    times=Counter(encode=perf_counter()-started,decode=0.,write=0.)
    count=Counter(images=1,image_encodes=1)
    arms={arm:Counter() for arm in ARMS}
    maxima=dict(endpoint_abs_error=0.,wt_mismatch_distance_from_half=0.)
    changes={}
    for begin in range(0,len(items),2):
        batch=items[begin:begin+2]
        native=[api.candidate(x) for x in batch]
        start=perf_counter()
        with api.torch.inference_mode():
            binary, probabilities=api.refiner.refine_batch([x[0] for x in native],refine_iters=5,smooth_kernel=1,
                                binarize_input=True,multimask_output=True,return_logits=True,max_batch=2)
        times["decode"]+=perf_counter()-start
        count["decoder_batches"]+=1
        start=perf_counter()
        for j,(candidate,(mask,coords,area),returned,prob) in enumerate(zip(batch,native,binary,probabilities,strict=True)):
            pos,k=candidate["position"],candidate["k"]
            ref=np.array(old["candidates"][begin+j]["probabilities"],dtype=np.float64)
            q=prob.flat[coords]
            require(q.shape==ref.shape and np.isfinite(ref).all(), "Old endpoint dimensions or values differ")
            different=np.count_nonzero(q!=ref)
            maxima["endpoint_abs_error"]=max(maxima["endpoint_abs_error"],float(np.abs(q-ref).max()))
            count.update(decoded_candidates=1,endpoint_pairs=len(coords),endpoint_probabilities=q.size,
                         exact_endpoint_probabilities=q.size-different,endpoint_probability_differences=different,
                         native_count_sum=k,candidate_refinement_iterations=5)
            results,tally=policies(mask,coords,prob,returned,k)
            count.update(tally)
            records={}
            for arm,out in results.items():
                adds=int(np.count_nonzero(out & ~mask))
                drops=int(np.count_nonzero(mask & ~out))
                flips=adds+drops
                changed=out!=mask
                inside=int(np.count_nonzero(changed.flat[coords]))
                old_fg=mask[changed]
                score=prob[changed]
                positive=int(np.count_nonzero((old_fg & (score<.5)) | (~old_fg & (score>.5))))
                zero=int(np.count_nonzero(score==.5))
                negative=flips-positive-zero
                if arm in ("nc","wc"):
                    require(flips==2*k, "Fixed-count arm failed to write the budget")
                else:
                    require(negative==0,"Author threshold retained a negative-utility flip")
                if arm in ("nc","nt"):
                    require(inside==flips, "Narrow write escaped the signed endpoint set")
                records[arm]=dict(rle=encode(out,api.codec))
                arms[arm].update(adds=adds,drops=drops,flips=flips,changed_candidates=int(flips>0),
                                 empty_outputs=int(not out.any()),signed_area_delta=adds-drops,
                                 positive_selected_flips=positive,zero_selected_flips=zero,negative_selected_flips=negative,
                                 nonpositive_selected_flips=zero+negative,
                                 inside_narrow_flips=inside,outside_narrow_flips=flips-inside,
                                 candidates_below_2k=int(flips<2*k),candidates_equal_2k=int(flips==2*k),
                                 candidates_above_2k=int(flips>2*k))
            for arm,previous in (("nc",old_area["changes"][str(pos)]["free"]["rle"]),("wt",whole[pos])):
                equal=(records[arm]["rle"]==previous)
                count["exact_"+arm+"_masks"]+=int(equal)
                if not equal:
                    disagreement=results[arm]!=api.codec.decode(previous).astype(bool)
                    count[arm+"_xor_pixels"]+=int(disagreement.sum())
                    if arm=="wt" and disagreement.any():
                        maxima["wt_mismatch_distance_from_half"]=max(maxima["wt_mismatch_distance_from_half"],
                                                                     float(np.abs(prob[disagreement]-.5).max()))
            count["whole_count_rank_certificates"]+=1
            changes[str(pos)]=records
        times["write"]+=perf_counter()-start
    return dict(ordinal=ordinal,image_id=image_id,changes=changes),count,arms,maxima,times


def read_rows(path, expected):
    data=path.read_bytes()
    require(sha256(data).hexdigest()==expected,"Wrong closed ledger: "+path.name)
    return list(map(json.loads,data.splitlines()))


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("source-root","baseline-worker","request","matched-endpoints","area-edits","whole-masks","protocol","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--protocol-sha256",required=True)
    parser.add_argument("--workers",type=int,choices=(4,),default=4)
    parser.add_argument("--threads",type=int,choices=(2,),default=2)
    a=parser.parse_args()
    for name,v in vars(a).items():
        if isinstance(v,Path):
            setattr(a,name,v.resolve(strict=name!="output"))
    old=baseline(a.baseline_worker)
    require(a.protocol_sha256==LOCK==old.file_sha(a.protocol) and not a.output.exists(), "Fixed protocol and fresh output required")
    os.environ.update(CUDA_VISIBLE_DEVICES="-1",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS=str(a.threads),
                      MKL_NUM_THREADS=str(a.threads),TORCH_FORCE_WEIGHTS_ONLY_LOAD="1")
    runtime={k:version(k) for k in RUNTIME}
    require(runtime==RUNTIME, "Expected qualified CPU environment")
    start=perf_counter()
    request=old.read_request(a.request,REQUEST)
    old.verify_author(a.source_root)
    jobs=list(zip(range(1,75),request["images"],read_rows(a.matched_endpoints,ENDPOINT),
                  read_rows(a.area_edits,AREA),read_rows(a.whole_masks,WHOLE),strict=True))
    totals=Counter()
    by_arm={arm:Counter() for arm in ARMS}
    maxima=Counter()
    times=Counter()
    a.output.mkdir()
    ledger=a.output/"FACTORIAL_EDITS_PRIVATE.jsonl"
    ledger_hash=sha256()
    with ledger.open("xb") as f, ProcessPoolExecutor(max_workers=a.workers,initializer=initialize,initargs=(a,)) as pool:
        for n,(row,counts,arms,errors,clock) in enumerate(pool.map(image_row,jobs),1):
            require(row["ordinal"]==n and row["image_id"]==request["images"][n-1]["image_id"], "Output image order changed")
            raw=old.encoded(row)
            require(f.write(raw)==len(raw),"Truncated image record")
            ledger_hash.update(raw)
            totals.update(counts)
            times.update(clock)
            for arm in ARMS:
                by_arm[arm].update(arms[arm])
            for key,val in errors.items():
                maxima[key]=max(maxima[key],val)
            if n==1 or n%5==0 or n==74:
                print(json.dumps(dict(phase="inference",images=n,candidates=totals["decoded_candidates"],
                      endpoint_disagreements=totals["endpoint_probability_differences"],
                      nc_disagreements=totals["decoded_candidates"]-totals["exact_nc_masks"],
                      wt_disagreements=totals["decoded_candidates"]-totals["exact_wt_masks"])),flush=True)
        f.flush()
        os.fsync(f.fileno())
    require(totals["images"]==totals["image_encodes"]==74 and totals["decoder_batches"]==2068
            and totals["decoded_candidates"]==4107 and totals["candidate_refinement_iterations"]==20535
            and totals["native_count_sum"]==29900 and totals["endpoint_pairs"]==129621
            and totals["endpoint_probabilities"]==259242 and totals["whole_count_rank_certificates"]==4107
            and by_arm["nc"]["flips"]==by_arm["wc"]["flips"]==59800, "Candidate, on-grid or support coverage differs")
    for arm,b in by_arm.items():
        require(b["positive_selected_flips"]+b["nonpositive_selected_flips"]==
                b["inside_narrow_flips"]+b["outside_narrow_flips"]==b["flips"]
                and b["nonpositive_selected_flips"]==b["negative_selected_flips"]+b["zero_selected_flips"]
                and b["candidates_below_2k"]+b["candidates_equal_2k"]+b["candidates_above_2k"]==4107,
                "Selected-utility, support or original-count census differs")
    masks_sha=ledger_hash.hexdigest()
    require(masks_sha==old.file_sha(ledger),"Complete mask ledger readback failed")
    if not (totals["exact_endpoint_probabilities"]==259242 and totals["exact_nc_masks"]==totals["exact_wt_masks"]==4107):
        errors=dict(status="RSDD74_PHOENIX_SUPPORT_BUDGET_ARCHIVE_DISAGREEMENT",counts=totals,maximum_errors=maxima)
        (a.output/"DISAGREEMENT_SAFE.json").write_bytes(old.encoded(errors))
        raise ValueError("Archive equality failed; see DISAGREEMENT_SAFE.json")
    safe=dict(status=STATUS,phase="inference",protocol_sha256=LOCK,worker_sha256=old.file_sha(Path(__file__)),
              baseline_worker_sha256=BASE,request_sha256=REQUEST,endpoint_sha256=ENDPOINT,area_edits_sha256=AREA,
              whole_masks_sha256=WHOLE,masks_sha256=masks_sha,input_lineage=request["input_lineage"],
              source_replay=request["source_replay"],author_commit=old.AUTHOR,weight_sha256=old.WEIGHT_SHA,
              endpoint_probabilities_exact=True,nc_matches_archived_free=True,wt_matches_archived_lvis=True,
              counts=totals,counts_by_arm=by_arm,maximum_errors=maxima,arms=list(ARMS),
              workers=a.workers,threads=a.threads,seed=old.SEED,batch_size=2,refine_iters=5,smooth_kernel=1,
              runtime=runtime,native_detector_forwards=0,optimizer_updates=0,target_labels_opened=False,
              seconds=perf_counter()-start,summed_worker_seconds=times)
    data=old.encoded(safe)
    with (a.output/"RESULT_SAFE.json").open("xb") as f:
        require(f.write(data)==len(data),"Incomplete footer")
    print(json.dumps(dict(status=STATUS,result_sha256=old.digest(data))),flush=True)


if __name__=="__main__":
    main()
