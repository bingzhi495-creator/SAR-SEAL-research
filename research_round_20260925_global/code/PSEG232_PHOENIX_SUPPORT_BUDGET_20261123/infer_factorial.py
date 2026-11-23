"""Apply the frozen Phoenix support/count policies on each original-size PSeg-SSDD test chip."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from hashlib import sha256
from importlib.metadata import version
from itertools import accumulate
import json
from math import fsum
import os
from pathlib import Path
import random
from time import perf_counter
from types import ModuleType

import numpy as np


ENGINE_SHA = "8d46b1d16bcba274576f03448e7cf74b9d332a9bf8848b090aa659409fa91c80"
LOCK = "5307bcbc32f94f11d27092945e1993692b6ef8347e85a2d3231080dae54c81df"
STATUS = "PSEG232_PHOENIX_SUPPORT_BUDGET_INFERENCE_COMPLETE"
REQUEST_STATUS = "PSEG232_PHOENIX_SUPPORT_BUDGET_REQUEST"
# Every finite probability returned as binary32 is an integer multiple of 2^-149.
HALF_UNITS = 1 << 148


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def probability_units(value):
    num,den=float(value).as_integer_ratio()
    return num << (150-den.bit_length())


def exact_fixed_count(native, prob, add, drop, k, *, top_sites, diagnostic):
    """Compare every signed allocation in exact binary32 units before applying ordinal ties."""
    flat=prob.ravel()
    selected_add=top_sites(add,flat,True,2*k)
    selected_drop=top_sites(drop,flat,False,2*k)
    au=list(accumulate((probability_units(flat[i])-HALF_UNITS for i in selected_add),initial=0))
    du=list(accumulate((HALF_UNITS-probability_units(flat[i]) for i in selected_drop),initial=0))
    feasible=range(max(0,2*k-len(selected_drop)),min(2*k,len(selected_add))+1)
    neg,_,a=min((-(au[a]+du[2*k-a]),abs(2*a-2*k),a) for a in feasible)
    # Replay the legacy binary64 decision on the same ranked prefixes for diagnosis only.
    old_a=flat[selected_add].astype(np.float64)-.5
    old_d=.5-flat[selected_drop].astype(np.float64)
    _,_,oa=min((-fsum([*old_a[:x],*old_d[:2*k-x]]),abs(2*x-2*k),x) for x in feasible)
    od=2*k-oa
    # Within-sign ordering is fixed. A strict global rank violation requires a better omitted opposite sign.
    violation=(oa>0 and od<len(old_d) and old_a[oa-1]<old_d[od]) or (od>0 and oa<len(old_a) and old_d[od-1]<old_a[oa])
    diagnostic.update(legacy_rounded_split_differs=int(oa!=a),legacy_rank_violation=int(violation),
                      legacy_true_objective_inferior=int(au[oa]+du[od] < -neg))
    output=native.copy(order="C")
    output.flat[selected_add[:a]]=True
    output.flat[selected_drop[:2*k-a]]=False
    return output,-neg


def verify_exact_cut(native, prob, wide):
    """Independently compare true extrema on selected and omitted native sites."""
    flips=wide!=native
    selected,omitted=[],[]
    for is_drop in (False,True):
        for chosen in (True,False):
            values=prob[(native==is_drop)&(flips==chosen)]
            if values.size==0:
                continue
            if is_drop:
                extremum=values.max() if chosen else values.min()
                utility=HALF_UNITS-probability_units(extremum)
            else:
                extremum=values.min() if chosen else values.max()
                utility=probability_units(extremum)-HALF_UNITS
            (selected if chosen else omitted).append(utility)
    require(selected and (not omitted or min(selected)>=max(omitted)), "Failed exact whole-support rank certificate")


def load_engine(path):
    data=path.read_bytes()
    require(sha256(data).hexdigest()==ENGINE_SHA,"Exact qualified policies and native-grid writer required")
    mod=ModuleType("_rsdd_support_budget")
    mod.__file__=str(path)
    exec(compile(data,str(path),"exec"),mod.__dict__)
    # Adapt this private in-process instance. The archived module on disk and its strict certificate stay intact.
    mod.last_allocation_diagnostic={}
    mod.fixed_count=partial(exact_fixed_count,top_sites=mod.top_sites,diagnostic=mod.last_allocation_diagnostic)
    return mod


def decode_candidate(cand, hw, codec):
    rle=cand["segmentation"]
    require(set(cand)=={"position","segmentation","k","coords"} and set(rle)=={"size","counts"}
            and rle["size"]==list(hw) and type(rle["counts"]) is str, "Expected original native-grid input mask")
    mask=codec.decode(rle).astype(bool)
    coords=np.asarray(cand["coords"])
    area=int(np.count_nonzero(mask))
    require(mask.shape==hw and int(codec.area(rle))==area and 32<=area<1024
            and coords.ndim==2 and coords.shape[1]==2 and coords.dtype.kind in "iu" and len(coords)<=32
            and type(cand["k"]) is int and 0<cand["k"]<=min(8,area//10,len(coords))
            and np.all((coords>=0)&(coords<mask.size)) and np.unique(coords).size==coords.size
            and not mask.flat[coords[:,0]].any() and mask.flat[coords[:,1]].all(), "Invalid signed endpoint/area/count")
    return mask,coords


def certify(mask, coords, prob, binary, k, engine, codec):
    # The pinned reference policies operate on general (H,W) arrays using original native row-major indexing.
    masks,stats=engine.policies(mask,coords,prob,binary,k)
    # The pinned policy function calls the solver first on narrow, then on whole support.
    stats.update(engine.last_allocation_diagnostic)
    verify_exact_cut(mask,prob,masks["wc"])
    q=prob.flat[coords]
    half_true=np.flatnonzero((q==.5) & binary.flat[coords])
    witness=dict(probabilities=q.tolist(),half_foreground=half_true.tolist())
    require(np.array_equal(np.asarray(witness["probabilities"],dtype=np.float32),q),"Sparse posterior precision loss")
    restored=q>.5
    restored.flat[half_true]=True
    require(np.array_equal(masks["nt"].flat[coords],restored)
            and np.array_equal(masks["wt"],binary), "Threshold projection differs from the author writer")
    records={}
    counts={}
    for arm,out in masks.items():
        changed=out!=mask
        fg=mask[changed]
        selected=prob[changed]
        flips=len(fg)
        drops=int(np.count_nonzero(fg))
        adds=flips-drops
        pos=int(np.count_nonzero((~fg & (selected>.5))|(fg & (selected<.5))))
        zero=int(np.count_nonzero(selected==.5))
        neg=flips-pos-zero
        inside=int(np.count_nonzero(changed.flat[coords]))
        require((arm not in ("nc","wc") or flips==2*k) and (arm not in ("nt","wt") or neg==0)
                and (arm not in ("nc","nt") or inside==flips), "Incorrect support, count, or threshold")
        records[arm]=dict(rle=engine.encode(out,codec))
        counts[arm]=dict(adds=adds,drops=drops,flips=flips,changed_candidates=int(flips!=0),empty_outputs=int(not out.any()),
                         signed_area_delta=adds-drops,positive_selected_flips=pos,negative_selected_flips=neg,
                         zero_selected_flips=zero,nonpositive_selected_flips=neg+zero,
                         inside_narrow_flips=inside,outside_narrow_flips=flips-inside,
                         candidates_below_2k=int(flips<2*k),candidates_equal_2k=int(flips==2*k),
                         candidates_above_2k=int(flips>2*k))
    stats.update(whole_count_rank_certificates=1,exact_dyadic_rank_certificates=1,endpoint_pairs=len(coords),endpoint_probabilities=q.size,
                 exact_nt_projections=1,exact_wt_author_masks=1)
    return records,witness,stats,counts


def read_request(args):
    raw,end_raw,receipt_raw=(p.read_bytes() for p in (args.request,args.request_terminal,args.request_result))
    receipt_sha=sha256(receipt_raw).hexdigest()
    doc,end,receipt=(json.loads(data) for data in (raw,end_raw,receipt_raw))
    require(type(end["actual_exit"]) is int and end["actual_exit"]==0 and end["output_footer_verified"] is True
            and end["phase"]==receipt["phase"]=="request" and end["worker_sha256"]==receipt["worker_sha256"]
            and end["result_sha256"]==receipt_sha and receipt["status"]==REQUEST_STATUS+"_COMPLETE"
            and receipt["target_labels_opened"] is False and receipt["detector_forwards"]==receipt["optimizer_updates"]==0
            and sha256(raw).hexdigest()==receipt["request_sha256"]==args.request_sha256,
            "A closed image-only request and observed successful footer are required")
    require(doc["status"]==REQUEST_STATUS and receipt["protocol_sha256"]==doc["protocol_sha256"]==LOCK
            and receipt["input_lineage"]==doc["input_lineage"] and receipt["source_replay"]==doc["source_replay"],
            "Historical candidate and pixel-support provenance differs")
    counts=receipt["counts"]
    expected={"images":232,"mapped_candidates":23200,"area_eligible":11209,"positive_budget_candidates":11054,
              "swaps_per_arm":73770}
    require(all(counts[k]==v for k,v in expected.items()) and len(doc["images"])==232,
            "Unexpected historical/native candidate roster")
    ids, actual = set(), Counter()
    for ordinal,image in enumerate(doc["images"],1):
        ident=image["image_id"]
        require(set(image)=={"image_id","image_file","encoded_sha256","decoded_rgb_sha256","hw","candidates"}
                and type(ident) in (str,int) and ident not in ids and image["image_file"]==f"image_{ordinal:03d}.png"
                and len(image["hw"])==2 and all(type(s) is int and s>0 for s in image["hw"]),
                "Image order, identity or original native geometry differs")
        ids.add(ident)
        positions=[c["position"] for c in image["candidates"]]
        require(positions==sorted(set(positions)) and all(type(pos) is int and 1<=pos<=100 for pos in positions),
                "Active positions must stay in original score order")
        actual.update(positive_budget_candidates=len(positions),swaps_per_arm=sum(c["k"] for c in image["candidates"]),
                      endpoint_pairs=sum(len(c["coords"]) for c in image["candidates"]))
    require(all(actual[k]==counts[k] for k in actual), "Request endpoint census differs from its receipt")
    return doc,receipt,receipt_sha,sha256(end_raw).hexdigest()


def initialize(args):
    global engine,author
    engine=load_engine(args.factorial_worker)
    author=engine.baseline(args.baseline_worker)
    author.init_worker(args)


def image_row(task):
    ordinal,img=task
    api=author
    hw=tuple(img["hw"])
    path=api.request_dir/img["image_file"]
    require(path.resolve(strict=True)==path and api.file_sha(path)==img["encoded_sha256"], "Encoded RGB changed")
    with api.Image.open(path) as f:
        require(f.format=="PNG" and f.mode=="RGB" and f.size==(hw[1],hw[0]),"Native PNG dimensions differ")
        rgb=np.asarray(f,dtype=np.uint8).copy()
    require(rgb.shape==(*hw,3) and api.digest(rgb.tobytes())==img["decoded_rgb_sha256"],"Decoded native RGB changed")
    random.seed(api.SEED)
    np.random.seed(api.SEED)
    api.torch.random.default_generator.manual_seed(api.SEED)
    start=perf_counter()
    with api.torch.inference_mode():
        api.refiner.set_image(rgb)
    clock=Counter(encode=perf_counter()-start,decode=0.,write=0.)
    total=Counter(images=1,image_encodes=1,non_square_images=int(hw[0]!=hw[1]))
    by_arm={arm:Counter() for arm in engine.ARMS}
    changes,readout={},[]
    items=img["candidates"]
    for offset in range(0,len(items),2):
        batch=items[offset:offset+2]
        native=[decode_candidate(c,hw,api.codec) for c in batch]
        start=perf_counter()
        with api.torch.inference_mode():
            binary,field=api.refiner.refine_batch([m for m,_ in native],refine_iters=5,smooth_kernel=1,
                            binarize_input=True,multimask_output=True,return_logits=True,max_batch=2)
        clock["decode"]+=perf_counter()-start
        start=perf_counter()
        for cand,(mask,coords),pred,q in zip(batch,native,binary,field,strict=True):
            arms,witness,stats,count=certify(mask,coords,q,pred,cand["k"],engine,api.codec)
            total.update(stats)
            total.update(decoded_candidates=1,candidate_refinement_iterations=5,native_count_sum=cand["k"])
            changes[str(cand["position"])]=arms
            readout.append(dict(position=cand["position"],**witness))
            for arm,v in count.items():
                by_arm[arm].update(v)
        total["decoder_batches"]+=1
        clock["write"]+=perf_counter()-start
    idrow=dict(ordinal=ordinal,image_id=img["image_id"])
    return dict(**idrow,changes=changes),dict(**idrow,candidates=readout),total,by_arm,clock


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for key in ("source-root","baseline-worker","factorial-worker","request","request-result","request-terminal","protocol","output"):
        p.add_argument("--"+key,required=True,type=Path)
    p.add_argument("--protocol-sha256",required=True)
    p.add_argument("--request-sha256",required=True)
    p.add_argument("--workers",type=int,choices=(4,),default=4)
    p.add_argument("--threads",type=int,choices=(2,),default=2)
    args=p.parse_args()
    for key,val in vars(args).items():
        if isinstance(val,Path):
            setattr(args,key,val.resolve(strict=key!="output"))
    ref=load_engine(args.factorial_worker)
    old=ref.baseline(args.baseline_worker)
    require(not args.output.exists() and args.protocol_sha256==LOCK==old.file_sha(args.protocol),
            "Fresh output directory and exact protocol required")
    start=perf_counter()
    os.environ.update(CUDA_VISIBLE_DEVICES="-1",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS=str(args.threads),
                      MKL_NUM_THREADS=str(args.threads),TORCH_FORCE_WEIGHTS_ONLY_LOAD="1")
    runtime={k:version(k) for k in ref.RUNTIME}
    require(runtime==ref.RUNTIME,"Previously qualified CPU environment required")
    request,source,rsha,tsha=read_request(args)
    old.verify_author(args.source_root)
    args.output.mkdir()
    path=args.output/"FACTORIAL_EDITS_PRIVATE.jsonl"
    posterior=args.output/"NARROW_POSTERIORS_PRIVATE.jsonl"
    counts,clock=Counter(),Counter()
    by_arm={arm:Counter() for arm in ref.ARMS}
    eh,ph=sha256(),sha256()
    images=request["images"]
    with path.open("xb") as edits,posterior.open("xb") as witness,ProcessPoolExecutor(
            max_workers=args.workers,initializer=initialize,initargs=(args,)) as pool:
        for ordinal,(row,sparse,stats,arms,times) in enumerate(pool.map(image_row,enumerate(images,1)),1):
            require(row["ordinal"]==sparse["ordinal"]==ordinal and row["image_id"]==sparse["image_id"]==images[ordinal-1]["image_id"],
                    "Parallel outputs changed original image order")
            for file,digest,obj in ((edits,eh,row),(witness,ph,sparse)):
                data=old.encoded(obj)
                require(file.write(data)==len(data),"Incomplete aggregate output ledger")
                digest.update(data)
            counts.update(stats)
            clock.update(times)
            for arm,val in arms.items():
                by_arm[arm].update(val)
            if ordinal==1 or ordinal%10==0 or ordinal==232:
                print(json.dumps(dict(phase="inference",images=ordinal,decoded_candidates=counts["decoded_candidates"])),flush=True)
        for file in (edits,witness):
            file.flush()
            os.fsync(file.fileno())
    require(counts["images"]==counts["image_encodes"]==232 and counts["decoded_candidates"]==11054 and counts["native_count_sum"]==73770
            and counts["candidate_refinement_iterations"]==55270
            and counts["whole_count_rank_certificates"]==counts["exact_dyadic_rank_certificates"]==11054
            and counts["exact_nt_projections"]==counts["exact_wt_author_masks"]==11054 and counts["threshold_disagreements_away_from_half"]==0
            and counts["decoder_batches"]==sum((len(im["candidates"])+1)//2 for im in images)
            and counts["endpoint_pairs"]==source["counts"]["endpoint_pairs"]
            and counts["endpoint_probabilities"]==2*counts["endpoint_pairs"]
            and by_arm["nc"]["flips"]==by_arm["wc"]["flips"]==147540, "Population or allocated flip count changed")
    for b in by_arm.values():
        require(b["flips"]==b["positive_selected_flips"]+b["nonpositive_selected_flips"]==b["outside_narrow_flips"]+b["inside_narrow_flips"]
                and b["nonpositive_selected_flips"]==b["zero_selected_flips"]+b["negative_selected_flips"]
                and b["candidates_below_2k"]+b["candidates_equal_2k"]+b["candidates_above_2k"]==11054,
                "Incorrect counts across signed supports or threshold decisions")
    require(eh.hexdigest()==old.file_sha(path) and ph.hexdigest()==old.file_sha(posterior), "Ledger digest readback failed")
    result=dict(phase="inference",status=STATUS,worker_sha256=old.file_sha(Path(__file__)),protocol_sha256=LOCK,
                factorial_worker_sha256=ENGINE_SHA,baseline_worker_sha256=ref.BASE,request_sha256=args.request_sha256,
                request_result_sha256=rsha,request_terminal_sha256=tsha,request_worker_sha256=source["worker_sha256"],
                input_lineage=request["input_lineage"],source_replay=request["source_replay"],masks_sha256=eh.hexdigest(),
                narrow_posteriors_sha256=ph.hexdigest(),arms=list(ref.ARMS),counts=counts,counts_by_arm=by_arm,
                seed=old.SEED,threads=args.threads,workers=args.workers,batch_size=2,refine_iters=5,smooth_kernel=1,
                author_commit=old.AUTHOR,weight_sha256=old.WEIGHT_SHA,runtime=runtime,summed_worker_seconds=clock,
                threshold_projection_exact=True,whole_binary_exact=True,signed_split_objective="exact_binary32_dyadic",
                optimizer_updates=0,native_detector_forwards=0,
                target_labels_opened=False,seconds=perf_counter()-start)
    data=old.encoded(result)
    with (args.output/"RESULT_SAFE.json").open("xb") as f:
        require(f.write(data)==len(data),"Incomplete aggregate receipt")
    print(json.dumps(dict(status=STATUS,result_sha256=old.digest(data))),flush=True)


if __name__=="__main__":
    main()
