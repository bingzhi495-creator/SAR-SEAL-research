"""Return the two original area-policy whole-mask outputs on the frozen 74-image request."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import re
from time import perf_counter
from types import ModuleType


PRIOR_WORKER = "ea98a849ae181f06f3ffc0e41b29d2a91a7f867b4a4bcba6909c957f27db2b99"
NATIVE_PROJECTION = "91f95a50e19c9809b9a84f9e5825d2b6a31476d970dae727f144e852b60d4d20"
LOCK = "57cf64a2881325446cd61f548933a9924fc611a3633d871d17ddb641845be40e"
STATUS = "PHOENIX_RSDD74_WHOLE_INFERENCE_COMPLETE"
RUNTIME = {"FastGeodis":"1.0.5","numpy":"1.26.4","torch":"2.10.0+cu130","Pillow":"12.3.0","pycocotools":"2.0.11"}
POLICIES = ("lvis","coco")


def eligible(area, pixels, policy):
    if policy == "lvis":
        return area > 0
    if policy == "coco":
        return 100*area > pixels
    raise ValueError("Unknown Phoenix instance area policy")


def whole_policy(candidates, policy, decode_batch, encode_mask):
    """Run this branch on exactly its own eligible, ordered full-frame masks."""
    import numpy as np

    active = [(pos, mask) for pos, mask in candidates if eligible(int(mask.sum()), mask.size, policy)]
    counts = Counter(decoded_candidates=len(active), pass_through=len(candidates)-len(active),
                     decoder_batches=0, empty_outputs=0, unchanged_outputs=0, adds=0, drops=0)
    records = []
    decode_seconds = 0.
    for start in range(0, len(active), 2):
        group = active[start:start+2]
        began = perf_counter()
        result = decode_batch([raw for _, raw in group])
        decode_seconds += perf_counter()-began
        counts["decoder_batches"] += 1
        for (position, source), out in zip(group, result, strict=True):
            if not isinstance(out, np.ndarray) or out.dtype != np.bool_ or out.shape != source.shape:
                raise ValueError("Official Phoenix must return a native-size boolean output")
            adds = int(np.count_nonzero(out & ~source))
            drops = int(np.count_nonzero(source & ~out))
            counts.update(adds=adds, drops=drops, empty_outputs=int(not out.any()), unchanged_outputs=int(adds+drops == 0))
            records.append(dict(position=position, segmentation=encode_mask(out)))
    return records, counts, decode_seconds


def require(test, reason):
    if not test:
        raise ValueError(reason)


def read_closed_request(request, result, terminal, request_sha, protocol_sha):
    data, receipt_bytes, terminal_bytes = (p.read_bytes() for p in (request, result, terminal))
    receipt, end = json.loads(receipt_bytes), json.loads(terminal_bytes)
    result_sha = sha256(receipt_bytes).hexdigest()
    require(type(end["actual_exit"]) is int and end["actual_exit"] == 0 and end["output_footer_verified"] is True
            and end["phase"] == receipt["phase"] == "request" and end["result_sha256"] == result_sha
            and end["worker_sha256"] == receipt["worker_sha256"]
            and receipt["status"] == "PHOENIX_RSDD74_WHOLE_REQUEST_COMPLETE"
            and receipt["target_labels_opened"] is False and receipt["protocol_sha256"] == protocol_sha
            and receipt["request_sha256"] == request_sha == sha256(data).hexdigest(), "Invalid closed request")
    doc = json.loads(data)
    require(doc["status"] == "PHOENIX_RSDD74_WHOLE_REQUEST" and doc["protocol_sha256"] == protocol_sha,
            "Different request protocol")
    return doc, result_sha, sha256(terminal_bytes).hexdigest()


def encode_rle(mask, codec, np):
    rle = codec.encode(np.asfortranarray(mask, dtype=np.uint8))
    require(np.array_equal(codec.decode(rle), mask), "Encoded output changed a native pixel")
    return dict(size=list(mask.shape), counts=rle["counts"].decode("ascii"))


def decode_rle(rle, codec, np):
    require(type(rle) is dict and set(rle) == {"size","counts"} and rle["size"] == [512,512]
            and type(rle["counts"]) is str, "Expected original native 512-square COCO RLE")
    mask = codec.decode(dict(size=rle["size"], counts=rle["counts"].encode("ascii"))).astype(bool)
    require(mask.shape == (512,512) and int(mask.sum()) == int(codec.area(rle)), "Invalid native mask")
    return mask


def project_image(ordinal, row):
    expect = {"image_file","encoded_sha256","decoded_rgb_sha256","candidates"}
    require(set(row) in (expect, expect | {"image_id"}) and row["image_file"] == f"image_{ordinal:03}.png"
            and all(re.fullmatch("[0-9a-f]{64}", row[k]) for k in ("encoded_sha256","decoded_rgb_sha256")),
            "Unexpected roster entry")
    rows = row["candidates"]
    require(type(rows) is list and len(rows) <= 100 and all(
        set(x) == {"position","segmentation"} and type(x["position"]) is int and x["position"] == pos
        for pos,x in enumerate(rows,1)), "Full ordered compact candidate population required")
    return dict(ordinal=ordinal, decoded_rgb_sha256=row["decoded_rgb_sha256"], candidates=rows)


def load_baseline(path):
    src = path.read_bytes()
    require(sha256(src).hexdigest() == PRIOR_WORKER, "Prior qualified Phoenix worker fingerprint differs")
    module = ModuleType("_previous_phoenix")
    module.__file__ = str(path)
    exec(compile(src, str(path), "exec"), module.__dict__)
    return module


def initialize(args):
    global author
    author = load_baseline(args.baseline_worker)
    author.init_worker(args)


def image_row(task):
    ordinal, row = task
    api = author
    np, torch = api.np, api.torch
    path = api.request_dir / row["image_file"]
    require(path.resolve(strict=True) == path and api.file_sha(path) == row["encoded_sha256"],
            "Image file failed closed request digest")
    with api.Image.open(path) as f:
        require(f.format == "PNG" and f.mode == "RGB" and f.size == (512,512), "Expected lossless 512-square native RGB")
        rgb = np.array(f,dtype=np.uint8)
    require(api.digest(rgb.tobytes()) == row["decoded_rgb_sha256"], "RGB pixels differ from the original")
    masks = [(e["position"], decode_rle(e["segmentation"],api.codec,np)) for e in row["candidates"]]
    counts = dict(images=1,image_encodes=1,native_candidates=len(masks),
                  native_empty=sum(not m.any() for _,m in masks))
    started = perf_counter()
    random.seed(api.SEED)
    np.random.seed(api.SEED)
    torch.random.default_generator.manual_seed(api.SEED)
    with torch.inference_mode():
        api.refiner.set_image(rgb)
    seconds = {"encode": perf_counter()-started}

    def decoder(chunk):
        with torch.inference_mode():
            return api.refiner.refine_batch(chunk, refine_iters=5, smooth_kernel=1,
                                            binarize_input=True, multimask_output=True, return_logits=False,
                                            max_batch=2)

    def encode(mask):
        return encode_rle(mask,api.codec,np)

    entry = dict(ordinal=ordinal)
    for policy in POLICIES:
        # Each rule sees the same native inputs and image initial RNG state.
        random.seed(api.SEED)
        np.random.seed(api.SEED)
        torch.random.default_generator.manual_seed(api.SEED)
        entry[policy], counts[policy], seconds[policy] = whole_policy(masks,policy,decoder,encode)
    return entry, counts, seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for k in ("source-root","baseline-worker","request","request-result","request-terminal","protocol","output"):
        parser.add_argument("--"+k,required=True,type=Path)
    for k in ("request-sha256","protocol-sha256"):
        parser.add_argument("--"+k,required=True)
    parser.add_argument("--workers",type=int,choices=(4,),default=4)
    parser.add_argument("--threads",type=int,choices=(2,),default=2)
    args=parser.parse_args()
    for name,value in vars(args).items():
        if isinstance(value,Path):
            setattr(args,name,value.resolve(strict=name != "output"))
    previous=load_baseline(args.baseline_worker)
    require(previous.file_sha(args.protocol) == args.protocol_sha256 == LOCK and not args.output.exists(),
            "Prelocked protocol and fresh output directory required")
    started=perf_counter()
    os.environ.update(CUDA_VISIBLE_DEVICES="-1",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS=str(args.threads),
                      MKL_NUM_THREADS=str(args.threads),TORCH_FORCE_WEIGHTS_ONLY_LOAD="1")
    runtime={key:version(key) for key in RUNTIME}
    require(runtime == RUNTIME,"Previously qualified CPU runtime required")
    request,receipt_sha,terminal_sha=read_closed_request(args.request,args.request_result,args.request_terminal,
                                                         args.request_sha256,args.protocol_sha256)
    images=request["images"]
    projection=sha256()
    for ordinal,row in enumerate(images,1):
        projection.update(previous.encoded(project_image(ordinal,row)))
    require(len(images) == 74 and projection.hexdigest() == NATIVE_PROJECTION,
            "The request must preserve all original native mask positions and exact RGB identities")
    previous.verify_author(args.source_root)
    args.output.mkdir()
    ledger=args.output/"WHOLE_MASKS_PRIVATE.jsonl"
    totals,branches,clock=Counter(),{k:Counter() for k in POLICIES},Counter()
    ledger_digest=sha256()
    with ledger.open("xb") as f, ProcessPoolExecutor(
            max_workers=args.workers,initializer=initialize,initargs=(args,)) as pool:
        for ordinal,(line,counts,seconds) in enumerate(pool.map(image_row,enumerate(images,1)),1):
            require(line["ordinal"] == ordinal, "Worker output differs from the original manifest order")
            for policy in POLICIES:
                per=counts.pop(policy)
                branches[policy].update(per)
                require(len(line[policy])==per["decoded_candidates"] and per["decoded_candidates"]+
                        per["pass_through"]==len(images[ordinal-1]["candidates"]), "Incomplete policy branch")
            totals.update(counts)
            clock.update(seconds)
            raw=previous.encoded(line)
            require(f.write(raw)==len(raw),"Incomplete policy output")
            ledger_digest.update(raw)
            if ordinal == 1 or ordinal%5 == 0 or ordinal == 74:
                print(json.dumps(dict(phase="inference",images=ordinal,
                                      lvis=branches["lvis"]["decoded_candidates"],
                                      coco=branches["coco"]["decoded_candidates"])),flush=True)
        f.flush()
        os.fsync(f.fileno())
    require(totals == dict(images=74,image_encodes=74,native_candidates=7346,native_empty=1267)
            and (branches["lvis"]["decoded_candidates"],branches["lvis"]["decoder_batches"]) == (6079,3057)
            and (branches["coco"]["decoded_candidates"],branches["coco"]["decoder_batches"]) == (172,89)
            and ledger_digest.hexdigest()==previous.file_sha(ledger), "Closed image, area, policy or writeback census differs")
    counts={**totals,**branches}
    for policy in POLICIES:
        b=counts[policy]
        b["changed_outputs"]=b["decoded_candidates"]-b["unchanged_outputs"]
        b["all_unchanged_outputs"]=b["pass_through"]+b["unchanged_outputs"]
        b["signed_area_delta"]=b["adds"]-b["drops"]
        b["candidate_refinement_iterations"]=b["decoded_candidates"]*5
    result=dict(status=STATUS,phase="inference",worker_sha256=previous.file_sha(Path(__file__)),
                protocol_sha256=LOCK,request_sha256=args.request_sha256,request_result_sha256=receipt_sha,
                request_terminal_sha256=terminal_sha,baseline_worker_sha256=PRIOR_WORKER,
                native_projection_sha256=NATIVE_PROJECTION,author_commit=previous.AUTHOR,weight_sha256=previous.WEIGHT_SHA,
                author_manifest_sha256=previous.MANIFEST_SHA,counts=counts,masks_sha256=ledger_digest.hexdigest(),
                seconds=perf_counter()-started,summed_worker_seconds=dict(clock),workers=args.workers,threads=args.threads,
                runtime=runtime,seed=previous.SEED,batch_size=2,refine_iters=5,smooth_kernel=1,multimask_output=True,
                binarize_input=True,binary_writer=True,policy_decodes_separate=True,
                optimizer_updates=0,native_detector_forwards=0,target_labels_opened=False)
    raw=previous.encoded(result)
    with (args.output/"RESULT_SAFE.json").open("xb") as f:
        require(f.write(raw)==len(raw),"Incomplete result")
    print(json.dumps(dict(status=STATUS,result_sha256=previous.digest(raw))),flush=True)


if __name__=="__main__":
    main()
