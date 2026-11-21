"""Within-image, counterbalanced comparison of the unchanged and sparse natural refinements."""
from collections import Counter
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import torch


HERE = Path(__file__).resolve().parent


def load_local(name):
    path = HERE/(name+".py")
    spec = importlib.util.spec_from_file_location(name,path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def write_json(name, record):
    raw = json.dumps(record, indent=2,allow_nan=False).encode()+b"\n"
    with (HERE/name).open("xb") as file:
        file.write(raw)
    return sha256(raw).hexdigest()


def run():
    prof, sparse = load_local("profile_stage"), load_local("sparse_sampling")
    b = prof.load_baseline()
    a,m,blocks,evidence,models = prof.get_iterator(b)
    counts = Counter()
    opt = sparse.replacement(m)
    dual = sparse.replacement(m, audit=counts)
    results, io_seconds = [], 0.
    origin = perf_counter()
    for ordinal, frozen in enumerate(evidence,1):
        t = perf_counter()
        item, coco, boxes, planes = next(blocks)
        rgb = m.legacy.rgb(a,item)
        io_seconds += perf_counter()-t
        b.check(item["image_id"] == frozen["image_id"] and sha256(rgb.tobytes()).hexdigest() == frozen["rgb_sha256"],
                "Exact accepted image")
        wanted = prof.exact_reference(coco,frozen)
        params = (models,coco,boxes,planes,rgb)
        arms = ("original","optimized") if ordinal % 2 else ("optimized","original")
        for arm in arms:
            mod = m if arm == "original" else dual
            b.check(b.natural(mod,*params) == wanted, "Warm complete native rows " + arm)
        pairs = {"original":[], "optimized":[]}
        for repeat in range(2):
            for arm in (arms if repeat==0 else arms[::-1]):
                mod = m if arm=="original" else opt
                predictions, wall_ms, cpu_ms = b.timed(lambda: b.natural(mod,*params))
                b.check(predictions == wanted, "Timed complete native rows " + arm)
                pairs[arm].append((wall_ms,cpu_ms))
        counts["candidates"] += len(coco)
        counts["edited_candidates"] += sum("free_natural" in edits for edits in frozen["changes"].values())
        counts["swaps"] += sum(edits["free_natural"]["swaps"] for edits in frozen["changes"].values()
                               if "free_natural" in edits)
        results.append(pairs)
        if ordinal%32==0 or ordinal==232:
            print(json.dumps(dict(event="FROZEN_NATURAL_EXACT_COMPARISON_PROGRESS",images=ordinal)),flush=True)
    b.check(next(blocks,None) is None and len(results)==232 and counts["candidates"]==23200
            and counts["edited_candidates"]==10466 and counts["swaps"]==55040 and
            counts["sampled_values"]>0 and not torch.cuda.is_initialized(), "Complete input and published output population")
    orig, accel = (np.asarray([row[k] for row in results]) for k in ("original","optimized"))
    stats = {}
    for j,clock in enumerate(("wall","process_cpu")):
        for label,array in (("original",orig),("optimized",accel),("saving",orig-accel)):
            stats[clock+"_"+label] = b.summary(array[:,:,j].mean(axis=1))
    base, new = orig[:,:,0].mean(axis=1), accel[:,:,0].mean(axis=1)
    observations_sha = write_json("PAIRED_TIMES_SAFE.json",dict(units="milliseconds",axes=["image in official order",
                       "repetition","wall/process CPU"],original=orig.tolist(),optimized=accel.tolist()))
    outcome = dict(status="PSEG_TEST232_EXACT_SPARSE_SAMPLER_TIMING_COMPLETE",images=len(results),
                   counts=dict(counts),all_warm_sample_uint16_bits_equal=True,all_complete_candidate_records_exact=True,
                   warm_and_measured_parity_passes_per_arm=696,measured_pairs_per_image=2,stage_summary=stats,
                   faster_images=int((base>new).sum()),slower_images=int((base<new).sum()),
                   tied_images=int((base==new).sum()),mean_wall_saving_percent=float(100*(1-new.sum()/base.sum())),
                   ratio_original_over_optimized=float(base.sum()/new.sum()),
                   repetition_totals_wall_seconds={k:(arr[:,:,0].sum(axis=0)/1000).tolist()
                   for k,arr in (("original",orig),("optimized",accel),("saving",orig-accel))},
                   capture_loading_and_rgb_seconds=io_seconds,elapsed_body_seconds=perf_counter()-origin,
                   paired_times_sha256=observations_sha,
                   input_refs_sha256={name:sha256((HERE/name).read_bytes()).hexdigest() for name in
                       ("profile_stage.py","compare.py","sparse_sampling.py","PROFILE_SAFE.json","PROTOCOL.md")},
                   legacy_natural_worker_sha256=os.environ.get("DFINE_BASELINE_SOURCE_SHA",prof.SOURCE),
                   prior_replay_readback_sha256=b.ACCEPT,python=platform.python_version(),torch=torch.__version__,
                   numpy=np.__version__,torch_intra_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),
                   cpu="Intel Core i7-13700KF",concurrent_HRSID_CPU_context=True,detector_forwards=0,
                   optimizer_steps=0,GT_opened=False,AP=False,cuda_initialized=False)
    digest = write_json("RESULT_SAFE.json",outcome)
    print(json.dumps(dict(status=outcome["status"],result_sha256=digest,percent=outcome["mean_wall_saving_percent"])),flush=True)


if __name__ == "__main__":
    run()
