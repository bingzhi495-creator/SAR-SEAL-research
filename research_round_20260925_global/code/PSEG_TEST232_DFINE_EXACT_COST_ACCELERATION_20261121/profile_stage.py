"""Measure frozen natural policy components on 32 fixed, equally spaced official images."""
from collections import Counter
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
from time import perf_counter, perf_counter_ns
from types import SimpleNamespace

import numpy as np
import torch
from pycocotools import mask as codec


HERE = Path(__file__).resolve().parent
SOURCE = "410e4f456c081ab27e5acf8af8d1c6a789e206e4d36b52d7bc8c7232d5f4595e"


def load_baseline():
    path = Path(os.environ["DFINE_BASELINE_SOURCE"]).resolve(strict=True)
    src = path.read_bytes()
    if sha256(src).hexdigest() != os.environ.get("DFINE_BASELINE_SOURCE_SHA", SOURCE):
        raise ValueError("Exact prebound natural CPU baseline")
    spec = importlib.util.spec_from_file_location("_unchanged_cpu_natural", path)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(src,str(path),"exec"),mod.__dict__)
    mod.ROOT = Path(os.environ["DFINE_CLOSED_INPUT_ROOT"]).resolve(strict=True)
    mod.OWNER = Path(os.environ["DFINE_CAPTURE_OWNER"]).resolve(strict=True)
    mod.REPLAY = Path(os.environ["DFINE_CLOSED_REPLAY"]).resolve(strict=True)
    mod.HERE = Path(os.environ["DFINE_BASELINE_RUN"]).resolve(strict=True)
    return mod


def exact_reference(coco, block):
    out = list(coco)
    for key, edits in block["changes"].items():
        if "free_natural" in edits:
            i = int(key)-1
            out[i] = dict(coco[i], segmentation=edits["free_natural"]["rle"])
    return out


def get_iterator(baseline):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    a,m,roster,receipt,_,evidence,models,_ = baseline.read_closed_inputs()
    return a,m,iter(m.legacy.blocks(a,m.author,roster,receipt,True)),evidence,models


def profile():
    baseline = load_baseline()
    a,m,blocks,evidence,models = get_iterator(baseline)
    selected = set(np.linspace(0,231,32,dtype=int).tolist())
    times = Counter()
    calls = Counter()
    def timed(name, fn):
        def wrap(*args,**kwargs):
            t = perf_counter_ns()
            val = fn(*args,**kwargs)
            times[name] += perf_counter_ns()-t
            calls[name] += 1
            return val
        return wrap
    pair,rank,free,learner,head,center,scale = models
    pair.image_features = timed("image_features",pair.image_features)
    free.choose_free = timed("choose_free",free.choose_free)
    learner.probabilities = timed("head",learner.probabilities)
    m.old.edit = timed("encode_edited_rle",m.old.edit)
    codec.decode = timed("decode_stock_rle",codec.decode)
    original = m.native.make_native
    def adapt(*args):
        native, sampling = original(*args)
        sampling.editable_pools = timed("native_mask_support",sampling.editable_pools)
        sampling.sample_probability = timed("sample_160_plane",sampling.sample_probability)
        native.paired_points = timed("point_feature_inclusive",native.paired_points)
        return native, sampling
    m.native.make_native = timed("shape_adapter",adapt)
    whole = 0
    for i,(item,coco,audit,planes) in enumerate(blocks):
        if i not in selected:
            continue
        rgb = m.legacy.rgb(a,item)
        want = exact_reference(coco,evidence[i])
        start = perf_counter_ns()
        got = baseline.natural(m,models,coco,audit,planes,rgb)
        whole += perf_counter_ns()-start
        baseline.check(got==want, "Exact original full RLE profile replay")
    times["point_construction_other"] = times.pop("point_feature_inclusive") - times["native_mask_support"] - times["sample_160_plane"]
    times["other"] = whole - sum(times.values())
    result = dict(status="PSEG_TEST232_FROZEN_NATURAL_PROFILE_COMPLETE",images=len(selected),population=232,
                  ordinals=[i+1 for i in sorted(selected)],measure="single-thread wall, timed subfunctions, excludes file IO",
                  components_seconds={n:v/1e9 for n,v in times.items()},wall_seconds=whole/1e9,
                  fractions={n:v/whole for n,v in times.items()},calls=dict(calls),exact_closed_output=True,
                  original_benchmark_source_sha256=os.environ.get("DFINE_BASELINE_SOURCE_SHA",SOURCE),
                  GT_opened=False,detector_forwards=0)
    payload=json.dumps(result,indent=2,allow_nan=False).encode()+b"\n"
    with (HERE/"PROFILE_SAFE.json").open("xb") as stream:
        stream.write(payload)
    print(json.dumps(dict(status=result["status"],sha256=sha256(payload).hexdigest(), fractions=result["fractions"])),flush=True)


if __name__ == "__main__":
    profile()
