"""Prepare label-free native signed points on already partitioned TRAIN915 image-only captures."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np
from pycocotools import mask


LOCK = "f454da151071eb0e05f38ee0a75cb4051cc2268e7667e4051408f05342b5a39a"
VERIFY = "e461828b04f166d98030210fe42df0d590d24e2452ea5da8630fb0fc3219d795"
VWORKER = "12b289e1307de56151cf016b8f9f8100eaffab8f6ae3da4a3b890647029dd449"
CAPTURE = "6f58ba8b486950431f4aebebf237e60029b447e5ec21a7f51ad2202d065f904a"
CWORKER = "0f10812a7d2409f69334670e7dae16a9ef51b4f84a20ee1278a4187d06e4c9a0"
MANIFEST = "701c89d7350f6a078337d6074d69960799ade868a4f64e546897e07137c92bf4"
ROSTER = "f4e23e5eb7f58a881db48e4a89d2902a253aa945e7a92ab7d1674c8b02b3fb17"
SPLIT = "91095aacbbbc89003ffd6eb8f25506ce99a64eee5ac369a7753203bbf962bcf9"
SPLIT_RESULT = "a02f0ac9e48bc4a54fd55a7d05853ddceee04fa69c4887ea279f85a09ae05872"
STATUS = "SPSD_PSEG_DFINE_TRAIN915_LABEL_FREE_POINTS_COMPLETE"
SOURCE = {
  "base": ("DFINE_HRSID_TO_SSDD_DEV187_CAPTURE_20261120/author_export_reference.py","5be622cdec5c889ca8520aab5547f6f15f01a54568a0263423a816dca7f96626"),
  "replay": ("DFINE_HRSID_TO_SSDD_POINT_TRANSFER_20261120/ssdd_replay.py","ee1db45898c42262ba263e15a14ee24c72ef1ee00d5015b5bd243daaeeee01d3"),
  "native": ("DFINE_HRSID_TO_SSDD_POINT_TRANSFER_20261120/native_points.py","475ecbdd932a2129f160e48bd1d838653b971df6cd69b9edda103c037ee3fcdc"),
  "pair": ("DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/fit109_paired_point/pair_math.py","8fe7a2c146c08c89f377b1f20b321276a60356de4bdbb27be2a00544dfc77962"),
  "rank": ("DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/fit109_probability_rank/rank_math.py","cb438f8937f8e29dbdcbc441abdf5a788ac9c442fd63f1b3ad504b8ae0fcca0b"),
  "attraction": ("FIT109_CLASSICAL_ATTRACTION_20261121/spatial_attraction.py","8eb162a3bd544929843d7f2aeef174f1f1ba03a2c593b6bcda49289cd94ad3c6"),
}


def need(test, reason):
    if not test:
        raise ValueError(reason)


def digest(data):
    return sha256(data).hexdigest()


def pin(path,checksum):
    data=path.read_bytes()
    need(digest(data)==checksum,"Exact closed input: "+path.name)
    return data


def module(path,expected):
    obj=ModuleType("_train_point_prep_"+path.stem)
    obj.__file__=str(path)
    obj.raw=pin(path,expected)
    exec(compile(obj.raw,str(path),"exec"),obj.__dict__)
    return obj


def recorded(root, kind, phase, safe, worker):
    data=(root/kind/"TERMINAL.json").read_bytes()
    term=json.loads(data)
    need(term["phase"]==phase and term["actual_exit"]==0 and term["output_footer_verified"] is True
         and term["result_sha256"]==safe and term["worker_sha256"]==worker,"Exact original closed "+phase)
    return digest(data)


def admitted(a):
    capture_terminal=recorded(a.train_root,"observed_capture","capture",CAPTURE,CWORKER)
    verify_terminal=recorded(a.train_root,"observed_readback","verify",VERIFY,VWORKER)
    capture=json.loads(pin(a.train_root/"capture/RESULT_SAFE.json",CAPTURE))
    receipt=json.loads(pin(a.train_root/"READBACK_SAFE.json",VERIFY))
    need(receipt["status"]=="DFINE_HRSID_E24_PSEG_TRAIN915_PROBABILITY_INDEPENDENTLY_ACCEPTED_V1"
         and receipt["export_result_sha256"]==CAPTURE and receipt["export_terminal_sha256"]==capture_terminal
         and receipt["manifest_sha256"]==MANIFEST and receipt["FIT"] is True
         and receipt["GT_opened"] is receipt["AP"] is receipt["TEST"] is False
         and receipt["image_bytes_and_decoded_rgb_recomputed"] is receipt["identical_closed_candidate_streams"] is True
         and receipt["exact_cross_test_decoded"]==receipt["exact_cross_test_encoded"]==0
         and receipt["post_nms_rows"]==receipt["exact_author_binary_replay"]==capture["post_nms_rows"]
         and receipt["rows_at_100"]==receipt["probability_rows"]==capture["exact_top100_binary_replay"]==91500,
         "Exact train-only capture and readback counts")
    man=json.loads(pin(a.train_root/"intake_retry/TRAIN_IMAGE_ONLY_MANIFEST_PRIVATE.json",MANIFEST))
    partition=json.loads(pin(a.split_dir/"SOURCE_PARTITION_PRIVATE.json",SPLIT))
    split_result=json.loads(pin(a.split_dir/"RESULT_SAFE.json",SPLIT_RESULT))
    need(split_result["status"]=="DUAL_REFERENCE_CLEAN_GROUP_SPLIT_COMPLETE"
         and split_result["private_partition_sha256"]==SPLIT and split_result["fit_gate_group_overlap"]==0
         and split_result["independently_verified_roster_sha256"]==partition["parent_roster_sha256"]==man["source_roster_sha256"]==ROSTER
         and split_result["historical_test_loaded"] is split_result["label_or_model_used_to_assign"] is False,
         "Closed exactly grouped source partition")
    split=partition["partition"]
    groups={}
    rows={r["image_ref"]["path"]:r for r in man["items"]}
    for kind,n,ng in (("fit",732,721),("source_gate",183,180)):
        names=split[kind]
        need(len(set(names))==len(names)==n,"Exact "+kind+" roster size")
        groups[kind]={rows[k]["source_group"] for k in names}
        need(len(groups[kind])==ng and split_result["partitions"][kind]["images"]==n
             and split_result["partitions"][kind]["decoded_groups"]==ng,"Exact "+kind+" group size")
    fits,gates=set(split["fit"]),set(split["source_gate"])
    need(not fits&gates and fits|gates==set(rows)
         and not groups["fit"]&groups["source_gate"]
         and not ({rows[k]["image_ref"]["sha256"] for k in fits}&{rows[k]["image_ref"]["sha256"] for k in gates})
         and not ({rows[k]["rgb_size_prefixed_sha256"] for k in fits}&{rows[k]["rgb_size_prefixed_sha256"] for k in gates}),
         "Exact byte and decoded disjoint partitions across all 915 images")
    groups_by_side=partition["decoded_groups"]
    need(set(groups_by_side)=={"fit","source_gate"},"Closed per-partition decoded grouping")
    for kind in ("fit","source_gate"):
        bound=groups_by_side[kind]
        flat=[name for group in bound for name in group]
        need(len(bound)==len(groups[kind]) and len(flat)==len(set(flat))==len(split[kind])
             and set(flat)==set(split[kind])
             and all(len({rows[name]["source_group"] for name in group})==1 for group in bound)
             and len({rows[group[0]]["source_group"] for group in bound})==len(bound),
             "Exact independent readback grouping within each partition")
    return receipt, man["items"], fits, verify_terminal


def save(path, payload):
    with path.open("xb") as file:
        need(file.write(payload)==len(payload),"Complete output file write")


def encode(doc):
    return (json.dumps(doc,sort_keys=True,allow_nan=False)+"\n").encode("utf-8")


def stack(values, shape, dtype):
    return np.concatenate(values,axis=0) if values else np.empty(shape,dtype=dtype)


def image_points(modules, row, coco, audits, planes, rgb, counts):
    pair, rank, native, attraction=(modules[k] for k in ("pair","rank","native","attraction"))
    adapted, probability=native.make_native(pair,rank,*row["original_hw"])
    native_gray=pair.image_features(rgb)
    ranks=[]
    offsets=[0]
    action_offsets=[0]
    coords, features, layers, actions = [], [], [], []
    for i,(source,d,plane) in enumerate(zip(coco,audits,planes,strict=True),1):
        area=d["mask_area"]
        counts["all_native_candidates"]+=1
        if not 32<=area<1024:
            continue
        counts["area_eligible"]+=1
        stock=mask.decode(source["segmentation"]).astype(bool,order="C")
        value=adapted.paired_points(stock,d["box_xyxy"],area,source["score"],plane,native_gray,probability)
        if value is None:
            counts["zero_support"]+=1
            continue
        c,f,depth,discord,half=value
        n=len(c)
        need(c.shape==(n,2) and c.dtype==np.int32 and len(np.unique(c))==2*n and 1<=n<=32
             and f.shape==(n,2,14) and f.dtype==np.float32 and np.isfinite(f).all()
             and depth.shape==(n,2) and depth.dtype==np.int8,"Exact original contracts for native sampled pair features")
        dynamic,static=attraction.select(stock,c,min(8,area//10,n))
        indices=np.asarray(static,dtype=np.int16).reshape((-1,2))
        k=len(dynamic)
        need(k==len(indices) and np.all((0<=indices)&(indices<n))
             and len(np.unique(indices[:,0]))==len(np.unique(indices[:,1]))==k,
             "Exact realized static count and distinct signed positions")
        ranks.append(i)
        offsets.append(offsets[-1]+n)
        action_offsets.append(action_offsets[-1]+k)
        coords.append(c)
        features.append(f)
        layers.append(depth)
        actions.append(indices)
        counts.update(supported=1,static_positive_candidates=int(k>0),
                      paired_sites=n,static_exchanges=k,threshold_discordant=discord,threshold_half=half)
    arrays={
        "native_ranks":np.array(ranks,dtype=np.int16),
        "point_offsets":np.array(offsets,dtype=np.int32),
        "signed_coordinates":stack(coords,(0,2),np.int32),
        "features":stack(features,(0,2,14),np.float32),
        "layers":stack(layers,(0,2),np.int8),
        "static_offsets":np.array(action_offsets,dtype=np.int32),
        "static_side_indices":stack(actions,(0,2),np.int16),
    }
    out=BytesIO()
    np.savez(out,**arrays)
    return out.getvalue(),len(ranks),offsets[-1],action_offsets[-1]


def prepare(a):
    start=perf_counter()
    receipt,roster,train_names,closed=admitted(a)
    mods={key:module(a.code/path,sha) for key,(path,sha) in SOURCE.items()}
    mods["base"].cpu_runtime()
    streams=SimpleNamespace(capture=a.train_root/"capture",images=a.train_root/"intake_retry/FIT_IMAGES")
    stats={key:Counter() for key in ("fit","source_gate")}
    indexes={key:sha256() for key in stats}
    a.output.mkdir()
    for key in stats:
        (a.output/key/"POINTS_PRIVATE").mkdir(parents=True)
    with (a.output/"fit/POINT_INDEX_PRIVATE.jsonl").open("xb") as fit_index, (
            a.output/"source_gate/POINT_INDEX_PRIVATE.jsonl").open("xb") as gate_index:
        for ordinal, (row,coco,audit,planes) in enumerate(mods["replay"].blocks(streams,mods["base"],roster,receipt,True),1):
            name=row["image_ref"]["path"]
            kind,idx=("fit",fit_index) if name in train_names else ("source_gate",gate_index)
            n=stats[kind]
            rgb=mods["replay"].rgb(streams,row)
            need(digest(rgb.tobytes(order="C"))==row["rgb_sha256"],"Exact independently accepted original RGB")
            data,support,points,swaps=image_points(mods,row,coco,audit,planes,rgb,n)
            n.update(images=1,point_bytes=len(data))
            relative=f"POINTS_PRIVATE/{ordinal:04}.npz"
            save(a.output/kind/relative,data)
            meta=dict(native_archive_ordinal=ordinal,image_id=row["image_id"],source_group=row["source_group"],
                      original_hw=row["original_hw"],rgb_sha256=row["rgb_sha256"],
                      filename=relative,sha256=digest(data),bytes=len(data),supported_candidates=support,
                      paired_sites=points,realized_static_exchanges=swaps)
            raw=encode(meta)
            need(idx.write(raw)==len(raw),"Complete output points index")
            indexes[kind].update(raw)
            if ordinal%100==0 or ordinal==915:
                print(json.dumps(dict(progress_images=ordinal,fit_images=stats["fit"]["images"],
                                      source_gate_images=stats["source_gate"]["images"])),flush=True)
        for stream in (fit_index,gate_index):
            stream.flush()
            os.fsync(stream.fileno())
    need(stats["fit"]["images"]==732 and stats["source_gate"]["images"]==183
         and sum(r["all_native_candidates"] for r in stats.values())==91500
         and all(r["area_eligible"]==r["supported"]+r["zero_support"] for r in stats.values()),
         "Exact full preparation and eligibility accounting")
    data={
      "status":STATUS,"protocol_sha256":LOCK,"worker_sha256":digest(Path(__file__).read_bytes()),
      "capture_result_sha256":CAPTURE,"capture_readback_sha256":VERIFY,"capture_readback_terminal_sha256":closed,
      "image_only_manifest_sha256":MANIFEST,"group_partition_sha256":SPLIT,"group_partition_result_sha256":SPLIT_RESULT,
      "reference_source_sha256":{k:v[1] for k,v in SOURCE.items()},
      "partitions":{k:dict(counts=dict(v),point_index_sha256=indexes[k].hexdigest()) for k,v in stats.items()},
      "detector_forwards":0,"optimizer_updates":0,"train_labels_opened":False,"test_images_opened":False,
      "test_labels_opened":False,"AP":False,"elapsed_seconds":perf_counter()-start,
    }
    out=encode(data)
    save(a.output/"RESULT_SAFE.json",out)
    print(json.dumps(dict(status=STATUS,result_sha256=digest(out))),flush=True)


def main():
    p=ArgumentParser(allow_abbrev=False)
    for field in ("train-root","split-dir","code","protocol","output"):
        p.add_argument("--"+field,type=Path,required=True)
    args=p.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1",
         "Isolated CPU point preparation")
    pin(args.protocol,LOCK)
    prepare(args)


if __name__=="__main__":
    main()
