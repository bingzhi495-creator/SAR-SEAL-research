"""Capture the accepted 5203 RF-DETR EMA on an original Instance-RSDD image roster."""
import argparse
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace


LOCK = "f7ac19ff63c2c84fb5e2aac772f5efa0bf16a8e8ac519b0a2469d74b6bfd05cf"
CKPT = "36205b47714057ee24a3916f2bc0d307779ad4aa60e5b39b949bbdf381f31054"
ACCEPTANCE = "508d101d54fe658c1940b959e1ce9cd518772f22c9230269d61e7d0958b100c6"
LOADER = "02c79adf3810fcf8f74076594a9c6f686580cb6ced1981f0a3e3490fb271d61d"
PORTABLE = "45512c959eb914ae7713264ecff48eb437a2a52729810b34308abc52079148df"
STATUS = "RSDD_CARRIER5203_CAPTURE_COMPLETE"
NATIVE_SAFE = "NATIVE_SAFE.json"
READBACK = "READBACK_SAFE.json"
RESULT = "RESULT_SAFE.json"
VERSIONS = {"torch":"2.10.0+cu130","rfdetr":"1.12.0","numpy":"1.26.4","Pillow":"12.3.0","pycocotools":"2.0.11"}
SETS = {
    74: dict(intake="conversion",observer="observed_conversion",partition="author train",parents=13,
             sensors={"GF-3":42,"TerraSAR-X":32},phase="prepare-edits",
             status="INSTANCE_RSDD_RESERVE74_CLEAN_CONVERSION_COMPLETE_V1",
             term="5b421508e82d20b05b67ee4cad26c0a59dded454231ae828529fa6dee924d282",
             result="8af3fed7614adf4ec9c61b3c940551721ae88a627fde71c0f8050f22d3eb74fd",
             manifest="154d0d748b96df0d56ac249adcb8b6ae63275a09dfd7a0d7480839005e044338",
             roster="2614e567e9ef1b2c791c1b357b03adfe7302ece53e911c1963c40e623ad405fc"),
    740: dict(intake="intake",observer="observed_intake",partition="author test",parents=112,
              sensors={"GF-3":583,"TerraSAR-X":157},phase="capture",status="RSDD740_CLEAN_IMAGE_INPUT_COMPLETE",
              term="87409d20f6aac091f9fe4b7bcb7d1b5ad4710606a380b1eff2c45a271b36839c",
              result="dc98ec423d3066305ee5b27893abf701f68c93fd4fc47633018da77f7f483b86",
              manifest="16a546e96d6e76d3afbfbe4f380b182505494818b35f1c8bcc960e6d4d1b2184",
              roster="72c4b9af8f8b0bd0df31a3a3f7df6a082390688f0942eed23f106316a2d5d98b")}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def pinned(path, expected):
    raw = path.read_bytes()
    need(digest(raw) == expected, "Unexpected bytes for "+path.name)
    return raw


def module(path, expected):
    loaded = ModuleType(path.stem)
    loaded.__file__ = str(path)
    exec(compile(pinned(path, expected), str(path), "exec"), loaded.__dict__)
    return loaded


def write(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()
    need(len(raw)<500000, "Aggregate receipt exceeds the declared limit")
    with path.open("xb") as stream:
        need(stream.write(raw)==len(raw), "Short receipt write")
        stream.flush()
        os.fsync(stream.fileno())
    return digest(raw)


def load_population(code, stock_root, cohort):
    """Recover only the observed image manifest and prior author image-only roster."""
    cfg = SETS[int(cohort)]
    shared = module(code/"INSTANCE_RSDD_B5_REPRO_20261123/portable_eval.py", PORTABLE)
    native = module(code/shared.NATIVE/"capture_instance_rsdd.py", shared.CAPTURE_SOURCE)
    intake = stock_root/cfg["intake"]
    term = json.loads(pinned(stock_root/cfg["observer"]/"TERMINAL.json",cfg["term"]))
    receipt = json.loads(pinned(intake/RESULT,cfg["result"]))
    need(type(term["actual_exit"]) is int and term["actual_exit"]==0 and term["output_footer_verified"] is True
         and term["phase"]==cfg["phase"] and term["result_sha256"]==cfg["result"] and receipt["status"]==cfg["status"]
         and receipt["image_manifest_sha256"]==cfg["manifest"] and receipt["images"]==receipt["exact_crc_and_rgb"]
         ==receipt["rgb_unique"]==int(cohort) and receipt["sensor_counts"]==cfg["sensors"], "Original image intake not accepted")
    manifest = json.loads(pinned(intake/shared.MANIFEST,cfg["manifest"]))
    if int(cohort)==74:
        need(receipt["private_annotations_opened"]==0 and receipt["roster_sha256"]==cfg["roster"]==shared.ROSTER,
             "Reserve image-only roster differs")
        items,manifest_sha = shared.load_intake(SimpleNamespace(roster=stock_root/"reserve.csv",intake=intake),native)
        need(manifest_sha==cfg["manifest"], "Image manifest changed during verification")
    else:
        need(receipt["target_polygon_rows_opened"]==0 and receipt["test_roster_sha256"]==cfg["roster"]
             ==native.AUTHOR_ROSTER_SHA, "Test image-only roster differs")
        author = json.loads(pinned(intake/"test_image_roster.json",cfg["roster"]))
        items = native.roster(author,manifest,size=740)
    need(len(items)==len({i["rgb_sha256"] for i in items})==int(cohort)
         and len({i["acquisition_group"] for i in items})==cfg["parents"]
         and Counter(i["sensor"] for i in items)==cfg["sensors"], "Image, sensor or parent census differs")
    lineage = dict(manifest_sha256=cfg["manifest"],roster_sha256=cfg["roster"],
                   intake_result_sha256=cfg["result"],intake_terminal_sha256=cfg["term"])
    return shared,native,intake,items,lineage


def source_model(code, shared, native, checkpoint, acceptance):
    """Use the qualified seed-5203 training marker/EMA loader and original reserve val576 capture."""
    loader = module(code/"RFDETR_COMMON5203_DEV705_FROZEN_20261121/capture_dev.py",LOADER)
    arm = loader.endpoint(json.loads(pinned(acceptance, ACCEPTANCE)))
    need(arm["checkpoint_sha256"]==CKPT and checkpoint.name=="checkpoint_23.ckpt" and checkpoint.stat().st_size==546526388,
         "Accepted complete center 5203 E24 checkpoint required")
    carrier = module(code/"RFDETR_CENTER5202_FULL_CAPTURE_20261120/capture.py",loader.FULL)
    core = loader.load_core(code,carrier)
    evaluator = loader.bind_loader(core,code,arm)
    post = core.load("post")
    # Only the fresh in-memory bindings change. The native writer, EMA model construction,
    # native COCO transform, author joint top-100 and inherited admission checks run unchanged.
    carrier.CKPT = native.SOURCE_CHECKPOINT = arm["checkpoint_sha256"]
    core.CHECKPOINTS = {"center":checkpoint}
    core.TRAIN_PROTOCOL = arm["phase"]["protocol"]
    carrier.load_core = lambda _: core
    carrier.setup = lambda args, admitted: (evaluator,post)
    common = native.reference(code/shared.UEIKAP/"capture.py")
    common.exact_source = lambda _: carrier
    return common


def native_capture(a, shared, native, intake, items):
    import torch
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="0","Select the original single CUDA device")
    native.install_guard(intake,items)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    from rfdetr.models.postprocess import PostProcess  # installs original B5 RF-DETR precision policy
    need(torch.get_float32_matmul_precision()=="high" and torch.backends.cuda.matmul.allow_tf32,
         "Original Instance-RSDD matmul policy required")
    common = source_model(a.code,shared,native,a.checkpoint,a.training_receipt)
    result = native.capture(SimpleNamespace(private_root=a.code,intake=intake,output=a.output),items,torch,common)
    need(result["model_forward_calls"]==len(items) and not torch.backends.cudnn.allow_tf32
         and torch.backends.cuda.matmul.allow_tf32,"Unexpected native forward or CUDA precision")
    return result


def records(path, expected):
    checksum = sha256()
    with path.open("rb") as stream:
        for raw in stream:
            checksum.update(raw)
            yield json.loads(raw)
    need(checksum.hexdigest()==expected,"Completed native stream digest differs: "+path.name)


def read_capture(capture, terminal, items, lineage):
    """Bind an observed closed capture. Consume the returned (audit, prediction) iterator fully."""
    terminal_raw=terminal.read_bytes()
    measured=json.loads(terminal_raw)
    final=pinned(capture/RESULT,measured["result_sha256"])
    safe=json.loads(final)
    cfg=SETS[len(items)]
    worker=digest(Path(__file__).read_bytes())
    need(type(measured["actual_exit"]) is int and measured["actual_exit"]==0 and measured["output_footer_verified"] is True
         and measured["phase"]==safe["phase"]=="capture" and measured["worker_sha256"]==safe["worker_sha256"]==worker
         and safe["status"]==STATUS and safe["cohort"]==safe["images"]==safe["model_forward_calls"]==len(items)
         and safe["source_partition"]==cfg["partition"] and safe["manifest_sha256"]==cfg["manifest"]
         and safe["checkpoint_sha256"]==CKPT and safe["training_receipt_sha256"]==ACCEPTANCE
         and safe["protocol_sha256"]==LOCK and safe["target_labels_opened"] is False and safe["optimizer_updates"]==0
         and safe["independent_cpu_readback"] is True and type(safe["independent_readback_exit"]) is int
         and safe["independent_readback_exit"]==0 and all(safe[k]==v for k,v in lineage.items()),
         "Observed closed capture and independent CPU readback required")
    native=json.loads(pinned(capture/NATIVE_SAFE,safe["native_result_sha256"]))
    check=json.loads(pinned(capture/READBACK,safe["readback_result_sha256"]))
    keys=("predictions_sha256","audit_sha256","counts","worker_sha256","protocol_sha256","checkpoint_sha256",*lineage)
    need(check["status"]=="RSDD_CARRIER5203_CPU_READBACK_COMPLETE" and check["exact_native_parity"] is True
         and check["model_forward_calls"]==0 and check["native_result_sha256"]==safe["native_result_sha256"]
         and all(native[k]==check[k]==safe[k] for k in keys),"Plane verification lineage differs")
    def pairs():
        pred=records(capture/"PREDICTIONS_PRIVATE.jsonl",safe["predictions_sha256"])
        audit=records(capture/"CANDIDATE_AUDIT_PRIVATE.jsonl",safe["audit_sha256"])
        for item,row,p in zip(items,audit,pred,strict=True):
            need(item["image_id"]==row["image_id"]==p["image_id"] and item["rgb_sha256"]==row["rgb_sha256"]
                 and item["image_ref"]["sha256"]==row["image_sha256"],"Native capture row order or identity differs")
            yield row,p
    return safe,pairs(),dict(capture_result_sha256=digest(final),capture_terminal_sha256=digest(terminal_raw),
                             readback_result_sha256=safe["readback_result_sha256"])


def validate_rows(native,common,items,intake,cap,result,torch):
    """Check native identities and detached posterior signs in an isolated CPU process."""
    import numpy as np
    from pycocotools import mask as codec
    preds = records(cap/native.PRED,result["predictions_sha256"])
    audit = records(cap/native.AUDIT,result["audit_sha256"])
    counts = dict(mapped=0,eligible=0,overrides=0,rank_gaps=0)
    for ordinal,(item,row,pred) in enumerate(zip(items,audit,preds,strict=True),1):
        native.decode_native(intake,item)
        hw = item["original_hw"]
        ident = item["image_id"]
        need(ident==row["image_id"]==pred["image_id"] and row["image_sha256"]==item["image_ref"]["sha256"]
             and all(row[k]==item[k] for k in ("source_image_name","sensor","acquisition_group","rgb_sha256","original_hw")),
             "Independent source image join failed")
        probs = np.asarray(row["raw_query_probabilities"],dtype=np.float32)
        boxes = np.asarray(row["raw_boxes_cxcywh"],dtype=np.float32)
        need(probs.shape==(164,2) and np.all((0<=probs)&(probs<=1)) and boxes.shape==(164,4)
             and np.isfinite(boxes).all(),"Invalid full detector output layout")
        slots = np.argsort(-probs.ravel(),kind="stable")[:100]
        candidates = [(r,int(q)//2) for r,q in enumerate(slots,1) if q%2==0]
        ranks,ids = [r for r,_ in candidates],[q for _,q in candidates]
        xyxy = native.native_xyxy(boxes,hw)
        need(pred["native_ranks"]==row["native_ranks"]==ranks and pred["raw_query_count"]==164
             and pred["native_count"]==100 and pred["unmapped_slot1_count"]==100-len(ranks)
             and row["query_ids"]==ids and row["native_boxes_xyxy"]==xyxy[ids].tolist(),"Independent Q164 ranks or boxes differ")
        pool = []
        for pos,((rank,q),co,cand) in enumerate(zip(candidates,pred["coco"],row["candidates"],strict=True),1):
            mask = co["segmentation"]
            area = int(codec.area(mask))
            need(co["image_id"]==ident and co["category_id"]==1 and mask["size"]==hw and co["score"]==float(probs[q,0])
                 and cand==dict(position=pos,native_rank=rank,query_id=q,mask_area=area,score=co["score"],
                                box_xyxy=xyxy[q].tolist()),"Native RLE area/score/candidate identity differs")
            if 32<=area<1024:
                pool.append((rank,q,mask))
        need(row["logit_ranks"]==[r for r,_,_ in pool] and row["logit_query_ids"]==[q for _,q,_ in pool]
             and row["logits_ref"]["path"]==f"planes/{ordinal:06d}.npy","Detached native logits order/path differs")
        planes = common.read_array(cap,row["logits_ref"])
        need(planes.shape==(len(pool),144,144),"Native posterior plane shape differs")
        overrides = {v["native_rank"]:v for v in row["posterior_overrides"]}
        need(len(overrides)==len(row["posterior_overrides"]),"Duplicated CUDA posterior fallback")
        for i,(rank,q,rle) in enumerate(pool):
            stock = codec.decode(rle).astype(bool)
            recovered = native.rasterize(torch.from_numpy(planes[i:i+1]),hw,torch)[0].numpy()>0
            if np.array_equal(stock,recovered):
                need(rank not in overrides,"Unnecessary native posterior fallback")
                continue
            ref = overrides.pop(rank)
            need(ref["query_id"]==q and ref["path"]==f"planes/{ordinal:06d}_r{rank:03d}_p.npy","CUDA override ID differs")
            p = common.read_array(cap,ref)
            need(p.shape==stock.shape and np.all((0<=p)&(p<=1)) and not np.any((p>.5)&~stock)
                 and not np.any((p<.5)&stock),"Native CUDA fallback sign differs")
        need(not overrides,"Unconsumed CUDA posterior fallback")
        native.count_rows(counts,candidates,pool,row["posterior_overrides"])
    need(counts==result["counts"] and counts["mapped"]+counts["rank_gaps"]==len(items)*100,
         "Independent candidate or signed posterior count differs")
    return counts


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("--cohort",type=int,required=True,choices=tuple(SETS))
    for key in ("code","stock-root","checkpoint","training-receipt","protocol","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--protocol-sha256",required=True)
    parser.add_argument("--readback-native-sha256",help=argparse.SUPPRESS)
    a=parser.parse_args()
    for key,value in vars(a).copy().items():
        if isinstance(value,Path):
            setattr(a,key,value.resolve(strict=key!="output"))
    need(a.protocol_sha256==LOCK and digest(a.protocol.read_bytes())==LOCK,"Exact frozen study protocol required")
    need(sys.version_info[:3]==(3,10,13) and all(version(k)==v for k,v in VERSIONS.items()),"Original B5 runtime required")
    start=perf_counter()
    shared,native,intake,items,lineage=load_population(a.code,a.stock_root,a.cohort)
    self_sha=digest(Path(__file__).read_bytes())
    config=SETS[a.cohort]
    binding=dict(cohort=a.cohort,source_partition=config["partition"],images=len(items),parents=config["parents"],
                 sensors=config["sensors"],protocol_sha256=LOCK,worker_sha256=self_sha,checkpoint_sha256=CKPT,
                 training_receipt_sha256=ACCEPTANCE,detector_seed=5203,source_detector_optimizer_updates=17640,
                 source_detector_ema_updates=17640,optimizer_updates=0,target_labels_opened=False,GT_opened=False,
                 **lineage)
    if a.readback_native_sha256:
        import torch
        need(os.environ.get("CUDA_VISIBLE_DEVICES")=="-1" and not torch.cuda.is_initialized(),"Independent CPU required")
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        result=json.loads(pinned(a.output/NATIVE_SAFE,a.readback_native_sha256))
        need(all(result[k]==v for k,v in binding.items()) and result["model_forward_calls"]==len(items)
             and result["status"]=="RSDD_CARRIER5203_NATIVE_WRITTEN","Observed native input differs")
        native.install_guard(intake,items)
        common=native.reference(a.code/shared.UEIKAP/"capture.py")
        counts=validate_rows(native,common,items,intake,a.output,result,torch)
        need(not torch.cuda.is_initialized(),"CPU readback initialized CUDA")
        out=dict(binding,status="RSDD_CARRIER5203_CPU_READBACK_COMPLETE",counts=counts,model_forward_calls=0,
                 native_result_sha256=a.readback_native_sha256,predictions_sha256=result["predictions_sha256"],
                 audit_sha256=result["audit_sha256"],exact_native_parity=True,elapsed_seconds=perf_counter()-start)
        footer=write(a.output/READBACK,out)
        print(json.dumps(dict(status=out["status"],result_sha256=footer)),flush=True)
        return
    need(not a.output.exists() and a.output.parent.is_dir(),"Expected a fresh capture directory")
    result=native_capture(a,shared,native,intake,items)
    elapsed=perf_counter()-start
    result.update(binding,status="RSDD_CARRIER5203_NATIVE_WRITTEN",native_capture_seconds=elapsed)
    native_hash=write(a.output/NATIVE_SAFE,result)
    command=[sys.executable,"-I","-B","-X","utf8",str(Path(__file__).resolve()),*sys.argv[1:],
             "--readback-native-sha256",native_hash]
    child=subprocess.run(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES="-1"),stdout=subprocess.PIPE,check=True)
    checked=(a.output/READBACK).read_bytes()
    check=json.loads(checked)
    footer=json.loads(child.stdout.splitlines()[-1])
    need(footer==dict(status="RSDD_CARRIER5203_CPU_READBACK_COMPLETE",result_sha256=digest(checked))
         and check["native_result_sha256"]==native_hash and check["exact_native_parity"] is True
         and check["counts"]==result["counts"] and check["model_forward_calls"]==0
         and all(check[k]==result[k] for k in (*binding,"predictions_sha256","audit_sha256")),
         "Successful independent CPU readback and footer required")
    result.update(status=STATUS,phase="capture",native_result_sha256=native_hash,independent_cpu_readback=True,
                  independent_readback_exit=child.returncode,readback_result_sha256=digest(checked),
                  independent_readback_seconds=check["elapsed_seconds"],elapsed_seconds=perf_counter()-start,
                  loader_sha256=LOADER,native_writer_sha256=shared.CAPTURE_SOURCE,portable_source_sha256=PORTABLE,
                  operator_versions=VERSIONS)
    footer=write(a.output/RESULT,result)
    print(json.dumps(dict(status=STATUS,result_sha256=footer)),flush=True)


if __name__=="__main__":
    main()
