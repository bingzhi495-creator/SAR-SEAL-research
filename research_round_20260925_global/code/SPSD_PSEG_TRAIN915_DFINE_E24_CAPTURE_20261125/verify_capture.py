"""Independently read back the closed frozen D-FINE TRAIN915 capture and original JPEG streams."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
from struct import pack
import sys
from time import perf_counter
from types import ModuleType

from PIL import Image, ImageOps

BASE = "5be622cdec5c889ca8520aab5547f6f15f01a54568a0263423a816dca7f96626"
SIDECAR = "53dd5867d4353e3569552627d252877e4222bfcb2e6c7f60ca5bb53f3eabc940"
EMA = "82ca80e2cb9a0b9673a136fad3d7de3097b7327f4f76e3a2d3e7184ab07e7fd8"
ACCEPT = "7401f031c996298caff1761c6ef691fae0c301e65c87c15f51d186dcf81f25d8"
WORKER = "0f10812a7d2409f69334670e7dae16a9ef51b4f84a20ee1278a4187d06e4c9a0"
UPSTREAM = "11df18cb1eb0dbcff66b576d0c082b45896e5e254adb66b98fdfaff366d7391a"
LOCK = "0565109236be7fd538bca5e35259fbf109a132fcd8fea65208f4c6fdb2350c56"
MANIFEST = "701c89d7350f6a078337d6074d69960799ade868a4f64e546897e07137c92bf4"
INVENTORY = "2a7aaf3013c77ddd59804a446140153234a4279807ef92f1b3eb3c014534ed58"
INTAKE = "190d206a1b4d3690323d7fbc4db6c0cd6398a1f53f993ab0c3819c3b0396ffde"
ADMISSION = "e40636ec2cfb4c42718673a553f069f5f4e2833e3e81264c9d44a43ece7b9927"
STATUS = "DFINE_HRSID_E24_PSEG_TRAIN915_PROBABILITY_INDEPENDENTLY_ACCEPTED_V1"


def require(ok,msg):
    if not ok:
        raise ValueError(msg)


def pin(path, expected, limit=1024**2):
    require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= limit, "Bounded " + path.name)
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == expected, "Exact closed digest " + path.name)
    return raw


def module(path, expected):
    scope=ModuleType("_readback_" + path.stem)
    scope.__file__=str(path)
    exec(compile(pin(path,expected),str(path),"exec"),scope.__dict__)
    return scope


def line(stream,digest,limit):
    raw=stream.readline(limit)
    require(raw.endswith(b"\n"), "Complete bounded native row")
    digest.update(raw)
    return json.loads(raw)


def main():
    p=ArgumentParser(allow_abbrev=False)
    for name in ("export","intake","capture-terminal","protocol","base","sidecar","output"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    require(sys.flags.isolated and sys.dont_write_bytecode and os.environ["CUDA_VISIBLE_DEVICES"] == "-1",
            "Exact isolated CPU-only readback")
    before=perf_counter()
    pin(a.protocol,LOCK)
    terminal_raw=a.capture_terminal.read_bytes()
    terminal=json.loads(terminal_raw)
    require(terminal["phase"]=="capture" and terminal["actual_exit"]==0
            and terminal["output_footer_verified"] is True and terminal["worker_sha256"]==WORKER,
            "Original capture tool has naturally exited with verified footer")
    result=json.loads(pin(a.export/"RESULT_SAFE.json",terminal["result_sha256"]))
    require(result["status"]=="DFINE_HRSID_E24_PSEG_TRAIN915_PROBABILITY_EXPORT_COMPLETE_V1"
            and result["export_source_sha256"]==WORKER and result["upstream_worker_sha256"]==UPSTREAM
            and result["protocol_sha256"]==LOCK and result["manifest_sha256"]==MANIFEST
            and result["image_inventory_sha256"]==INVENTORY and result["ema_sha256"]==EMA and result["e24_acceptance_sha256"]==ACCEPT
            and result["author_commit"]=="b782ece206487792f36c165ca65548154cd12094"
            and result["images"]==result["model_forwards"]==result["probability_files"]==result["native_dimensions_verified"]==915
            and result["raw_queries"]==result["internal_top_rows"]==915*300
            and result["exact_author_binary_replay"]==result["post_nms_rows"]
            and result["exact_top100_binary_replay"]==result["rows_at_100"]==result["probability_rows"]
            and result["original_author_postprocess_parity"] is True
            and result["confidence_threshold"]==0. and result["box_nms_iou"]==.70 and result["mask_threshold"]==.50
            and result["keep_ratio"] is result["rect"] is False
            and result["model_precision"]=="FP32_TF32_disabled" and result["mask_resize_precision"]=="author_CUDA_FP16"
            and result["FIT"] is True and result["DEV"] is result["TEST"] is result["GT_opened"] is result["AP"] is False
            and result["optimizer_updates"]==0,
            "Exactly closed train-only capture, counts, model, and GPU author parity")
    preparer=json.loads(pin(a.intake/"RESULT_SAFE.json",INTAKE))
    intake_terminal=json.loads((a.intake.parent/"observed_intake_retry/TERMINAL.json").read_bytes())
    require(intake_terminal["actual_exit"]==0 and intake_terminal["output_footer_verified"] is True
            and intake_terminal["result_sha256"]==INTAKE and preparer["image_manifest_sha256"]==MANIFEST
            and preparer["admission_sha256"]==ADMISSION and preparer["images"]==915
            and preparer["source_groups"]==901 and preparer["exact_cross_test_encoded"]==preparer["exact_cross_test_decoded"]==0,
            "Closed admitted TRAIN image-only bytes")
    roster=json.loads(pin(a.intake/"TRAIN_IMAGE_ONLY_MANIFEST_PRIVATE.json",MANIFEST))
    rows=roster["items"]
    require(roster["FIT"] is True and roster["DEV"] is roster["TEST"] is roster["GT_opened"] is roster["AP"] is False
            and len(rows)==len(set(roster["expected_ids"]))==len({i["source_group"] for i in rows})+14==915
            and [row["image_id"] for row in rows]==roster["expected_ids"],
            "Exact source-only manifest and grouping")
    images=a.intake/"FIT_IMAGES"
    names=[row["image_ref"]["path"] for row in rows]
    require(names==sorted(set(names)) and {p.name for p in images.iterdir()}==set(names),"All original admitted train images")
    base, sidecar=module(a.base,BASE),module(a.sidecar,SIDECAR)
    base.cpu_runtime()
    ph,ah,ih,inventory=sha256(),sha256(),sha256(),sha256()
    counters=Counter()
    planes=set()
    rgb_groups={}
    with (a.export/base.PREDICTIONS).open("rb") as pstream, (a.export/base.AUDIT).open("rb") as astream, (
            a.export/"PROBABILITIES_PRIVATE.jsonl").open("rb") as istream:
        for ordinal,row in enumerate(rows,1):
            identity=row["image_id"]
            ref=row["image_ref"]
            name=ref["path"]
            h,w=row["original_hw"]
            require(len(name)==10 and name.endswith(".jpg") and name[:6].isdigit() and int(name[:6])%10 not in (1,9),
                    "Only original train split member")
            jpg=pin(images/name,ref["sha256"],4*1024**2)
            require(len(jpg)==ref["bytes"],"Exact staged JPEG length")
            inventory.update(f"{name}\t{len(jpg)}\t{ref['sha256']}\n".encode("ascii"))
            with Image.open(BytesIO(jpg)) as image:
                require(image.format=="JPEG" and image.mode=="RGB","Original color JPEG")
                image=ImageOps.exif_transpose(image).convert("RGB")
                require(image.size==(w,h),"Original source native dimensions")
                raw=image.tobytes()
            require(sha256(raw).hexdigest()==row["rgb_sha256"]
                    and sha256(pack(">II",w,h)+raw).hexdigest()==row["rgb_size_prefixed_sha256"],
                    "Exact independent decoded RGB identity")
            group=row["source_group"]
            require(rgb_groups.setdefault(row["rgb_size_prefixed_sha256"],group)==group,"Exact decoded duplication grouping")
            pred,audit,index=line(pstream,ph,4*1024**2),line(astream,ah,1024**2),line(istream,ih,2**15)
            require(index["image_id"]==identity and set(index)=={"image_id","query_ids","probability"},
                    "Native source image and candidate-index alignment")
            n,empty=base.validate_rows(identity,(h,w),pred,audit)
            query=index["query_ids"]
            require(len(query)==len(set(query))==n and all(type(q) is int and 0<=q<300 for q in query),
                    "All post-NMS query IDs retain exact order, range and uniqueness")
            k=min(100,n)
            reference=index["probability"]
            arr=sidecar.verify_probability_file(a.export,ordinal,reference,k)
            require(len(arr)==k,"Exact sidecar shape")
            planes.add(Path(reference["file"]).name)
            counters.update(post_nms_rows=n,rows_at_100=k,empty_post_nms=empty,
                            empty_at_100=sum(d["mask_area"]==0 for d in audit["candidates"][:k]),
                            zero_prediction_images=int(n==0),probability_bytes=reference["bytes"],
                            non_square_native_images=int(h!=w),post_nms_below100_images=int(n<100))
            if ordinal%150==0 or ordinal==915:
                print(json.dumps(dict(verified_images=ordinal,sidecar_rows=counters["rows_at_100"])),flush=True)
        require(not pstream.read(1) and not astream.read(1) and not istream.read(1),"No trailing extra capture rows")
    require({p.name for p in (a.export/"PROBABILITY_PRIVATE").iterdir()}==planes and len(planes)==915
            and len(rgb_groups)==len(set(rgb_groups.values()))==901 and inventory.hexdigest()==INVENTORY
            and ph.hexdigest()==result["predictions_sha256"] and ah.hexdigest()==result["candidate_audit_sha256"]
            and ih.hexdigest()==result["probability_index_sha256"]
            and all(counters[k]==result[k] for k in ("post_nms_rows","rows_at_100","empty_post_nms",
                  "empty_at_100","zero_prediction_images","probability_bytes")),
            "Independent exact whole stream and metadata parity")
    receipt=dict(status=STATUS,FIT=True,DEV=False,TEST=False,GT_opened=False,AP=False,
            original_os_exit=0,original_os_wait=True,job_drained=True,images=915,source_groups=901,
            model_forwards=0,optimizer_updates=0,protocol_sha256=LOCK,worker_sha256=WORKER,
            readback_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
            manifest_sha256=MANIFEST,image_inventory_sha256=INVENTORY,ema_sha256=EMA,e24_acceptance_sha256=ACCEPT,
            export_result_sha256=terminal["result_sha256"],export_terminal_sha256=sha256(terminal_raw).hexdigest(),
            observed_capture_worker_pid=terminal["worker_pid"],
            predictions_sha256=ph.hexdigest(),candidate_audit_sha256=ah.hexdigest(),probability_index_sha256=ih.hexdigest(),
            probability_files=915,probability_rows=counters["rows_at_100"],
            exact_author_binary_replay=result["exact_author_binary_replay"],
            exact_top100_binary_replay=result["exact_top100_binary_replay"],original_author_postprocess_parity=True,
            identical_closed_candidate_streams=True, image_bytes_and_decoded_rgb_recomputed=True,
            exact_cross_test_encoded=0,exact_cross_test_decoded=0,elapsed_seconds=perf_counter()-before,**counters)
    raw=(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+"\n").encode("utf-8")
    with a.output.open("xb") as file:
        file.write(raw)
        file.flush()
        os.fsync(file.fileno())
    print(json.dumps(dict(status=STATUS,result_sha256=sha256(raw).hexdigest())),flush=True)


if __name__=="__main__":
    main()
