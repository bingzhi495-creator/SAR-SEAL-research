"""Frozen RF-DETR center5202 E24 FIT109 cache extension and DEV705 capture.

This worker is launched only by an independently qualified bounded owner.
The --check command validates closed source and admission metadata without
loading a model, pixels, logits or ground truth.
"""
from argparse import ArgumentParser
from contextlib import nullcontext
from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType
from zipfile import ZipFile

MIB = 1024**2
CKPT = "011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11"
PAIR = "ec6c36973a763344d0f063727e7e07c697a132573e6235eaa8525482e8d8da9f"
COHORT = "70e9a0e8fb30f5ba70b48b9067c1b95d07bfec23d6b248d92c5983727acceb6e"
CACHE_MANIFEST = "af43530e7f8f118261b5523cdde738cca467ead813d38f1d4c989c8f4102e184"
CACHE_ROWS = "2ed2f67671705a8b5b502a3661190c6c16b354c8a4d940d28a8c67943786f9df"
CACHE_OWNER = "e85ea39f0ff1e808a5611bebc2d915aa34c9ef3b799260c28e89f04629e81b67"
CACHE_READBACK = "47bed472492920b3f8950c60ea77fcf3b5904665f8bcf6a374bfc5bc1274289f"
DEV_MANIFEST = "39a40d089d4a54448fe6efca8ea5cc98ad1eb7c0587815f3dd9a3b097173e78f"
DEV_STOCK = "da55f2d3f9bbf08b26d5fae76606b2496a5b482c5fb61f4fab53c23172e5c06e"
DEV_REPORT = "05c5c11274f7800504b5ab1fb1dcb8db39c60316f51dc732e3f20ee063078c19"
DEV_PAIR = "d080a27ade36369a8fc8d2cb003c37cc3670d75860e83b30e365bd47e77543c0"
DEV_CLOSURE = "HRSID_MASK_COMMON_FIT_E24_5202_PRIVATE"
CORE_SOURCE = "91d02064170b8d2ca8c6cda1578bd95d8b8a18e4c4c103312b6ef199636e342a"


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def exact(path, digest, limit=4*MIB):
    need(path.is_absolute() and path.resolve(strict=True) == path and path.is_file()
         and 0 < path.stat().st_size <= limit, "Canonical bounded closed input")
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    need((before.st_size,before.st_mtime_ns) == (after.st_size,after.st_mtime_ns)
         and sha256(data).hexdigest() == digest, "Exact stable closed input " + path.name)
    return data


def document(path, digest, limit=4*MIB):
    return json.loads(exact(path,digest,limit))


def encoded(value):
    return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def seal(path, raw):
    with path.open("xb") as stream:
        need(stream.write(raw) == len(raw), "Complete owned artifact")
        stream.flush()
        os.fsync(stream.fileno())
    return sha256(raw).hexdigest()


def load_core(root):
    os.environ["SAR_SEAL_CLOSED_FIT_ROOT"] = str(root)
    path = root/"github_release_20261011/research_round_20260925_global/code/HRSID_FIT114_CORE36_CENTER5202_RAW_CAPTURE_20261112/capture.py"
    scope = ModuleType("_exact_core36_reuse")
    scope.__file__ = str(path)
    exec(compile(exact(path,CORE_SOURCE),str(path),"exec"),scope.__dict__)
    return scope


def native_block(image_id, query):
    ids, ranks = query["native_ids"], query["native_ranks"]
    need(len(ids) == len(ranks) and len(set(ids)) == len(ids)
         and ranks == sorted(set(ranks)) and all(1 <= r <= 100 for r in ranks),
         "Original class-zero joint native rank gaps")
    rows = [dict(image_id=image_id, category_id=1, score=query["ship_scores"][q],
                 segmentation=query["all_rles"][q]) for q in ids]
    return dict(image_id=image_id,raw_query_count=164,native_count=100,
                unmapped_slot1_count=100-len(ids),native_ranks=ranks,coco=rows)


def eligible_candidates(query, codec):
    area = codec.area(query["all_rles"])
    return [(q,r,int(area[q])) for q,r in zip(query["native_ids"],query["native_ranks"],strict=True)
            if 32 <= area[q] < 1024]


def setup(args, core):
    admission = core.load("admission")
    phase = admission.load_phase(core.source("phase")[0])
    accepted = admission.read_pair(phase,core.RECEIPTS/"PAIR_READY_PRIVATE.json",PAIR,
        native_path=core.RECEIPTS/"NATIVE_ARM_ACCEPTED_PRIVATE.json",
        center_path=core.RECEIPTS/"CENTER_ARM_ACCEPTED_PRIVATE.json",
        mode="center",training_protocol=core.TRAIN_PROTOCOL)
    need(accepted["checkpoint_sha256"] == CKPT, "Exact independently accepted center E24")
    filtered, base = phase.load_evaluator(core.source("filtered")[0],core.source("base")[0])
    admission.bind_evaluator(phase,base,filtered,accepted)
    base.FIT_AUDIT = core.source("fit_audit")[0]
    base.RECIPE = core.CODE.parent/"configs/RFDETR_SEG_NANO_BASELINE_576_E100_20261107.yaml"
    return base, core.load("post")


def metadata(args, core):
    if args.mode == "fit":
        raw = document(args.root/"HRSID_FIT_NATIVE_QUERY_RETENTION_ACTUAL_PRIVATE_20261109_01/COHORT_PRIVATE.json",COHORT)
        rows = sorted((r for r in raw["images"] if r["core"]),key=lambda r:r["decoded_rgb_sha256"])
        need(raw["fit_only"] is True and len(rows) == len({r["group"] for r in rows})
             == len({r["decoded_rgb_sha256"] for r in rows}) == 109, "Original label-free FIT109")
        old = args.root/"HRSID_FIT114_CORE36_CENTER5202_RAW_CAPTURE_DRAFT_PRIVATE_20261112_01"
        cache = document(old/"IMAGE_ONLY_MANIFEST_PRIVATE.json",CACHE_MANIFEST)
        need(cache["images"] == [{k:r[k] for k in core.ROSTER_FIELDS} for r in rows[:36]],
             "Same deterministic CORE36 subset and image fingerprints")
        report = document(old/"INDEPENDENT_READBACK_SAFE.json",CACHE_READBACK)
        need(report["raw_logit_raster_RLE_exact"] == report["raw_logit_rows"] == 2130
             and report["original_prediction_sha256"] == CACHE_ROWS
             and report["original_owner_receipt_sha256"] == CACHE_OWNER, "Exact accepted CPU cache")
        return rows
    root = args.root/"HRSID_OFFICIAL_PNG_DEV705_AUDIT_ACTUAL_PRIVATE_20261109_01"
    rows = document(root/"DEV_IMAGE_ONLY_MANIFEST_PRIVATE.json",DEV_MANIFEST)
    previous = args.root/"HRSID_MASK_COMMON_FIT_E24_DEV705_5202_PRIVATE_20261110_01"
    pair = document(previous/"CLOSED_EXPORT_SCORE_PAIR_PRIVATE.json",DEV_PAIR)
    receipt = pair["arms"]["center"]
    report = document(previous/"EXPORT_CENTER_ACTUAL_PRIVATE/DEV_NATIVE_EXPORT_SAFE.json",DEV_REPORT)
    need(len(rows) == len({r["image_id"] for r in rows}) == report["images"] == 705
         and receipt["original_os_wait"] is True and receipt["original_os_exit"] == 0
         and receipt["resource_stop"] is False and receipt["predictions"]["sha256"] == DEV_STOCK
         and report["checkpoint_sha256"] == CKPT and report["image_manifest_sha256"] == DEV_MANIFEST
         and report["predictions_sha256"] == DEV_STOCK
         and pair["training_pair"]["sha256"] == PAIR, "Exact accepted original GPU stock lineage")
    return rows


def read_cache(args):
    root = args.root/"HRSID_FIT114_CORE36_CENTER5202_RAW_CAPTURE_ACTUAL_PRIVATE_20261112_01"
    document(root/"PAIR_SAFE.json",CACHE_OWNER)
    capture = root/"center/capture"
    rows = [json.loads(line) for line in exact(capture/"QUERIES_PRIVATE.jsonl",CACHE_ROWS,64*MIB).splitlines()]
    need(len(rows) == len({r["image_id"] for r in rows}) == 36, "Exact original cache census")
    return capture,{r["image_id"]:r for r in rows}


def model_and_transform(args,core,base,torch):
    from rfdetr.datasets.coco import make_coco_transforms_square_div_64
    for name,digest in base.MODEL_SOURCES.items():
        exact(Path(import_module(name).__file__),digest,MIB)
    model,config,train,_ = base.load_model(core.CHECKPOINTS["center"],CKPT,torch,args.output.parent,core.TRAIN_PROTOCOL)
    if args.mode == "dev":
        model = model.to("cuda")
    model.eval()
    transform = make_coco_transforms_square_div_64("val",576,multi_scale=False,expanded_scales=False,
        skip_random_resize=True,patch_size=config.patch_size,num_windows=config.num_windows,
        aug_config=None,scale_jitter=train.scale_jitter,gpu_postprocess=False)
    return model,transform,config


def original_image(args,row,core,fit,archive):
    from rfdetr.datasets.io_utils import decode_pil_image_bytes
    import numpy as np
    if args.mode == "fit":
        data = exact(core.LAYOUT/"train"/row["native_basename"],row["encoded_sha256"],4*MIB)
    else:
        info = archive.getinfo(row["member"])
        need(row["member"] == "images/"+fit.image_stem(row["file_name"])+".png"
             and info.file_size == row["bytes"] and f"{info.CRC:08x}" == row["crc32"],
             "Exact original qualified DEV PNG member")
        data = fit.checked_member(archive,row["member"],fit.PNG_LIMIT)
        need(sha256(data).hexdigest() == row["encoded_sha256"],"Exact original DEV encoded digest")
    image,scales = decode_pil_image_bytes(data,draft_size=None)
    need(image.mode == "RGB" and image.size == (800,800) and scales == (1.,1.)
         and sha256(np.asarray(image).tobytes(order="C")).hexdigest() == row["decoded_rgb_sha256"],
         "Exact original native RGB")
    return image


def run(args,core,base,post,rows):
    import numpy as np
    import torch
    from pycocotools import mask as codec
    from rfdetr.models.postprocess import PostProcess
    from rfdetr.utilities.tensors import nested_tensor_from_tensor_list
    need(sys.version_info[:3] == (3,10,13) and torch.__version__ == "2.10.0+cu130"
         and version("rfdetr") == "1.12.0" and version("Pillow") == "12.3.0"
         and version("pycocotools") == "2.0.11" and np.__version__ == "1.26.4",
         "Exact original inference runtime")
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(1 if args.mode == "fit" else 2)
    torch.set_num_interop_threads(1)
    if args.mode == "fit":
        need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and not torch.cuda.is_initialized(),
             "Same original CPU-only cache carrier")
    else:
        reservation = document(args.gpu_reservation,args.gpu_reservation_sha,16*MIB)
        row = reservation[args.reservation_key]
        need(reservation["GPU_owner_or_worker_active"] is True and reservation["gpu_training_active"] is False
             and row["active_attempt"] == args.reservation_key and row["resource_closed"] is False
             and row["DEV"] is True and row["FIT"] is row["TEST"] is row["AP"] is False
             and all(k == args.reservation_key or not isinstance(v,dict) or v.get("active_attempt") is None
                     for k,v in reservation.items()), "Original exclusive bounded GPU owner")
        need(torch.cuda.is_available() and torch.cuda.device_count() == 1 and torch.cuda.mem_get_info()[0]>=base.CAP,
             "Original native device admission")
        torch.backends.cudnn.allow_tf32 = False
        total = torch.cuda.mem_get_info()[1]
        torch.cuda.set_per_process_memory_fraction(min(.35,base.CAP/total))
    protocol = exact(Path(__file__).with_name("PROTOCOL.md"),args.protocol_sha,64*1024)
    started = perf_counter()
    model,transform,config = model_and_transform(args,core,base,torch)
    fit = core.load("fit_audit")
    cache_root,cache = read_cache(args) if args.mode == "fit" else (None,{})
    args.output.mkdir()
    (args.output/"RAW_LOGITS").mkdir()
    stock_hash,query_hash = sha256(),sha256()
    forwards = cache_hits = mapped = logits_total = logit_bytes = 0
    image_manifest = []
    archive_context = ZipFile(args.archive) if args.mode == "dev" else nullcontext()
    if args.mode == "dev":
        need(args.archive.stat().st_size == fit.ARCHIVE_BYTES,"Exact original archive size")
        with args.archive.open("rb") as stream:
            stream.seek(fit.CENTRAL_OFFSET)
            need(sha256(stream.read(fit.CENTRAL_BYTES)).hexdigest() == fit.CENTRAL_SHA,
                 "Exact original archive central directory")
    with archive_context as archive,(args.output/"QUERIES_PRIVATE.jsonl").open("xb") as queries,(
            args.output/"STOCK_NATIVE_PREDICTIONS_PRIVATE.jsonl").open("xb") as stock,torch.inference_mode():
        for ordinal,row in enumerate(rows,1):
            identity = row["image_id"]
            if identity in cache:
                block = dict(cache[identity])
                need(block["encoded_sha256"] == row["encoded_sha256"]
                     and block["image_content_sha256"] == row["decoded_rgb_sha256"],"Exact cache input")
                ref = block["logit_capture"]
                values_raw = exact(cache_root/ref["relative_path"],ref["sha256"],9*MIB)
                values = np.load(BytesIO(values_raw),allow_pickle=False)
                cache_hits += 1
            else:
                image = original_image(args,row,core,fit,archive)
                tensor,target = transform(image,None)
                need(target is None and tensor.shape == (3,576,576) and tensor.dtype == torch.float32,
                     "Unchanged stock val576")
                sample = nested_tensor_from_tensor_list([tensor],block_size=config.patch_size*config.num_windows)
                need(sample.mask is not None and not bool(sample.mask.any()),"Native original no-padding view")
                raw = model(sample.to("cuda" if args.mode == "dev" else "cpu"))
                query = post.aligned_query_rows(raw,(800,800),torch=torch,np=np,codec=codec,processor=PostProcess)
                selected = eligible_candidates(query,codec)
                ids = torch.tensor([q for q,_,_ in selected],device=raw["pred_masks"].device,dtype=torch.long)
                values = raw["pred_masks"][0].index_select(0,ids).contiguous().cpu().numpy().astype("<f4",copy=False)
                data = BytesIO()
                np.save(data,values,allow_pickle=False)
                values_raw = data.getvalue()
                probs = raw["pred_logits"][0].sigmoid()
                joint = torch.argsort(probs.flatten(),descending=True,stable=True)[:100]
                group = row["group"] if args.mode == "fit" else Path(row["file_name"]).name.split("_",1)[0]
                name = row["native_basename"] if args.mode == "fit" else Path(row["file_name"]).stem+".png"
                block = dict(image_id=identity,source_prefix=group,native_basename=name,
                    image_content_sha256=row["decoded_rgb_sha256"],encoded_sha256=row["encoded_sha256"],
                    **query,raw_boxes_cxcywh=raw["pred_boxes"][0].tolist(),
                    raw_classifier_logits=raw["pred_logits"][0].tolist(),joint_query_ids=(joint//2).tolist(),
                    joint_labels=(joint%2).tolist(),joint_scores=probs.flatten().index_select(0,joint).tolist())
                forwards += 1
            selected = eligible_candidates(block,codec)
            need(values.dtype == np.dtype("<f4") and values.shape == (len(selected),144,144)
                 and bool(np.isfinite(values).all()),"Exact eligible raw native logits")
            for start in range(0,len(selected),8):
                planes = torch.from_numpy(values[start:start+8,None])
                if args.mode == "dev":
                    planes = planes.to("cuda")
                raster = (torch.nn.functional.interpolate(planes,size=(800,800),
                    mode="bilinear",align_corners=False)>0)[:,0].cpu().numpy()
                for mask,(qid,_,_) in zip(raster,selected[start:start+8],strict=True):
                    rle = codec.encode(np.asfortranarray(mask,dtype=np.uint8))
                    need(dict(size=rle["size"],counts=rle["counts"].decode("ascii")) == block["all_rles"][qid],
                         "Every saved native logit reproduces same-device original stock RLE")
            filename = f"RAW_LOGITS/{ordinal:03d}.npy"
            digest = seal(args.output/filename,values_raw)
            block["logit_capture"] = dict(relative_path=filename,sha256=digest,bytes=len(values_raw),
                shape=list(values.shape),candidates=[dict(logit_row=i,query_id=q,native_rank=rank,
                    score=block["ship_scores"][q],predicted_area=area)
                    for i,(q,rank,area) in enumerate(selected)])
            stock_row = native_block(identity,block)
            stock_line = base.encoded(stock_row)
            query_line = encoded(block)
            need(stock.write(stock_line) == len(stock_line) and queries.write(query_line) == len(query_line),
                 "Complete original stock and continuous native carrier")
            stock_hash.update(stock_line)
            query_hash.update(query_line)
            mapped += len(stock_row["coco"])
            logits_total += len(selected)
            logit_bytes += len(values_raw)
            image_manifest.append(dict(image_id=identity,encoded_sha256=row["encoded_sha256"],
                decoded_rgb_sha256=row["decoded_rgb_sha256"],origin="accepted_CPU_CORE36" if identity in cache else
                "new_same_CPU_FP32" if args.mode == "fit" else "new_original_CUDA_FP32"))
            if ordinal % 10 == 0:
                print(json.dumps(dict(event="FROZEN_RF_FULL_CAPTURE_PROGRESS",images=ordinal,mode=args.mode,
                                      forwards=forwards,cache_hits=cache_hits)),flush=True)
        for stream in (stock,queries):
            stream.flush()
            os.fsync(stream.fileno())
    if args.mode == "dev":
        need(stock_hash.hexdigest() == DEV_STOCK and mapped == 67386,"Bitwise original DEV705 native stream parity")
    else:
        need(cache_hits == 36 and forwards == 73 and not torch.cuda.is_initialized(),"Exact CPU36+73 FIT109 scope")
    manifest_hash = seal(args.output/"IMAGE_MANIFEST_PRIVATE.json",encoded(image_manifest))
    result = dict(status="RFDETR_CENTER5202_FULL_CAPTURE_COMPLETE_V1",mode=args.mode,
        images=len(rows),model_forwards=forwards,cache_hits=cache_hits,checkpoint_sha256=CKPT,
        training_pair_sha256=PAIR,device="cpu" if args.mode == "fit" else "cuda",precision="FP32_TF32_disabled",
        native_joint_rows=len(rows)*100,native_mapped_rows=mapped,raw_logit_rows=logits_total,
        raw_logit_bytes=logit_bytes,raw_logit_files=len(rows),raw_logit_raster_RLE_exact=logits_total,
        protocol_sha256=args.protocol_sha,queries_sha256=query_hash.hexdigest(),
        stock_predictions_sha256=stock_hash.hexdigest(),image_manifest_sha256=manifest_hash,
        eligible_area=[32,1024],mask_rule="bilinear(raw_logits,align_corners=False)>0",
        feature_probability_rule="sigmoid(bilinear(raw_logits,align_corners=False))",
        historical_CUDA_stock_exact=args.mode=="dev",GT_opened=False,AP=False,FIT=args.mode=="fit",
        DEV=args.mode=="dev",TEST=False,optimizer_updates=0,elapsed_seconds=perf_counter()-started,
        worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    digest = seal(args.output/"RESULT_SAFE.json",encoded(result))
    print(json.dumps(dict(status=result["status"],result_sha256=digest)),flush=True)


def main():
    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--mode",choices=("fit","dev"),required=True)
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--run",action="store_true")
    for name in ("output","archive","gpu-reservation"):
        parser.add_argument("--"+name,type=Path)
    for name in ("gpu-reservation-sha","reservation-key","protocol-sha"):
        parser.add_argument("--"+name)
    args = parser.parse_args()
    need(args.check != args.run and sys.flags.isolated and sys.dont_write_bytecode,"One explicit isolated command")
    core = load_core(args.root)
    base,post = setup(args,core)
    rows = metadata(args,core)
    if args.check:
        print(json.dumps(dict(status="RF_FULL_CAPTURE_SOURCE_ADMISSION_CHECKED",mode=args.mode,images=len(rows),
                              model_loaded=False,images_opened=False,GT_opened=False,AP=False)),flush=True)
        return
    need(args.protocol_sha is not None and args.output is not None and args.output.is_absolute() and not args.output.exists()
         and args.output.parent.resolve(strict=True)==args.output.parent,"Fresh canonical capture output")
    need(args.mode=="fit" or all((args.archive,args.gpu_reservation,args.gpu_reservation_sha,args.reservation_key)),
         "Original full CUDA owner inputs")
    def offline(event,_):
        if event.startswith("socket."):
            raise PermissionError("Offline frozen source capture")
        if event == "open" and args.mode=="fit" and isinstance(_[0],(str,bytes)):
            name = Path(os.fsdecode(_[0]))
            if name.name.lower().endswith(".coco.json") or any(p.lower() in ("dev","test","valid") for p in name.parts):
                raise PermissionError("FIT-only data roles")
    sys.addaudithook(offline)
    run(args,core,base,post,rows)


if __name__=="__main__":
    main()

