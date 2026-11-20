"""Export and score the fixed closed D-FINE-seg-S HRSID DEV705 E24 endpoint."""

from argparse import ArgumentParser
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha1, sha256
from importlib.metadata import version
from io import BytesIO, StringIO
import json
from math import isfinite
import os
from pathlib import Path, PurePosixPath
import re
import sys
from time import perf_counter
from types import ModuleType
from zipfile import ZipFile


TRAIN_PROTOCOL = "684ab3a18dddc3a9887819050dc545c28736bbb3ccc664e83be9c67ef2b1d0a3"
FIT_ORDER_SHA = "c6bd988e2728f0f50e44dcd52134a970ca1a135c300ff15d828bc9950771dc0f"
FIT_MAP_SHA = "5731bf5b6422171f9c90108b32a0607f131164f6e785e1b2a476c0ce82391939"
ACCEPTED_E24 = "DFINE_SEG_S_HRSID_E24_CUDA_INDEPENDENT_READBACK_PASSED"
EXPORT_COMPLETE = "HRSID_DFINE_E24_DEV705_EXPORT_COMPLETE_V1"
EXPORT_OS_CLOSED = "HRSID_DFINE_E24_DEV705_EXPORT_OS_CLOSED_V1"
EXPORT_ACCEPTED = "HRSID_DFINE_E24_DEV705_EXPORT_INDEPENDENTLY_ACCEPTED_V1"
IMAGES, ANNOTATIONS, EMPTY, QUERIES, UPDATES = 705, 1782, 4, 300, 17640
MIB = 1024**2
FIT_VISITS = 70560
AUTHOR_COMMIT = "b782ece206487792f36c165ca65548154cd12094"
AUTHOR_TREE_SHA = "7381520384d7c55858eb19eb347bc45b7c23fa012e9cc088122b803fe8fd4712"
AUTHOR_TREE = "65876566e86bbb3378732fcd4a8d3f80d020333c"
AUTHOR_SOURCE = {
    "dfine_seg/infer/torch_model.py": "33241142ac5ed1f798319e55a8ce83ebd0f1a7deecf8fce4eb0a67b57fa2cdfd",
    "dfine_seg/api/ckpt.py": "d24481b777892b8a5e10f67ae6203a9b9de818b470d387d9f0a39746f1a60434",
    "dfine_seg/model/dfine.py": "0e95334e7e580e4af6325832e3874ea95a55ad85eca7b7a518727d628460422d",
    "dfine_seg/model/configs.py": "e5f2ecdd14e0cc8c15bafcaf39a0d484a08a815defa2cd5649fa11c744fc55d6",
}
AUTHOR_META = dict(dfine_seg_version="0.7.0",model_name="s",task="segment",num_classes=1,in_channels=3,
                   label_to_name={0:"ship"},img_size=[640,640],keep_ratio=False)
FILES = {
    "DEV_INPUT_AUDIT_SAFE.json": "2a02c89e84da572ebf3f1fec859609376a56c57f0fdf5efd84cfb5abcfc41439",
    "ROOT_INDEPENDENT_DEV705_SAFE.json": "a650238185dfe32227c720f31d6ba2439d115902e5a4037be965626ac0180cfe",
    "DEV_IMAGE_ONLY_MANIFEST_PRIVATE.json": "39a40d089d4a54448fe6efca8ea5cc98ad1eb7c0587815f3dd9a3b097173e78f",
    "DEV_ORIGINAL_GT_PRIVATE.json": "393b3da233991dc5fafc98b4ddeaa341b042d9571434ef8ce53c3f2fdada5628",
    "DEV_EMPTY_RASTERS_PRIVATE.json": "f3f2d644b2e0ce475b4a46b382ec7ec51f2efa4609c8f85b4f8f552ff57d5889",
}
DEV_MANIFEST = "DEV_IMAGE_ONLY_MANIFEST_PRIVATE.json"
PREDICTIONS = "DEV_MASK_PREDICTIONS_PRIVATE.jsonl"
AUDIT = "DEV_CANDIDATE_AUDIT_PRIVATE.jsonl"
RESULT = "RESULT_SAFE.json"
SCORE_ADAPTER_SHA = "f9fc79403d12d8840c3269b04ba22d7584b8eb5e3c592ead2b6359ad7df65748"
ARCHIVE_BYTES, CENTRAL_OFFSET, CENTRAL_BYTES = 5698302315,5697629273,672944
CENTRAL_SHA = "4a54303e8cce24f0330cddd2e6d3c12205c8b0002859d0b5bf534efb531a2dce"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return isinstance(value,str) and re.fullmatch("[0-9a-f]{64}",value) is not None


def decode(raw):
    def unique(pairs):
        obj = dict(pairs)
        need(len(obj) == len(pairs), "Duplicate JSON field")
        return obj

    def reject(token):
        raise ValueError("Nonfinite JSON token: "+token)

    return json.loads(raw,object_pairs_hook=unique,parse_constant=reject)


def encoded(value):
    return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def identity(stat):
    return (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)


def canonical_file(path, limit, exact=None):
    need(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(),
         "Expected canonical existing file: "+path.name)
    stat = path.stat()
    need((0 if exact == 0 else 1) <= stat.st_size <= limit and (exact is None or stat.st_size == exact),
         "Expected bounded pinned byte count: "+path.name)
    return identity(stat)


def pinned(path, expected_sha, limit=MIB, exact=None):
    before = canonical_file(path,limit,exact)
    need(digest(expected_sha), "Exact lowercase SHA-256 required")
    with path.open("rb") as handle:
        raw = handle.read()
        need(identity(os.fstat(handle.fileno())) == before == identity(path.stat()),
             "Pinned input changed: "+path.name)
    need(sha256(raw).hexdigest() == expected_sha,"Wrong exact pinned bytes: "+path.name)
    return raw


def write_new(path, value):
    raw = encoded(value)
    with path.open("xb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    return sha256(raw).hexdigest()


def pack_candidates(image_id, hw, labels, scores, boxes, masks):
    import numpy as np
    from pycocotools import mask as codec

    n = len(labels)
    need(0 <= n <= QUERIES and labels.shape == scores.shape == (n,) and boxes.shape == (n,4)
         and labels.dtype == np.int64 and all(k == 0 for k in labels) and all(isfinite(s) and 0 <= s <= 1 for s in scores)
         and all(a >= b for a,b in zip(scores,scores[1:])) and bool(np.isfinite(boxes).all()),
         "Exact author one-class native post-NMS order")
    if n:
        need(masks is not None and masks.shape == (n,*hw) and masks.dtype == np.uint8,
             "Exact original-size author binary masks")
    predictions,audit = [],[]
    for i,(score,box) in enumerate(zip(scores,boxes,strict=True),1):
        raster = masks[i-1]
        need(int(raster.max()) <= 1, "Author mask must remain binary")
        rle = codec.encode(np.asfortranarray(raster))
        area = int(codec.area(rle))
        rle["counts"] = rle["counts"].decode("ascii")
        score = float(score)
        predictions.append(dict(image_id=image_id,category_id=1,score=score,segmentation=rle))
        audit.append(dict(native_rank=i,score=score,box_xyxy=box.tolist(),mask_area=area))
    return (dict(image_id=image_id,coco=predictions),
            dict(image_id=image_id,raw_queries=QUERIES,internal_top=QUERIES,post_nms=n,candidates=audit))


def validate_rows(image_id, hw, prediction, audit):
    from pycocotools import mask as codec

    rows,candidates = prediction["coco"],audit["candidates"]
    n = len(rows)
    need(set(prediction) == {"image_id","coco"} and set(audit) == {"image_id","raw_queries","internal_top","post_nms","candidates"}
         and prediction["image_id"] == audit["image_id"] == image_id
         and type(rows) is type(candidates) is list and 0 <= n <= QUERIES
         and audit["post_nms"] == len(candidates) == n and audit["raw_queries"] == audit["internal_top"] == QUERIES,
         "Exact closed prediction/audit image roster")
    empty = 0
    prior = 1.
    for rank,(row,detail) in enumerate(zip(rows,candidates,strict=True),1):
        score,rle = row["score"],row["segmentation"]
        need(set(row) == {"image_id","category_id","score","segmentation"}
             and row["image_id"] == image_id and row["category_id"] == 1
             and type(score) in (float,int) and isfinite(score) and 0 <= score <= prior
             and set(rle) == {"size","counts"} and rle["size"] == list(hw)
             and type(rle["counts"]) is str and bool(rle["counts"]), "Exact native COCO RLE-only row")
        area = int(codec.area(rle))
        box = detail["box_xyxy"]
        need(set(detail) == {"native_rank","score","box_xyxy","mask_area"}
             and detail["native_rank"] == rank and detail["score"] == score and detail["mask_area"] == area
             and type(box) is list and len(box) == 4 and all(type(x) in (float,int) and isfinite(x) for x in box),
             "Exact separate native box/rank audit")
        prior = score
        empty += area == 0
    return n,empty



def admitted_e24(receipt):
    need(receipt["status"] == ACCEPTED_E24 and receipt["protocol_sha256"] == TRAIN_PROTOCOL
         and receipt["fit_order_sha256"] == FIT_ORDER_SHA and receipt["fit_id_map_sha256"] == FIT_MAP_SHA
         and receipt["FIT"] is True and receipt["DEV"] is receipt["TEST"] is receipt["AP"] is False
         and receipt["original_exit"] == 0 and receipt["job_drained"] is True
         and receipt["completed_epochs"] == 24 and receipt["fit_visits"] == FIT_VISITS
         and receipt["optimizer_updates"] == receipt["ema_updates"] == receipt["scheduler_updates"] == UPDATES
         and type(receipt["ema_checkpoint_bytes"]) is int and 0 < receipt["ema_checkpoint_bytes"] < 256*MIB
         and all(digest(receipt[k]) for k in ("worker_source_sha256","result_sha256","original_tool_wait_sha256",
                                             "worker_terminal_sha256","ema_checkpoint_sha256")),
         "Exact independent FIT-only E24 acceptance required")
    return receipt


def dev_json(root, name, limit=4*MIB):
    return decode(pinned(root/name,FILES[name],limit))


def read_manifest(dev_root):
    audit = dev_json(dev_root,"DEV_INPUT_AUDIT_SAFE.json")
    independent = dev_json(dev_root,"ROOT_INDEPENDENT_DEV705_SAFE.json")
    need(audit["decoded_dev_images"] == independent["images"] == IMAGES
         and audit["attempted_dev_rasters"] == independent["unchanged_annotations"] == ANNOTATIONS
         and audit["empty_dev_rasters"] == independent["independent_empty_rasters"] == EMPTY
         and independent["no_exact_FIT_encoded_or_RGB_overlap"] is True
         and independent["official_TEST_opened"] is False
         and audit["image_manifest_sha256"] == FILES[DEV_MANIFEST]
         and audit["original_dev_gt_sha256"] == FILES["DEV_ORIGINAL_GT_PRIVATE.json"]
         and audit["empty_raster_findings_sha256"] == FILES["DEV_EMPTY_RASTERS_PRIVATE.json"],
         "Exact image-only DEV705 admission required")
    manifest = dev_json(dev_root,DEV_MANIFEST)
    need(len(manifest) == len({r["image_id"] for r in manifest}) == len({r["member"] for r in manifest}) == IMAGES,
         "Exact complete unique DEV705 image-only roster")
    return manifest


CODEC_WHEEL_SHA = "8e159232adae3aef6b4e2d37b008bff107b26e9ed3b48e70ea6482302834bd34"


def prepare_export_codec(wheel, target):
    raw = pinned(wheel,CODEC_WHEEL_SHA,MIB)
    with ZipFile(BytesIO(raw)) as package:
        files = package.infolist()
        for info in files:
            path = PurePosixPath(info.filename)
            need(not path.is_absolute() and ".." not in path.parts and "\\" not in info.filename
                 and path.parts[0] in {"pycocotools","pycocotools-2.0.11.dist-info"}
                 and info.file_size <= MIB, "Official codec wheel layout")
            if not info.is_dir():
                local = target.joinpath(*path.parts)
                local.parent.mkdir(parents=True,exist_ok=True)
                with local.open("xb") as stream:
                    stream.write(package.read(info))
    sys.path.insert(0,str(target))
    need(version("pycocotools") == "2.0.11","Exact export RLE codec")


def admit_source(source, tree_path):
    need(source.is_absolute() and source.resolve(strict=True) == source and source.is_dir(),"Exact canonical author source root")
    tree = decode(pinned(tree_path,AUTHOR_TREE_SHA))
    need(tree["status"] == "AUTHOR_FIXED_TAG_AND_TREE" and tree["commit"] == AUTHOR_COMMIT and tree["tree"] == AUTHOR_TREE,
         "Exact previously verified upstream Git tree")
    entries = [r for r in tree["entries"] if r["type"] == "blob" and r["path"].startswith("dfine_seg/")]
    expected = {r["path"] for r in entries}
    need(len(entries) == len(expected) == 71 and set(AUTHOR_SOURCE) <= expected,"Exact author package file roster")
    actual = {p.relative_to(source).as_posix() for p in (source/"dfine_seg").rglob("*")
              if p.is_file() and p.suffix in {".py",".json",".yaml"}}
    need(actual == expected,"No unadmitted package modules or configuration files")
    for item in entries:
        name = item["path"]
        path = source/name
        canonical_file(path,MIB,item["size"])
        raw = path.read_bytes()
        header = f"blob {len(raw)}\x00".encode("ascii")
        need(sha1(header+raw).hexdigest() == item["sha"],"Wrong author Git blob: "+name)
        if name in AUTHOR_SOURCE:
            need(sha256(raw).hexdigest() == AUTHOR_SOURCE[name],"Wrong pinned author inference source: "+name)
    sys.path.insert(0,str(source))


def read_png(archive, item):
    import numpy as np
    from PIL import Image

    hw = (item["height"],item["width"])
    member = item["member"]
    expected = "images/"+PurePosixPath(item["file_name"]).stem+".png"
    info = archive.getinfo(member)
    need(hw == (800,800) and member == expected and 0 < info.file_size == item["bytes"] <= 8*MIB
         and f"{info.CRC:08x}" == item["crc32"], "Exact qualified DEV PNG archive member")
    raw = archive.read(info)
    need(sha256(raw).hexdigest() == item["encoded_sha256"],"Exact qualified DEV PNG bytes")
    with Image.open(BytesIO(raw)) as original:
        need(original.format == "PNG" and original.size == hw[::-1],"Original DEV PNG canvas")
        rgb = np.asarray(original.convert("RGB"),dtype=np.uint8)
    need(rgb.shape == (*hw,3) and sha256(rgb.tobytes(order="C")).hexdigest() == item["decoded_rgb_sha256"],
         "Exact audited DEV RGB pixels")
    return rgb,hw


def check_wrapper(wrapper, accepted_state, torch, expected_device="cuda"):
    need(wrapper.model_name == "s" and wrapper.task == "segment" and wrapper.n_outputs == 1
         and wrapper.names == {0:"ship"} and wrapper.channels == 3 and wrapper.input_size == (640,640)
         and wrapper.device == expected_device and wrapper.conf_thresh == 0. and wrapper.apply_nms
         and wrapper.nms_iou_thresh == .70 and wrapper.binarize_masks and wrapper.mask_threshold == .50
         and not wrapper.rect and not wrapper.keep_ratio and not wrapper.labels_to_use and not wrapper.model.training,
         "Exact author one-class native export configuration")
    actual = wrapper.model.state_dict()
    cache = {"decoder.anchors","decoder.valid_mask"}
    decoder = wrapper.model.decoder
    need(len(accepted_state) == 815 and len(actual) == 813
         and actual.keys() == accepted_state.keys()-cache and decoder.eval_spatial_size is None,
         "Only the two fixed training anchor buffers are dynamically recomputed by author inference")
    need(tuple(accepted_state["decoder.anchors"].shape) == (1,8400,4)
         and tuple(accepted_state["decoder.valid_mask"].shape) == (1,8400,1),
         "Expected archived 640-grid anchor-cache shapes")
    need(all(v.shape == accepted_state[k].shape and v.dtype == accepted_state[k].dtype
             and torch.equal(v.detach().cpu(),accepted_state[k]) for k,v in actual.items()),
         "All 813 loaded inference tensors must equal the independently accepted EMA checkpoint")


def run_export(args):
    acceptance = admitted_e24(decode(pinned(args.acceptance,args.acceptance_sha)))
    admit_source(args.author_source,args.author_tree)
    import torch
    import torchvision
    from PIL import __version__ as pillow
    from dfine_seg.infer.torch_model import TorchModel
    need(sys.version_info[:3] == (3,11,15) and torch.__version__ == "2.13.0+cu130"
         and torchvision.__version__ == "0.28.0+cu130" and pillow == "13.0.0"
         and Path(sys.modules[TorchModel.__module__].__file__).resolve() == args.author_source/"dfine_seg/infer/torch_model.py",
         "Exact qualified author inference runtime")
    need(torch.cuda.is_available() and torch.cuda.device_count() == 1,"One isolated CUDA evaluation device")
    device = torch.device("cuda:0")
    free,total = torch.cuda.mem_get_info(device)
    need(free >= 6*1024**3,"CUDA export admission floor")
    torch.cuda.set_per_process_memory_fraction(min(.35,6*1024**3/total),device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    manifest = read_manifest(args.dev_root)
    start = perf_counter()
    before_ema = canonical_file(args.ema,256*MIB,acceptance["ema_checkpoint_bytes"])
    archive_bytes = pinned(args.ema,acceptance["ema_checkpoint_sha256"],256*MIB,acceptance["ema_checkpoint_bytes"])
    checkpoint = torch.load(BytesIO(archive_bytes),map_location="cpu",weights_only=True)
    del archive_bytes
    need(set(checkpoint) == {"model","meta"} and checkpoint["meta"] == AUTHOR_META,"Exact stock E24 EMA envelope")
    wrapper = TorchModel(str(args.ema),conf_thresh=0.,rect=False,keep_ratio=False,apply_nms=True,
                         nms_iou_thresh=.70,binarize_masks=True,mask_threshold=.50,device="cuda",task="segment")
    check_wrapper(wrapper,checkpoint["model"],torch)
    del checkpoint
    need(identity(args.ema.stat()) == before_ema,"Accepted EMA archive unchanged during author load")
    args.output.mkdir(exist_ok=False)

    frames = []
    def observe(_, inputs, out):
        need(len(inputs) == 1 and tuple(inputs[0].shape) == (1,3,640,640) and inputs[0].dtype == torch.float32
             and tuple(out["pred_logits"].shape) == (1,QUERIES,1)
             and tuple(out["pred_boxes"].shape) == (1,QUERIES,4)
             and tuple(out["pred_masks"].shape) == (1,QUERIES,160,160)
             and out["pred_logits"].dtype == torch.float32,"Exact author FP32 one-image query carrier")
        frames.append(1)

    watch = wrapper.model.register_forward_hook(observe)
    before_archive = canonical_file(args.archive,ARCHIVE_BYTES,ARCHIVE_BYTES)
    post_nms = empty = at100 = empty_at100 = zero_images = 0
    prediction_digest,audit_digest = sha256(),sha256()
    seconds = 0.
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    prepare_export_codec(args.codec_wheel,args.output/"CODEC_PRIVATE")
    with args.archive.open("rb") as stream, (args.output/PREDICTIONS).open("xb") as out, (args.output/AUDIT).open("xb") as audit:
        need(identity(os.fstat(stream.fileno())) == before_archive,"Closed archive handle")
        stream.seek(CENTRAL_OFFSET)
        need(sha256(stream.read(CENTRAL_BYTES)).hexdigest() == CENTRAL_SHA,"Exact official PNG ZIP central directory")
        with ZipFile(stream) as archive, torch.inference_mode():
            for ordinal,item in enumerate(manifest,1):
                rgb,hw = read_png(archive,item)
                t = perf_counter()
                result, = wrapper(rgb,bgr=False,conf_thresh=0.)
                torch.cuda.synchronize(device)
                seconds += perf_counter()-t
                need(len(frames) == ordinal,"Exactly one author model forward per DEV chip")
                masks = result.get("masks")
                pred,detail = pack_candidates(item["image_id"],hw,result["labels"].cpu().numpy(),
                                              result["scores"].cpu().numpy(),result["boxes"].cpu().numpy(),
                                              None if masks is None else masks.cpu().numpy())
                n = detail["post_nms"]
                audit_rows = detail["candidates"]
                post_nms += n
                empty += sum(r["mask_area"] == 0 for r in audit_rows)
                at100 += min(100,n)
                empty_at100 += sum(r["mask_area"] == 0 for r in audit_rows[:100])
                zero_images += n == 0
                raw = encoded(pred)
                out.write(raw)
                prediction_digest.update(raw)
                raw = encoded(detail)
                audit.write(raw)
                audit_digest.update(raw)
                if ordinal%100 == 0:
                    print(json.dumps(dict(event="DFINE_DEV705_EXPORT_PROGRESS",images=ordinal)),flush=True)
        need(before_archive == identity(os.fstat(stream.fileno())) == identity(args.archive.stat()),
             "Original official PNG ZIP remained unchanged")
        for handle in (out,audit):
            handle.flush()
            os.fsync(handle.fileno())
    watch.remove()
    need(len(frames) == IMAGES,"Exact full DEV705 author forwards")
    result = dict(
        status=EXPORT_COMPLETE,protocol_sha256=args.protocol_sha,training_protocol_sha256=TRAIN_PROTOCOL,
        export_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),e24_acceptance_sha256=args.acceptance_sha,
        ema_sha256=acceptance["ema_checkpoint_sha256"],ema_updates=UPDATES,author_commit=AUTHOR_COMMIT,
        manifest_sha256=FILES[DEV_MANIFEST],predictions_sha256=prediction_digest.hexdigest(),
        candidate_audit_sha256=audit_digest.hexdigest(),images=IMAGES,model_forwards=IMAGES,synthetic_warmups=1,
        raw_queries=IMAGES*QUERIES,internal_top_rows=IMAGES*QUERIES,post_nms_rows=post_nms,
        empty_post_nms=empty,rows_at_100=at100,empty_at_100=empty_at100,zero_prediction_images=zero_images,
        confidence_threshold=0.,box_nms_iou=.70,mask_threshold=.50,mask_box_cleanup=True,
        model_precision="FP32_TF32_disabled",mask_resize_precision="author_CUDA_FP16",
        loaded_state_entries=813,archived_dynamic_anchor_buffers=2,
        runtime=dict(torch=torch.__version__,torchvision=torchvision.__version__,pillow=pillow,pycocotools="2.0.11"),
        FIT=False,DEV=True,TEST=False,GT_opened=False,AP=False,forward_postprocess_seconds=seconds,
        elapsed_seconds=perf_counter()-start,peak_allocated=torch.cuda.max_memory_allocated(device),
    )
    output_sha = write_new(args.output/RESULT,result)
    print(json.dumps(dict(status=result["status"],result_sha256=output_sha),sort_keys=True),flush=True)



def export_receipt(args):
    e24 = admitted_e24(decode(pinned(args.acceptance,args.acceptance_sha)))
    report = decode(pinned(args.export/RESULT,args.export_result_sha))
    need(report["status"] == EXPORT_COMPLETE and report["protocol_sha256"] == args.protocol_sha
         and report["training_protocol_sha256"] == TRAIN_PROTOCOL
         and report["e24_acceptance_sha256"] == args.acceptance_sha
         and report["ema_sha256"] == e24["ema_checkpoint_sha256"] and report["ema_updates"] == UPDATES
         and report["author_commit"] == AUTHOR_COMMIT and report["manifest_sha256"] == FILES[DEV_MANIFEST]
         and report["images"] == report["model_forwards"] == IMAGES and report["synthetic_warmups"] == 1
         and report["raw_queries"] == report["internal_top_rows"] == IMAGES*QUERIES
         and report["confidence_threshold"] == 0. and report["box_nms_iou"] == .70
         and report["mask_threshold"] == .50 and report["mask_box_cleanup"] is True
         and report["model_precision"] == "FP32_TF32_disabled" and report["mask_resize_precision"] == "author_CUDA_FP16"
         and report["loaded_state_entries"] == 813 and report["archived_dynamic_anchor_buffers"] == 2
         and report["FIT"] is report["TEST"] is report["GT_opened"] is report["AP"] is False
         and report["DEV"] is True and digest(report["predictions_sha256"]) and digest(report["candidate_audit_sha256"])
         and report["export_source_sha256"] == sha256(Path(__file__).read_bytes()).hexdigest(),
         "Exact closed fixed E24 label-free author export required")
    return report


def validate_streams(root, manifest, report, *, keep):
    pred_path,audit_path = root/PREDICTIONS,root/AUDIT
    p_before,a_before = canonical_file(pred_path,512*MIB),canonical_file(audit_path,128*MIB)
    ph,ah = sha256(),sha256()
    totals = dict(post_nms_rows=0,empty_post_nms=0,rows_at_100=0,empty_at_100=0,zero_prediction_images=0)
    flat = []
    with pred_path.open("rb") as pred_stream,audit_path.open("rb") as audit_stream:
        for item in manifest:
            p_line,a_line = pred_stream.readline(4*MIB),audit_stream.readline(MIB)
            need(p_line.endswith(b"\n") and a_line.endswith(b"\n"),"Complete bounded DEV streams")
            ph.update(p_line)
            ah.update(a_line)
            prediction,audit = decode(p_line),decode(a_line)
            n,empty = validate_rows(item["image_id"],(item["height"],item["width"]),prediction,audit)
            totals["post_nms_rows"] += n
            totals["empty_post_nms"] += empty
            totals["rows_at_100"] += min(100,n)
            totals["empty_at_100"] += sum(r["mask_area"] == 0 for r in audit["candidates"][:100])
            totals["zero_prediction_images"] += n == 0
            if keep:
                flat.extend(prediction["coco"])
        need(not pred_stream.read(1) and not audit_stream.read(1),"Exact DEV population without trailing rows")
        need(identity(os.fstat(pred_stream.fileno())) == p_before == identity(pred_path.stat())
             and identity(os.fstat(audit_stream.fileno())) == a_before == identity(audit_path.stat()),
             "Closed DEV streams remained unchanged")
    need(ph.hexdigest() == report["predictions_sha256"] and ah.hexdigest() == report["candidate_audit_sha256"]
         and all(report[k] == v for k,v in totals.items()), "Exact closed prediction, audit and count digests")
    return flat,totals


def cpu_runtime():
    need(sys.flags.isolated and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
         and version("pycocotools") == "2.0.11" and version("numpy") == "1.26.4",
         "Exact isolated CPU official COCO runtime")


def run_verify_export(args):
    cpu_runtime()
    report = export_receipt(args)
    terminal = decode(pinned(args.terminal,args.terminal_sha))
    need(terminal["status"] == EXPORT_OS_CLOSED and terminal["original_os_wait"] is True
         and terminal["original_os_exit"] == 0 and terminal["job_drained"] is True and terminal["resource_stop"] is False
         and terminal["export_result_sha256"] == args.export_result_sha
         and terminal["protocol_sha256"] == args.protocol_sha and digest(terminal["original_tool_wait_sha256"]),
         "Independent verification requires original export OS wait and drained job")
    _,counts = validate_streams(args.export,read_manifest(args.dev_root),report,keep=False)
    args.output.mkdir(exist_ok=False)
    result = dict(
        status=EXPORT_ACCEPTED,protocol_sha256=args.protocol_sha,training_protocol_sha256=TRAIN_PROTOCOL,
        e24_acceptance_sha256=args.acceptance_sha,export_result_sha256=args.export_result_sha,
        export_terminal_sha256=args.terminal_sha,original_tool_wait_sha256=terminal["original_tool_wait_sha256"],
        predictions_sha256=report["predictions_sha256"],candidate_audit_sha256=report["candidate_audit_sha256"],
        ema_sha256=report["ema_sha256"],images=IMAGES,original_os_wait=True,original_os_exit=0,
        job_drained=True,resource_stop=False,FIT=False,DEV=True,TEST=False,GT_opened=False,AP=False,**counts,
    )
    out_sha = write_new(args.output/RESULT,result)
    print(json.dumps(dict(status=result["status"],result_sha256=out_sha),sort_keys=True),flush=True)


def independent_export(args, report):
    witness = decode(pinned(args.export_acceptance,args.export_acceptance_sha))
    need(witness["status"] == EXPORT_ACCEPTED and witness["protocol_sha256"] == args.protocol_sha
         and witness["training_protocol_sha256"] == TRAIN_PROTOCOL and witness["e24_acceptance_sha256"] == args.acceptance_sha
         and witness["export_result_sha256"] == args.export_result_sha and witness["ema_sha256"] == report["ema_sha256"]
         and witness["images"] == IMAGES and witness["predictions_sha256"] == report["predictions_sha256"]
         and witness["candidate_audit_sha256"] == report["candidate_audit_sha256"]
         and witness["original_os_wait"] is True and witness["original_os_exit"] == 0 and witness["job_drained"] is True
         and witness["resource_stop"] is False and witness["FIT"] is witness["TEST"] is witness["GT_opened"] is witness["AP"] is False
         and witness["DEV"] is True and digest(witness["export_terminal_sha256"]) and digest(witness["original_tool_wait_sha256"]),
         "Exact independently closed and label-free DEV export required")
    return witness


def score_variants(gt, predictions, empty_ids, evaluate_population):
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    image_ids = [r["id"] for r in gt["images"]]
    require_ids = {r["id"] for r in gt["annotations"]}
    need(empty_ids <= require_ids and len(require_ids) == len(gt["annotations"]),"Exact original unique annotation IDs")
    output = {}
    for kind,excluded in (("primary",set()),("sensitivity",empty_ids)):
        original = deepcopy(gt)
        original["annotations"] = [r for r in original["annotations"] if r["id"] not in excluded]
        coco = COCO()
        coco.dataset = original
        captured = []
        def evaluator(*args):
            instance = COCOeval(*args)
            captured.append(instance)
            return instance
        with redirect_stdout(StringIO()):
            coco.createIndex()
            metrics = evaluate_population(dict(COCO=COCO,COCOeval=evaluator),coco,predictions,image_ids)
        evaluation, = captured
        iou, = np.flatnonzero(np.isclose(evaluation.params.iouThrs,.90))
        values = evaluation.eval["precision"][iou,:,:,0,-1]
        valid = values[values > -1]
        fields = dict(zip(("AP","AP50","AP75","APs","APm","APl"),metrics[:6],strict=True))
        output[kind] = dict(annotations=len(original["annotations"]),AP90=float(valid.mean()) if valid.size else -1.,
                            AR100=metrics[8],all12=metrics,**fields)
    return output


def verify_labels(gt, findings, manifest):
    from pycocotools import mask
    from pycocotools.coco import COCO

    need(len(gt["images"]) == len(manifest) == IMAGES
         and len(gt["annotations"]) == ANNOTATIONS and [r["id"] for r in gt["categories"]] == [1]
         and [(r["id"],r["height"],r["width"]) for r in gt["images"]] ==
             [(r["image_id"],r["height"],r["width"]) for r in manifest],"Exact original DEV705 label roster")
    empty_ids = {r["annotation_id"] for r in findings if r["kind"] == "EMPTY_COCO_RASTER"}
    need(len(findings) == len(empty_ids) == EMPTY,"Exactly four pinned empty-raster IDs")
    with redirect_stdout(StringIO()):
        coco = COCO()
        coco.dataset = gt
        coco.createIndex()
    actual = {(r["id"],r["image_id"]) for r in gt["annotations"] if int(mask.area(coco.annToRLE(r))) == 0}
    need(actual == {(r["annotation_id"],r["image_id"]) for r in findings},"Exact four original empty GT raster witnesses")
    return empty_ids


def run_score(args):
    cpu_runtime()
    report = export_receipt(args)
    witness = independent_export(args,report)
    manifest = read_manifest(args.dev_root)
    predictions,counts = validate_streams(args.export,manifest,report,keep=True)
    need(all(witness[k] == value for k,value in counts.items()),"Independent exact native row counts")
    adapter = ModuleType("_dfine_dev705_original_coco_adapter")
    adapter.__file__ = str(args.score_adapter)
    source = pinned(args.score_adapter,SCORE_ADAPTER_SHA)
    exec(compile(source,adapter.__file__,"exec"),adapter.__dict__)

    args.output.mkdir(exist_ok=False)
    receipt = dict(
        status="HRSID_DFINE_E24_DEV705_SCORE_PRELABEL_V1",protocol_sha256=args.protocol_sha,
        export_result_sha256=args.export_result_sha,export_acceptance_sha256=args.export_acceptance_sha,
        e24_acceptance_sha256=args.acceptance_sha,ema_sha256=report["ema_sha256"],images=IMAGES,
        predictions_sha256=report["predictions_sha256"],candidate_audit_sha256=report["candidate_audit_sha256"],
        FIT=False,DEV=True,TEST=False,GT_opened=False,AP=False,**counts,
    )
    receipt_sha = write_new(args.output/"PRELABEL_RECEIPT_SAFE.json",receipt)
    print(json.dumps(dict(status=receipt["status"],receipt_sha256=receipt_sha),sort_keys=True),flush=True)
    gt = dev_json(args.dev_root,"DEV_ORIGINAL_GT_PRIVATE.json")
    empty_ids = verify_labels(gt,dev_json(args.dev_root,"DEV_EMPTY_RASTERS_PRIVATE.json"),manifest)
    started = perf_counter()
    outcomes = score_variants(gt,predictions,empty_ids,adapter.evaluate_population)
    result = dict(
        status="HRSID_DFINE_E24_DEV705_COCO_COMPLETE_V1",protocol_sha256=args.protocol_sha,
        training_protocol_sha256=TRAIN_PROTOCOL,e24_acceptance_sha256=args.acceptance_sha,
        ema_sha256=report["ema_sha256"],export_result_sha256=args.export_result_sha,
        export_acceptance_sha256=args.export_acceptance_sha,prelabel_receipt_sha256=receipt_sha,
        predictions_sha256=report["predictions_sha256"],candidate_audit_sha256=report["candidate_audit_sha256"],
        gt_sha256=FILES["DEV_ORIGINAL_GT_PRIVATE.json"],images=IMAGES,sensitivity_removed=EMPTY,
        max_dets=[1,10,100],units="fraction",FIT=False,DEV=True,TEST=False,GT_opened=True,AP=True,
        official_TEST_AP=False,score_seconds=perf_counter()-started,**counts,**outcomes,
    )
    out_sha = write_new(args.output/RESULT,result)
    print(json.dumps(dict(status=result["status"],result_sha256=out_sha),sort_keys=True),flush=True)


def main():
    parser = ArgumentParser(description=__doc__,allow_abbrev=False)
    modes = parser.add_subparsers(required=True,dest="mode")
    for mode,fn in (("export",run_export),("verify-export",run_verify_export),("score",run_score)):
        sub = modes.add_parser(mode,allow_abbrev=False)
        for name in ("dev-root","output","acceptance"):
            sub.add_argument("--"+name,type=Path,required=True)
        for name in ("acceptance-sha","protocol-sha"):
            sub.add_argument("--"+name,required=True)
        if mode != "export":
            sub.add_argument("--export",type=Path,required=True)
            sub.add_argument("--export-result-sha",required=True)
        sub.set_defaults(run=fn)
    for name in ("author-source","author-tree","archive","ema","codec-wheel"):
        modes.choices["export"].add_argument("--"+name,type=Path,required=True)
    modes.choices["verify-export"].add_argument("--terminal",type=Path,required=True)
    modes.choices["verify-export"].add_argument("--terminal-sha",required=True)
    modes.choices["score"].add_argument("--score-adapter",type=Path,required=True)
    modes.choices["score"].add_argument("--export-acceptance",type=Path,required=True)
    modes.choices["score"].add_argument("--export-acceptance-sha",required=True)
    args = parser.parse_args()

    def offline(event,_):
        if event.startswith("socket."):
            raise PermissionError("Offline D-FINE evaluation")
    sys.addaudithook(offline)
    need(sys.flags.isolated and sys.dont_write_bytecode and args.output.is_absolute()
         and args.output.parent.resolve(strict=True) == args.output.parent
         and not args.output.exists() and not args.output.is_symlink(),"Isolated no-bytecode and fresh canonical output")
    pinned(Path(__file__).resolve().with_name("PROTOCOL.md"),args.protocol_sha,64*1024)
    args.run(args)


if __name__ == "__main__":
    main()



