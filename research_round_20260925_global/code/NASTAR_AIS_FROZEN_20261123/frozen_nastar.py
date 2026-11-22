"""Frozen RF-DETR capture and independent physical writeback on NASTaR chips."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

PORTABLE = "45512c959eb914ae7713264ecff48eb437a2a52729810b34308abc52079148df"
PROTOCOL = "54c4094d79ade98f06b7147608eb84c66439e722ef13f59bc0f2c8793f1231f9"
ROSTER = "8782fde699b60e8ae4e2db70d2f59937d58d4279c0a7ddb6ce34db68b2ab27dc"
CAPTURE = "NASTAR_FROZEN_CAPTURE_COMPLETE"
REPLAY = "NASTAR_FROZEN_REPLAY_COMPLETE"


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


def digest(data):
    return sha256(data).hexdigest()


def pin(path, expected):
    data = path.read_bytes()
    require(digest(data) == expected, f"Changed input {path.name}")
    return data


def encode(payload):
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def portable_source(code):
    path = code / "INSTANCE_RSDD_B5_REPRO_20261123/portable_eval.py"
    operator = ModuleType("_nastar_portable_detector")
    operator.__file__ = str(path)
    exec(compile(pin(path, PORTABLE), str(path), "exec"), operator.__dict__)
    return operator


def load_items(args):
    from PIL import Image

    pin(args.protocol, PROTOCOL)
    lock = json.loads(pin(args.intake / "cohort_lock.json", ROSTER))
    manifest = json.loads(pin(args.intake / "intake.json", args.intake_sha256))
    require(manifest["lock_sha256"] == ROSTER and manifest["expected"] == manifest["ready"] == 142
            and manifest["failed"] == 0 and manifest["ready_passes"] == 24,
            "Original locked image intake must be fully observed")
    items = []
    for row, selected in zip(manifest["items"], lock["items"], strict=True):
        image_id = row["image_id"]
        name = f"{image_id:06d}.png"
        require(row["status"] == "ready" and row["image_path"] == "images/" + name
                and all(row[k] == selected[k] for k in ("image_id", "pass_id", "product", "patch_number")),
                "Submitted image must belong to the locked official raster cohort")
        raw = pin(args.intake / row["image_path"], row["png_sha256"])
        with Image.open(BytesIO(raw)) as image:
            require(image.format == "PNG" and image.mode == "L" and image.size == (512,512)
                    and digest(image.tobytes()) == row["decoded_sha256"], "Exact native uint8 grayscale canvas")
            rgb = digest(image.convert("RGB").tobytes())
        items.append(dict(image_id=image_id, source_image_name=name, acquisition_group=row["pass_id"], sensor="NovaSAR",
                          original_hw=[512,512], rgb_sha256=rgb,
                          image_ref=dict(path=row["image_path"], sha256=row["png_sha256"], bytes=len(raw))))
    require([r["image_id"] for r in items] == list(range(1,143)), "Complete ordered set of locked chips")
    return items


def read_rgb(directory, item):
    from PIL import Image
    from rfdetr.datasets.io_utils import decode_pil_image_bytes

    ref = item["image_ref"]
    raw = pin(directory / ref["path"], ref["sha256"])
    image, scale = decode_pil_image_bytes(raw, draft_size=None)
    with Image.open(BytesIO(raw)) as first:
        require(image.mode == "RGB" and image.size == (512,512) and scale == (1.,1.)
                and image.tobytes() == first.convert("RGB").tobytes()
                and digest(image.tobytes()) == item["rgb_sha256"],
                "Independent lossless grayscale replication with the original decoder")
    return image


def capture(args, items, operator):
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    from rfdetr.models.postprocess import PostProcess

    require(torch.get_float32_matmul_precision() == "high" and torch.backends.cuda.matmul.allow_tf32,
            "Exact original CUDA matmul import policy")
    native = operator.module(args.code / operator.NATIVE / "capture_instance_rsdd.py", operator.CAPTURE_SOURCE)
    native.install_guard(args.intake, items)
    native.decode_native = read_rgb
    carrier, core, base, post = operator.detector_components(args.code, args.checkpoint)
    carrier.load_core = lambda _: core
    carrier.setup = lambda a, prior: (base, post)
    common = native.reference(args.code / operator.UEIKAP / "capture.py")
    common.exact_source = lambda _: carrier
    return native.capture(SimpleNamespace(intake=args.intake, output=args.output, private_root=args.code),
                          items, torch, common)


def verify_capture(args, items, operator):
    import numpy as np
    from pycocotools import mask as codec

    terminal = json.loads(args.capture_terminal.read_bytes())
    raw = pin(args.capture / "RESULT_SAFE.json", terminal["result_sha256"])
    cap = json.loads(raw)
    require(terminal["actual_exit"] == 0 and terminal["output_footer_verified"] is True
            and terminal["worker_sha256"] == cap["worker_sha256"] and cap["status"] == CAPTURE
            and cap["manifest_sha256"] == args.intake_sha256 and cap["protocol_sha256"] == PROTOCOL
            and cap["lock_sha256"] == ROSTER and cap["checkpoint_sha256"] == operator.CHECKPOINT
            and cap["target_labels_opened"] is False and cap["model_forward_calls"] == len(items)
            and cap["optimizer_updates"] == 0,
            "Observed successful frozen label-free capture")
    preds = [json.loads(line) for line in pin(args.capture / operator.PREDICTIONS, cap["predictions_sha256"]).splitlines()]
    audits = [json.loads(line) for line in pin(args.capture / operator.AUDIT, cap["audit_sha256"]).splitlines()]
    native = operator.module(args.code / operator.NATIVE / "capture_instance_rsdd.py", operator.CAPTURE_SOURCE)
    totals = dict(mapped=0, eligible=0, overrides=0, rank_gaps=0)
    for item, row, pred in zip(items, audits, preds, strict=True):
        require(item["image_id"] == pred["image_id"] == row["image_id"] and row["image_sha256"] == item["image_ref"]["sha256"]
                and all(row[k] == item[k] for k in ("acquisition_group","sensor","source_image_name","rgb_sha256","original_hw")),
                "Image and native stream identities")
        probs = np.asarray(row["raw_query_probabilities"], dtype=np.float32)
        boxes = np.asarray(row["raw_boxes_cxcywh"], dtype=np.float32)
        require(probs.shape == (164,2) and ((0 <= probs) & (probs <= 1)).all()
                and boxes.shape == (164,4) and np.isfinite(boxes).all(), "Archived full original queries")
        keep = [(i,int(k)//2) for i,k in enumerate(np.argsort(-probs.ravel(),kind="stable")[:100],1) if k%2 == 0]
        ranks, queries = [r for r,q in keep], [q for r,q in keep]
        xyxy = native.native_xyxy(boxes, item["original_hw"])[queries].tolist()
        require(ranks == pred["native_ranks"] == row["native_ranks"] and queries == row["query_ids"]
                and xyxy == row["native_boxes_xyxy"] and pred["native_count"] == 100
                and pred["unmapped_slot1_count"] == 100-len(keep), "Original top-100 ship query maps")
        eligible = []
        for pos, ((rank,query), box, c, stock) in enumerate(zip(keep,xyxy,row["candidates"],pred["coco"],strict=True),1):
            area = int(codec.area(stock["segmentation"]))
            require(stock["image_id"] == item["image_id"] and stock["category_id"] == 1
                    and stock["segmentation"]["size"] == [512,512]
                    and stock["score"] == float(probs[query,0]) and
                    c == dict(position=pos,native_rank=rank,query_id=query,score=stock["score"],mask_area=area,box_xyxy=box),
                    "Exact native RLE, candidate score and linked query")
            if 32 <= area < 1024:
                eligible.append((rank,query))
        overrides = [(x["native_rank"],x["query_id"]) for x in row["posterior_overrides"]]
        require(eligible == list(zip(row["logit_ranks"],row["logit_query_ids"],strict=True))
                and len(overrides) == len(set(overrides)) and set(overrides) <= set(eligible),
                "One original plane per eligible candidate and only genuine overrides")
        native.count_rows(totals, keep, eligible, overrides)
    require(totals == cap["counts"], "Closed candidate population")
    return cap, digest(raw), list(zip(audits,preds,strict=True))


def initialize(args, items):
    global worker
    worker = portable_source(args.code)
    worker.initialize_replay(args.code, args.intake, items, args.capture, args.point_head)


def replay_image(pair):
    return worker.replay_both(pair)


def replay(args, items, operator):
    cap, cap_sha, population = verify_capture(args, items, operator)
    args.output.mkdir()
    total, sha = Counter(), sha256()
    with (args.output / operator.EDITS).open("xb") as out, ProcessPoolExecutor(
            max_workers=args.workers, initializer=initialize, initargs=(args,items)) as pool:
        for idx,(row,counts) in enumerate(pool.map(replay_image,population,chunksize=2),1):
            require(row["image_id"] == items[idx-1]["image_id"], "Stable writeback order")
            line = encode(row)
            out.write(line)
            sha.update(line)
            total.update(counts)
            if idx==1 or idx%20==0:
                print(json.dumps(dict(images=idx, phase="replay")), flush=True)
    require(total["mapped_candidates"] == cap["counts"]["mapped"] and total["images"] == len(items)
            and total["fixed_swaps"] == total["free_matched_count_swaps"], "Verified full same-count replay")
    return dict(capture_result_sha256=cap_sha, predictions_sha256=cap["predictions_sha256"], audit_sha256=cap["audit_sha256"],
                edits_sha256=sha.hexdigest(), counts=dict(total), independent_decisions_equal=True,
                point_head_sha256=operator.POINT_HEAD, model_forward_calls=0)


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = p.add_subparsers(dest="action",required=True)
    for name in ("capture","replay"):
        sub = commands.add_parser(name)
        for key in ("code","intake","protocol","output"):
            sub.add_argument("--"+key, type=Path, required=True)
        sub.add_argument("--intake-sha256", required=True)
        if name=="capture":
            sub.add_argument("--checkpoint",type=Path,required=True)
        else:
            for key in ("capture","capture_terminal","point_head"):
                sub.add_argument("--"+key.replace("_","-"), type=Path, required=True)
            sub.add_argument("--workers",type=int,choices=range(1,9),default=4)
    args = p.parse_args()
    for key,value in vars(args).items():
        if isinstance(value,Path):
            setattr(args,key,value.resolve(strict=key!="output"))
    require(not args.output.exists() and sys.version_info[:3]==(3,10,13), "Fresh output and qualified CPython")
    if args.action=="replay":
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    start=perf_counter()
    operator=portable_source(args.code)
    items=load_items(args)
    result = capture(args,items,operator) if args.action=="capture" else replay(args,items,operator)
    result.update(status=CAPTURE if args.action=="capture" else REPLAY, images=len(items),
                  manifest_sha256=args.intake_sha256, lock_sha256=ROSTER, protocol_sha256=PROTOCOL,
                  checkpoint_sha256=operator.CHECKPOINT, target_labels_opened=False, optimizer_updates=0,
                  worker_sha256=digest(Path(__file__).read_bytes()), wall_seconds=perf_counter()-start)
    raw=encode(result)
    (args.output / "RESULT_SAFE.json").write_bytes(raw)
    print(json.dumps(dict(status=result["status"],result_sha256=digest(raw))),flush=True)


if __name__=="__main__":
    main()
