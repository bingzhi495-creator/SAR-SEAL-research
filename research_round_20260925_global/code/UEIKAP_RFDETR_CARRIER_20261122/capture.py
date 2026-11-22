"""Frozen center5202 native RF-DETR forward on original UEIKAP PNGs, with label-free readback."""
import argparse
from contextlib import ExitStack
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


ROSTER_SHA = "1274563170a1dd6a7bc4326cf8d14a110d184d3d1687ec9bee22b6c5265be77f"
LOCK_SHA = "3ceb843a570028ccd2215cb5a211915b5ed80f82640786dffbf8cb03897da6e6"
LOCK_COMMIT = "7a7fdf39525e4b27cccdd8dba1528e5c5bb82a5e"
CARRIER_SHA = "97449f58d1ad1a62c52c6f100fc872690e51d465e85e241660befba7a9690a71"
PRED = "PREDICTIONS_PRIVATE.jsonl"
AUDIT = "CANDIDATE_AUDIT_PRIVATE.jsonl"
CAPTURE = "UEIKAP_RFDETR_CENTER5202_108_CAPTURE_COMPLETE_V1"
VERIFIED = "UEIKAP_RFDETR_CENTER5202_108_INDEPENDENT_READBACK_V1"


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def digest(raw):
    return sha256(raw).hexdigest()


def encoded(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def pinned(path, sha):
    data = path.read_bytes()
    need(digest(data) == sha, "Exact digest for " + path.name)
    return data


def write(path, raw):
    with path.open("xb") as target:
        need(target.write(raw) == len(raw), "Complete write")
        target.flush()
        os.fsync(target.fileno())
    return digest(raw)


def write_array(root, name, array):
    import numpy as np
    out = BytesIO()
    np.save(out, array, allow_pickle=False)
    raw = out.getvalue()
    return dict(path=name, sha256=write(root/name, raw), bytes=len(raw))


def read_array(root, ref):
    import numpy as np
    path = (root/ref["path"]).resolve(strict=True)
    need(path.is_relative_to(root) and path.is_file(), "Capture-local array")
    raw = pinned(path, ref["sha256"])
    stream = BytesIO(raw)
    result = np.load(stream, allow_pickle=False)
    need(ref["bytes"] == len(raw) and not stream.read(1) and result.dtype == np.float32
         and np.isfinite(result).all(), "Exact finite FP32 plane block")
    return result


def decode_image(a, item):
    import numpy as np
    from rfdetr.datasets.io_utils import decode_pil_image_bytes
    from PIL import Image
    ref = item["image_ref"]
    num = item["image_id"]
    need(ref["path"] == f"images/{num:06d}.png" and item["original_hw"] == [512,512], "Original chip roster")
    png = pinned(a.intake/ref["path"], ref["sha256"])
    need(len(png) == ref["bytes"], "Exact PNG size")
    img, scale = decode_pil_image_bytes(png, draft_size=None)
    with Image.open(BytesIO(png)) as independent:
        need(independent.format == "PNG" and independent.size == (512,512) and
             np.array_equal(np.asarray(independent.convert("RGB")), np.asarray(img)), "Independent native decode")
    need(img.mode == "RGB" and img.size == (512,512) and scale == (1.,1.), "Unscaled RGB")
    return img, digest(np.asarray(img).tobytes())


def exact_source(a):
    raw = pinned(a.carrier, CARRIER_SHA)
    source = ModuleType("_closed_native_rf_carrier")
    source.__file__ = str(a.carrier)
    exec(compile(raw, str(a.carrier), "exec"), source.__dict__)
    return source


def install_guard(a):
    def guard(event, args):
        if event in ("socket.connect", "socket.getaddrinfo", "socket.gethostbyname"):
            raise PermissionError("Offline inference")
        if event == "open" and isinstance(args[0], (str, bytes)):
            p = Path(os.fsdecode(args[0]))
            if (p.name.lower().endswith(".coco.json") or p.name.lower() == "labels_roster.json"
                    or any(v.lower() in ("labels","annotations") for v in p.parts)
                    or any(v.lower() in ("dev","test","valid","train") for v in p.parts) and
                    p.suffix.lower() in (".jpg",".png",".json",".zip",".tif")):
                raise PermissionError("Unadmitted image or label file")
    sys.addaudithook(guard)


def capture(a, items, torch):
    import numpy as np
    from pycocotools import mask as codec
    from rfdetr.models.postprocess import PostProcess
    from rfdetr.utilities.tensors import nested_tensor_from_tensor_list
    source = exact_source(a)
    core = source.load_core(a.private_root)
    args = SimpleNamespace(root=a.private_root, output=a.output, mode="dev")
    base, post = source.setup(args, core)
    need(torch.cuda.is_available() and torch.cuda.device_count() == 1, "One available CUDA device")
    free, total = torch.cuda.mem_get_info()
    need(free >= base.CAP, "Original native CUDA memory floor")
    torch.cuda.set_per_process_memory_fraction(min(.35, base.CAP / total))
    torch.backends.cudnn.allow_tf32 = False
    model, transform, config = source.model_and_transform(args, core, base, torch)
    need(not model.training and model.num_queries == 164 and model.group_detr == 13, "Exact source Q164")
    a.output.mkdir()
    (a.output/"planes").mkdir()
    counts = dict(mapped=0, eligible=0, overrides=0, rank_gaps=0)
    with ExitStack() as context:
        predictions = context.enter_context((a.output/PRED).open("xb"))
        audits = context.enter_context((a.output/AUDIT).open("xb"))
        context.enter_context(torch.inference_mode())
        for item in items:
            img, rgb_sha = decode_image(a, item)
            num = item["image_id"]
            hw = tuple(item["original_hw"])
            tensor, target = transform(img, None)
            need(target is None and tensor.shape == (3,576,576) and tensor.dtype == torch.float32, "Stock val576")
            batch = nested_tensor_from_tensor_list([tensor], block_size=config.patch_size * config.num_windows)
            need(batch.mask is not None and not bool(batch.mask.any()), "No stock padding")
            raw = model(batch.to("cuda"))
            query = post.aligned_query_rows(raw, hw, torch=torch, np=np, codec=codec, processor=PostProcess)
            stock = source.native_block(num, query)
            ranks, ids = query["native_ranks"], query["native_ids"]
            pred_area = codec.area(query["all_rles"])
            boxes = raw["pred_boxes"][0].cpu()
            cx,cy,w,h = boxes.unbind(-1)
            xyxy = (torch.stack((cx-.5*w,cy-.5*h,cx+.5*w,cy+.5*h), -1)*hw[0]).clamp(0, hw[0])
            candidates = [dict(position=i, native_rank=r, query_id=q, score=query["ship_scores"][q],
                               mask_area=int(pred_area[q]), box_xyxy=xyxy[q].tolist())
                          for i,(r,q) in enumerate(zip(ranks, ids, strict=True), 1)]
            eligible = [c for c in candidates if 32 <= c["mask_area"] < 1024]
            indices = torch.tensor([c["query_id"] for c in eligible], dtype=torch.long, device="cuda")
            selected = raw["pred_masks"][0].index_select(0, indices).contiguous()
            cpu = selected.cpu()
            ref = write_array(a.output, f"planes/{num:06d}.npy", cpu.numpy())
            overrides = []
            for i,c in enumerate(eligible):
                stock_mask = codec.decode(query["all_rles"][c["query_id"]]).astype(bool)
                kwargs = dict(size=hw, mode="bilinear", align_corners=False)
                gpu_raster = torch.nn.functional.interpolate(selected[i:i+1,None], **kwargs)[0,0]
                cpu_raster = torch.nn.functional.interpolate(cpu[i:i+1,None], **kwargs)[0,0]
                need(np.array_equal(stock_mask, (gpu_raster>0).cpu().numpy()), "GPU strict-sign stock parity")
                if not np.array_equal(stock_mask, (cpu_raster>0).numpy()):
                    override = write_array(a.output, f"planes/{num:06d}_r{c['native_rank']:03d}_p.npy",
                                           gpu_raster.sigmoid().cpu().numpy())
                    overrides.append(dict(native_rank=c["native_rank"], query_id=c["query_id"], **override))
            row = dict(image_id=num, original_hw=list(hw), source_image_name=f"{num:06d}.png",
                       image_sha256=item["image_ref"]["sha256"], rgb_sha256=rgb_sha,
                       candidates=candidates, native_ranks=ranks, query_ids=ids, native_boxes_xyxy=xyxy[ids].tolist(),
                       logit_ranks=[c["native_rank"] for c in eligible],
                       logit_query_ids=[c["query_id"] for c in eligible], logits_ref=ref, posterior_overrides=overrides,
                       raw_query_probabilities=raw["pred_logits"][0].sigmoid().cpu().tolist(),
                       raw_boxes_cxcywh=boxes.tolist())
            predictions.write(encoded(stock))
            audits.write(encoded(row))
            counts["mapped"] += len(candidates)
            counts["eligible"] += len(eligible)
            counts["overrides"] += len(overrides)
            counts["rank_gaps"] += 100-len(candidates)
            if num % 12 == 0:
                print(json.dumps(dict(images=num, **counts)), flush=True)
        for stream in (predictions, audits):
            stream.flush()
            os.fsync(stream.fileno())
    return dict(status=CAPTURE, counts=counts, checkpoint_sha256=source.CKPT, carrier_sha256=CARRIER_SHA,
                predictions_sha256=digest((a.output/PRED).read_bytes()), audit_sha256=digest((a.output/AUDIT).read_bytes()),
                model_forward_calls=len(items), cuda_peak_bytes=torch.cuda.max_memory_allocated(), cuda_admission_free=free)


def verify(a, items, torch):
    import numpy as np
    from pycocotools import mask as codec
    terminal = json.loads((a.output.parent/"observed_capture"/"TERMINAL.json").read_bytes())
    cap = a.output.parent/"capture"
    result = json.loads(pinned(cap/"RESULT_SAFE.json", terminal["result_sha256"]))
    need(terminal["actual_exit"] == 0 and terminal["output_footer_verified"] is True and result["status"] == CAPTURE
         and result["worker_sha256"] == terminal["worker_sha256"] == digest(Path(__file__).read_bytes())
         and result["protocol_sha256"] == LOCK_SHA and result["intake_sha256"] == ROSTER_SHA, "Closed capture")
    stocks = [json.loads(x) for x in pinned(cap/PRED, result["predictions_sha256"]).splitlines()]
    rows = [json.loads(x) for x in pinned(cap/AUDIT, result["audit_sha256"]).splitlines()]
    counts = dict(mapped=0, eligible=0, overrides=0, rank_gaps=0)
    for item, pred, row in zip(items, stocks, rows, strict=True):
        _, rgb = decode_image(a, item)
        num = item["image_id"]
        need(num == row["image_id"] == pred["image_id"] and rgb == row["rgb_sha256"] and row["original_hw"] == [512,512]
             and row["image_sha256"] == item["image_ref"]["sha256"] and row["source_image_name"] == f"{num:06d}.png",
             "Original decoded capture provenance")
        probs = np.asarray(row["raw_query_probabilities"], dtype=np.float32)
        need(probs.shape == (164,2) and np.isfinite(probs).all() and np.all((0<=probs)&(probs<=1)), "Joint source scores")
        order = np.argsort(-probs.ravel(), kind="stable")[:100]
        retained = [(r, int(k)//2) for r,k in enumerate(order,1) if k%2 == 0]
        boxes = np.asarray(row["raw_boxes_cxcywh"], dtype=np.float32)
        low = boxes[:,:2] - .5*boxes[:,2:]
        high = boxes[:,:2] + .5*boxes[:,2:]
        clipped = (np.concatenate((low,high),axis=1)*512).clip(0,512)
        ranks, ids = [r for r,q in retained], [q for r,q in retained]
        need(pred["native_ranks"] == row["native_ranks"] == ranks and row["query_ids"] == ids
             and row["native_boxes_xyxy"] == clipped[ids].tolist() and pred["raw_query_count"] == 164 and
             pred["native_count"] == 100 and pred["unmapped_slot1_count"] == 100-len(ranks), "Independent native rank gaps")
        pool = []
        for index, ((r,q), co, candidate) in enumerate(zip(retained, pred["coco"], row["candidates"], strict=True),1):
            need(co["image_id"] == num and co["category_id"] == 1 and co["segmentation"]["size"] == [512,512]
                 and co["score"] == float(probs[q,0]), "Mapped COCO identity and original score")
            area = int(codec.area(co["segmentation"]))
            need(candidate == dict(position=index,native_rank=r,query_id=q,mask_area=area,score=co["score"],
                                   box_xyxy=clipped[q].tolist()), "Exact position, score, ranked query and box")
            if 32 <= area < 1024:
                pool.append((r,q,co))
        need(row["logit_ranks"] == [x[0] for x in pool] and row["logit_query_ids"] == [x[1] for x in pool],
             "Independent eligible census")
        values = read_array(cap, row["logits_ref"])
        need(values.shape == (len(pool),144,144), "Original native logit plane layout")
        overrides = {v["native_rank"]:v for v in row["posterior_overrides"]}
        need(len(overrides) == len(row["posterior_overrides"]), "No duplicate overrides")
        for i,(r,q,co) in enumerate(pool):
            mask = codec.decode(co["segmentation"]).astype(bool)
            raster = torch.nn.functional.interpolate(torch.from_numpy(values[i:i+1,None]),size=mask.shape,
                                                      mode="bilinear",align_corners=False)[0,0]
            if not np.array_equal((raster>0).numpy(), mask):
                ov = overrides.pop(r)
                prob = read_array(cap, ov)
                need(ov["query_id"] == q and prob.shape == mask.shape and np.all((0<=prob)&(prob<=1))
                     and not np.any((prob>.5)&~mask) and not np.any((prob<.5)&mask), "Strict native posterior override")
            else:
                need(r not in overrides, "No unneeded posterior override")
        need(not overrides, "Exactly the needed original GPU posterior planes")
        counts["mapped"] += len(retained)
        counts["eligible"] += len(pool)
        counts["overrides"] += len(row["posterior_overrides"])
        counts["rank_gaps"] += 100-len(retained)
    need(counts == result["counts"] and not torch.cuda.is_initialized(), "Independent aggregate and CPU-only execution")
    a.output.mkdir()
    return dict(status=VERIFIED, counts=counts, model_forward_calls=0, capture_sha256=terminal["result_sha256"],
                predictions_sha256=result["predictions_sha256"], audit_sha256=result["audit_sha256"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase",choices=("capture","verify-capture"))
    for name in ("output","intake","protocol","private-root","carrier"):
        p.add_argument("--"+name, type=Path, required=True)
    a = p.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and a.output.is_absolute()
         and not a.output.exists() and a.output.name == a.phase, "Fresh isolated stage")
    pinned(a.protocol, LOCK_SHA)
    roster = json.loads(pinned(a.intake/"images_roster.json", ROSTER_SHA))
    items = roster["items"]
    need(roster["expected_ids"] == list(range(1,109)) == [r["image_id"] for r in items], "Pinned 108 PNGs")
    install_guard(a)
    import torch
    import numpy as np
    need(sys.version_info[:3] == (3,10,13) and torch.__version__ == "2.10.0+cu130" and
         version("rfdetr") == "1.12.0" and version("Pillow") == "12.3.0" and
         version("pycocotools") == "2.0.11" and np.__version__ == "1.26.4", "Exact native runtime")
    torch.set_num_threads(2 if a.phase=="capture" else 1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    result = capture(a,items,torch) if a.phase=="capture" else verify(a,items,torch)
    result.update(images=len(items), worker_sha256=digest(Path(__file__).read_bytes()), protocol_sha256=LOCK_SHA,
                  prelock_commit=LOCK_COMMIT, intake_sha256=ROSTER_SHA, GT_opened=False, AP=False,
                  FIT=False, DEV=False, TEST=True, author_TEST=False, optimizer_updates=0)
    sha = write(a.output/"RESULT_SAFE.json", encoded(result))
    print(json.dumps(dict(status=result["status"], result_sha256=sha)), flush=True)


if __name__ == "__main__":
    main()
