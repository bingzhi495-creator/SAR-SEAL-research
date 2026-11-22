"""Frozen HRSID RF-DETR inference and independent readback on author TEST images."""
import argparse
from contextlib import ExitStack
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
import json
import os
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace

import numpy as np


REFERENCE_SHA = "89c3294d79e1d69530401a4bf08e3400d52fbc6b77bc7967029cbd1fbbdbb116"
SOURCE_CHECKPOINT = "011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11"
PROTOCOL_SHA = "5c0c9410db15710e34fe453271bd3a99a1f1a50eb365bd354f884e40b88c8b85"
PRELOCK_COMMIT = "6665e15ebb2c064661957b51d0ee7334bae1bb6c"
AUTHOR_ROSTER_SHA = "72c4b9af8f8b0bd0df31a3a3f7df6a082390688f0942eed23f106316a2d5d98b"
CAPTURE_STATUS = "INSTANCE_RSDD_TEST740_RFDETR_CAPTURE_COMPLETE_V1"
VERIFY_STATUS = "INSTANCE_RSDD_TEST740_RFDETR_READBACK_COMPLETE_V1"
PRED = "PREDICTIONS_PRIVATE.jsonl"
AUDIT = "CANDIDATE_AUDIT_PRIVATE.jsonl"


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


def digest(raw):
    return sha256(raw).hexdigest()


def checked_digest(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def pinned(path, sha):
    require(checked_digest(sha), "SHA-256 required")
    raw = path.read_bytes()
    require(digest(raw) == sha, "Pinned bytes: " + path.name)
    return raw


def reference(path):
    source = ModuleType("_verified_ueikap_capture")
    source.__file__ = str(path)
    exec(compile(pinned(path, REFERENCE_SHA), str(path), "exec"), source.__dict__)
    return source


def roster(author, manifest, size=740):
    submitted, items = author["images"], manifest["items"]
    ids = [x["id"] for x in submitted]
    require(set(author) == {"images"} and len(ids) == len(set(ids)) == len(items) == size
            and ids == manifest["expected_ids"] == [item["image_id"] for item in items]
            and all(type(x) is int and x >= 0 for x in ids), "Full ordered author test image raster roster")
    names = set()
    for row, item in zip(submitted, items, strict=True):
        require(set(row) == {"id", "file_name", "height", "width"}, "Image-only author roster")
        name = row["file_name"]
        ref = item["image_ref"]
        hw = [row["height"], row["width"]]
        require(re.fullmatch(r"[0-9]+_[0-9]+_[0-9]+\.jpg", name) is not None
                and name not in names and item["source_image_name"] == name
                and ref["path"] == "images/" + name, "Exact unique native image path")
        require(hw == item["original_hw"] and all(type(d) is int and d > 0 for d in hw)
                and checked_digest(ref["sha256"]) and checked_digest(item["rgb_sha256"])
                and type(ref["bytes"]) is int and ref["bytes"] > 0, "Original encoded and decoded image identity")
        require(type(item["acquisition_group"]) is int and 1 <= item["acquisition_group"] <= 125
                and item["sensor"] in {"GF-3", "TerraSAR-X"}, "Published parent acquisition mapping")
        names.add(name)
    return items


def native_xyxy(boxes, hw):
    size = np.asarray(hw[::-1], dtype=np.float32)
    low = np.clip((boxes[:, :2] - 0.5 * boxes[:, 2:]) * size, 0, size)
    high = np.clip((boxes[:, :2] + 0.5 * boxes[:, 2:]) * size, 0, size)
    return np.concatenate((low, high), axis=1)


def decode_native(root, item):
    from PIL import Image
    from rfdetr.datasets.io_utils import decode_pil_image_bytes

    ref = item["image_ref"]
    path = (root / ref["path"]).resolve(strict=True)
    require(path.parent == (root / "images").resolve(strict=True), "Only admitted local JPEGs")
    raw = pinned(path, ref["sha256"])
    image, scale = decode_pil_image_bytes(raw, draft_size=None)
    pix = np.asarray(image)
    h, w = item["original_hw"]
    with Image.open(BytesIO(raw)) as independent:
        require(independent.format == "JPEG" and independent.size == (w, h)
                and np.array_equal(np.asarray(independent.convert("RGB")), pix), "Independent JPEG native decode")
    require(len(raw) == ref["bytes"] and image.mode == "RGB" and pix.dtype == np.uint8
            and pix.shape == (h, w, 3) and scale == (1.0, 1.0)
            and digest(pix.tobytes(order="C")) == item["rgb_sha256"], "Author chip bytes and decoded RGB")
    return image


def install_guard(intake, items):
    image_dir = (intake / "images").resolve(strict=True)
    allowed = {os.path.normcase(os.fspath(image_dir / item["source_image_name"])) for item in items}

    def audit(event, args):
        if event in ("socket.connect", "socket.getaddrinfo", "socket.gethostbyname"):
            raise PermissionError("Frozen offline inference")
        if event != "open" or not isinstance(args[0], (bytes, str)):
            return
        path = Path(os.fsdecode(args[0]))
        parts = {part.casefold() for part in path.parts}
        if parts & {"annotation_coco", "labels", "annotations"} or path.name.casefold().endswith(".coco.json"):
            raise PermissionError("No label files in frozen image stages")
        if path.suffix.casefold() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            if os.path.normcase(os.fspath(path.absolute())) not in allowed:
                raise PermissionError("Image is outside the admitted author test roster")

    sys.addaudithook(audit)


def read_inputs(a):
    require(a.protocol_sha == PROTOCOL_SHA and a.prelock_commit == PRELOCK_COMMIT
            and a.author_roster_sha == AUTHOR_ROSTER_SHA, "Published preregistration and author image-only roster")
    pinned(a.protocol, PROTOCOL_SHA)
    require(a.intake == a.manifest.parent and a.intake == a.intake_result.parent,
            "Manifest and verified intake result share the materialized image root")
    terminal = json.loads(pinned(a.intake_terminal, a.intake_terminal_sha))
    require(terminal["actual_exit"] == 0 and terminal["output_footer_verified"] is True
            and terminal["result_sha256"] == a.intake_result_sha, "Observed complete image intake")
    intake_result = json.loads(pinned(a.intake_result, a.intake_result_sha))
    require(intake_result["status"] == "INSTANCE_RSDD_TEST740_IMAGE_INTAKE_COMPLETE_V1"
            and intake_result["image_manifest_sha256"] == a.manifest_sha
            and intake_result["test_roster_sha256"] == AUTHOR_ROSTER_SHA
            and intake_result["test_images"] == intake_result["member_crc32_verified"] == intake_result["rgb_decode_verified"] == 740
            and intake_result["target_polygon_rows_opened"] == intake_result["optimizer_updates"] == intake_result["model_forwards"] == 0,
            "Verified intake binds the exact image-only population")
    items = roster(json.loads(pinned(a.author_roster, AUTHOR_ROSTER_SHA)),
                   json.loads(pinned(a.manifest, a.manifest_sha)))
    sensors = {s: sum(item["sensor"] == s for item in items) for s in ("GF-3", "TerraSAR-X")}
    require(sensors == intake_result["sensor_counts"] == {"GF-3": 583, "TerraSAR-X": 157}
            and len({x["acquisition_group"] for x in items}) == intake_result["acquisition_groups"] == 112,
            "Prespecified actual-release sensor and acquisition census")
    return items


def count_rows(counts, candidates, eligible, overrides):
    counts["mapped"] += len(candidates)
    counts["eligible"] += len(eligible)
    counts["overrides"] += len(overrides)
    counts["rank_gaps"] += 100 - len(candidates)


def rasterize(planes, size, torch):
    return torch.nn.functional.interpolate(planes[:, None], size=size, mode="bilinear", align_corners=False)[:, 0]


def capture(a, items, torch, common):
    from pycocotools import mask as codec
    from rfdetr.models.postprocess import PostProcess
    from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

    source = common.exact_source(a)
    require(source.CKPT == SOURCE_CHECKPOINT, "Exact frozen HRSID checkpoint")
    core = source.load_core(a.private_root)
    args = SimpleNamespace(root=a.private_root, output=a.output, mode="dev")
    base, post = source.setup(args, core)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "One available CUDA device")
    free, total = torch.cuda.mem_get_info()
    require(free >= base.CAP, "Original native CUDA memory floor")
    torch.cuda.set_per_process_memory_fraction(min(0.35, base.CAP / total))
    torch.backends.cudnn.allow_tf32 = False
    model, transform, config = source.model_and_transform(args, core, base, torch)
    require(not model.training and model.num_queries == 164 and model.group_detr == 13, "Exact source Q164")
    a.output.mkdir()
    (a.output / "planes").mkdir()
    totals = dict(mapped=0, eligible=0, overrides=0, rank_gaps=0)
    with ExitStack() as ctx:
        predictions = ctx.enter_context((a.output / PRED).open("xb"))
        audit = ctx.enter_context((a.output / AUDIT).open("xb"))
        ctx.enter_context(torch.inference_mode())
        for ordinal, item in enumerate(items, 1):
            image = decode_native(a.intake, item)
            num, hw = item["image_id"], tuple(item["original_hw"])
            tensor, target = transform(image, None)
            require(target is None and tensor.shape == (3, 576, 576) and tensor.dtype == torch.float32, "Native val576")
            batch = nested_tensor_from_tensor_list([tensor], block_size=config.patch_size * config.num_windows)
            require(batch.mask is not None and not bool(batch.mask.any()), "Identical author tensor padding")
            raw = model(batch.to("cuda"))
            query = post.aligned_query_rows(raw, hw, torch=torch, np=np, codec=codec, processor=PostProcess)
            prediction = source.native_block(num, query)
            ranks, ids = query["native_ranks"], query["native_ids"]
            areas = codec.area(query["all_rles"])
            boxes = raw["pred_boxes"][0].cpu().numpy()
            xyxy = native_xyxy(boxes, hw)
            candidates = [dict(position=i, native_rank=r, query_id=q, score=query["ship_scores"][q],
                               mask_area=int(areas[q]), box_xyxy=xyxy[q].tolist())
                          for i, (r, q) in enumerate(zip(ranks, ids, strict=True), 1)]
            eligible = [c for c in candidates if 32 <= c["mask_area"] < 1024]
            indices = torch.tensor([c["query_id"] for c in eligible], dtype=torch.long, device="cuda")
            selected = raw["pred_masks"][0].index_select(0, indices).contiguous()
            cpu = selected.cpu()
            require(cpu.shape == (len(eligible), 144, 144) and cpu.dtype == torch.float32, "Native logit plane shape")
            ref = common.write_array(a.output, f"planes/{ordinal:06d}.npy", cpu.numpy())
            overrides = []
            for start in range(0, len(eligible), 8):
                stop = start + 8
                gpu = rasterize(selected[start:stop], hw, torch)
                native_sign = (gpu > 0).cpu().numpy()
                cpu_sign = (rasterize(cpu[start:stop], hw, torch) > 0).numpy()
                indices_block = eligible[start:stop]
                stock = codec.decode([query["all_rles"][c["query_id"]] for c in indices_block]).transpose(2, 0, 1).astype(bool)
                require(np.array_equal(stock, native_sign), "GPU strict-sign stock parity")
                for i, c in enumerate(indices_block):
                    if not np.array_equal(stock[i], cpu_sign[i]):
                        override = common.write_array(a.output, f"planes/{ordinal:06d}_r{c['native_rank']:03d}_p.npy",
                                                      gpu[i].sigmoid().cpu().numpy())
                        overrides.append(dict(native_rank=c["native_rank"], query_id=c["query_id"], **override))
            row = dict(image_id=num, original_hw=list(hw), source_image_name=item["source_image_name"],
                       image_sha256=item["image_ref"]["sha256"], rgb_sha256=item["rgb_sha256"],
                       acquisition_group=item["acquisition_group"], sensor=item["sensor"],
                       candidates=candidates, native_ranks=ranks, query_ids=ids, native_boxes_xyxy=xyxy[ids].tolist(),
                       logit_ranks=[c["native_rank"] for c in eligible], logit_query_ids=[c["query_id"] for c in eligible],
                       logits_ref=ref, posterior_overrides=overrides,
                       raw_query_probabilities=raw["pred_logits"][0].sigmoid().cpu().tolist(), raw_boxes_cxcywh=boxes.tolist())
            predictions.write(common.encoded(prediction))
            audit.write(common.encoded(row))
            count_rows(totals, candidates, eligible, overrides)
            if ordinal % 20 == 0:
                print(json.dumps(dict(images=ordinal, **totals)), flush=True)
        for stream in (predictions, audit):
            stream.flush()
            os.fsync(stream.fileno())
    return dict(status=CAPTURE_STATUS, counts=totals, model_forward_calls=len(items),
                predictions_sha256=digest((a.output / PRED).read_bytes()), audit_sha256=digest((a.output / AUDIT).read_bytes()),
                cuda_peak_bytes=torch.cuda.max_memory_allocated(), cuda_admission_free=free)


def verify(a, items, torch, common):
    from pycocotools import mask as codec

    terminal = json.loads((a.output.parent / "observed_capture" / "TERMINAL.json").read_bytes())
    cap = a.output.parent / "capture"
    result = json.loads(pinned(cap / "RESULT_SAFE.json", terminal["result_sha256"]))
    require(terminal["actual_exit"] == 0 and terminal["output_footer_verified"] is True and result["status"] == CAPTURE_STATUS
            and result["worker_sha256"] == terminal["worker_sha256"] == digest(Path(__file__).read_bytes())
            and result["protocol_sha256"] == a.protocol_sha and result["intake_sha256"] == a.manifest_sha
            and result["model_forward_calls"] == len(items), "Observed closed label-free capture")
    predictions = [json.loads(x) for x in pinned(cap / PRED, result["predictions_sha256"]).splitlines()]
    rows = [json.loads(x) for x in pinned(cap / AUDIT, result["audit_sha256"]).splitlines()]
    totals = dict(mapped=0, eligible=0, overrides=0, rank_gaps=0)
    for item, pred, row in zip(items, predictions, rows, strict=True):
        decode_native(a.intake, item)
        num, hw = item["image_id"], item["original_hw"]
        require(num == row["image_id"] == pred["image_id"] and row["original_hw"] == hw
                and row["rgb_sha256"] == item["rgb_sha256"] and row["image_sha256"] == item["image_ref"]["sha256"]
                and all(row[k] == item[k] for k in ("source_image_name", "sensor", "acquisition_group")), "Exact image provenance")
        probs = np.asarray(row["raw_query_probabilities"], dtype=np.float32)
        boxes = np.asarray(row["raw_boxes_cxcywh"], dtype=np.float32)
        require(probs.shape == (164, 2) and np.isfinite(probs).all() and np.all((0 <= probs) & (probs <= 1))
                and boxes.shape == (164, 4) and np.isfinite(boxes).all(), "Original query probabilities and boxes")
        indices = np.argsort(-probs.ravel(), kind="stable")[:100]
        retained = [(r, int(k) // 2) for r, k in enumerate(indices, 1) if k % 2 == 0]
        ranks, ids = [r for r, _ in retained], [q for _, q in retained]
        xyxy = native_xyxy(boxes, hw)
        require(pred["native_ranks"] == row["native_ranks"] == ranks and row["query_ids"] == ids
                and row["native_boxes_xyxy"] == xyxy[ids].tolist() and pred["raw_query_count"] == 164
                and pred["native_count"] == 100 and pred["unmapped_slot1_count"] == 100 - len(ranks), "Independent joint ranks")
        pool = []
        for pos, ((r, q), co, cand) in enumerate(zip(retained, pred["coco"], row["candidates"], strict=True), 1):
            require(co["image_id"] == num and co["category_id"] == 1 and co["segmentation"]["size"] == hw
                    and co["score"] == float(probs[q, 0]), "Native COCO score and identity")
            area = int(codec.area(co["segmentation"]))
            require(cand == dict(position=pos, native_rank=r, query_id=q, mask_area=area, score=co["score"],
                                 box_xyxy=xyxy[q].tolist()), "Exact candidate audit")
            if 32 <= area < 1024:
                pool.append((r, q, co["segmentation"]))
        require(row["logit_ranks"] == [x[0] for x in pool] and row["logit_query_ids"] == [x[1] for x in pool], "Eligible native plane identity")
        values = common.read_array(cap, row["logits_ref"])
        require(values.shape == (len(pool), 144, 144), "Exact native plane layout")
        overrides = {v["native_rank"]: v for v in row["posterior_overrides"]}
        require(len(overrides) == len(row["posterior_overrides"]), "Unique native override")
        for i, (rank, q, rle) in enumerate(pool):
            stock = codec.decode(rle).astype(bool)
            reconstruct = rasterize(torch.from_numpy(values[i:i+1]), hw, torch)[0].numpy() > 0
            if not np.array_equal(stock, reconstruct):
                ov = overrides.pop(rank)
                prob = common.read_array(cap, ov)
                require(ov["query_id"] == q and prob.shape == stock.shape and np.all((0 <= prob) & (prob <= 1))
                        and not np.any((prob > .5) & ~stock) and not np.any((prob < .5) & stock), "Exact CUDA posterior fallback")
            else:
                require(rank not in overrides, "Unnecessary CUDA posterior fallback")
        require(not overrides, "No unconsumed native posterior fallbacks")
        count_rows(totals, retained, pool, row["posterior_overrides"])
    require(totals == result["counts"] and not torch.cuda.is_initialized(), "Complete independent CPU readback")
    a.output.mkdir()
    return dict(status=VERIFY_STATUS, counts=totals, model_forward_calls=0, capture_sha256=terminal["result_sha256"],
                predictions_sha256=result["predictions_sha256"], audit_sha256=result["audit_sha256"], exact_native_parity=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("phase", choices=("capture", "verify-capture"))
    for name in ("protocol", "author-roster", "intake", "manifest", "intake-result", "intake-terminal", "private-root", "carrier", "reference", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("protocol-sha", "prelock-commit", "author-roster-sha", "manifest-sha", "intake-result-sha", "intake-terminal-sha"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    require(sys.flags.isolated and sys.dont_write_bytecode and a.output.is_absolute() and not a.output.exists()
            and a.output.name == a.phase and a.intake.is_absolute(), "Fresh isolated stage")
    items = read_inputs(a)
    common = reference(a.reference)
    install_guard(a.intake, items)
    import torch
    require(sys.version_info[:3] == (3, 10, 13) and torch.__version__ == "2.10.0+cu130" and version("rfdetr") == "1.12.0"
            and version("Pillow") == "12.3.0" and version("pycocotools") == "2.0.11" and np.__version__ == "1.26.4", "Exact native runtime")
    torch.set_num_threads(2 if a.phase == "capture" else 1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    result = capture(a, items, torch, common) if a.phase == "capture" else verify(a, items, torch, common)
    result.update(images=len(items), worker_sha256=digest(Path(__file__).read_bytes()), protocol_sha256=a.protocol_sha,
                  prelock_commit=a.prelock_commit, intake_sha256=a.manifest_sha, intake_result_sha256=a.intake_result_sha,
                  intake_terminal_sha256=a.intake_terminal_sha,
                  author_roster_sha256=a.author_roster_sha, checkpoint_sha256=SOURCE_CHECKPOINT, carrier_sha256=common.CARRIER_SHA,
                  GT_opened=False, AP=False, FIT=False, DEV=False, TEST=True, author_TEST=True, optimizer_updates=0)
    sha = common.write(a.output / "RESULT_SAFE.json", common.encoded(result))
    print(json.dumps(dict(status=result["status"], result_sha256=sha)), flush=True)


if __name__ == "__main__":
    main()
