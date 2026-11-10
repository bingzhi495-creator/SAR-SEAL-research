"""Export frozen HRSID E24 EMA predictions on a fixed Schwegmann image-only cohort."""

import argparse
from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
from io import BytesIO
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import sys
from time import perf_counter
from types import ModuleType
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile


LOCK_STATUS = "P254_SCHWEGMANN_FROZEN_EXPORT_LOCK_V1"
EXPORT_STATUS = "P254_SCHWEGMANN_FROZEN_EXPORT_COMPLETE_PENDING_INDEPENDENT_ACCEPTANCE_V1"
PREDICTIONS_NAME, REPORT_NAME = "PREDICTIONS_PRIVATE.jsonl", "EXPORT_PRIVATE.json"
P250_SHA = "4acb5fbcaf35c5a32510f5f93ccf51f34e08641fd6d007c78c85becde7496727"
WINDOWS_RUNTIME_SOURCES = {
    "PIL._imaging": "16a7c881fd3624c8428078ce24460861e91fc5cc43de1b3614dc2531fe327e2e",
    "pycocotools._mask": "bcc52682fb8cd20dcc58b09f009566a07d0cf8bd27354aa7f386b5e18d5e5b20",
    "rfdetr._namespace": "7b5ab31bd5a8f99b2588c766a3a8c5193d79fd7326c84f3610282f33143f7280",
    "rfdetr.datasets.transforms": "a4070434e1cfcb50040f1c84abddf13e37fb8f93953c76c035c32058f8b6d2a1",
    "rfdetr.models.heads.segmentation": "5de847861e2bd2fab8853f442e263f4e9c0eb75c5b3fbc6a03bb25e9c7115e1e",
}
IMAGE_STATUS = "P254_SCHWEGMANN_IMAGE_ONLY_V1"
IMAGE_HW = (101, 101)
MAX_IMAGES, MAX_PNG = 2000, 128 * 1024
QUERY_WIDTH, NATIVE_ROWS, UPDATES = 164, 100, 17_640
ROI_XYXY = (40, 40, 61, 61)
SELECTION_RULE = "native_top100_then_slot0_highest_rank_box_center_in_fixed_roi"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def center_row(result, *, image_id, torch, np, codec):
    """Retain the first native slot-0 row whose original box center is in the ROI."""
    scores, labels, boxes, masks = (result[key] for key in ("scores", "labels", "boxes", "masks"))
    require(scores.dtype == boxes.dtype == torch.float32 and labels.dtype == torch.int64
            and masks.dtype == torch.bool and tuple(scores.shape) == tuple(labels.shape) == (NATIVE_ROWS,)
            and tuple(boxes.shape) == (NATIVE_ROWS, 4) and tuple(masks.shape) == (NATIVE_ROWS, 1, *IMAGE_HW),
            "Exact native original-canvas top100 result required")
    scores, labels, boxes = scores.tolist(), labels.tolist(), boxes.tolist()
    require(all(label in (0, 1) for label in labels)
            and all(math.isfinite(score) and 0 <= score <= 1 for score in scores)
            and all(a >= b for a, b in zip(scores, scores[1:]))
            and all(all(map(math.isfinite, box)) and box[0] <= box[2] and box[1] <= box[3] for box in boxes),
            "Expected stable score-ranked native slots and finite xyxy boxes")
    left, top, right, bottom = ROI_XYXY
    eligible = [i for i, (label, (x1, y1, x2, y2)) in enumerate(zip(labels, boxes, strict=True))
                if label == 0 and left <= (x1 + x2) / 2 < right and top <= (y1 + y2) / 2 < bottom]
    candidate = None
    if eligible:
        index = eligible[0]
        full_mask = masks[index, 0].detach().cpu().numpy()
        rle = codec.encode(np.asfortranarray(full_mask, dtype=np.uint8))
        require(rle["size"] == list(IMAGE_HW) and bool(rle["counts"]), "Original-canvas COCO RLE required")
        candidate = dict(native_rank=index + 1, score=scores[index], bbox_xyxy=boxes[index],
                         segmentation=dict(size=list(IMAGE_HW), counts=rle["counts"].decode("ascii")))
    return dict(image_id=image_id, raw_query_count=QUERY_WIDTH, native_count=NATIVE_ROWS,
                mapped_rows=labels.count(0), eligible_count=len(eligible), candidate=candidate)


def native_center_row(raw, *, image_id, torch, postprocess, np, codec):
    """Use stock all-slot top100 with no threshold, then apply center selection."""
    logits, boxes, masks = (raw[key] for key in ("pred_logits", "pred_boxes", "pred_masks"))
    require(tuple(logits.shape) == (1, QUERY_WIDTH, 2) and tuple(boxes.shape) == (1, QUERY_WIDTH, 4)
            and tuple(masks.shape) == (1, QUERY_WIDTH, 144, 144)
            and all(value.dtype == torch.float32 and bool(torch.isfinite(value).all())
                    for value in (logits, boxes, masks)), "Exact finite FP32 Q164 native output required")
    sizes = torch.tensor([IMAGE_HW], dtype=torch.int64, device=logits.device)
    result, = postprocess(
        {"pred_logits": logits, "pred_boxes": boxes, "pred_masks": masks}, sizes, score_threshold=None,
    )
    return center_row(result, image_id=image_id, torch=torch, np=np, codec=codec)


def manifest_items(manifest):
    require(type(manifest) is dict and set(manifest) == {"status", "images"}
            and manifest["status"] == IMAGE_STATUS and type(manifest["images"]) is list
            and 0 < len(manifest["images"]) <= MAX_IMAGES, "Exact bounded image-only manifest required")
    items = manifest["images"]
    for row in items:
        require(type(row) is dict and set(row) == {"image_id", "basename", "bytes", "sha256"}
                and type(row["image_id"]) is int and 0 < row["image_id"] <= 9999
                and row["basename"] == f"chip_{row['image_id']:04d}.png"
                and type(row["bytes"]) is int and 0 < row["bytes"] <= MAX_PNG
                and type(row["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]),
                "Expected positive chip IDs and exact flat PNG references")
    require(len({row["image_id"] for row in items}) == len(items)
            and len({row["sha256"] for row in items}) == len(items),
            "Image-only cohort requires distinct IDs and PNGs")
    return items


def validate_image_zip(archive, items):
    members = archive.infolist()
    require(len(members) == len(items) and {member.filename for member in members}
            == {item["basename"] for item in items}, "Exact image-only PNG ZIP member set required")
    for item in items:
        member = archive.getinfo(item["basename"])
        require(not member.is_dir() and not (member.flag_bits & 1)
                and member.file_size == item["bytes"] and member.compress_type in (ZIP_STORED, ZIP_DEFLATED),
                "Exact bounded image-only PNG ZIP metadata required")


def read_png(archive, item):
    data = archive.read(item["basename"])
    require(len(data) == item["bytes"] and sha256(data).hexdigest() == item["sha256"],
            "Exact locked image-only PNG bytes required")
    return data


def decode_png(data, np):
    from PIL import Image

    require(data.startswith(b"\x89PNG\r\n\x1a\n") and data[12:16] == b"IHDR"
            and data[24:26] == b"\x08\x02", "Exact source RGB8 PNG encoding required")
    with Image.open(BytesIO(data)) as image:
        require(image.format == "PNG" and image.mode == "RGB" and image.size == IMAGE_HW[::-1]
                and image.n_frames == 1 and image.getexif().get(274, 1) == 1,
                "Exact oriented 101x101 single-frame RGB PNG required")
        image.load()
        pixels = np.asarray(image)
        require(pixels.dtype == np.uint8 and pixels.shape == (*IMAGE_HW, 3)
                and np.array_equal(pixels[:, :, 0], pixels[:, :, 1])
                and np.array_equal(pixels[:, :, 0], pixels[:, :, 2]),
                "Locked calibrated gray8 pixels must be replicated across three RGB channels")
        return image.copy()


def infer_png(data, item, *, model, transform, block_size, postprocess, torch, np, codec, nested_tensor):
    image = decode_png(data, np)
    tensor, target = transform(image, None)
    require(target is None and tuple(tensor.shape) == (3, 576, 576) and tensor.dtype == torch.float32
            and tensor.device.type == "cpu" and bool(torch.isfinite(tensor).all()),
            "Exact finite source val576 image-only CPU transform required")
    sample = nested_tensor([tensor], block_size=block_size)
    require(sample.mask is not None and not bool(sample.mask.any()), "Unexpected source profile spatial padding")
    raw = model(sample)
    require(not torch.cuda.is_initialized(), "The CPU worker initialized CUDA")
    return native_center_row(raw, image_id=item["image_id"], torch=torch, postprocess=postprocess, np=np, codec=codec)


def _bounded_bytes(path, digest, limit, *, expected_size=None):
    require(path.is_absolute() and path.resolve(strict=True) == path
            and type(digest) is str and re.fullmatch(r"[0-9a-f]{64}", digest),
            "Expected canonical closed source path and SHA-256")

    def identity(meta):
        return meta.st_dev, meta.st_ino, meta.st_size, meta.st_mtime_ns, meta.st_ctime_ns

    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= limit
                and (expected_size is None or before.st_size == expected_size), "Exact bounded closed source size")
        data = stream.read(before.st_size + 1)
        require(len(data) == before.st_size and sha256(data).hexdigest() == digest
                and identity(before) == identity(os.fstat(stream.fileno())) == identity(path.stat()),
                "Closed source identity or SHA-256 changed")
    return data


def _unique_fields(pairs):
    value = dict(pairs)
    require(len(value) == len(pairs), "Duplicate JSON key in frozen export lock")
    return value


def read_lock(path, digest):
    raw = _bounded_bytes(path, digest, 64 * 1024)
    lock = json.loads(raw, object_pairs_hook=_unique_fields)
    require(type(lock) is dict and type(lock.get("sources")) is dict, "Exact frozen export lock required")
    old_ref = lock["sources"].get("p250_exporter")
    require(type(old_ref) is dict and set(old_ref) == {"path", "bytes", "sha256"}
            and type(old_ref["path"]) is str and type(old_ref["bytes"]) is int
            and 0 < old_ref["bytes"] <= 64 * 1024 and old_ref["sha256"] == P250_SHA,
            "Exact pinned P250 source-admission module reference required")
    old_path = Path(old_ref["path"])
    require(old_path.name == "export_frozen_hrsid_ssdd.py", "Exact P250 module basename required")
    old_bytes = _bounded_bytes(old_path, P250_SHA, 64 * 1024, expected_size=old_ref["bytes"])
    p250 = ModuleType("_p254_pinned_p250_exporter")
    p250.__file__ = str(old_path)
    exec(compile(old_bytes, str(old_path), "exec"), p250.__dict__)

    p250.exact(lock, dict(status=LOCK_STATUS, device="cpu"), "Exact Windows CPU frozen export lock", extra=(
        "evaluation_protocol", "source_commit", "seed", "mode", "checkpoint", "pair_receipt",
        "native_receipt", "center_receipt", "image_manifest", "image_zip", "sources", "output",
    ))
    require(p250.is_hex(lock["evaluation_protocol"], 40) and p250.is_hex(lock["source_commit"], 40)
            and type(lock["seed"]) is int and lock["seed"] in p250.SEEDS and lock["mode"] in p250.MODES
            and type(lock["output"]) is str and set(lock["sources"]) == {"worker", "p250_exporter", *p250.SOURCE_PINS},
            "Exact prospective protocol, arm, source roster and output required")
    worker, _ = p250.read_ref(lock["sources"]["worker"], 64 * 1024)
    require(worker == Path(__file__).resolve(strict=True), "Expected the exact running frozen inference worker")
    return lock, p250


def check_runtime(base, p250):
    require(platform.system() == "Windows" and platform.machine() == "AMD64"
            and sys.version_info[:3] == (3, 10, 13)
            and sys.flags.isolated == sys.flags.utf8_mode == 1 and sys.dont_write_bytecode
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
            and all(os.environ.get(name) == "1" for name in (
                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
            )), "Exact isolated offline Windows CPU runtime required")
    package_pins = dict(p250.PACKAGE_PINS, PyYAML="6.0.3", pydantic="2.14.3", transformers="5.1.0")
    require({name: version(name) for name in package_pins} == package_pins,
            "Exact Windows CPU source runtime package versions required")
    for name, digest in dict(base.MODEL_SOURCES, **WINDOWS_RUNTIME_SOURCES).items():
        path = Path(import_module(name).__file__).resolve(strict=True)
        p250.closed_bytes(path, digest, 32 * p250.MIB)


def _encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
            + "\n").encode("utf-8")


def _write(stream, raw):
    require(stream.write(raw) == len(raw), "Incomplete frozen output record")


def _flush(stream):
    stream.flush()
    os.fsync(stream.fileno())


def _emit(value):
    sys.stdout.buffer.write(_encoded(value))
    sys.stdout.buffer.flush()


def export(lock_path, lock_sha):
    started = perf_counter()
    lock, p250 = read_lock(lock_path, lock_sha)
    output = Path(lock["output"])
    require(output.is_absolute() and output.parent.resolve(strict=True) == output.parent
            and not output.exists() and not output.is_symlink(), "Fresh canonical frozen output required")
    checkpoint = p250.ref_path(lock["checkpoint"], p250.GIB, name="checkpoint_23.ckpt")
    _, manifest = p250.read_ref(lock["image_manifest"], 8 * p250.MIB)
    items = manifest_items(p250.object_from(manifest))
    _, image_zip = p250.read_ref(lock["image_zip"], 256 * p250.MIB)
    sys.addaudithook(p250.offline)
    base = p250.admit_source(lock)
    check_runtime(base, p250)

    import numpy as np
    import torch
    from pycocotools import mask as codec
    from rfdetr.datasets.coco import make_coco_transforms_square_div_64
    from rfdetr.models.postprocess import PostProcess
    from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

    require(not torch.is_autocast_enabled("cpu") and not torch.is_autocast_enabled("cuda")
            and not torch.cuda.is_initialized(), "FP32 CPU without autocast required")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    protocol = p250.TRAINING_PROTOCOLS[lock["seed"]]
    with ZipFile(BytesIO(image_zip)) as archive:
        validate_image_zip(archive, items)
        model, config, train, _ = base.load_model(checkpoint, lock["checkpoint"]["sha256"], torch,
                                                 output.parent, protocol)
        model.requires_grad_(False).eval()
        require(not model.training and model.num_queries == QUERY_WIDTH and model.group_detr == 13
                and all(param.device.type == "cpu" and param.dtype == torch.float32
                        for param in model.parameters()), "Exact frozen CPU HRSID source model required")
        transform = make_coco_transforms_square_div_64(
            "val", 576, multi_scale=False, expanded_scales=False, skip_random_resize=True,
            patch_size=config.patch_size, num_windows=config.num_windows, aug_config=None,
            scale_jitter=train.scale_jitter, gpu_postprocess=False,
        )
        postprocess = PostProcess(num_select=NATIVE_ROWS, upsample_masks_to_image_size=True)
        output.mkdir()
        digest = sha256()
        seen = mapped = eligible = chosen = chosen_empty = 0
        with (output / PREDICTIONS_NAME).open("xb") as stream, torch.inference_mode():
            for item in items:
                row = infer_png(
                    read_png(archive, item), item, model=model, transform=transform,
                    block_size=config.patch_size * config.num_windows, postprocess=postprocess,
                    torch=torch, np=np, codec=codec, nested_tensor=nested_tensor_from_tensor_list,
                )
                encoded = _encoded(row)
                _write(stream, encoded)
                digest.update(encoded)
                seen += 1
                mapped += row["mapped_rows"]
                eligible += row["eligible_count"]
                if row["candidate"] is not None:
                    chosen += 1
                    chosen_empty += int(codec.area(row["candidate"]["segmentation"]) == 0)
                if seen == 1 or seen % 16 == 0 or seen == len(items):
                    _emit(dict(event="P254_SCHWEGMANN_FROZEN_EXPORT_PROGRESS", images=seen,
                               expected_images=len(items)))
            _flush(stream)
    require(seen == len(items) and 0 <= chosen_empty <= chosen <= eligible <= mapped <= seen * NATIVE_ROWS,
            "Exact image census and internally consistent frozen candidates required")
    report = dict(
        status=EXPORT_STATUS, evaluation_protocol=lock["evaluation_protocol"], source_commit=lock["source_commit"],
        source_training_protocol=protocol, source_seed=lock["seed"], source_mode=lock["mode"],
        checkpoint_sha256=lock["checkpoint"]["sha256"], source_pair_receipt_sha256=lock["pair_receipt"]["sha256"],
        source_arm_receipt_sha256=lock[f'{lock["mode"]}_receipt']["sha256"],
        target_image_manifest_sha256=lock["image_manifest"]["sha256"], target_image_zip_sha256=lock["image_zip"]["sha256"],
        lock_sha256=lock_sha, worker_sha256=lock["sources"]["worker"]["sha256"], predictions_sha256=digest.hexdigest(),
        images=seen, raw_queries=seen * QUERY_WIDTH, native_top_rows=seen * NATIVE_ROWS, mapped_rows=mapped,
        eligible_rows=eligible, candidate_images=chosen, missing_candidate_images=seen - chosen,
        selected_empty_masks=chosen_empty, model_forward_calls=seen, precision="FP32_TF32_disabled", device="cpu",
        query_width=QUERY_WIDTH, num_select=NATIVE_ROWS, checkpoint_ema_updates=UPDATES,
        roi_xyxy=list(ROI_XYXY), selection_rule=SELECTION_RULE,
        training=False, target_adaptation=False, GT_opened=False, official_TEST_opened=False, AP=False,
        elapsed_seconds=perf_counter() - started,
    )
    with (output / REPORT_NAME).open("xb") as stream:
        _write(stream, _encoded(report))
        _flush(stream)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--lock-sha256", required=True)
    args = parser.parse_args()
    _emit(export(args.lock, args.lock_sha256))


if __name__ == "__main__":
    main()
