"""Export one closed HRSID E24 EMA on the fixed image-only SSDD DEV187."""

import argparse
from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
from importlib.util import find_spec
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import re
import stat
import sys
from time import perf_counter
from types import ModuleType
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile


MIB = 1024**2
IMAGES = 187
MAX_JPEG = 16 * MIB
MAX_PIXELS = 4_194_304
IMAGE_MANIFEST_SHA = "18df007e15e3e838e607f178f14e7a26f2b37b992ee39bf091e0b74cf084fb13"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def is_hex(value, count=64):
    return type(value) is str and re.fullmatch(rf"[0-9a-f]{{{count}}}", value) is not None


def exact(value, fields, reason, *, extra=()):
    require(type(value) is dict and set(value) == fields.keys() | set(extra)
            and all(type(value[k]) is type(v) and value[k] == v for k, v in fields.items()), reason)


def manifest_items(manifest, *, expected_images=IMAGES):
    exact(manifest, dict(schema="Root-RFDETR-DEV187-image-only-manifest-v1", population="internal_DEV187",
                         model_execution=False, AP=False, TEST=False),
          "Expected original image-only DEV manifest", extra=("expected_ids", "items"))
    ids, rows = manifest["expected_ids"], manifest["items"]
    require(type(ids) is list and type(rows) is list and len(ids) == len(rows) == expected_images
            and all(type(i) is int and i >= 0 for i in ids) and len(set(ids)) == expected_images,
            "Exact ordered logical DEV image IDs required")
    items, members = [], set()
    for image_id, row in zip(ids, rows, strict=True):
        exact(row, dict(image_id=image_id), "Exact original DEV image row required",
              extra=("image_ref", "original_hw"))
        source, hw = row["image_ref"], row["original_hw"]
        exact(source, {}, "Exact original DEV JPEG reference required", extra=("path", "bytes", "sha256"))
        path = source["path"]
        require(type(path) is str and type(source["bytes"]) is int and 0 < source["bytes"] <= MAX_JPEG
                and is_hex(source["sha256"]) and type(hw) is list and len(hw) == 2
                and all(type(n) is int and n > 0 for n in hw) and hw[0] * hw[1] <= MAX_PIXELS,
                "Bounded source JPEG bytes and original H/W required")
        windows = PureWindowsPath(path)
        old = windows if windows.drive else PurePosixPath(path)
        member = "valid/" + old.name
        require(old.is_absolute() and old.parent.name == "valid" and ".." not in old.parts
                and re.fullmatch(r"[A-Za-z0-9_.-]+\.jpg", old.name) is not None and member not in members,
                "Exact unique source DEV JPEG basename required")
        members.add(member)
        items.append(dict(image_id=image_id, original_hw=tuple(hw), member=member,
                          bytes=source["bytes"], sha256=source["sha256"]))
    return items


def validate_image_zip(archive, items):
    entries = archive.infolist()
    require(len(entries) == len(items) and len({entry.filename for entry in entries}) == len(entries)
            and {entry.filename for entry in entries} == {item["member"] for item in items},
            "Exact image-only ZIP member set required")
    for item in items:
        entry = archive.getinfo(item["member"])
        require(not entry.is_dir() and entry.flag_bits & 1 == 0 and entry.file_size == item["bytes"]
                and entry.compress_type in (ZIP_STORED, ZIP_DEFLATED), "Exact bounded original JPEG ZIP metadata")


def read_jpeg(archive, item):
    data = archive.read(item["member"])
    require(len(data) == item["bytes"] and sha256(data).hexdigest() == item["sha256"],
            "Exact original SSDD JPEG bytes required")
    return data


LOCK_STATUS = "FROZEN_HRSID_TO_SSDD_DEV187_EXPORT_LOCK_V1"
EXPORT_STATUS = "FROZEN_HRSID_TO_SSDD_DEV187_EXPORT_COMPLETE_PENDING_INDEPENDENT_ACCEPTANCE_V1"
PREDICTIONS_NAME = "DEV_NATIVE_PREDICTIONS_PRIVATE.jsonl"
REPORT_NAME = "DEV_NATIVE_EXPORT_SAFE.json"
SEEDS = (5201, 5202, 5203)
MODES = ("native", "center")
TRAINING_PROTOCOLS = {
    5201: "d66baf3c57fafdba4242ad1b55e08cf76bfd77f7",
    5202: "a49361b92977a605267025b1e7798420936cbe53",
    5203: "a49361b92977a605267025b1e7798420936cbe53",
}
QUERY_WIDTH, NATIVE_ROWS, UPDATES = 164, 100, 17_640
GIB = 1024**3
HOST_FLOOR, CUDA_FLOOR, CUDA_ALLOCATED, CUDA_RESERVED = 24 * GIB, 4 * GIB, 2 * GIB, 3 * GIB
SOURCE_PINS = {
    "phase": ("hrsid_mask_phase_dev705.py", "488f9ed73943cc842162c0d45822b6bac42aecabf37c8ecf0d4dde8e96c0959b"),
    "repeat": ("hrsid_mask_phase_repeat_dev705.py", "40d1ec2bd1ce35925eac3442ea7946f06138ffc2924c2d021c5a22da9bf687e1"),
    "filtered": ("hrsid_native_dev705_filtered.py", "9a69500d10d5b0a371ea504f1ba89b7335aca130c1a28d96c4f93a87b2a4b9c0"),
    "base": ("hrsid_native_dev705.py", "1ce0eaa2106fe52452129fad534d0fc11bb8ad5fc4611f70858dbdf6f87a5ec3"),
    "fit_audit": ("hrsid_png_fit_audit.py", "eb5f91a333ea7c7a1c12a87574d0c192f678aa80a742107a510654e3d5512df5"),
    "recipe": ("RFDETR_SEG_NANO_BASELINE_576_E100_20261107.yaml",
               "ba9508d9acab5ff68b10db8a1aff5d9001868d6dda00e39f299d6f5cc1b8011b"),
}
PACKAGE_PINS = {
    "torch": "2.10.0+cu130", "torchvision": "0.25.0+cu130", "rfdetr": "1.12.0",
    "pytorch-lightning": "2.6.6", "numpy": "1.26.4", "Pillow": "12.3.0", "pycocotools": "2.0.11",
}
BINARY_PINS = {
    ("PIL", "_imaging.*.so"): "2bdbf62e38af4c59c6563b9391f11d411e42f5ced5dc0991aaf31bdda72928a8",
    ("pycocotools", "_mask.*.so"): "53d99dd5a70f01aed23bcbfa59aa4f82f0dc68c48cbc4a676a9d5df9e31f1597",
}


def _identity(metadata):
    return metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns


def closed_bytes(path, expected_sha, limit, *, expected_size=None):
    require(path.is_absolute() and path.resolve(strict=True) == path and is_hex(expected_sha),
            "Expected canonical closed input and its SHA-256")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= limit
                and (expected_size is None or before.st_size == expected_size), "Exact bounded closed input size")
        data = stream.read(before.st_size + 1)
        require(len(data) == before.st_size and sha256(data).hexdigest() == expected_sha
                and _identity(before) == _identity(os.fstat(stream.fileno())) == _identity(path.stat()),
                "Closed input SHA-256 or identity changed")
    return data


def _unique(pairs):
    result = dict(pairs)
    require(len(result) == len(pairs), "Duplicate field in frozen export input")
    return result


def object_from(data):
    value = json.loads(data, object_pairs_hook=_unique)
    require(type(value) is dict, "Expected a frozen export JSON object")
    return value


def ref_path(ref, limit, *, name=None):
    exact(ref, {}, "Expected closed path/bytes/SHA-256 reference", extra=("path", "bytes", "sha256"))
    require(type(ref["path"]) is str and type(ref["bytes"]) is int and 0 < ref["bytes"] <= limit
            and is_hex(ref["sha256"]), "Bounded closed artifact reference required")
    path = Path(ref["path"])
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file()
            and path.stat().st_size == ref["bytes"] and (name is None or path.name == name),
            "Exact transported artifact path and size required")
    return path


def read_ref(ref, limit, *, name=None):
    path = ref_path(ref, limit, name=name)
    return path, closed_bytes(path, ref["sha256"], limit, expected_size=ref["bytes"])


def read_lock(path, digest):
    lock = object_from(closed_bytes(path, digest, 64 * 1024))
    exact(lock, dict(status=LOCK_STATUS), "Exact frozen transfer lock required", extra=(
        "evaluation_protocol", "source_commit", "seed", "mode", "device", "checkpoint", "pair_receipt",
        "native_receipt", "center_receipt", "image_manifest", "image_zip", "sources", "output",
    ))
    require(is_hex(lock["evaluation_protocol"], 40) and is_hex(lock["source_commit"], 40)
            and type(lock["seed"]) is int and lock["seed"] in SEEDS and lock["mode"] in MODES
            and lock["device"] in ("cpu", "cuda") and type(lock["output"]) is str,
            "Prospectively locked protocol, source, arm, seed, device and output required")
    sources = lock["sources"]
    require(type(sources) is dict and set(sources) == {"worker", *SOURCE_PINS},
            "Exact frozen worker and source evaluator roster required")
    worker, _ = read_ref(sources["worker"], 64 * 1024)
    require(worker == Path(__file__).resolve(strict=True), "Exact running frozen export worker required")
    return lock


def load_module(name, path, data):
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(data, str(path), "exec"), module.__dict__)
    return module


def admit_source(lock):
    paths, contents = {}, {}
    for role, (name, digest) in SOURCE_PINS.items():
        ref = lock["sources"][role]
        path, contents[role] = read_ref(ref, 128 * 1024, name=name)
        require(ref["sha256"] == digest, "Original source evaluator or official recipe digest differs")
        paths[role] = path
    receipt_paths = {}
    for role in ("pair_receipt", "native_receipt", "center_receipt"):
        receipt_paths[role], _ = read_ref(lock[role], 8192)
    pair_ref = lock["pair_receipt"]
    seed, mode = lock["seed"], lock["mode"]
    if seed == 5201:
        phase = load_module("_frozen_hrsid_phase", paths["phase"], contents["phase"])
        arm = phase.read_pair(receipt_paths["pair_receipt"], pair_ref["sha256"],
                              native_path=receipt_paths["native_receipt"], center_path=receipt_paths["center_receipt"],
                              mode=mode)
        repeat = None
    else:
        repeat = load_module("_frozen_hrsid_repeat", paths["repeat"], contents["repeat"])
        phase = repeat.load_phase(paths["phase"])
        arm = repeat.read_pair(phase, receipt_paths["pair_receipt"], pair_ref["sha256"],
                               native_path=receipt_paths["native_receipt"], center_path=receipt_paths["center_receipt"],
                               mode=mode, seed=seed)
    require(phase.TRAINING_PROTOCOL == TRAINING_PROTOCOLS[seed]
            and arm["checkpoint_sha256"] == lock["checkpoint"]["sha256"]
            and arm["phase"]["seed"] == seed and arm["phase"]["mask_resize_mode"] == mode,
            "Transported checkpoint must be the independently accepted source arm")
    filtered, base = phase.load_evaluator(paths["filtered"], paths["base"])
    if repeat is None:
        phase.bind_evaluator(base, filtered, arm)
    else:
        repeat.bind_evaluator(phase, base, filtered, arm, seed=seed)
    # Keep the original model/endpoint/EMA checks, using only byte-identical source copies.
    base.RECIPE = paths["recipe"]
    base.FIT_AUDIT = paths["fit_audit"]
    require((base.WIDTH, base.NATIVE_ROWS, base.STEPS) == (QUERY_WIDTH, NATIVE_ROWS, UPDATES),
            "Exact HRSID Q164 source inference profile")
    return base


def offline(event, _):
    if event in ("socket.connect", "socket.getaddrinfo"):
        raise RuntimeError("Frozen transfer export cannot access the network")


def check_runtime(device, base):
    require(platform.system() == "Linux" and platform.machine() == "aarch64"
            and sys.version_info[:3] == (3, 12, 14)
            and sys.flags.isolated == sys.flags.utf8_mode == 1 and sys.dont_write_bytecode
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ("0" if device == "cuda" else "-1")
            and all(os.environ.get(name) == "1" for name in (
                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
            )), "Exact isolated offline DGX inference runtime required")
    require({name: version(name) for name in PACKAGE_PINS} == PACKAGE_PINS, "Exact DGX package versions required")
    for name, digest in base.MODEL_SOURCES.items():
        path = Path(import_module(name).__file__).resolve(strict=True)
        closed_bytes(path, digest, MIB)
    for (module, pattern), digest in BINARY_PINS.items():
        root = Path(next(iter(find_spec(module).submodule_search_locations))).resolve(strict=True)
        files = list(root.glob(pattern))
        require(len(files) == 1, "Exact DGX codec extension required")
        closed_bytes(files[0], digest, 32 * MIB)


def host_available():
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            _, count, unit = line.split()
            require(unit == "kB", "Unexpected Linux memory units")
            return int(count) * 1024
    raise ValueError("Linux host available memory witness missing")


def device_admission(torch, name):
    require(host_available() >= HOST_FLOOR, "Host available memory is below the frozen inference floor")
    require(not torch.is_autocast_enabled("cpu") and not torch.is_autocast_enabled("cuda"),
            "Frozen source eval requires FP32 without autocast")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if name == "cpu":
        require(not torch.cuda.is_initialized(), "CPU export must keep CUDA uninitialized")
        return torch.device("cpu")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "One admitted CUDA device required")
    device = torch.device("cuda:0")
    free, total = torch.cuda.mem_get_info(device)
    require(free >= CUDA_FLOOR and total > CUDA_RESERVED, "CUDA immediate free memory is below the inference floor")
    torch.cuda.set_per_process_memory_fraction(CUDA_RESERVED / total, device)
    torch.cuda.reset_peak_memory_stats(device)
    return device


def peak_memory(torch, device):
    if device.type == "cpu":
        require(not torch.cuda.is_initialized(), "CPU export initialized CUDA")
        return 0, 0
    allocated = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)
    require(0 <= allocated <= CUDA_ALLOCATED and allocated <= reserved <= CUDA_RESERVED,
            "Frozen inference exceeded the locked CUDA allocation or reservation cap")
    return allocated, reserved


def infer_image(raw_jpeg, item, *, model, transform, postprocess, base, config, torch, np, codec,
                decode_image, nested_tensor, device):
    hw = item["original_hw"]
    image, scales = decode_image(raw_jpeg, draft_size=None)
    require(image.mode == "RGB" and image.size == (hw[1], hw[0]) and scales == (1.0, 1.0),
            "Exact original SSDD JPEG decode and dimensions required")
    tensor, target = transform(image, None)
    require(target is None and tuple(tensor.shape) == (3, 576, 576) and tensor.dtype == torch.float32
            and bool(torch.isfinite(tensor).all()), "Source 576 image-only native transform changed")
    sample = nested_tensor([tensor], block_size=config.patch_size * config.num_windows)
    require(sample.mask is not None and not bool(sample.mask.any()), "Unexpected source profile spatial padding")
    outputs = model(sample.to(device))
    peak_memory(torch, device)
    require(tuple(outputs["pred_masks"].shape) == (1, QUERY_WIDTH, 144, 144),
            "Expected unaltered Q164 source mask-head geometry")
    return base.native_rows(outputs, image_id=item["image_id"], hw=hw, torch=torch,
                            postprocess=postprocess, np=np, codec=codec)


def export(lock_path, lock_sha):
    started = perf_counter()
    lock = read_lock(lock_path, lock_sha)
    output = Path(lock["output"])
    require(output.is_absolute() and output.parent.resolve(strict=True) == output.parent
            and not output.exists() and not output.is_symlink(), "Fresh canonical frozen export directory required")
    checkpoint_path = ref_path(lock["checkpoint"], GIB, name="checkpoint_23.ckpt")
    _, manifest = read_ref(lock["image_manifest"], MIB)
    require(lock["image_manifest"]["sha256"] == IMAGE_MANIFEST_SHA, "Exact frozen SSDD DEV187 manifest required")
    items = manifest_items(object_from(manifest))
    require(len({item["sha256"] for item in items}) == 183, "Expected 183 distinct original JPEG byte objects")
    _, image_zip = read_ref(lock["image_zip"], 64 * MIB)
    sys.addaudithook(offline)
    base = admit_source(lock)
    check_runtime(lock["device"], base)

    import numpy as np
    import torch
    from pycocotools import mask as codec
    from rfdetr.datasets.coco import make_coco_transforms_square_div_64
    from rfdetr.datasets.io_utils import decode_pil_image_bytes
    from rfdetr.models.postprocess import PostProcess
    from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

    with ZipFile(BytesIO(image_zip)) as archive:
        validate_image_zip(archive, items)
        device = device_admission(torch, lock["device"])
        training_protocol = TRAINING_PROTOCOLS[lock["seed"]]
        model, config, train, _ = base.load_model(checkpoint_path, lock["checkpoint"]["sha256"], torch,
                                                 output.parent, training_protocol)
        model = model.to(device).eval()
        require(not model.training and model.num_queries == QUERY_WIDTH and model.group_detr == 13,
                "Exact frozen HRSID source model required")
        peak_memory(torch, device)
        transform = make_coco_transforms_square_div_64(
            "val", 576, multi_scale=False, expanded_scales=False, skip_random_resize=True,
            patch_size=config.patch_size, num_windows=config.num_windows, aug_config=None,
            scale_jitter=train.scale_jitter, gpu_postprocess=False,
        )
        postprocess = PostProcess(num_select=NATIVE_ROWS, upsample_masks_to_image_size=True)
        output.mkdir()
        digest = sha256()
        mapped = slot1 = empty = count = 0
        with (output / PREDICTIONS_NAME).open("xb") as stream, torch.inference_mode():
            for item in items:
                result = infer_image(read_jpeg(archive, item), item, model=model, transform=transform,
                                     postprocess=postprocess, base=base, config=config, torch=torch, np=np, codec=codec,
                                     decode_image=decode_pil_image_bytes, nested_tensor=nested_tensor_from_tensor_list,
                                     device=device)
                count += 1
                mapped += len(result["coco"])
                slot1 += result["unmapped_slot1_count"]
                empty += int(not result["coco"])
                line = base.encoded(result)
                require(stream.write(line) == len(line), "Short write of frozen image prediction")
                digest.update(line)
                if count in (1, 50, 100, 150, IMAGES):
                    print(json.dumps(dict(event="FROZEN_HRSID_TO_SSDD_EXPORT_PROGRESS", images=count)), flush=True)
            stream.flush()
            os.fsync(stream.fileno())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    allocated, reserved = peak_memory(torch, device)
    require(count == IMAGES and mapped + slot1 == IMAGES * NATIVE_ROWS, "Exact frozen export population required")
    report = dict(
        status=EXPORT_STATUS, evaluation_protocol=lock["evaluation_protocol"], source_training_protocol=training_protocol,
        source_commit=lock["source_commit"], source_seed=lock["seed"], source_mode=lock["mode"],
        checkpoint_sha256=lock["checkpoint"]["sha256"], source_pair_receipt_sha256=lock["pair_receipt"]["sha256"],
        source_arm_receipt_sha256=lock[f'{lock["mode"]}_receipt']["sha256"],
        target_image_manifest_sha256=IMAGE_MANIFEST_SHA, target_image_zip_sha256=lock["image_zip"]["sha256"],
        predictions_sha256=digest.hexdigest(), images=count, raw_queries=count * QUERY_WIDTH,
        native_top_rows=count * NATIVE_ROWS, mapped_rows=mapped, unmapped_slot1_rows=slot1,
        empty_image_predictions=empty, model_forward_calls=count, precision="FP32_TF32_disabled", device=lock["device"],
        query_width=QUERY_WIDTH, num_select=NATIVE_ROWS, checkpoint_ema_updates=UPDATES,
        training=False, target_adaptation=False, GT_opened=False, official_TEST_opened=False, AP=False,
        elapsed_seconds=perf_counter() - started, peak_allocated_bytes=allocated, peak_reserved_bytes=reserved,
    )
    with (output / REPORT_NAME).open("xb") as stream:
        stream.write(base.encoded(report))
        stream.flush()
        os.fsync(stream.fileno())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--lock-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.lock, args.lock_sha256), sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
