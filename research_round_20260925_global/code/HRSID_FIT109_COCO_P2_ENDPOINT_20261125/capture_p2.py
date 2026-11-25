"""Capture and independently read back ImageNet/COCO backbone features on exact FIT109 PNGs."""

import argparse
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter

import numpy as np
from PIL import Image


COUNT = 109
NATIVE_HW = (800, 800)
FEATURE_SHAPE = (256, 200, 200)
FEATURE_DTYPE = "<f4"
STRIDE = 4
COHORT_SHA = "70e9a0e8fb30f5ba70b48b9067c1b95d07bfec23d6b248d92c5983727acceb6e"
WEIGHT_SHA = "bf2d0c1efbc936eeee2bc95a48e80ebc86b891f61b0106485937fc29f9315fc0"
WEIGHT_NAME = "maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
WEIGHT_URL = "https://download.pytorch.org/models/" + WEIGHT_NAME
MANIFEST = "P2_MANIFEST_PRIVATE.jsonl"
RESULT = "RESULT_SAFE.json"
CAPTURE = "HRSID_FIT109_COCO_V1_P2_CAPTURED_V1"
VERIFIED = "HRSID_FIT109_COCO_V1_P2_VERIFIED_V1"


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(data):
    return sha256(data).hexdigest()


def file_sha(path):
    result = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def pin(path, expected):
    raw = path.read_bytes()
    need(digest(raw) == expected, "Exact pinned bytes differ: " + path.name)
    return raw


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def source_rows(cohort):
    locked = json.loads(pin(cohort, COHORT_SHA))
    rows = sorted((row for row in locked["images"] if row["core"]), key=lambda x: x["decoded_rgb_sha256"])
    need(locked["fit_only"] is True and locked["core_images"] == COUNT
         and len(rows) == len({row["image_id"] for row in rows}) == len({row["group"] for row in rows})
         == len({row["decoded_rgb_sha256"] for row in rows}) == COUNT,
         "The exact archived 109-image and 109-source core roster")
    for row in rows:
        name = row["native_basename"]
        need(re.fullmatch(r"P[0-9]+_[A-Za-z0-9_]+\.png", name) is not None
             and row["group"] == name.split("_", 1)[0] and type(row["encoded_bytes"]) is int
             and 0 < row["encoded_bytes"] <= 4 * 1024**2, "Admitted original native PNG identity")
    return rows


def decode_native(image_dir, row):
    path = (image_dir / row["native_basename"]).resolve(strict=True)
    need(path.parent == image_dir.resolve(strict=True), "Admitted materialized FIT image folder only")
    raw = pin(path, row["encoded_sha256"])
    need(len(raw) == row["encoded_bytes"], "Exact archive member byte length")
    with Image.open(BytesIO(raw)) as source:
        need(source.format == "PNG" and source.size == NATIVE_HW[::-1], "Original native 800px PNG")
        image = source.convert("RGB")
    native = np.asarray(image)
    need(native.shape == (*NATIVE_HW, 3) and native.dtype == np.uint8
         and digest(native.tobytes(order="C")) == row["decoded_rgb_sha256"],
         "Identical original native RGB pixels")
    return image


def image_guard(paths, rows):
    allowed_png = {os.path.normcase(os.fspath((paths / row["native_basename"]).resolve(strict=True))) for row in rows}

    def audit(event, args):
        if event in ("socket.connect", "socket.getaddrinfo", "socket.gethostbyname"):
            raise PermissionError("Frozen offline capture")
        if event != "open" or not isinstance(args[0], (str, bytes)):
            return
        file = Path(os.fsdecode(args[0]))
        parts = {part.casefold() for part in file.parts}
        if parts & {"annotations", "annotation_coco", "labels"} or file.suffix.casefold() in {".zip", ".ckpt"}:
            raise PermissionError("No annotation, detector artifact or archive container input")
        if file.suffix.casefold() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            if os.path.normcase(os.fspath(file.absolute())) not in allowed_png:
                raise PermissionError("Image outside exact FIT109 roster")

    sys.addaudithook(audit)


def verify_runtime():
    expected = {"torch": "2.10.0+cu130", "torchvision": "0.25.0+cu130",
                "numpy": "1.26.4", "Pillow": "12.3.0"}
    got = {name: version(name) for name in expected}
    need(sys.version_info[:3] == (3, 10, 13) and got == expected and sys.flags.isolated
         and sys.dont_write_bytecode, "Exact qualified B5 Python with torchvision GPU/CPU support")
    return got


def feature_info(path, native):
    need(native.shape == FEATURE_SHAPE and native.dtype.str == FEATURE_DTYPE
         and native.flags.c_contiguous and np.isfinite(native).all(),
         "Finite untouched float32 CHW P2")
    with path.open("xb") as fp:
        np.save(fp, native, allow_pickle=False)
    return dict(path="features/" + path.name, dtype=FEATURE_DTYPE, shape=list(FEATURE_SHAPE),
                bytes=path.stat().st_size, sha256=file_sha(path))


def row_info(ordinal, row):
    return dict(ordinal=ordinal, image_id=row["image_id"], source_prefix=row["group"], file_name=row["native_basename"],
                encoded_sha256=row["encoded_sha256"], decoded_rgb_sha256=row["decoded_rgb_sha256"],
                native_hw=list(NATIVE_HW), transformed_hw=list(NATIVE_HW), padded_hw=list(NATIVE_HW), p2_stride=STRIDE)


def pick_device(request, torch):
    if request != "cpu" and torch.cuda.is_available() and torch.cuda.device_count() == 1:
        free, total = torch.cuda.mem_get_info(0)
        if free >= 6 * 1024**3:
            torch.cuda.set_per_process_memory_fraction(min(.35, 6 * 1024**3 / total), 0)
            return "cuda:0", free
    need(request != "cuda", "CUDA hardware or six-GiB free admission failed")
    return "cpu", None


def capture(args, rows):
    import inspect
    import torch
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
    from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

    need(args.weights.name == WEIGHT_NAME and args.weights.parent.name == "checkpoints"
         and args.weights.stat().st_size == 178090079 and file_sha(args.weights) == WEIGHT_SHA,
         "Exact official COCO-V1 checkpoint before any loading")
    weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
    need(weights.url == WEIGHT_URL, "Installed official torchvision enumeration and URL")
    torch.hub.set_dir(str(args.weights.parent.parent))
    torch.set_num_interop_threads(1)
    torch.set_num_threads(4 if args.device == "cpu" else 2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device, free = pick_device(args.device, torch)
    image_guard(args.images, rows)
    model = maskrcnn_resnet50_fpn(weights=weights, progress=False).eval().to(device)
    convert = weights.transforms()
    transform = model.transform
    need(tuple(transform.min_size) == (800,) and transform.max_size == 1333
         and transform.size_divisible == 32 and transform.fixed_size is None
         and transform.image_mean == [0.485, 0.456, 0.406] and transform.image_std == [0.229, 0.224, 0.225]
         and not model.training and not model.backbone.training and not transform.training
         and sum(p.numel() for p in model.parameters()) == weights.meta["num_params"] == 44401393,
         "Exact stock COCO-V1 detector architecture and preprocessing")

    def reject_detector(*args):
        raise RuntimeError("RPN/ROI heads may not execute during P2-only capture")

    model.rpn.register_forward_pre_hook(reject_detector)
    model.roi_heads.register_forward_pre_hook(reject_detector)
    args.output.mkdir()
    feature_dir = args.output / "features"
    feature_dir.mkdir()
    total_bytes = 0
    with (args.output / MANIFEST).open("xb") as stream, torch.inference_mode():
        for ordinal, row in enumerate(rows, 1):
            image = decode_native(args.images, row)
            array, targets = transform([convert(image).to(device)], None)
            need(targets is None and array.image_sizes == [NATIVE_HW]
                 and tuple(array.tensors.shape) == (1, 3, *NATIVE_HW) and array.tensors.dtype == torch.float32,
                 "Single-image stock transform retains native resolution and zero padding")
            features = model.backbone(array.tensors)
            need(list(features) == ["0", "1", "2", "3", "pool"]
                 and tuple(features["0"].shape) == (1, *FEATURE_SHAPE) and features["0"].dtype == torch.float32,
                 "Original torch ResNet-50 FPN level 0 P2 stride")
            values = np.asarray(features["0"][0].contiguous().cpu().numpy(), dtype=FEATURE_DTYPE, order="C")
            ref = feature_info(feature_dir / f"{ordinal:04d}.npy", values)
            stream.write(encode(dict(row_info(ordinal, row), feature=ref)))
            total_bytes += ref["bytes"]
            if ordinal % 10 == 0:
                print(json.dumps(dict(images=ordinal, feature_bytes=total_bytes)), flush=True)
    components = [maskrcnn_resnet50_fpn, type(convert), type(transform), type(model.backbone), FeaturePyramidNetwork]
    source_hashes = {part.__module__: file_sha(Path(inspect.getsourcefile(part))) for part in components}
    return dict(status=CAPTURE, backbone_forwards=COUNT, model_forward_calls=0, files=COUNT,
                manifest_sha256=file_sha(args.output / MANIFEST), feature_file_bytes=total_bytes, device=device,
                gpu_free_at_admission=free, cuda_peak_bytes=torch.cuda.max_memory_allocated() if free else None,
                torch_source_sha256=source_hashes,
                input_transform=dict(min_size=[800], max_size=1333, size_divisible=32,
                                     mean=transform.image_mean, std=transform.image_std, resize=False, pad=False))


def verify(args, rows, worker_sha):
    original = args.output.parent / "capture"
    observed = json.loads((args.output.parent / "observed_capture/TERMINAL.json").read_bytes())
    received_sha = observed["result_sha256"]
    received = json.loads(pin(original / RESULT, received_sha))
    need(observed["phase"] == received["phase"] == "capture" and observed["actual_exit"] == 0
         and observed["output_footer_verified"] is True
         and observed["worker_sha256"] == received["worker_sha256"] == worker_sha
         and received["status"] == CAPTURE and received["protocol_sha256"] == args.protocol_sha
         and received["weight_sha256"] == WEIGHT_SHA and received["input_cohort_sha256"] == COHORT_SHA
         and received["images"] == received["backbone_forwards"] == COUNT
         and received["target_labels_opened"] is received["fit_labels_opened"] is False
         and received["rf_outputs_opened"] is False and received["model_forward_calls"] == 0,
         "Original observed official model capture closed successfully")
    records = [json.loads(line) for line in pin(original / MANIFEST, received["manifest_sha256"]).splitlines()]
    feature_dir = original / "features"
    need({path.name for path in feature_dir.iterdir()} == {f"{ordinal:04d}.npy" for ordinal in range(1, COUNT + 1)},
         "Complete exact private P2 file roster")
    image_guard(args.images, rows)
    total_bytes = 0
    for ordinal, (item, record) in enumerate(zip(rows, records, strict=True), 1):
        decode_native(args.images, item)
        expected = row_info(ordinal, item)
        need(all(record[k] == v for k, v in expected.items()), "Original feature/image registration and order")
        ref = record["feature"]
        feature = feature_dir / f"{ordinal:04d}.npy"
        need(ref["path"] == "features/" + feature.name and ref["dtype"] == FEATURE_DTYPE
             and ref["shape"] == list(FEATURE_SHAPE) and feature.stat().st_size == ref["bytes"]
             and file_sha(feature) == ref["sha256"], "Exact feature file identity and private shape")
        value = np.load(feature, mmap_mode="r", allow_pickle=False)
        need(value.shape == FEATURE_SHAPE and value.dtype.str == FEATURE_DTYPE
             and value.flags.c_contiguous and np.isfinite(value).all(), "Checked finite exact FP32 NPY")
        total_bytes += ref["bytes"]
    need(total_bytes == received["feature_file_bytes"] and len(records) == COUNT, "All feature files independently verified")
    args.output.mkdir()
    return dict(status=VERIFIED, capture_sha256=received_sha, manifest_sha256=received["manifest_sha256"],
                files=COUNT, backbone_forwards=0, model_forward_calls=0, feature_file_bytes=total_bytes,
                exact_decoded_rgb_and_feature_file_replay=True, capture_device=received["device"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    phases = parser.add_subparsers(dest="phase", required=True)
    for phase in ("capture", "verify-capture"):
        command = phases.add_parser(phase)
        for name in ("cohort", "images", "protocol", "output"):
            command.add_argument("--" + name, type=Path, required=True)
        command.add_argument("--protocol-sha", required=True)
        if phase == "capture":
            command.add_argument("--weights", type=Path, required=True)
            command.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    for name in ("cohort", "images", "protocol") + (("weights",) if args.phase == "capture" else ()):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.output = args.output.resolve()
    need(args.output.name == args.phase and not args.output.exists(), "New phase output directory")
    started = perf_counter()
    pin(args.protocol, args.protocol_sha)
    runtime = verify_runtime()
    rows = source_rows(args.cohort)
    worker_sha = file_sha(Path(__file__))
    value = capture(args, rows) if args.phase == "capture" else verify(args, rows, worker_sha)
    value.update(phase=args.phase, worker_sha256=worker_sha, protocol_sha256=args.protocol_sha,
                 input_cohort_sha256=COHORT_SHA, weight_sha256=WEIGHT_SHA, weight_url=WEIGHT_URL,
                 images=COUNT, source_prefixes=COUNT, runtime=runtime, feature_shape=list(FEATURE_SHAPE),
                 feature_dtype=FEATURE_DTYPE, fpn_level="0", p2_stride=STRIDE,
                 coordinates="native integer centers (r,c) to (r/4,c/4); bilinear with border clamp",
                 target_labels_opened=False, fit_labels_opened=False, rf_outputs_opened=False,
                 optimizer_updates=0, rpn_forwards=0, roi_forwards=0, seconds=perf_counter() - started)
    raw = encode(value)
    with (args.output / RESULT).open("xb") as stream:
        stream.write(raw)
    print(json.dumps(dict(status=value["status"], result_sha256=digest(raw))), flush=True)


if __name__ == "__main__":
    main()
