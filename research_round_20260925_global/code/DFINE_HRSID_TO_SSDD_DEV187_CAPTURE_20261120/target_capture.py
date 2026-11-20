"""Run frozen HRSID D-FINE E24 author inference on original SSDD DEV187 JPEGs."""
from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

BASE_SHA = "5be622cdec5c889ca8520aab5547f6f15f01a54568a0263423a816dca7f96626"
CAPTURE_SHA = "8a7dd193e1d4e14eae528cba453941af9bc7d646646f34888726ec9287e0ce12"
SIDECAR_SHA = "53dd5867d4353e3569552627d252877e4222bfcb2e6c7f60ca5bb53f3eabc940"
MANIFEST_SHA = "18df007e15e3e838e607f178f14e7a26f2b37b992ee39bf091e0b74cf084fb13"
ACCEPT_SHA = "7401f031c996298caff1761c6ef691fae0c301e65c87c15f51d186dcf81f25d8"
EMA_SHA = "82ca80e2cb9a0b9673a136fad3d7de3097b7327f4f76e3a2d3e7184ab07e7fd8"
STATUS = "DFINE_HRSID_E24_SSDD_DEV187_PROBABILITY_EXPORT_COMPLETE_V1"
MIB = 1024**2


def load_source(filename, digest):
    path = Path(__file__).resolve().with_name(filename)
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != digest:
        raise ValueError("Exact reused source " + filename)
    module = ModuleType("_frozen_" + path.stem)
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def manifest_rows(path, base):
    raw = base.pinned(path, MANIFEST_SHA, MIB, 49477)
    doc = base.decode(raw)
    base.need(set(doc) == {"schema", "population", "model_execution", "AP", "TEST", "expected_ids", "items"}
              and doc["schema"] == "Root-RFDETR-DEV187-image-only-manifest-v1"
              and doc["population"] == "internal_DEV187" and doc["model_execution"] is doc["AP"] is doc["TEST"] is False
              and len(doc["expected_ids"]) == len(doc["items"]) == len(set(doc["expected_ids"])) == 187,
              "Exact frozen image-only DEV187 population")
    for expected, row in zip(doc["expected_ids"], doc["items"], strict=True):
        hw, ref = row["original_hw"], row["image_ref"]
        base.need(set(row) == {"image_id", "original_hw", "image_ref"} and row["image_id"] == expected
                  and len(hw) == 2 and all(type(n) is int and n > 0 for n in hw) and hw[0]*hw[1] <= 2*MIB
                  and set(ref) == {"path", "bytes", "sha256"} and 0 < ref["bytes"] <= 16*MIB,
                  "Exact bounded logical image and original dimensions")
    return doc["items"]


def read_rgb(root, row, base):
    import numpy as np
    from PIL import Image, ImageOps
    from pathlib import PurePosixPath, PureWindowsPath

    ref = row["image_ref"]
    original = PureWindowsPath(ref["path"]) if PureWindowsPath(ref["path"]).drive else PurePosixPath(ref["path"])
    base.need(original.is_absolute() and original.parent.name == "valid" and original.name.endswith(".jpg"),
              "Exact image-only source role")
    raw = base.pinned(root / "valid" / original.name, ref["sha256"], 16*MIB, ref["bytes"])
    with Image.open(BytesIO(raw)) as image:
        base.need(image.format == "JPEG", "Original full-resolution JPEG")
        rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB")).copy()
    base.need(rgb.dtype == np.uint8 and rgb.shape == (*row["original_hw"], 3),
              "Exact original EXIF-oriented RGB dimensions")
    return rgb


def run(args):
    base = load_source("author_export_reference.py", BASE_SHA)
    capture = load_source("native_capture_reference.py", CAPTURE_SHA)
    sidecar = load_source("probability_sidecar.py", SIDECAR_SHA)
    protocol = base.pinned(Path(__file__).with_name("PROTOCOL.md"), args.protocol_sha, 64*1024)
    accepted = base.admitted_e24(base.decode(base.pinned(args.acceptance, ACCEPT_SHA)))
    base.need(args.acceptance_sha == ACCEPT_SHA and accepted["ema_checkpoint_sha256"] == EMA_SHA,
              "Exact original HRSID E24 source model")
    base.admit_source(args.author_source, args.author_tree)
    import torch
    import torchvision
    import numpy as np
    from PIL import __version__ as pillow
    from dfine_seg.infer.torch_model import TorchModel

    base.need(sys.version_info[:3] == (3,11,15) and torch.__version__ == "2.13.0+cu130"
              and torchvision.__version__ == "0.28.0+cu130" and pillow == "13.0.0"
              and torch.cuda.is_available() and torch.cuda.device_count() == 1,
              "Exact qualified CUDA inference runtime")
    device = torch.device("cuda:0")
    free, total = torch.cuda.mem_get_info(device)
    base.need(free >= 6*1024**3, "Inherited CUDA worker memory floor")
    torch.cuda.set_per_process_memory_fraction(min(.35, 6*1024**3/total), device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    rows = manifest_rows(args.manifest, base)
    base.need(args.dev_root.resolve(strict=True) == args.dev_root and args.dev_root.is_dir(),
              "Exact original image-only local root")
    start = perf_counter()
    checkpoint = torch.load(BytesIO(base.pinned(args.ema, EMA_SHA, 256*MIB, accepted["ema_checkpoint_bytes"])),
                            map_location="cpu", weights_only=True)
    base.need(set(checkpoint) == {"model", "meta"} and checkpoint["meta"] == base.AUTHOR_META,
              "Original author source E24 EMA envelope")
    model = TorchModel(str(args.ema), conf_thresh=0., rect=False, keep_ratio=False, apply_nms=True,
                       nms_iou_thresh=.70, binarize_masks=True, mask_threshold=.50, device="cuda", task="segment")
    base.check_wrapper(model, checkpoint["model"], torch)
    del checkpoint
    author = sys.modules[TorchModel.__module__]
    forward, nms, hook, original_nms = capture.bind_observers(model, author, torch)
    args.output.mkdir()
    base.prepare_export_codec(args.codec_wheel, args.output / "CODEC_PRIVATE")
    ph, ah = sha256(), sha256()
    n_all = n_top = n_empty = n_empty_top = zero = forward_count = 0
    seconds = 0.
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with sidecar.Capture(args.output) as archive, (args.output / base.PREDICTIONS).open("xb") as predictions, (
            args.output / base.AUDIT).open("xb") as audit, torch.inference_mode():
        for ordinal, row in enumerate(rows, 1):
            rgb = read_rgb(args.dev_root, row, base)
            before = perf_counter()
            pred, observed, planes = capture.capture_row(rgb, row, model, author, torch, np, base, forward, nms)
            torch.cuda.synchronize()
            seconds += perf_counter()-before
            detail = archive.write(ordinal, row["image_id"], observed, planes)
            count, empty = base.validate_rows(row["image_id"], row["original_hw"], pred, detail)
            n_all += count
            n_top += min(100,count)
            n_empty += empty
            n_empty_top += sum(c["mask_area"] == 0 for c in detail["candidates"][:100])
            zero += count == 0
            forward_count += 1
            for stream, digest, value in ((predictions, ph, pred), (audit, ah, detail)):
                raw = base.encoded(value)
                base.need(stream.write(raw) == len(raw), "Complete original native candidate stream")
                digest.update(raw)
            if ordinal % 25 == 0:
                print(json.dumps(dict(event="SSDD_DEV187_NATIVE_CAPTURE_PROGRESS", images=ordinal,
                                      post_nms_rows=n_all, probability_rows=n_top)), flush=True)
        for stream in (predictions, audit):
            sidecar.flush(stream)
    hook.remove()
    author.nms = original_nms
    stats = archive.receipt()
    base.need(forward_count == stats["probability_files"] == 187 and n_top == stats["probability_rows"],
              "Exact full original candidate and probability scope")
    result = dict(status=STATUS, protocol_sha256=args.protocol_sha,
        export_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        manifest_sha256=MANIFEST_SHA, e24_acceptance_sha256=ACCEPT_SHA, ema_sha256=EMA_SHA,
        author_commit=base.AUTHOR_COMMIT, images=187, model_forwards=187, synthetic_warmups=1,
        raw_queries=187*300, internal_top_rows=187*300, post_nms_rows=n_all, rows_at_100=n_top,
        empty_post_nms=n_empty, empty_at_100=n_empty_top, zero_prediction_images=zero,
        exact_author_binary_replay=n_all, exact_top100_binary_replay=n_top,
        original_author_postprocess_parity=True, predictions_sha256=ph.hexdigest(), candidate_audit_sha256=ah.hexdigest(),
        original_rgb_rule="PIL EXIF transpose then RGB, full native resolution",
        native_dimensions_verified=187, confidence_threshold=0., box_nms_iou=.70, mask_threshold=.50,
        keep_ratio=False, rect=False, model_precision="FP32_TF32_disabled", mask_resize_precision="author_CUDA_FP16",
        probability_semantics="decoder_sigmoid_FP32_pre_crop", elapsed_seconds=perf_counter()-start,
        forward_postprocess_seconds=seconds, peak_allocated=torch.cuda.max_memory_allocated(),
        optimizer_updates=0, FIT=False, DEV=True, TEST=False, GT_opened=False, AP=False, **stats)
    digest = base.write_new(args.output / "RESULT_SAFE.json", result)
    print(json.dumps(dict(status=STATUS, result_sha256=digest), sort_keys=True), flush=True)


def main():
    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument("mode", choices=("export",))
    for field in ("dev-root", "manifest", "output", "acceptance", "author-source", "author-tree", "ema", "codec-wheel"):
        parser.add_argument("--" + field, type=Path, required=True)
    for field in ("acceptance-sha", "protocol-sha"):
        parser.add_argument("--" + field, required=True)
    args = parser.parse_args()
    def offline(event, _):
        if event.startswith("socket."):
            raise PermissionError("Offline label-free frozen target inference")
    sys.addaudithook(offline)
    if not sys.flags.isolated or not sys.dont_write_bytecode or args.output.exists():
        raise ValueError("Isolated original fresh operation")
    run(args)


if __name__ == "__main__":
    main()

