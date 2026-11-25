"""Observe and read back the pinned SRNet author's exact pre-resize probabilities."""
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np

CAPTURE_WORKER = "ee08ae69961a547464899e1c5480a9def54d243d7daf89a6707898397432ffdc"
CAPTURE_RESULT = "dd38b03fa1e05ef014d9063d0a7cf436387b2451178b087673e77c6c4b5beb89"
CAPTURE_TERMINAL = "dc68f6100f08dc5bcc66b78de82c8bb6ac6e4da4b3711b4511ccf8c569fcc6ed"
STATUS = "SRNET_RSDD74_AUTHOR_PREGRID_CAPTURE_COMPLETE"
READBACK = "SRNET_RSDD74_AUTHOR_PREGRID_READBACK_COMPLETE"
STATE = None


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, expected):
    raw = path.read_bytes()
    require(digest(raw) == expected, "Bound SHA-256 mismatch: " + path.name)
    return raw


def old_module(root):
    path = root / "native_capture.py"
    code = ModuleType("closed_native_capture")
    code.__file__ = str(path)
    exec(compile(pin(path, CAPTURE_WORKER), str(path), "exec"), vars(code))
    return code


def observe_author_resize(utils, author_call, count, resized_hw, native_hw):
    """Keep the author inputs/call/output unchanged and retain the tensors around the resize."""
    import torch

    functional = utils.F
    seen = []

    class Observer:
        def __getattr__(self, name):
            return getattr(functional, name)

        def interpolate(self, values, size=None, **kwargs):
            require(not seen and values.shape == (1, count, *resized_hw) and values.dtype == torch.float32
                    and tuple(size) == native_hw and kwargs == {"mode": "bilinear"},
                    "Pinned author resize consumes unpadded FP32 probability, half-pixel bilinear")
            result = functional.interpolate(values, size, **kwargs)
            seen.append((values[0].detach().contiguous().clone(), result[0].detach().contiguous().clone()))
            return result

    utils.F = Observer()
    try:
        stock = author_call()
    finally:
        utils.F = functional
    require(len(seen) == 1, "Single original author rescale for every nonempty candidate row")
    return stock, *seen[0]


def closed_old(root):
    term = json.loads(pin(root / "observed_capture/TERMINAL.json", CAPTURE_TERMINAL))
    result = json.loads(pin(root / "actual/RESULT_SAFE.json", CAPTURE_RESULT))
    require(term["phase"] == "capture" and term["actual_exit"] == 0 and term["output_footer_verified"] is True
            and term["result_sha256"] == CAPTURE_RESULT and term["worker_sha256"] == result["worker_sha256"] == CAPTURE_WORKER
            and result["status"] == "SRNET_SSDD_E24_RSDD74_NATIVE_CAPTURE_COMPLETE"
            and result["model_forwards"] == 74 and result["counts"]["images"] == 74 and result["counts"]["candidates"] == 89
            and result["reference_files_opened"] == result["optimizer_updates"] == 0 and result["AP"] is False
            and result["all_native_rle_and_probability_masks_exact"] is True
            and result["all_original_rank_score_and_rotated_box_same_call_exact"] is True
            and result["all_archive_jpeg_and_rgb_exact"] is True, "Exact closed previous image-only capture")
    index = [json.loads(x) for x in pin(root / "actual/INDEX_PRIVATE.jsonl", result["index_sha256"]).splitlines()]
    require(len(index) == 74 and sum(len(x["candidates"]) for x in index) == 89, "Whole frozen candidate ledger")
    return result, index


def packed_file(root, folder, ordinal, ref, name):
    expected = f"{folder}/{ordinal:03d}.npz"
    require(ref["path"] == expected, "Exact per-ordinal archive file")
    raw = pin(root / expected, ref["sha256"])
    require(len(raw) == ref["bytes"], "Exact per-ordinal archive compressed byte count")
    with np.load(BytesIO(raw), allow_pickle=False) as data:
        require(data.files == [name], "Single expected compressed array")
        value = data[name]
    require(value.dtype == np.float32 and np.isfinite(value).all() and
            (value >= -1e-6).all() and (value <= 1 + 1e-6).all(), "Finite FP32 probability map")
    return value


def equal_bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype == np.float32 and np.array_equal(a.view(np.uint32), b.view(np.uint32))


def initialize(args, image_index, image_root):
    global STATE
    base = old_module(args.original)
    base.initialize(args, image_index, image_root)
    STATE = SimpleNamespace(args=args, base=base, context=base.STATE)


def capture_one(task):
    row, old = task
    ctx, args, base = STATE.context, STATE.args, STATE.base
    torch = ctx.torch
    ordinal = old["ordinal"]
    require(ctx.dataset.img_ids[ordinal - 1] == row["image_id"] == old["image_id"], "Original image order")
    prepared = ctx.dataset.prepare_test_img(ordinal - 1)
    tensor, = prepared["img"]
    container, = prepared["img_metas"]
    if not torch.is_tensor(tensor):
        tensor = tensor.data
    meta = container.data
    resized = tuple(meta["img_shape"][:2])
    padded = tuple(meta["pad_shape"][:2])
    native = tuple(meta["ori_shape"][:2])
    require(tensor.shape == (3, 608, 608) and tensor.dtype == torch.float32 and tensor.device.type == "cpu"
            and padded == (608, 608) and native == (512, 512) and meta["flip"] is False
            and all(0 < c <= p for c, p in zip(resized, padded, strict=True)), "Exact qualified image geometry")
    original = ctx.head.get_mask
    artifacts = []

    def tap(rois, labels, crops, angle_version, output_size, img_meta, thr, num_classes, rescale):
        require(angle_version == "le90" and output_size == 28 and thr == .5 and num_classes == 1 and rescale is True
                and crops.shape == (len(rois), 1, 28, 28) and torch.all(labels == 0)
                and torch.isfinite(crops).all() and torch.all((crops >= 0) & (crops <= 1))
                and (tuple(img_meta["img_shape"][:2]), tuple(img_meta["pad_shape"][:2]), tuple(img_meta["ori_shape"][:2]))
                == (resized, padded, native), "Unaltered original rotated projection inputs")
        stock, pregrid, final = observe_author_resize(ctx.utils, lambda: original(
            rois, labels, crops, angle_version, output_size, img_meta, thr, num_classes, rescale), len(rois), resized, native)
        scaled = rois.detach().clone()
        scaled[:, :4] /= scaled.new_tensor(img_meta["scale_factor"])
        corners = ctx.utils.obb2poly(scaled, angle_version).reshape(len(rois), 4, 2).numpy()
        artifacts.append((scaled.numpy(), corners, pregrid.numpy(), final.numpy()))
        return stock

    ctx.head.get_mask = tap
    try:
        with torch.inference_mode():
            per_image, = ctx.model.simple_test(tensor[None], [meta], rescale=True, thr=.5)
    finally:
        ctx.head.get_mask = original

    model_boxes, stock = per_image
    require(len(model_boxes) == len(stock) == 1 and len(model_boxes[0]) == len(stock[0]) == len(old["candidates"])
            and len(artifacts) == int(bool(stock[0])), "Exact original candidate counts with empty rows")
    if artifacts:
        scaled, corners, pregrid, final = artifacts[0]
        require(np.array_equal(scaled, model_boxes[0]), "Same-call natural stock boxes and confidence")
        restored = [base.candidate_record(rank, box, corner, mask, p, ctx.codec)
                    for rank, (box, corner, mask, p) in enumerate(zip(scaled, corners, stock[0], final, strict=True), 1)]
        require(restored == old["candidates"], "All original candidate ranks, scores, boxes, AABBs and RLEs")
    else:
        pregrid = np.empty((0, *resized), dtype=np.float32)
        final = np.empty((0, *native), dtype=np.float32)
    previous = packed_file(args.original / "actual", "PROBABILITIES_PRIVATE", ordinal, old["probability_file"], "probability")
    require(equal_bits(final, previous), "Actual author native FP32 bits equal the closed native archive")
    require(pregrid.shape == (len(stock[0]), *resized) and np.isfinite(pregrid).all()
            and (pregrid >= -1e-6).all() and (pregrid <= 1 + 1e-6).all(), "Lossless finite pre-resize probabilities")
    path = f"PREGRID_PRIVATE/{ordinal:03d}.npz"
    buffer = BytesIO()
    np.savez_compressed(buffer, pregrid=pregrid)
    raw = buffer.getvalue()
    ref = dict(path=path, bytes=len(raw), sha256=digest(raw))
    with (args.output / path).open("xb") as stream:
        stream.write(raw)
    note = f"{resized[0]}x{resized[1]}"
    result = dict(old, pregrid_file=ref, resized_hw=list(resized), padded_hw=list(padded), native_hw=list(native),
                  final_resize_called=bool(artifacts))
    counts = dict(images=1, candidates=len(stock[0]), resized_images=int(bool(artifacts)),
                  resized_candidates=len(stock[0]), no_candidate_images=int(not artifacts),
                  exact_native_f32_candidates=len(stock[0]), pregrid_positions=pregrid.size,
                  same_shape_resize_images=int(bool(artifacts) and resized == native))
    return result, counts, note


def capture(cmd, old, index):
    started = perf_counter()
    base = old_module(cmd.original)
    pilot = base.load_pilot(cmd.pilot)
    term = json.loads(pin(cmd.pilot / "observed_pilot/TERMINAL.json", base.PILOT_TERMINAL))
    result = json.loads(pin(cmd.pilot / "actual/RESULT_SAFE.json", base.PILOT_RESULT))
    require(term["phase"] == "verify" and term["actual_exit"] == 0 and term["output_footer_verified"] is True
            and term["result_sha256"] == base.PILOT_RESULT and term["worker_sha256"] == base.PILOT_CODE
            and result["all_independent_projection_masks_exact"] is True and result["reference_files_opened"] == 0
            and old["pilot_result_sha256"] == base.PILOT_RESULT and old["weights_sha256"] == pilot.CHECKPOINT,
            "Original closed projection pilot")
    pilot.admit(cmd)
    for name, expected in pilot.PINS.items():
        pin(cmd.source / name, expected)
    pin(cmd.source / "mmrotate/core/bbox/transforms.py", base.BOX_TRANSFORMS)
    manifest = json.loads(pin(cmd.intake / "intake/IMAGE_MANIFEST_PRIVATE.json", pilot.MANIFEST))
    rows = manifest["items"]
    require(old["intake_image_manifest_sha256"] == pilot.MANIFEST and
            [r["image_id"] for r in rows] == [r["image_id"] for r in index], "Exact frozen image manifest and order")
    images, image_rows = pilot.bind_images(cmd.intake, rows)
    from PIL import Image, ImageOps
    for row, previous in zip(rows, index, strict=True):
        with Image.open(images / row["source_image_name"]) as im:
            rgb = ImageOps.exif_transpose(im).convert("RGB")
            require(rgb.size == (512, 512) and digest(rgb.tobytes()) == row["rgb_sha256"] == previous["rgb_sha256"]
                    and row["image_ref"]["sha256"] == previous["image_sha256"], "Exact frozen JPEG and RGB image")
    cmd.output.mkdir()
    (cmd.output / "PREGRID_PRIVATE").mkdir()
    image_index = cmd.output / "IMAGE_ONLY_COCO_PRIVATE.json"
    pilot.write_json(image_index, dict(images=image_rows, categories=[dict(id=1, name="ship")], annotations=[]))
    print(json.dumps(dict(stage="ADMITTED_EXACT_74_IMAGE_ONLY_ROWS")), flush=True)
    counts, sizes = Counter(), Counter()
    checksum = sha256()
    with (cmd.output / "INDEX_PRIVATE.jsonl").open("xb") as stream, ProcessPoolExecutor(
            max_workers=cmd.workers, initializer=initialize, initargs=(cmd, image_index, images)) as pool:
        for new, stats, size in pool.map(capture_one, zip(rows, index, strict=True), chunksize=1):
            require(new["ordinal"] == counts["images"] + 1, "Frozen archive ordinal sequence")
            payload = (json.dumps(new, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
            stream.write(payload)
            checksum.update(payload)
            counts.update(stats)
            sizes[size] += 1
            if counts["images"] % 10 == 0:
                print(json.dumps(dict(stage="AUTHOR_PREGRID_CAPTURED", images=counts["images"])), flush=True)
    require(counts["images"] == 74 and counts["candidates"] == counts["resized_candidates"] ==
            counts["exact_native_f32_candidates"] == 89 and
            counts["no_candidate_images"] == old["counts"]["no_candidates"], "Complete exact unchanged candidate matrix")
    record = safe_common(cmd, started, STATUS, old)
    record.update(index_sha256=checksum.hexdigest(), pregrid_dtype="float32", pregrid_origin_yx=[0, 0],
                  pregrid_definition="author rotated grid_sample result after top-left img_shape unpadding",
                  interpolation="bilinear_half_pixel", reference_files_opened=0, model_forwards=74,
                  optimizer_updates=0, cpu_worker_processes=cmd.workers, torch_threads_per_process=1,
                  source_fit_result_sha256=old["source_fit_result_sha256"],
                  weights_sha256=pilot.CHECKPOINT, author_commit=pilot.AUTHOR_COMMIT,
                  resized_image_shape_histogram=dict(sizes), counts=dict(counts),
                  all_original_native_probability_f32_bits_exact=True, all_original_stock_fields_exact=True)
    finish(cmd, record)


def safe_common(cmd, started, status, old):
    return dict(status=status, worker_sha256=digest(Path(__file__).read_bytes()),
                protocol_sha256=cmd.protocol_sha256, prelock_commit=cmd.prelock_commit,
                original_capture_result_sha256=CAPTURE_RESULT, original_capture_terminal_sha256=CAPTURE_TERMINAL,
                original_capture_index_sha256=old["index_sha256"],
                image_manifest_sha256=old["intake_image_manifest_sha256"],
                AP=False, ground_truth_opened=False, elapsed_seconds=perf_counter() - started)


def finish(cmd, value):
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    with (cmd.output / "RESULT_SAFE.json").open("xb") as stream:
        stream.write(payload)
    print(json.dumps(dict(status=value["status"], result_sha256=digest(payload))), flush=True)


def verify(cmd, old, index):
    started = perf_counter()
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(torch.__version__ == "1.12.0+cpu" and torch.get_num_threads() == 1
            and not torch.cuda.is_initialized(), "Original exact CPU resize runtime")
    terminal = json.loads(pin(cmd.capture_terminal, cmd.capture_terminal_sha256))
    rendered = json.loads(pin(cmd.capture / "RESULT_SAFE.json", cmd.capture_result_sha256))
    require(terminal["phase"] == "capture" and terminal["actual_exit"] == 0 and terminal["output_footer_verified"] is True
            and terminal["result_sha256"] == cmd.capture_result_sha256 and
            terminal["worker_sha256"] == rendered["worker_sha256"] == digest(Path(__file__).read_bytes())
            and rendered["status"] == STATUS and rendered["protocol_sha256"] == cmd.protocol_sha256
            and rendered["prelock_commit"] == cmd.prelock_commit and rendered["original_capture_result_sha256"] == CAPTURE_RESULT
            and rendered["original_capture_index_sha256"] == old["index_sha256"] and rendered["all_original_stock_fields_exact"]
            and rendered["all_original_native_probability_f32_bits_exact"] and rendered["model_forwards"] == 74
            and rendered["reference_files_opened"] == rendered["optimizer_updates"] == 0, "Natural author pregrid capture closure")
    new = [json.loads(x) for x in pin(cmd.capture / "INDEX_PRIVATE.jsonl", rendered["index_sha256"]).splitlines()]
    counts = Counter()
    for row, previous in zip(new, index, strict=True):
        require({k: row[k] for k in previous} == previous, "Original candidate ledger fields survive exact")
        ordinal = previous["ordinal"]
        resized, padded, native = (tuple(row[k]) for k in ("resized_hw", "padded_hw", "native_hw"))
        require(padded == (608, 608) and native == (512, 512) and len(resized) == 2
                and all(0 < a <= b for a, b in zip(resized, padded, strict=True)), "Original resized/unpadded shape bounds")
        pregrid = packed_file(cmd.capture, "PREGRID_PRIVATE", ordinal, row["pregrid_file"], "pregrid")
        target = packed_file(cmd.original / "actual", "PROBABILITIES_PRIVATE", ordinal, previous["probability_file"], "probability")
        count = len(previous["candidates"])
        require(pregrid.shape == (count, *resized) and target.shape == (count, *native)
                and row["final_resize_called"] is bool(count), "Known spacing and identity of every candidate plane")
        if count:
            final = F.interpolate(torch.from_numpy(pregrid)[None], native, mode="bilinear", align_corners=False)[0].numpy()
            require(equal_bits(final, target), "Independent saved pregrid resize reconstructs archived native FP32 exactly")
        counts.update(images=1, candidates=count, exact_native_f32_candidates=count, no_candidate_images=int(not count),
                      pregrid_positions=pregrid.size)
    require(counts["images"] == 74 and all(rendered["counts"][k] == v for k, v in counts.items()),
            "Whole original order, row count and archived probability checks")
    cmd.output.mkdir()
    value = safe_common(cmd, started, READBACK, old)
    value.update(capture_result_sha256=cmd.capture_result_sha256, capture_terminal_sha256=cmd.capture_terminal_sha256,
                 index_sha256=rendered["index_sha256"], all_original_stock_fields_exact=True,
                 all_original_native_probability_f32_bits_exact=True, reference_files_opened=0,
                 model_forwards=0, optimizer_updates=0, counts=dict(counts))
    finish(cmd, value)


def main():
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    phases = parser.add_subparsers(dest="phase", required=True)
    for phase in ("capture", "verify"):
        sub = phases.add_parser(phase, allow_abbrev=False)
        paths = ["original", "protocol", "output"]
        strings = ["protocol-sha256", "prelock-commit"]
        if phase == "capture":
            paths.extend(("source", "fit", "intake", "pilot"))
            sub.add_argument("--workers", type=int, choices=(1, 2, 3, 4), required=True)
        else:
            paths.extend(("capture", "capture-terminal"))
            strings.extend(("capture-terminal-sha256", "capture-result-sha256"))
        for name in paths:
            sub.add_argument("--" + name, type=lambda x: Path(x).resolve(), required=True)
        for name in strings:
            sub.add_argument("--" + name, required=True)
    cmd = parser.parse_args()
    require(sys.flags.isolated and sys.dont_write_bytecode and not cmd.output.exists() and
            sys.version_info[:3] == (3, 10, 13) and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and
            all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")) and
            re.fullmatch("[0-9a-f]{40}", cmd.prelock_commit), "Qualified isolated single-thread CPU and published prelock")
    pin(cmd.protocol, cmd.protocol_sha256)
    old, index = closed_old(cmd.original)
    (capture if cmd.phase == "capture" else verify)(cmd, old, index)


if __name__ == "__main__":
    main()

