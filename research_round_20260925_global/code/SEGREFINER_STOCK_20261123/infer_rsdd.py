"""Run stock SegRefiner-LR on the label-free Instance-RSDD RF-DETR capture."""
import argparse
from collections import Counter
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
import json
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
from PIL import Image
from pycocotools import mask as codec


LOCK_NAME = "SEGREFINER_STOCK_PRELOCK_20261123.md"
LOCK_SHA256 = "30007c832d47c6e086445d474d4f319642854de20da7844706ebac8e40a7df9c"
EXPOSURE_SHA256 = "78e2dd5de0901e996087b319dbefe78549f702333fd87a542b2d70a69534867c"
EXPOSURE_STATUS = "RSDD74_SEGREFINER_STOCK_EXPOSURE_COMPLETE"
INPUTS_SHA256 = "2c3b84f852bb4f8234d935347708a2e71eba4466d0eb111471bd254aa2f45e58"
AUTHOR_COMMIT = "53419a2d38ea3da0b6e2be77e5b45e139195a0b3"
STATUS = "RSDD74_SEGREFINER_STOCK_INFERENCE_COMPLETE"
SMOKE_STATUS = "RSDD74_SEGREFINER_STOCK_SMOKE_COMPLETE"
LEDGER = "REFINED_MASKS_PRIVATE.jsonl"
SEED = 5217
MINIMUM_AREA = 512
FULL_COUNTS = {"images": 74, "native_candidates": 7346, "eligible_candidates": 3074}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path, expected):
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == expected, "Input digest differs for " + path.name)
    return raw


def file_digest(path):
    identity = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            identity.update(chunk)
    return identity.hexdigest()


def source(path, expected):
    raw = pin(path, expected)
    module = ModuleType(path.stem)
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def encode(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def eligible_count(row):
    return sum(c["mask_area"] >= MINIMUM_AREA for c in row[1]["candidates"])


def input_population(a, exposure):
    loader = source(a.code / "POINTREND_SIGNED_QUOTA_SWAP_20261123/inputs.py", INPUTS_SHA256)
    rows, lineage = loader.load_population(a)
    counts = dict(images=len(rows), native_candidates=sum(len(row[2]["coco"]) for row in rows),
                  eligible_candidates=sum(map(eligible_count, rows)))
    declared = exposure["views"]["full"]["native_candidates"]
    require(lineage == exposure["input_lineage"] and counts == FULL_COUNTS
            and declared["total"] == counts["native_candidates"] and declared["eligible"] == counts["eligible_candidates"]
            and len({row[0]["image_id"] for row in rows}) == counts["images"], "Frozen native exposure differs")
    return rows, lineage, counts


def decode_rgb(root, item):
    ref = item["image_ref"]
    path = (root / ref["path"]).resolve(strict=True)
    require(path.parent == (root / "images").resolve(strict=True), "Only originally admitted JPEGs may be read")
    raw = pin(path, ref["sha256"])
    require(len(raw) == ref["bytes"], "Original JPEG byte count differs")
    with Image.open(BytesIO(raw)) as image:
        require(image.format == "JPEG" and image.size == tuple(item["original_hw"][::-1]),
                "Original native JPEG dimensions differ")
        rgb = np.array(image.convert("RGB"), dtype=np.uint8, order="C")
    require(rgb.shape == (*item["original_hw"], 3)
            and sha256(rgb.tobytes()).hexdigest() == item["rgb_sha256"], "Decoded admitted RGB differs")
    return rgb


def refine_image(refiner, rgb, item, audit, prediction, synchronize):
    stock = prediction["coco"]
    candidates = audit["candidates"]
    n = len(stock)
    h, w = item["original_hw"]
    require(n > 0 and n == len(candidates) and rgb.shape == (h, w, 3) and rgb.dtype == np.uint8
            and [c["position"] for c in candidates] == list(range(1, n + 1)), "Original candidate ordinals or RGB differ")
    native = np.asarray(codec.decode([row["segmentation"] for row in stock]).transpose(2, 0, 1), dtype=bool)
    original_sizes = np.count_nonzero(native, axis=(1, 2))
    require(native.shape == (n, h, w) and original_sizes.tolist() == [c["mask_area"] for c in candidates],
            "Decoded native masks or areas differ")
    gate = original_sizes >= MINIMUM_AREA
    boxes = np.asarray([(*c["box_xyxy"], c["score"]) for c in candidates], dtype=np.float32)
    labels = np.asarray([row["category_id"] - 1 for row in stock], dtype=np.int64)
    require(boxes.shape == (n, 5) and np.isfinite(boxes).all() and np.all(labels == 0),
            "Exact original single-class boxes and scores required")

    started = perf_counter()
    refined = refiner.refine(rgb, native.copy(order="C"), boxes, labels)
    synchronize()
    inference_seconds = perf_counter() - started
    require(isinstance(refined, np.ndarray) and refined.dtype == np.bool_ and refined.shape == native.shape,
            "Refiner must return boolean native masks in original ordinal order")
    require(np.array_equal(refined[~gate], native[~gate]), "The original pixels of a bypassed mask changed")
    packed = codec.encode(np.asfortranarray(refined.transpose(1, 2, 0), dtype=np.uint8))
    changes = {}
    changed = 0
    for slot, (row, rle, use) in enumerate(zip(stock, packed, gate, strict=True), start=1):
        canonical = dict(size=rle["size"], counts=rle["counts"].decode("ascii"))
        if use:
            changes[str(slot)] = {"rle": canonical}
            changed += int(canonical != row["segmentation"])
        else:
            require(canonical == row["segmentation"], "Canonical RLE bytes of a bypassed mask changed")
    counts = dict(images=1, native_candidates=n, eligible_candidates=len(changes), eligible_masks_changed=changed,
                  bypassed_candidates=n - len(changes), eligible_native_pixels=int(original_sizes[gate].sum()),
                  eligible_refined_pixels=int(np.count_nonzero(refined[gate])))
    return dict(image_id=item["image_id"], changes=changes), counts, inference_seconds


def environment(torch, device):
    libs = {}
    for name in ("numpy", "Pillow", "pycocotools", "mmcv", "mmcv-full", "mmdet"):
        try:
            libs[name] = version(name)
        except PackageNotFoundError:
            continue
    return dict(torch=str(torch.__version__), cuda_runtime=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                torch_default_floating_dtype=str(torch.get_default_dtype()), box_dtype="float32", libraries=libs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code", "stock-root", "point-root", "exposure", "author-dir", "checkpoint", "protocol", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("protocol-sha256", "runtime-sha256", "checkpoint-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true", help="Run only the first canonically ordered image with a gate-positive candidate")
    a = parser.parse_args()
    for key, value in vars(a).items():
        if isinstance(value, Path):
            setattr(a, key, value.resolve(strict=key != "output"))
    require(a.protocol.name == LOCK_NAME and a.protocol_sha256 == LOCK_SHA256 and not a.output.exists(),
            "Published author protocol and fresh output required")
    pin(a.protocol, LOCK_SHA256)
    exposure = json.loads(pin(a.exposure, EXPOSURE_SHA256))
    require(exposure["status"] == EXPOSURE_STATUS and exposure["protocol_sha256"] == LOCK_SHA256
            and exposure["minimum_native_mask_area_pixels"] == MINIMUM_AREA
            and exposure["all_native_candidates_verified_before_labels"] is True, "Admitted stock gate diagnostic required")
    a.cohort, a.support_root = 74, None
    rows, lineage, population_counts = input_population(a, exposure)
    print(encode(dict(phase="native_population_verified", **population_counts)).decode().strip(), flush=True)
    selected = [next(row for row in rows if eligible_count(row) > 0)] if a.smoke else rows

    import torch
    device = torch.device(a.device)
    require((a.smoke or device.type == "cuda")
            and device.type in {"cuda", "cpu"} and (device.type != "cuda" or torch.cuda.is_available()),
            "CUDA device required for the full scientific run")
    metadata = environment(torch, device)
    require(file_digest(a.checkpoint) == a.checkpoint_sha256, "Author checkpoint digest differs")
    runtime = source(Path(__file__).with_name("author_runtime.py"), a.runtime_sha256)
    start = perf_counter()
    refiner = runtime.AuthorRefiner(a.author_dir, a.checkpoint, str(device))
    synchronize = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    synchronize()
    load_seconds = perf_counter() - start
    torch.manual_seed(SEED)

    a.output.mkdir()
    stream_digest = sha256()
    run_counts = Counter()
    decode_seconds = inference_seconds = 0.0
    with (a.output / LEDGER).open("xb") as ledger, torch.inference_mode():
        for index, (item, audit, prediction, _, _) in enumerate(selected, start=1):
            start = perf_counter()
            rgb = decode_rgb(a.stock_root / "conversion", item)
            decode_seconds += perf_counter() - start
            payload, observed, duration = refine_image(refiner, rgb, item, audit, prediction, synchronize)
            require(observed["eligible_candidates"] == eligible_count((item, audit)),
                    "Realized gate support differs from frozen audit")
            encoded = encode(payload)
            require(ledger.write(encoded) == len(encoded), "Incomplete private output row")
            stream_digest.update(encoded)
            run_counts.update(observed)
            inference_seconds += duration
            if index == 1 or index % 10 == 0 or index == len(selected):
                print(encode(dict(phase="masks_verified", images=index, eligible=run_counts["eligible_candidates"]))
                      .decode().strip(), flush=True)

    require(run_counts["native_candidates"] == run_counts["eligible_candidates"] + run_counts["bypassed_candidates"]
            and (a.smoke or all(run_counts[name] == v for name, v in FULL_COUNTS.items())),
            "Realized writer and full-stream counts differ")
    status = SMOKE_STATUS if a.smoke else STATUS
    result = dict(
        status=status, phase="inference", worker_sha256=file_digest(Path(__file__)), protocol_sha256=LOCK_SHA256,
        author_commit=AUTHOR_COMMIT, runtime_sha256=a.runtime_sha256, checkpoint_sha256=a.checkpoint_sha256,
        exposure_sha256=EXPOSURE_SHA256, input_lineage=lineage, full_population_counts=population_counts,
        counts=dict(run_counts), refined_masks_sha256=stream_digest.hexdigest(), refined_masks_filename=LEDGER,
        smoke=a.smoke, scientific_replay=not a.smoke, target_labels_opened=False, optimizer_updates=0,
        detector_forwards=0, segrefiner_image_calls=run_counts["images"], random_seed=SEED,
        minimum_native_mask_area_pixels=MINIMUM_AREA, exact_original_bypass=True, candidate_ordinals_preserved=True,
        environment=metadata, model_load_seconds=load_seconds, jpeg_decode_seconds=decode_seconds,
        transfer_and_inference_seconds=inference_seconds,
    )
    raw = encode(result)
    with (a.output / "RESULT_SAFE.json").open("xb") as receipt:
        require(receipt.write(raw) == len(raw), "Incomplete safe inference receipt")
    print(encode(dict(status=status, result_sha256=sha256(raw).hexdigest())).decode().strip(), flush=True)


if __name__ == "__main__":
    main()
