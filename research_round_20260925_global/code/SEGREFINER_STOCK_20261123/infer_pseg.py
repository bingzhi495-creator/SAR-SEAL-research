"""Run stock SegRefiner-LR on the frozen, label-free PSeg-SSDD native predictions."""
import argparse
from collections import Counter
from hashlib import sha256
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import platform
import random
import sys
from time import perf_counter

import numpy as np
from PIL import Image


EXPOSURE_WORKER = "0ef0bb5f6e0780680f417d3e8ed85824eb83e49a4893fdf51ff0e91905fdd1b3"
EXPOSURE_SAFE = "9d6aea9df9aaf0c8f79a8ec48c27f9dfcd77b89256645e85cdcfbaceff652f7b"
REQUEST_RESULT = "55d1a8f95d9008fc00b609503942f022e0e393407a478d50ce7cea3d4dc3c7f8"
REQUEST_TERMINAL = "bf5b06188a2e18c4c420de53616e3c2944f8adab51273f21a9da99dc9d7ffa4d"
REQUEST_PAYLOAD = "c02de790d0634afdc950ecc4252ccac792daeaec969a60f0f8d933691a0f4c29"
AUTHOR_COMMIT = "53419a2d38ea3da0b6e2be77e5b45e139195a0b3"
STATUS = "PSEG232_SEGREFINER_STOCK_INFERENCE_COMPLETE"
SEED = 5217
NATIVE_GATE = 512
EXPECTED = (232, 23200, 5067)


def line(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest_file(path):
    result = sha256()
    with path.open("rb") as stream:
        while content := stream.read(1024 * 1024):
            result.update(content)
    return result.hexdigest()


def load_module(path, expected):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError(f"Closed module digest differs for {path.name}")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def read_rgb(source, request, image, ordinal, record, item):
    filename = f"image_{ordinal:03d}.png"
    source.require(image["image_file"] == filename and image["image_id"] == item["image_id"]
                   and image["hw"] == record["size"] and image["decoded_rgb_sha256"] == record["rgb"],
                   "Prior RGB image and original native rank block do not align")
    content = source.pin(request / filename, image["encoded_sha256"])
    with Image.open(BytesIO(content)) as picture:
        source.require(picture.mode == "RGB" and picture.format == "PNG" and picture.size == tuple(record["size"][::-1])
                       and source.digest(picture.tobytes()) == record["rgb"], "Expected exact decoded native RGB")
        return np.asarray(picture, dtype=np.uint8).copy(order="C")


def native_inputs(source, record, audit):
    detections = record["native"]
    details = audit["candidates"][:100]
    masks = np.stack([source.decoded(det["segmentation"], record["size"]) for det in detections])
    boxes = np.asarray([(*detail["box_xyxy"], det["score"]) for detail, det in zip(details, detections, strict=True)], dtype=np.float32)
    labels = np.asarray([det["category_id"] for det in detections], dtype=np.int64)
    source.require(boxes.shape == (100, 5) and np.isfinite(boxes).all() and labels.shape == (100,)
                   and masks.dtype == np.bool_ and masks.shape == (100, *record["size"])
                   and np.array_equal(masks.sum(axis=(1, 2)), record["areas"]), "Invalid original masks, detections or categories")
    return masks, boxes, labels


def changed_rows(source, output, masks, native, eligible):
    source.require(isinstance(output, np.ndarray) and output.shape == masks.shape and output.dtype == np.bool_,
                   "Author runtime must return bool native masks in their original candidate order")
    source.require(np.array_equal(output[~eligible], masks[~eligible]), "A candidate below the author area gate was modified")
    rows = {}
    different = 0
    for i, det in enumerate(native):
        revised = source.encode(output[i])
        if eligible[i]:
            rows[str(i+1)] = {"rle": revised}
            different += int(revised != det["segmentation"])
        else:
            source.require(revised == det["segmentation"], "Bypassed mask must preserve its canonical native bytes")
    return rows, different


def qualify_request(source, args, capture, comp):
    receipt, result_hash, terminal_hash = source.observed(
        args.request, args.request_terminal, "request", source.REQUEST_STATUS + "_COMPLETE", source.REQUEST_WORKER,
        (REQUEST_RESULT, REQUEST_TERMINAL))
    source.require(receipt["request_sha256"] == REQUEST_PAYLOAD and receipt["target_labels_opened"] is False
                   and receipt["protocol_sha256"] == source.LOCK and receipt["optimizer_updates"] == receipt["detector_forwards"] == 0,
                   "Previously observed source-only PSeg transport receipt required")
    payload = json.loads(source.pin(args.request / "REQUEST_PRIVATE.json", REQUEST_PAYLOAD))
    lineage, origin = receipt["input_lineage"], receipt["source_replay"]
    source.require(payload["status"] == source.REQUEST_STATUS and payload["protocol_sha256"] == source.LOCK
                   and payload["input_lineage"] == lineage and payload["source_replay"] == origin
                   and lineage["manifest_sha256"] == capture["manifest_sha256"]
                   and lineage["acceptance_sha256"] == capture["acceptance_sha256"]
                   and lineage["predictions_sha256"] == capture["input_predictions_sha256"]
                   and lineage["candidate_audit_sha256"] == capture["candidate_audit_sha256"]
                   and origin["original_readback_sha256"] == comp.READBACK_SHA
                   and origin["classical_worker_sha256"] == source.COMPARE
                   and len(payload["images"]) == receipt["counts"]["images"] == EXPECTED[0], "Request lineage differs from native cohort")
    return payload["images"], result_hash, terminal_hash


def run(args):
    start = perf_counter()
    exposure = load_module(Path(__file__).with_name("exposure_pseg.py"), EXPOSURE_WORKER)
    source = exposure.load_source(args.factorial_source)
    protocol_sha = source.digest(args.protocol.read_bytes())
    source.require(sys.flags.isolated and sys.dont_write_bytecode and not args.output.exists()
                   and args.protocol.name == exposure.PROTOCOL_NAME and protocol_sha == args.protocol_sha256 == exposure.PROTOCOL_SHA256,
                   "Isolated execution, published protocol and fresh output required")
    established = json.loads(source.pin(args.exposure_safe, EXPOSURE_SAFE))
    source.require(established["status"] == exposure.STATUS and established["worker_sha256"] == EXPOSURE_WORKER
                   and established["protocol_sha256"] == protocol_sha
                   and established["stock_min_original_mask_area_pixels"] == NATIVE_GATE,
                   "Previous exposure must bind the same protocol and area gate")
    comp, scorer, _, _, capture, roster, audits, original, blocks = source.historical(args)
    records = exposure.native_records(source, blocks, roster, audits, original)
    excluded = set(scorer.EXCLUDED)
    for view, names in (("official", set(records)), ("exclude_exact_fit_overlap", set(records) - excluded)):
        counts = exposure.area_census((area, records[name]["rgb"]) for name in names for area in records[name]["areas"])
        source.require(counts == established["views"][view]["original_candidates"], "Previously accepted area exposure differs")
    official = established["views"]["official"]["original_candidates"]
    source.require((len(records), official["total"], official["stock_area_admitted"]) == EXPECTED
                   and capture["manifest_sha256"] == established["manifest_sha256"]
                   and capture["input_predictions_sha256"] == established["native_predictions_sha256"]
                   and capture["candidate_audit_sha256"] == established["native_audit_sha256"], "Exposed source cohort differs")
    images, request_hash, terminal_hash = qualify_request(source, args, capture, comp)
    source.require(digest_file(args.checkpoint) == args.checkpoint_sha256, "Pinned author's checkpoint required")
    runtime = load_module(Path(__file__).with_name("author_runtime.py"), args.runtime_sha256)

    import torch

    torch_device = torch.device(args.device)
    if torch_device.type == "cuda":
        source.require(torch.cuda.is_available(), "Committed CUDA runtime is unavailable")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    initialized = perf_counter()
    refiner = runtime.AuthorRefiner(args.author_dir, args.checkpoint, args.device)
    setup_seconds = perf_counter() - initialized
    counts = Counter()
    author_seconds = 0.0
    args.output.mkdir()
    ledger = sha256()
    with (args.output / "EDITS_PRIVATE.jsonl").open("xb") as stream:
        for ordinal, (image, item, audit) in enumerate(zip(images, roster, audits, strict=True), 1):
            record = records[item["image_ref"]["path"]]
            rgb = read_rgb(source, args.request, image, ordinal, record, item)
            masks, boxes, labels = native_inputs(source, record, audit)
            eligible = np.asarray(record["areas"]) >= NATIVE_GATE
            started = perf_counter()
            output = refiner.refine(rgb, masks.copy(), boxes.copy(), labels.copy())
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)
            author_seconds += perf_counter() - started
            rows, different = changed_rows(source, output, masks, record["native"], eligible)
            source.require(len(rows) == int(eligible.sum()), "Only stock area-eligible rows may be serialized")
            raw = line(dict(ordinal=ordinal, image_id=item["image_id"], changes=rows))
            source.require(stream.write(raw) == len(raw), "Incomplete private inference ledger")
            ledger.update(raw)
            counts.update(images=1, native_candidates=len(masks), area_admitted=len(rows),
                          exact_native_bypass=len(masks)-len(rows), changed_area_admitted=different,
                          unchanged_area_admitted=len(rows)-different)
            if ordinal == 1 or ordinal % 10 == 0 or ordinal == EXPECTED[0]:
                print(json.dumps(dict(phase="native_stock_progress", **counts)), flush=True)
    source.require(tuple(counts[key] for key in ("images", "native_candidates", "area_admitted")) == EXPECTED,
                   "Unexpected realized image, candidate or author area-gate count")
    return dict(status=STATUS, phase="inference", worker_sha256=source.digest(Path(__file__).read_bytes()),
                protocol_sha256=protocol_sha, runtime_sha256=args.runtime_sha256, author_commit=AUTHOR_COMMIT,
                checkpoint_sha256=args.checkpoint_sha256, exposure_safe_sha256=EXPOSURE_SAFE,
                request_result_sha256=request_hash, request_terminal_sha256=terminal_hash, request_sha256=REQUEST_PAYLOAD,
                historical_source_sha256=exposure.SOURCE_SHA256, manifest_sha256=capture["manifest_sha256"],
                native_predictions_sha256=capture["input_predictions_sha256"], native_audit_sha256=capture["candidate_audit_sha256"],
                stock_min_original_mask_area_pixels=NATIVE_GATE, edits_sha256=ledger.hexdigest(), counts=dict(counts),
                runtime=dict(device=args.device, gpu=torch.cuda.get_device_name(torch_device) if torch_device.type == "cuda" else None,
                             python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                             torch_cuda=torch.version.cuda, seed=SEED),
                setup_seconds=setup_seconds, author_inference_seconds=author_seconds, total_seconds=perf_counter()-start,
                target_labels_opened=False, detector_forwards=0, optimizer_updates=0, bypass_verified_native_exact=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("factorial-source", "code", "owner", "capture", "classical-root", "request", "request-terminal", "exposure-safe",
                 "author-dir", "checkpoint", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--original-readback", type=Path)
    for name in ("checkpoint-sha256", "runtime-sha256", "protocol-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = run(args)
    raw = line(result)
    with (args.output / "RESULT_SAFE.json").open("xb") as stream:
        if stream.write(raw) != len(raw):
            raise OSError("Incomplete safe inference receipt")
    print(json.dumps(dict(status=STATUS, result_sha256=sha256(raw).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
