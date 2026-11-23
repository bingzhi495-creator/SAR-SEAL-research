"""Run the pinned Phoenix binary writer on every nonempty stock Umbra mask."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import re
import sys
from time import perf_counter
from types import ModuleType


LOCK = "49fb049a6986d1f3d34adb7469b3239f5742701723e84f4dce8e60a4425210cb"
PROTOCOL_NAME = "UMBRA12_PHOENIX_WHOLE_PRELOCK_20261123.md"
WHOLE_WORKER = "1a7b423bd681edf3c83ab1c5a71125e3fd20c9bee9b7d3f344a2eae39ae088cd"
CARRIER_CLOSURE = "29f6f58a12cf72cb0da444c24be9939d0dacbab56d6a664f289a1a5f5b88a665"
ROSTER = "e626c38ea53727a2d235a80e675735898bbce5defce698f04fb602d674a87a56"
LINEAGE_FIELDS = {"carrier_closure_sha256", "carrier_protocol_sha256", "anonymous_roster_sha256",
                  "native_intake_manifest_sha256", "native_prediction_ledger_sha256", "native_candidate_audit_sha256",
                  "native_edit_ledger_sha256", "checkpoint_sha256", "ordered_native_population_sha256"}
REQUEST = "PHOENIX_UMBRA12_WHOLE_REQUEST"
REQUEST_COMPLETE = REQUEST + "_COMPLETE"
STATUS = "PHOENIX_UMBRA12_WHOLE_INFERENCE_COMPLETE"
LEDGER = "WHOLE_MASKS_PRIVATE.jsonl"
HW = (1024, 1024)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha_text(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def validate_request(document, protocol_sha):
    require(set(document) == {"status", "protocol_sha256", "lineage", "counts", "images"}
            and document["status"] == REQUEST and document["protocol_sha256"] == protocol_sha,
            "Expected the exact label-free Umbra request schema")
    lineage = document["lineage"]
    require(isinstance(lineage, dict) and set(lineage) == LINEAGE_FIELDS
            and all(sha_text(value) for value in lineage.values())
            and lineage["carrier_closure_sha256"] == CARRIER_CLOSURE and lineage["anonymous_roster_sha256"] == ROSTER,
            "Expected the exact prior native closure and immutable SHA-256 lineage bindings")
    images = document["images"]
    require(isinstance(images, list) and len(images) == 12, "All original twelve Umbra windows required")
    tally = Counter(images=12, native_candidates=0, native_empty=0, nonempty_candidates=0)
    decoded = set()
    projection = sha256()
    for ordinal, row in enumerate(images, 1):
        require(set(row) == {"ordinal", "image_id", "image_file", "hw", "encoded_sha256", "decoded_rgb_sha256",
                             "native_candidate_count", "candidates"}
                and type(row["ordinal"]) is type(row["image_id"]) is int
                and row["ordinal"] == row["image_id"] == ordinal
                and row["image_file"] == f"image_{ordinal:03}.png" and row["hw"] == list(HW)
                and sha_text(row["encoded_sha256"]) and sha_text(row["decoded_rgb_sha256"]),
                "Expected original native input geometry and compact acquisition order")
        decoded.add(row["decoded_rgb_sha256"])
        candidates = row["candidates"]
        n = row["native_candidate_count"]
        require(type(n) is int and 0 <= n <= 100 and isinstance(candidates, list) and len(candidates) <= n
                and all(set(item) == {"position", "segmentation"} and type(item["position"]) is int
                        for item in candidates), "Expected exact nonempty candidate projection")
        positions = [item["position"] for item in candidates]
        require(positions == sorted(set(positions)) and all(1 <= pos <= n for pos in positions),
                "Projected prompts must preserve original compact ship slots")
        tally.update(native_candidates=n, native_empty=n-len(candidates), nonempty_candidates=len(candidates))
        row_text = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        projection.update(row_text.encode("utf-8"))
    counts = document["counts"]
    require(len(decoded) == 12 and isinstance(counts, dict) and all(type(x) is int for x in counts.values())
            and counts == dict(tally) and projection.hexdigest() == lineage["ordered_native_population_sha256"],
            "Closed native input population differs")
    return images


def read_closed_request(args):
    raw = args.request.read_bytes()
    receipt_raw = args.request_result.read_bytes()
    terminal_raw = args.request_terminal.read_bytes()
    receipt, end = json.loads(receipt_raw), json.loads(terminal_raw)
    receipt_sha = sha256(receipt_raw).hexdigest()
    require(args.request.name == "REQUEST_PRIVATE.json" and args.request_result.name == "RESULT_SAFE.json"
            and args.request_result.parent == args.request.parent and args.request_terminal.name == "TERMINAL.json"
            and type(end["actual_exit"]) is int and end["actual_exit"] == 0 and end["output_footer_verified"] is True
            and end["phase"] == receipt["phase"] == "request" and end["result_sha256"] == receipt_sha
            and sha_text(receipt["worker_sha256"]) and end["worker_sha256"] == receipt["worker_sha256"]
            and receipt["status"] == REQUEST_COMPLETE and receipt["target_labels_opened"] is False
            and receipt["protocol_sha256"] == args.protocol_sha256
            and sha256(raw).hexdigest() == args.request_sha256 == receipt["request_sha256"],
            "Successful observed request with matching file identities required")
    document = json.loads(raw)
    validate_request(document, args.protocol_sha256)
    require(all(receipt[k] == document[k] for k in ("lineage", "counts")), "Request and observer lineage differ")
    return document, receipt_sha, sha256(terminal_raw).hexdigest(), receipt["worker_sha256"]


def load_whole(path):
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == WHOLE_WORKER, "Exact previously qualified whole-mask actor required")
    module = ModuleType("_prior_phoenix_whole")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def initialize(args):
    global whole, author
    whole = load_whole(args.whole_worker)
    author = whole.load_baseline(args.baseline_worker)
    author.init_worker(args)


def decode_rle(rle, codec, np):
    require(type(rle) is dict and set(rle) == {"size", "counts"} and rle["size"] == list(HW)
            and type(rle["counts"]) is str and bool(rle["counts"]), "Expected a native COCO RLE prompt")
    mask = codec.decode(dict(size=rle["size"], counts=rle["counts"].encode("ascii"))).astype(np.bool_)
    require(mask.shape == HW and int(mask.sum()) == int(codec.area(rle)) > 0,
            "Every projected prompt must be nonempty on the original native grid")
    return mask


def image_row(task):
    ordinal, row = task
    api = author
    np, torch = api.np, api.torch
    path = api.request_dir / row["image_file"]
    require(path.resolve(strict=True) == path and api.file_sha(path) == row["encoded_sha256"],
            "Image must match observed request identity")
    with api.Image.open(path) as image:
        require(image.format == "PNG" and image.mode == "RGB" and image.size == HW[::-1],
                "Expected native lossless RGB with no resized input")
        rgb = np.array(image, dtype=np.uint8)
    require(api.digest(rgb.tobytes()) == row["decoded_rgb_sha256"]
            and np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) and np.array_equal(rgb[:, :, 0], rgb[:, :, 2]),
            "Expected native grayscale replicated exactly into RGB")

    masks = [(item["position"], decode_rle(item["segmentation"], api.codec, np)) for item in row["candidates"]]
    native_count = row["native_candidate_count"]
    empty_count = native_count - len(masks)

    def reset_seed():
        random.seed(api.SEED)
        np.random.seed(api.SEED)
        torch.random.default_generator.manual_seed(api.SEED)

    reset_seed()
    start = perf_counter()
    if masks:
        with torch.inference_mode():
            api.refiner.set_image(rgb)
    encode_seconds = perf_counter() - start if masks else 0.0

    def decoder(chunk):
        with torch.inference_mode():
            return api.refiner.refine_batch(chunk, refine_iters=5, smooth_kernel=1,
                                            binarize_input=True, multimask_output=True, return_logits=False,
                                            max_batch=2)

    reset_seed()
    records, branch, decode_seconds = whole.whole_policy(
        masks, "lvis", decoder, lambda mask: whole.encode_rle(mask, api.codec, np))
    require(branch["pass_through"] == 0 and branch["decoded_candidates"] == len(masks)
            and [entry["position"] for entry in records] == [pos for pos, _ in masks],
            "Author return must preserve every original nonempty slot, even an empty final mask")
    branch["pass_through"] = empty_count
    counts = dict(images=1, image_encodes=int(bool(masks)), native_candidates=native_count,
                  native_empty=empty_count, nonempty_candidates=len(masks))
    return dict(ordinal=ordinal, image_id=row["image_id"], lvis=records), counts, branch, encode_seconds, decode_seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("source-root", "whole-worker", "baseline-worker", "request", "request-result", "request-terminal",
                 "protocol", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("request-sha256", "protocol-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--workers", type=int, choices=(4,), default=4)
    parser.add_argument("--threads", type=int, choices=(2,), default=2)
    args = parser.parse_args()
    require(sys.flags.isolated and sys.dont_write_bytecode, "Use isolated Python with bytecode writes disabled")
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve(strict=name != "output"))
    old = load_whole(args.whole_worker)
    previous = old.load_baseline(args.baseline_worker)
    require(args.protocol.name == PROTOCOL_NAME and previous.file_sha(args.protocol) == args.protocol_sha256 == LOCK
            and not args.output.exists(), "Exact preregistration and fresh output directory required")
    started = perf_counter()
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS=str(args.threads),
                      MKL_NUM_THREADS=str(args.threads), TORCH_FORCE_WEIGHTS_ONLY_LOAD="1")
    runtime = {key: version(key) for key in old.RUNTIME}
    require(runtime == old.RUNTIME, "Previously qualified DGX CPU runtime required")
    request, receipt_sha, terminal_sha, request_worker_sha = read_closed_request(args)
    images = request["images"]
    previous.verify_author(args.source_root)
    args.output.mkdir()
    ledger = args.output / LEDGER
    totals = Counter(images=0, image_encodes=0, native_candidates=0, native_empty=0, nonempty_candidates=0)
    lvis = Counter(decoded_candidates=0, pass_through=0, decoder_batches=0, empty_outputs=0, unchanged_outputs=0,
                   adds=0, drops=0)
    ledger_digest = sha256()
    encode_seconds = decode_seconds = 0.0
    with ledger.open("xb") as handle, ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                                                           initargs=(args,)) as executor:
        for ordinal, (record, counts, branch, enc, dec) in enumerate(executor.map(image_row, enumerate(images, 1)), 1):
            expected = images[ordinal-1]
            require(record["ordinal"] == record["image_id"] == expected["image_id"]
                    and len(record["lvis"]) == branch["decoded_candidates"] == len(expected["candidates"])
                    and branch["decoded_candidates"] + branch["pass_through"] == expected["native_candidate_count"],
                    "Incomplete original mask positions or frame order")
            totals.update(counts)
            lvis.update(branch)
            encode_seconds += enc
            decode_seconds += dec
            raw = previous.encoded(record)
            require(handle.write(raw) == len(raw), "Incomplete private mask ledger")
            ledger_digest.update(raw)
            if ordinal in (1, 4, 8, 12):
                print(json.dumps(dict(phase="inference", images=ordinal,
                                      decoded_candidates=lvis["decoded_candidates"])), flush=True)
        handle.flush()
        os.fsync(handle.fileno())
    require(all(totals[key] == count for key, count in request["counts"].items())
            and totals["image_encodes"] == sum(bool(row["candidates"]) for row in images)
            and lvis["decoded_candidates"] == totals["nonempty_candidates"] and lvis["pass_through"] == totals["native_empty"]
            and lvis["decoder_batches"] == sum((len(row["candidates"])+1)//2 for row in images)
            and ledger_digest.hexdigest() == previous.file_sha(ledger), "Closed image, policy and writeback census differs")
    lvis["changed_outputs"] = lvis["decoded_candidates"] - lvis["unchanged_outputs"]
    lvis["all_unchanged_outputs"] = lvis["pass_through"] + lvis["unchanged_outputs"]
    lvis["signed_area_delta"] = lvis["adds"] - lvis["drops"]
    lvis["candidate_refinement_iterations"] = lvis["decoded_candidates"] * 5
    result = dict(status=STATUS, phase="inference", worker_sha256=previous.file_sha(Path(__file__)),
                  protocol_sha256=LOCK, request_sha256=args.request_sha256, request_result_sha256=receipt_sha,
                  request_terminal_sha256=terminal_sha, request_worker_sha256=request_worker_sha,
                  whole_worker_sha256=WHOLE_WORKER, baseline_worker_sha256=old.PRIOR_WORKER,
                  lineage=request["lineage"], author_commit=previous.AUTHOR, weight_sha256=previous.WEIGHT_SHA,
                  author_manifest_sha256=previous.MANIFEST_SHA, counts={**totals, "lvis": dict(lvis)},
                  masks_sha256=ledger_digest.hexdigest(), seconds=perf_counter()-started,
                  summed_worker_seconds=dict(encode=encode_seconds, lvis=decode_seconds),
                  workers=args.workers, threads=args.threads, runtime=runtime, seed=previous.SEED, batch_size=2,
                  refine_iters=5, smooth_kernel=1, multimask_output=True, binarize_input=True, binary_writer=True,
                  optimizer_updates=0, native_detector_forwards=0, target_labels_opened=False)
    raw = previous.encoded(result)
    with (args.output / "RESULT_SAFE.json").open("xb") as handle:
        require(handle.write(raw) == len(raw), "Incomplete inference receipt")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(dict(status=STATUS, result_sha256=previous.digest(raw))), flush=True)


if __name__ == "__main__":
    main()
