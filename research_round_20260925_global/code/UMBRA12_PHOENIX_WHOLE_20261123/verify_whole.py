"""Independent original-slot and native-pixel readback of the closed Umbra Phoenix writer."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from types import ModuleType

import numpy as np


HW = (1024, 1024)
LOCK_NAME = "UMBRA12_PHOENIX_WHOLE_PRELOCK_20261123.md"
LOCK = "49fb049a6986d1f3d34adb7469b3239f5742701723e84f4dce8e60a4425210cb"
DECODE_SOURCE = "5a638df3638a8647dc49b81a1f63ee9d72bee2d41dc1e94e276b5a0d2016fdc1"
AUTHOR_REVISION = "13ee3996ef3d0292fab144316cc8e03b69c0ecfa"
WEIGHT = "9b6ba211e0d6d434085d3d5e379eb527d8b56ef413628f42cb587153cc51af7d"
STATUS = "PHOENIX_UMBRA12_WHOLE_GEOMETRY_READBACK_COMPLETE"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, expected):
    raw = path.read_bytes()
    require(digest(raw) == expected, f"Expected exact observed bytes for {path.name}")
    return raw


def load_native_decoder(code):
    source = code / "PSEG232_PHOENIX_SUPPORT_BUDGET_20261123/score_factorial.py"
    worker = ModuleType("umbra_strict_rle")
    worker.__file__ = str(source)
    exec(compile(pin(source, DECODE_SOURCE), str(source), "exec"), worker.__dict__)
    return worker


def observed(run, phase, worker_path):
    worker_sha = digest(worker_path.read_bytes())
    receipt_path = run / phase / "RESULT_SAFE.json"
    terminal_raw = (run / ("observed_" + phase) / "TERMINAL.json").read_bytes()
    end = json.loads(terminal_raw)
    receipt = json.loads(pin(receipt_path, end["result_sha256"]))
    require(end["phase"] == receipt["phase"] == phase and type(end["actual_exit"]) is int and end["actual_exit"] == 0
            and end["output_footer_verified"] is True and receipt["worker_sha256"] == end["worker_sha256"] == worker_sha
            and receipt["protocol_sha256"] == LOCK and receipt["target_labels_opened"] is False,
            "Expected the observed label-free request or refiner execution")
    return receipt, end["result_sha256"], digest(terminal_raw)


def verify_row(decoder, req, written):
    require(set(written) == {"ordinal", "image_id", "lvis"}
            and req["ordinal"] == written["ordinal"] and req["image_id"] == written["image_id"],
            "Returned mask image identities differ")
    stock = req["candidates"]
    rows = written["lvis"]
    n = req["native_candidate_count"]
    require(len(rows) == len(stock) and len(stock) <= n <= 100, "Whole-mask eligibility census differs")
    positions = [row["position"] for row in stock]
    require(all(type(p) is int and 1 <= p <= n for p in positions)
            and positions == sorted(set(positions)) == [row["position"] for row in rows],
            "Returned compact candidate positions differ")
    c = Counter(decoded_candidates=len(stock), pass_through=n-len(stock), decoder_batches=(len(stock)+1)//2,
                empty_outputs=0, unchanged_outputs=0, adds=0, drops=0)
    for candidate, out in zip(stock, rows, strict=True):
        require(set(candidate) == set(out) == {"position", "segmentation"}, "Unexpected candidate field")
        before = decoder.decoded(candidate["segmentation"], HW)
        after = decoder.decoded(out["segmentation"], HW)
        require(bool(before.any()), "Only a nonempty original mask may be changed by the admitted branch")
        adds = int(np.count_nonzero(after & ~before))
        drops = int(np.count_nonzero(before & ~after))
        c.update(adds=adds, drops=drops, empty_outputs=int(not after.any()), unchanged_outputs=int(adds+drops == 0))
    return c


def verify(args):
    start = perf_counter()
    require(args.protocol.name == LOCK_NAME and not args.output.exists(), "Expected the fixed source protocol and a fresh run")
    pin(args.protocol, LOCK)
    req, req_sha, req_end_sha = observed(args.run, "request", args.request_worker)
    returned, infer_sha, infer_end_sha = observed(args.run, "inference", args.inference_worker)
    require(req["status"] == "PHOENIX_UMBRA12_WHOLE_REQUEST_COMPLETE"
            and returned["status"] == "PHOENIX_UMBRA12_WHOLE_INFERENCE_COMPLETE"
            and req["request_sha256"] == returned["request_sha256"]
            and req["worker_sha256"] == returned["request_worker_sha256"]
            and req_sha == returned["request_result_sha256"] and req_end_sha == returned["request_terminal_sha256"]
            and req["lineage"] == returned["lineage"]
            and returned["author_commit"] == AUTHOR_REVISION and returned["weight_sha256"] == WEIGHT
            and tuple(returned[k] for k in ("workers", "threads", "seed", "batch_size", "refine_iters", "smooth_kernel"))
            == (4, 2, 5202, 2, 5, 1)
            and returned["multimask_output"] is returned["binarize_input"] is returned["binary_writer"] is True
            and returned["native_detector_forwards"] == returned["optimizer_updates"] == 0,
            "Request and author whole-mask refiner settings differ")
    request = json.loads(pin(args.run / "request" / "REQUEST_PRIVATE.json", req["request_sha256"]))
    require(request["status"] == "PHOENIX_UMBRA12_WHOLE_REQUEST" and request["protocol_sha256"] == LOCK
            and request["lineage"] == req["lineage"] and request["counts"] == req["counts"],
            "Frozen whole-mask request metadata differs")
    output = pin(args.run / "inference" / "WHOLE_MASKS_PRIVATE.jsonl", returned["masks_sha256"])
    decoder = load_native_decoder(args.code)
    counts = Counter()
    images = request["images"]
    require(len(images) == 12, "Expected all twelve locked images")
    for ordinal, (original, refined) in enumerate(zip(images, map(json.loads, output.splitlines()), strict=True), 1):
        require(original["ordinal"] == ordinal and original["hw"] == list(HW), "Frozen native shape or order differs")
        counts.update(verify_row(decoder, original, refined))
    counts["changed_outputs"] = counts["decoded_candidates"] - counts["unchanged_outputs"]
    counts["all_unchanged_outputs"] = counts["pass_through"] + counts["unchanged_outputs"]
    counts["signed_area_delta"] = counts["adds"] - counts["drops"]
    counts["candidate_refinement_iterations"] = 5 * counts["decoded_candidates"]
    require(counts == returned["counts"]["lvis"] and returned["counts"]["images"] == 12 and
            all(returned["counts"][key] == value for key, value in req["counts"].items()) and
            returned["counts"]["image_encodes"] == sum(bool(row["candidates"]) for row in images) and
            counts["decoded_candidates"] == req["counts"]["nonempty_candidates"] and
            counts["pass_through"] == req["counts"]["native_empty"] and
            counts["pass_through"] + counts["decoded_candidates"] == req["counts"]["native_candidates"],
            "Returned count, edit or verified eligibility census differs")
    return dict(status=STATUS, phase="edits", worker_sha256=digest(Path(__file__).read_bytes()), protocol_sha256=LOCK,
                capture_lineage=req["lineage"], whole_mask_ledger_sha256=returned["masks_sha256"],
                request_result_sha256=req_sha, request_terminal_sha256=req_end_sha,
                inference_result_sha256=infer_sha, inference_terminal_sha256=infer_end_sha,
                exact_original_compact_positions=True, exact_native_binary_pixel_counts=True,
                images=12, optimizer_updates=0, detector_forwards=0, target_labels_opened=False,
                author_inference_seconds=returned["seconds"], verify_seconds=perf_counter()-start,
                coverage_counts=counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ("code", "run", "request-worker", "inference-worker", "protocol", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    result = verify(args)
    raw = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    args.output.mkdir()
    with (args.output / "RESULT_SAFE.json").open("xb") as stream:
        require(stream.write(raw) == len(raw), "Incomplete whole-mask verification receipt")
    print(json.dumps(dict(status=STATUS, result_sha256=digest(raw))), flush=True)


if __name__ == "__main__":
    main()
