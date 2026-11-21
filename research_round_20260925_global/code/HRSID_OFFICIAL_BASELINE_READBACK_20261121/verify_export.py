"""Independently read the closed native official TEST export after original session 88756 exits."""

import argparse
from collections import Counter
from functools import lru_cache
from hashlib import sha256
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import re
from struct import pack
import sys

import numpy as np
from pycocotools import mask as codec


EXPORT_SOURCE = "0f9aa1f4be24d604d38db965b8c038031278c669c1a0f10f91174c925b6e4423"
INPUT = "15aabe26b7580c55a5b5f3e4ae1558cac84294ae950f283c826de5b0720e53ee"
MANIFEST = "dd42cbf5ef415dc152bab9160c5eb385ea33255d697fba2dddf49917eda3ee1b"
ENDPOINT = "011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11"
ARCHIVE = "f089fe3417a6c58f3e71f4bc35f8895ac6b136f29c58191b4a90442ab630d159"
PHASE = "HRSID_OFFICIAL_PNG_TEST_NATIVE_BASELINE_EXPORT_ONLY_V1"
RUNTIME = dict(python="3.10.13", numpy="1.26.4", pycocotools="2.0.11", torch="2.10.0+cu130",
               Pillow="12.3.0", rfdetr="1.12.0")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def valid_sha(value):
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def pinned(path, digest):
    require(valid_sha(digest), "64-digit lowercase SHA-256 required")
    with path.open("rb") as src:
        data = src.read(1024**2 + 1)
    require(0 < len(data) <= 1024**2 and sha256(data).hexdigest() == digest, "Pinned bytes of " + path.name)
    return data


def check_terminal(receipt):
    require(receipt.get("status") == "HRSID_OFFICIAL_TEST_NATIVE_EXPORT_ORIGINAL_TERMINAL"
            and str(receipt.get("tool_session_handle")) == "88756"
            and receipt.get("original_process_terminal") is True and type(receipt.get("original_exit")) is int
            and receipt["original_exit"] == 0 and receipt.get("export_source_sha256") == EXPORT_SOURCE
            and all(valid_sha(receipt.get(k)) for k in ("admission_sha256", "result_sha256")),
            "Successful root-observed original terminal for the frozen worker and admission")


def check_bindings(result, admission, receipt):
    common = dict(export_source_sha256=EXPORT_SOURCE, phase=PHASE, device="cpu", endpoint_sha256=ENDPOINT,
                  input_readback_sha256=INPUT, manifest_sha256=MANIFEST, archive_sha256=ARCHIVE)
    require(all(result.get(k) == v == admission.get(k) for k,v in common.items())
            and admission.get("status") == "ROOT_ADMITTED_HRSID_OFFICIAL_TEST_NATIVE_BASELINE_EXPORT_V1"
            and result.get("status") == "HRSID_OFFICIAL_TEST_NATIVE_EXPORT_COMPLETE"
            and result.get("admission_sha256") == receipt["admission_sha256"], "Matching original admission and result")
    require(result["TEST"] is True and all(result[k] is False for k in ("FIT","DEV","GT_opened","AP"))
            and result["images"] == result["model_forward_calls"] == 1962 and result["optimizer_updates"] == 0
            and result["native_joint_rows"] == 196200 and result["precision"] == "float32"
            and type(result["mapped_rows"]) is int and 0 <= result["mapped_rows"] <= 196200
            and type(result["prediction_bytes"]) is int and result["prediction_bytes"] > 0
            and valid_sha(result["predictions_sha256"]), "Original complete label-free native export")


@lru_cache(maxsize=1024)
def raster_evidence(h, w, counts):
    """Bound-check COCO differential runs before C decoding. Verify a canonical round-trip."""
    require(type(counts) is str and bool(counts), "Nonempty compressed RLE string")
    coverage = foreground = n = offset = value = previous = previous_two = 0
    for char in counts:
        code = ord(char) - 48
        require(0 <= code <= 63 and offset <= 25, "Valid RLE group")
        value += (code & 31) << offset
        offset += 5
        if code & 32:
            continue
        if code & 16:
            value -= 1 << offset
        length = value + (previous_two if n > 2 else 0)
        coverage += length
        require(0 <= length and coverage <= h*w and n <= h*w, "Bounded RLE runs")
        if n & 1:
            foreground += length
        previous_two, previous = previous, length
        n += 1
        offset = value = 0
    require(offset == 0 and coverage == h*w, "RLE covers exact native canvas")
    encoded = dict(size=[h,w], counts=counts.encode("ascii"))
    pixels = codec.decode(encoded)
    require(pixels.shape == (h,w) and pixels.dtype == np.uint8 and int(np.count_nonzero(pixels)) == foreground
            and codec.encode(pixels)["counts"] == encoded["counts"], "RLE area and canonical raster reconstruction")
    packed = np.packbits(pixels.ravel(order="F"), bitorder="little")
    return foreground, sha256(packed).digest()


def check_stream(src, manifest, result):
    file_hash, block_hash, geometry_hash = sha256(), sha256(), sha256()
    stats = Counter(mapped_rows=0, images_without_predictions=0, empty_masks=0, summed_mask_area=0,
                    unmapped_slots=0, identical_duplicate_prediction_pairs=0)
    seen = {}
    nbytes = 0
    for row in manifest:
        image_id, h, w = row["image_id"], row["height"], row["width"]
        line = src.readline()
        require(line.endswith(b"\n"), "Complete ordered image record")
        nbytes += len(line)
        file_hash.update(line)
        block_hash.update(sha256(line).digest())
        block = json.loads(line)
        require(set(block) == {"image_id","raw_query_count","native_count","unmapped_slot1_count","native_ranks","coco"}
                and type(block["image_id"]) is int and block["image_id"] == image_id,
                "Original ordered image identity and block schema")
        ranks, rows = block["native_ranks"], block["coco"]
        gaps = block["unmapped_slot1_count"]
        require(type(rows) is list and type(ranks) is list and type(gaps) is int
                and type(block["raw_query_count"]) is int and block["raw_query_count"] == 164
                and type(block["native_count"]) is int and block["native_count"] == 100
                and len(rows) == len(ranks) and all(type(r) is int and 1 <= r <= 100 for r in ranks)
                and ranks == sorted(set(ranks)) and 0 <= gaps == 100 - len(rows), "Native ranks and slot accounting")
        stats["unmapped_slots"] += gaps
        stats["mapped_rows"] += len(rows)
        stats["images_without_predictions"] += int(not rows)
        image_hash = sha256()
        last = 1.
        for rank, prediction in zip(ranks, rows):
            require(set(prediction) == {"image_id","category_id","score","segmentation"}
                    and type(prediction["image_id"]) is int and prediction["image_id"] == image_id
                    and type(prediction["category_id"]) is int and prediction["category_id"] == 1, "Native image/category")
            score = prediction["score"]
            require(type(score) in (float,int) and math.isfinite(score) and 0 <= score <= last, "Native score order")
            last = score
            seg = prediction["segmentation"]
            require(type(seg) is dict and set(seg) == {"size","counts"} and seg["size"] == [h,w], "Native RLE geometry")
            area, digest = raster_evidence(h, w, seg["counts"])
            stats["empty_masks"] += int(area == 0)
            stats["summed_mask_area"] += area
            image_hash.update(pack(">Bd", rank, score) + digest)
        value = image_hash.digest()
        geometry_hash.update(value)
        identity = row["decoded_rgb_sha256"]
        if identity in seen:
            require(value == seen[identity], "Duplicate original RGBs must have identical native outputs")
            stats["identical_duplicate_prediction_pairs"] += 1
        else:
            seen[identity] = value
    require(src.read(1) == b"" and nbytes == result["prediction_bytes"]
            and stats["mapped_rows"] == result["mapped_rows"] and file_hash.hexdigest() == result["predictions_sha256"],
            "Verified original native file SHA, length, candidate count and exact EOF")
    return dict(stats, prediction_bytes=nbytes, predictions_sha256=file_hash.hexdigest(),
                ordered_prediction_block_digests_sha256=block_hash.hexdigest(),
                ordered_rank_score_raster_sha256=geometry_hash.hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("terminal","admission","export-source","result","manifest","predictions","output"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--terminal-sha256", required=True)
    args = parser.parse_args()
    receipt = json.loads(pinned(args.terminal, args.terminal_sha256))
    check_terminal(receipt)
    require(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
            "Isolated CPU readback")
    runtime = dict(python=platform.python_version(), **{key:version(key) for key in RUNTIME if key != "python"})
    require(runtime == RUNTIME, "Exact admitted native runtime distributions")
    require(not args.output.exists() and args.output.parent.is_dir(), "Fresh independent receipt file")
    pinned(args.export_source, EXPORT_SOURCE)
    admission = json.loads(pinned(args.admission, receipt["admission_sha256"]))
    result = json.loads(pinned(args.result, receipt["result_sha256"]))
    check_bindings(result, admission, receipt)
    manifest = json.loads(pinned(args.manifest, MANIFEST))
    ids = [r["image_id"] for r in manifest]
    require(len(manifest) == len(set(ids)) == 1962 and all(type(i) is int for i in ids)
            and all(r["height"] == r["width"] == 800 and valid_sha(r["decoded_rgb_sha256"]) for r in manifest)
            and len({r["decoded_rgb_sha256"] for r in manifest}) == 1961, "Complete original image-only manifest")
    with args.predictions.open("rb") as src:
        before = os.fstat(src.fileno())
        require(before.st_size == result["prediction_bytes"], "Closed prediction size")
        metrics = check_stream(src, manifest, result)
        after = os.fstat(src.fileno())
        require((before.st_size,before.st_mtime_ns) == (after.st_size,after.st_mtime_ns),
                "Stable closed predictions during readback")
    require(metrics["identical_duplicate_prediction_pairs"] == 1
            and metrics["mapped_rows"] + metrics["unmapped_slots"] == 196200, "Complete original census")
    closure = dict(status="HRSID_OFFICIAL_TEST_NATIVE_EXPORT_INDEPENDENT_READBACK_ACCEPTED",
                   original_process_terminal=True, original_exit=0, tool_session_handle=88756,
                   original_terminal_sha256=args.terminal_sha256, result_sha256=receipt["result_sha256"],
                   admission_sha256=receipt["admission_sha256"], export_source_sha256=EXPORT_SOURCE,
                   independent_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                   endpoint_sha256=ENDPOINT, manifest_sha256=MANIFEST, input_readback_sha256=INPUT,
                   images=len(ids), GT_opened=False, AP=False, independent_model_forwards=0,
                   independent_optimizer_updates=0, runtime_versions=runtime, **metrics)
    raw = (json.dumps(closure, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()
    with args.output.open("xb") as dest:
        dest.write(raw)
        dest.flush()
        os.fsync(dest.fileno())
    print(json.dumps(dict(status=closure["status"], readback_sha256=sha256(raw).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
