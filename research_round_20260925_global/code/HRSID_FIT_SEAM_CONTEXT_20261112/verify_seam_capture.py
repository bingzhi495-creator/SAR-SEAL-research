"""Read back closed seam capture without loading a model or labels."""

from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import re
from time import perf_counter
from types import ModuleType

from PIL import Image


PROTOCOL = "6b0955af80a701175f2a5b1ebdb4f9d82d239df2e18586e5037dc414c850eba2"
MANIFEST = "ec6a168757ebcfc2915b837e8853c286dd0df6f6750b6f8476fde210fc44450b"
CENTER_5202 = "84402a6cf29a6ae428853d3049a7759f97e342defa31ab719b74667ce932ef3a"
RLE_HELPER = "2fb790afa6efc5852222cbfd046bb154380741add9ab33b0b7d96410de03fe20"
PRIOR_CAPTURE = "be37834f7ec657c938e15326ba04a9ac4a6dedaa4bf7a16c173490e93eb49cba"
FIELDS = {"ordinal", "group", "basename", "encoded_sha256", "decoded_rgb_sha256", "encoded_bytes"}


def need(valid, reason):
    if not valid:
        raise ValueError(reason)


def exact(path, digest, max_bytes):
    need(path.is_file() and 0 < path.stat().st_size <= max_bytes, "Unexpected artifact path or size")
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == digest, "Unexpected closed digest: " + path.name)
    return raw


def rle_module(path):
    module = ModuleType("_seam_readback_independent_rle")
    exec(compile(exact(path, RLE_HELPER, 2**20), str(path), "exec"), module.__dict__)
    return module


def inspect_native(block, helper):
    rows = block["native"]
    need(len(rows) == 100, "Original two-slot top-100 census")
    last = None
    by_query = {}
    class_queries = set()
    ships = []
    ordinal = block["ordinal"]
    for rank, row in enumerate(rows, start=1):
        query, slot = row["query"], row["class_slot"]
        score = row["score"]
        need(row["uid"] == f"{ordinal}:{rank}" and row["rank"] == rank and
             0 <= query < 164 and slot in (0, 1) and (query, slot) not in class_queries,
             "Unique stock rank and query-slot identity")
        class_queries.add((query, slot))
        key = (-score, 2*query + slot)
        need(math.isfinite(score) and 0 <= score <= 1 and (last is None or last <= key),
             "Stock score order and stable tie policy")
        last = key
        need(len(row["box"]) == 4 and all(math.isfinite(x) for x in row["box"]) and
             row["box"][0] <= row["box"][2] and row["box"][1] <= row["box"][3],
             "Finite stock xyxy geometry")
        identity = row["area"], row["rle"], row["box"]
        if query not in by_query:
            need(len(helper.foreground_indices(row["rle"])) == row["area"],
                 "Independent native RLE area differs")
            by_query[query] = identity
        else:
            need(by_query[query] == identity, "Shared query geometry differs across classes")
        if slot == 0:
            ships.append(row)
    candidates = [r for r in ships if r["area"] > 0]
    need(block["mapped"] == len(ships) and block["candidates"] == candidates,
         "Exact native nonempty ship-slot subset")
    return len(ships), len(candidates), len(by_query)


def compare_prior(stage):
    safe = json.loads(exact(stage / "RESULT_SAFE.json", PRIOR_CAPTURE, 2**20))
    arm = next(a for a in safe["arms"] if a["seed"] == 5202)
    need(safe["GT_opened"] is False and arm["checkpoint_sha256"] == CENTER_5202,
         "Prior stock same-carrier scope")
    raw = exact(stage / "5202" / "CANDIDATES_PRIVATE.jsonl", arm["prediction_sha256"], 128 * 2**20)
    return {r["basename"]:r for r in map(json.loads, raw.splitlines())}


def main(args):
    started = perf_counter()
    helper = rle_module(args.rle_helper)
    stage = args.closed_stage
    safe = json.loads(exact(stage / "RESULT_SAFE.json", args.result_sha256, 2**20))
    need(safe["status"] == "HRSID_FIT_SEAM_CONTEXT_CAPTURE_ACTUAL" and safe["FIT"] is True
         and all(safe[key] is False for key in ("DEV", "TEST", "GT_opened", "AP"))
         and safe["protocol_sha256"] == PROTOCOL and safe["manifest_sha256"] == MANIFEST
         and safe["checkpoint_sha256"] == CENTER_5202 and safe["optimizer_updates"] == 0,
         "Exact label-free seam capture result")
    manifest = json.loads(exact(args.manifest, MANIFEST, 2**20))
    need(manifest["status"] == "HRSID_FIT_SEAM_CONTEXT_IMAGE_MANIFEST_V1" and
         manifest["FIT"] is True and all(manifest[k] is False for k in ("DEV", "TEST", "GT_fields_present")),
         "Exact image-only manifest")
    images = manifest["images"]
    raw = exact(stage / "CANDIDATES_PRIVATE.jsonl", safe["prediction_sha256"], 128 * 2**20)
    blocks = [json.loads(line) for line in raw.splitlines()]
    need(len(blocks) == len(images) == safe["images"] == safe["model_forwards"] == 144
         and len({r["group"] for r in images}) == safe["source_groups"] == 40
         and len({r["basename"] for r in images}) == 144 and safe["native_rows"] == 14400
         and safe["raw_queries"] == 23616, "Fixed scope and candidate census")
    history = compare_prior(args.previous_capture) if args.previous_capture is not None else {}
    mapped = nonempty = queries = comparable = 0
    for ordinal, (record, block) in enumerate(zip(images, blocks, strict=True), start=1):
        name = record["basename"]
        need(set(record) == FIELDS and ordinal == record["ordinal"] == block["ordinal"]
             and all(record[k] == block[k] for k in ("basename", "group"))
             and record["decoded_rgb_sha256"] == block["rgb_sha256"] and
             re.fullmatch(r"P\d+_\d+_\d+_\d+_\d+\.png", name), "Exact manifest row identity")
        png = exact(args.fit_images / name, record["encoded_sha256"], 4 * 2**20)
        with Image.open(BytesIO(png)) as image:
            need(image.format == "PNG" and image.size == (800, 800) and len(png) == record["encoded_bytes"]
                 and sha256(image.convert("RGB").tobytes()).hexdigest() == record["decoded_rgb_sha256"],
                 "Independent original PNG/RGB identity differs")
        m, n, q = inspect_native(block, helper)
        mapped += m
        nonempty += n
        queries += q
        if name in history:
            previous = history[name]
            fields = ("query", "rank", "score", "area", "rle")
            need(previous["rgb_sha256"] == block["rgb_sha256"] and previous["mapped"] == block["mapped"] and
                 [tuple(c[k] for k in fields) for c in previous["candidates"]] ==
                 [tuple(c[k] for k in fields) for c in block["candidates"]],
                 "Reused same-carrier native image does not match old output")
            comparable += 1
    need((mapped, nonempty) == (safe["mapped"], safe["nonempty"]), "Independent candidate counts differ")
    out = dict(status="HRSID_FIT_SEAM_CONTEXT_CAPTURE_INDEPENDENT_ACCEPTED",
        result_sha256=args.result_sha256, predictions_verified=True, FIT=True, DEV=False, TEST=False,
        GT_opened=False, model_executed=False, native_png_and_rgb_hashes_checked=144,
        source_groups=40, images=144, native_rows=14400, independent_rle_query_masks_checked=queries,
        mapped=mapped, nonempty=nonempty, comparable_same_carrier_prior_images=comparable,
        manifest_sha256=MANIFEST, prediction_sha256=safe["prediction_sha256"], worker_seconds=perf_counter()-started)
    payload = (json.dumps(out, sort_keys=True, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as file:
        file.write(payload)
    print(json.dumps(dict(out, readback_sha256=sha256(payload).hexdigest()), sort_keys=True))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("closed-stage", "manifest", "fit-images", "rle-helper", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--previous-capture", type=Path)
    parser.add_argument("--result-sha256", required=True)
    main(parser.parse_args())
