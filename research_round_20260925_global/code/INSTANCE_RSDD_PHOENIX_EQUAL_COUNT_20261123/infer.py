"""Run the fixed Phoenix CPU path and retain only admitted endpoint probabilities."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha1, sha256
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import sys
from time import perf_counter

LOCK = "0160a2ee707bb95d2c8342cec229cc0c06e0edb4d9860e857701c23cf6b61052"
AUTHOR = "13ee3996ef3d0292fab144316cc8e03b69c0ecfa"
MANIFEST_SHA = "92d5084f8c7d7bceb949fd099489edc5cd835b939dae49959c1df4de026fb853"
WEIGHT_SHA = "9b6ba211e0d6d434085d3d5e379eb527d8b56ef413628f42cb587153cc51af7d"
REQUEST_SHA_STATUS = "PHOENIX_RSDD74_EQUAL_COUNT_REQUEST"
STATUS = "PHOENIX_RSDD74_EQUAL_COUNT_INFERENCE_COMPLETE"
HW = (512, 512)
SEED = 5202
REFINE_ITERS = 5
BATCH = 2
AUTHOR_WEIGHT = "phoenix_efficientvit_xl1.pt"
OUTPUT_NAME = "ENDPOINT_PROBABILITIES_PRIVATE.jsonl"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def file_sha(path):
    checksum = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def encoded(data):
    return (json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def verify_author(root):
    manifest_path = root / "SOURCE_ACQUISITION_SAFE.json"
    body = manifest_path.read_bytes()
    require(digest(body) == MANIFEST_SHA, "Exact previously accepted author acquisition manifest")
    manifest = json.loads(body)
    require(manifest["revision"] == AUTHOR and len(manifest["files"]) == 46, "Fixed author revision required")
    for name, blob in manifest["files"].items():
        path = root / name
        require(path.resolve(strict=True) == path and path.is_relative_to(root), "Regular pinned source path required")
        raw = path.read_bytes()
        require(sha1(f"blob {len(raw)}\x00".encode("ascii") + raw).hexdigest() == blob, "Author git blob mismatch")
    weight = root / AUTHOR_WEIGHT
    require(weight.stat().st_size == 814868059 and file_sha(weight) == WEIGHT_SHA, "Pinned official instance weight required")


def block_network(event, _):
    if event.startswith("socket."):
        raise PermissionError("Phoenix worker has no external dependency calls")


def init_worker(a):
    global refiner, torch, np, codec, Image, request_dir
    import FastGeodis
    import numpy as np
    import torch
    from PIL import Image
    from pycocotools import mask as codec

    request_dir = a.request.parent
    os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    require(version("FastGeodis") == "1.0.5" and not torch.cuda.is_initialized(), "Qualified CPU geodesic runtime")
    sys.path.insert(0, str(a.source_root))
    import phoenix
    require(Path(phoenix.__file__).resolve() == a.source_root / "phoenix/__init__.py", "Pinned local source import")
    checkpoint = a.source_root / AUTHOR_WEIGHT
    require(not torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint), "Tensor-only official checkpoint")
    with torch.inference_mode():
        model = phoenix.build_phoenix(checkpoint=str(checkpoint), encoder="efficientvit_xl1", device="cpu", strict=True)
    refiner = phoenix.PhoenixRefiner(model)
    sys.addaudithook(block_network)


def candidate(item):
    rle = item["segmentation"]
    require(set(rle) == {"size", "counts"} and rle["size"] == list(HW) and isinstance(rle["counts"], str),
            "Original full-grid stock COCO RLE")
    mask = codec.decode(dict(size=rle["size"], counts=rle["counts"].encode("ascii"))).astype(bool)
    pts = np.asarray(item["coords"])
    area = int(mask.sum())
    require(mask.shape == HW and 32 <= area < 1024 and type(item["k"]) is int
            and pts.ndim == 2 and pts.shape[1] == 2 and pts.dtype.kind in "iu"
            and 1 <= item["k"] <= min(8, area // 10, len(pts)) <= len(pts) <= 32
            and (pts >= 0).all() and (pts < mask.size).all()
            and len(np.unique(pts)) == pts.size and not mask.flat[pts[:, 0]].any() and mask.flat[pts[:, 1]].all(),
            "Admitted native signed position and exact positive count required")
    return mask, pts, area


def image_row(data):
    ordinal, row = data
    name = f"image_{ordinal:03}.png"
    require(row["image_file"] == name, "Canonical image roster filename")
    path = request_dir / name
    require(path.resolve(strict=True) == path and file_sha(path) == row["encoded_sha256"], "Bound native RGB file")
    with Image.open(path) as f:
        require(f.format == "PNG" and f.mode == "RGB" and f.size == HW[::-1], "Lossless native RGB input")
        rgb = np.array(f, dtype=np.uint8)
    require(digest(rgb.tobytes()) == row["decoded_rgb_sha256"], "Exact source RGB bytes")
    items = row["candidates"]
    totals = Counter(images=1)
    begin = perf_counter()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.random.default_generator.manual_seed(SEED)
    if items:
        with torch.inference_mode():
            refiner.set_image(rgb)
        totals["image_encodes"] += 1
    encode_seconds = perf_counter() - begin if items else 0.0
    start = perf_counter()
    results = []
    for offset in range(0, len(items), BATCH):
        chapter = items[offset:offset + BATCH]
        loaded = [candidate(x) for x in chapter]
        with torch.inference_mode():
            masks, maps = refiner.refine_batch(
                [x[0] for x in loaded], refine_iters=REFINE_ITERS, smooth_kernel=1,
                binarize_input=True, multimask_output=True, return_logits=True, max_batch=BATCH,
            )
        totals["decoder_batches"] += 1
        for item, (source, coords, area), refined, prob in zip(chapter, loaded, masks, maps, strict=True):
            require(refined.shape == prob.shape == source.shape == HW and refined.dtype == np.bool_
                    and prob.dtype == np.float32 and np.isfinite(prob).all()
                    and (prob >= 0.0).all() and (prob <= 1.0).all(),
                    "Finite original-grid f32 selected probability")
            q = prob.flat[coords]
            tied = [len(q) - len(np.unique(q[:, c])) for c in range(2)]
            totals.update(decoded_candidates=1, candidate_refinement_iterations=REFINE_ITERS,
                          native_count_sum=item["k"], endpoint_pairs=len(q),
                          default_area_rule_active=int(area * 100 > source.size),
                          equal_score_extra_add=tied[0], equal_score_extra_drop=tied[1],
                          constant_add_scores=int(tied[0] == len(q) - 1),
                          constant_drop_scores=int(tied[1] == len(q) - 1),
                          full_output_empty=int(not refined.any()))
            results.append(dict(position=item["position"], probabilities=q.tolist()))
    return dict(image_id=row["image_id"], candidates=results), totals, encode_seconds, perf_counter() - start


def read_request(path, wanted):
    raw = path.read_bytes()
    require(digest(raw) == wanted, "Exact requested image-only roster required")
    obj = json.loads(raw)
    require(set(obj) == {"status", "protocol_sha256", "input_lineage", "source_replay", "images"}
            and obj["status"] == REQUEST_SHA_STATUS and obj["protocol_sha256"] == LOCK, "Request schema differs")
    images = obj["images"]
    require(len(images) == len({row["image_id"] for row in images}) == 74
            and sum(len(row["candidates"]) for row in images) == 4107, "Complete fixed reserve roster required")
    for ordinal, row in enumerate(images, 1):
        require(set(row) == {"image_id", "image_file", "encoded_sha256", "decoded_rgb_sha256", "candidates"}
                and type(row["image_id"]) is int and row["image_file"] == f"image_{ordinal:03}.png",
                "Exact original image identity")
        positions = [x["position"] for x in row["candidates"]]
        ranks = [x["native_rank"] for x in row["candidates"]]
        require(positions == sorted(set(positions)) and ranks == sorted(set(ranks))
                and all(type(p) is int and 1 <= p <= r <= 100 for p, r in zip(positions, ranks, strict=True)),
                "Fixed full ledger order projected onto positive-count candidates")
        require(all(set(x) == {"position", "native_rank", "query_id", "score", "segmentation", "k", "coords"}
                    and type(x["query_id"]) is int and 0 <= x["query_id"] < 164
                    and isinstance(x["score"], (int, float)) and math.isfinite(x["score"]) and 0 <= x["score"] <= 1
                    for x in row["candidates"]), "Candidate roster schema differs")
    return obj


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("source-root", "request", "protocol", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--threads", type=int, choices=range(1, 3), default=2)
    a = parser.parse_args()
    for key, val in vars(a).items():
        if isinstance(val, Path):
            setattr(a, key, val.resolve(strict=key != "output"))
    require(a.protocol_sha256 == LOCK == file_sha(a.protocol) and not a.output.exists(), "Fixed lock and unique output")
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS=str(a.threads),
                      MKL_NUM_THREADS=str(a.threads), TORCH_FORCE_WEIGHTS_ONLY_LOAD="1")
    runtime = {name: version(name) for name in ("FastGeodis", "numpy", "torch", "Pillow", "pycocotools")}
    require(runtime["FastGeodis"] == "1.0.5" and runtime["numpy"] == "1.26.4"
            and runtime["torch"] == "2.10.0+cu130", "Prior qualified DGX CPU runtime")
    start = perf_counter()
    verify_author(a.source_root)
    request = read_request(a.request, a.request_sha256)
    a.output.mkdir()
    sums, hashes = Counter(), sha256()
    encode_s = decode_s = 0.0
    with (a.output / OUTPUT_NAME).open("xb") as f, ProcessPoolExecutor(
            max_workers=a.workers, initializer=init_worker, initargs=(a,)) as pool:
        for i, (row, tally, enc, dec) in enumerate(pool.map(image_row, enumerate(request["images"], 1)), 1):
            require(row["image_id"] == request["images"][i - 1]["image_id"], "Exact output roster order required")
            line = encoded(row)
            require(f.write(line) == len(line), "Complete endpoint record")
            hashes.update(line)
            sums.update(tally)
            encode_s += enc
            decode_s += dec
            if i % 5 == 0 or i in (1, 74):
                print(json.dumps(dict(phase="inference", images=i, decoded_candidates=sums["decoded_candidates"])), flush=True)
        f.flush()
        os.fsync(f.fileno())
    require(sums["images"] == 74 and sums["decoded_candidates"] == 4107 and sums["native_count_sum"] == 29900
            and sums["default_area_rule_active"] == 0 and sums["candidate_refinement_iterations"] == 4107 * REFINE_ITERS,
            "Full fixed candidate census, author default and actual model iterations")
    result = dict(status=STATUS, phase="inference", protocol_sha256=LOCK, author_commit=AUTHOR, weight_sha256=WEIGHT_SHA,
                  request_sha256=a.request_sha256, worker_sha256=file_sha(Path(__file__)), endpoint_sha256=hashes.hexdigest(),
                  source_replay=request["source_replay"], input_lineage=request["input_lineage"], counts=dict(sums),
                  seconds=perf_counter() - start, summed_image_encode_seconds=encode_s, summed_decode_seconds=decode_s,
                  workers=a.workers, threads=a.threads, runtime=runtime, seed=SEED, batch_size=BATCH, refine_iters=REFINE_ITERS,
                  optimizer_updates=0, native_detector_forwards=0, target_labels_opened=False)
    raw = encoded(result)
    with (a.output / "RESULT_SAFE.json").open("xb") as f:
        f.write(raw)
    print(json.dumps(dict(status=STATUS, result_sha256=digest(raw))), flush=True)


if __name__ == "__main__":
    main()
