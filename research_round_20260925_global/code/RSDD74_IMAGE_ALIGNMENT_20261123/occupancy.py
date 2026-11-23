"""Replay frozen occupancy heads on the original and two horizontally displaced image inputs."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType


LOCK = "b5a74bebb9f2ab99c523d57a6074cd978fb23916dda8c9075ea008e51177f2e6"
BUILDER = "da5ff070df83b3f4ab190baf494a71daee1f1fb9eff335aa255c2a20984b5021"
REQUEST_RESULT = "08ba5d901c7c6d0b17a8b17cd8a3bb5f646139e66e2aae2459554cacb0fabd2a"
REQUEST_TERMINAL = "f0e8fbaa487e0819c5f6e7ac2ecaadcee86edbf7d871accb27600021089008de"
REQUEST = "a40c0b8a3b6473a4c4f1f6c58c04f9cf6070f34c3ef7331783134a1dd77447ac"
SEEDS = (5217, 5218, 5219)
SHIFTS = ("original", "minus", "plus")
STATUS = "RSDD74_IMAGE_ALIGNMENT_OCCUPANCY_COMPLETE"
RUNTIME = {"torch": "2.10.0+cu130", "numpy": "1.26.4", "Pillow": "12.3.0",
           "pycocotools": "2.0.11", "rfdetr": "1.12.0"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_builder(code):
    path = code / "INSTANCE_RSDD_PHOENIX_EQUAL_COUNT_20261123/build_request.py"
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == BUILDER, "Historical builder differs")
    module = ModuleType("_phoenix_original_request")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def source_population(a, builder):
    from PIL import Image

    summary, result_sha, terminal_sha = builder.observed(
        a.request, a.request_terminal, "request", builder.STATUS, REQUEST_RESULT, REQUEST_TERMINAL, BUILDER)
    require(summary["phase"] == "request" and summary["protocol_sha256"] == builder.LOCK
            and summary["request_sha256"] == REQUEST and summary["counts"] == builder.TOTALS
            and summary["target_labels_opened"] is False and summary["detector_forwards"] == 0,
            "Qualified image-only request required")
    request = json.loads(builder.pin(a.request / "REQUEST_PRIVATE.json", REQUEST))
    rows, ledger, lineage, source = builder.native_inputs(a)
    require(request["status"] == builder.REQUEST and request["protocol_sha256"] == builder.LOCK
            and request["source_replay"] == summary["source_replay"] == source
            and request["input_lineage"] == summary["input_lineage"] == lineage, "Original CFA/request linkage differs")
    images = request["images"]
    for ordinal, (image, row, archive) in enumerate(zip(images, rows, ledger, strict=True), 1):
        item = row[0]
        require(image["image_id"] == item["image_id"] == archive["image_id"]
                and image["image_file"] == f"image_{ordinal:03d}.png"
                and image["decoded_rgb_sha256"] == item["rgb_sha256"], "Original population order differs")
        body = builder.pin(a.request / image["image_file"], image["encoded_sha256"])
        with Image.open(BytesIO(body)) as png:
            require(png.format == "PNG" and png.mode == "RGB" and png.size == (512, 512)
                    and sha256(png.tobytes()).hexdigest() == item["rgb_sha256"], "Original PNG/stock RGB differ")
    return rows, ledger, images, dict(request_sha256=REQUEST, request_result_sha256=result_sha,
                                     request_terminal_sha256=terminal_sha, source_replay=source, input_lineage=lineage)


def initialize(a, items):
    global cfa, w
    builder = load_builder(a.code)
    cfa = builder.module(a.code / "INSTANCE_RSDD_CLASSICAL_FAIR_MATCH_20261123/replay.py", builder.CFA_SHA)
    cfa.initialize(a, items)
    w = cfa.w
    require(set(w.models) == set(SEEDS) and w.torch.get_float32_matmul_precision() == "high",
            "Original head seeds and numerical precision required")


def shifted_rgb(rgb):
    """Native row indexing stays fixed. Replicate the edge exposed by each horizontal shift."""
    import numpy as np

    require(rgb.shape == (512, 512, 3) and rgb.dtype == np.uint8, "Original native RGB required")
    minus = np.empty_like(rgb)
    minus[:, :-1] = rgb[:, 1:]
    minus[:, -1] = rgb[:, -1]
    plus = np.empty_like(rgb)
    plus[:, 1:] = rgb[:, :-1]
    plus[:, 0] = rgb[:, 0]
    return {"original": rgb, "minus": minus, "plus": plus}


def appearance_at(planes, coords):
    np = w.np
    # paired_points appends intensity, local 3/5 means, and signed x/y image gradients.
    y, x = np.divmod(coords, 512)
    return np.moveaxis(planes[:, y, x], 0, -1)


def edit_candidate(stock, coords, features, image_planes, original, k, counts):
    np = w.np
    cap = min(8, int(stock.sum()) // 10, len(coords))
    require(0 < k <= cap and features.shape == (len(coords), 2, 14) and features.dtype == np.float32
            and np.array_equal(features[:, :, 9:], appearance_at(image_planes["original"], coords)),
            "Original 14-channel occupancy input differs")
    edits = {}
    for shift in SHIFTS:
        # The physical coordinates, all nine non-image channels, and the feature dtype are frozen.
        current = features.copy()
        current[:, :, 9:] = appearance_at(image_planes[shift], coords)
        require(np.array_equal(current[:, :, :9], features[:, :, :9])
                and (shift != "original" or np.array_equal(current, features)), "Non-image point inputs differ")
        for seed in SEEDS:
            head, center, scale = w.models[seed]
            q = w.step.predict(head, current, center, scale)
            require(q.shape == coords.shape and np.isfinite(q).all(), "Invalid endpoint probabilities")
            counts["point_head_forwards"] += 1
            chosen = w.methods.free.choose_free(q, cap, k)
            adds, drops = cfa.common.select(coords, q, k, k)
            require(np.array_equal(coords[[i for i,j in chosen], 0], adds)
                    and np.array_equal(coords[[j for i,j in chosen], 1], drops),
                    "Independent native endpoint ties or signed order differ")
            entry = cfa.write_mask(stock, coords, chosen, k)
            baseline = original[f"{seed}/learned"]
            if shift == "original":
                require(entry == baseline, "Reconstructed original signed mask differs from the verified CFA archive")
                counts[f"{seed}_original_exact"] += 1
            else:
                edits[f"{seed}/{shift}"] = entry
                counts[f"{seed}_{shift}_differs_from_original"] += int(entry != baseline)
    return edits


def image_replay(work):
    np = w.np
    rowset, archive, request = work
    item, audit, pred, previous, unused = rowset
    image_id = item["image_id"]
    require(unused is None and image_id == audit["image_id"] == pred["image_id"] == previous["image_id"]
            == archive["image_id"] == request["image_id"] and not audit["posterior_overrides"]
            and audit["original_hw"] == [512, 512], "Original image/capture identity differs")
    rgb = w.base.load_rgb(w.a.image_root, audit)
    require(sha256(rgb.tobytes(order="C")).hexdigest() == item["rgb_sha256"] == request["decoded_rgb_sha256"],
            "Rendered image/request identity differs")
    image_planes = {s: w.methods.pair.image_features(p) for s,p in shifted_rgb(rgb).items()}
    planes = w.base.load_array(w.a.stock_root / "capture", audit["logits_ref"])
    logits = dict(zip(audit["logit_ranks"], planes, strict=True))
    require(planes.shape == (len(logits), 144, 144), "Native detached probability planes differ")
    lookup = {r["position"]: r for r in request["candidates"]}
    require(list(lookup) == sorted(lookup) and len(lookup) == len(request["candidates"])
            and set(archive["changes"]) == {str(p) for p in lookup}, "Positive-budget roster differs")
    changes = {}
    counts = Counter(images=1, mapped_candidates=len(pred["coco"]), eligible_planes=len(logits))
    for pos, (info, det) in enumerate(zip(audit["candidates"], pred["coco"], strict=True), 1):
        rank, area = info["native_rank"], int(w.codec.area(det["segmentation"]))
        require((info["position"], rank, info["query_id"], info["score"]) ==
                (pos, audit["native_ranks"][pos-1], audit["query_ids"][pos-1], det["score"])
                and area == info["mask_area"] and (rank in logits) == (32 <= area < 1024),
                "Candidate rank, area, score or query linkage differs")
        box = np.asarray(info["box_xyxy"], dtype=np.float32)
        if rank not in logits or np.any(box[:2] == box[2:]):
            require(pos not in lookup, "Ineligible or degenerate candidate acquired a budget")
            continue
        stock = w.codec.decode(det["segmentation"]).astype(bool)
        probability = w.base.native_probability(logits[rank], stock, None)
        data = w.pair.paired_points(stock, box, area, det["score"], probability, image_planes["original"], w.sampler)
        coords = data[0] if data is not None else np.empty((0, 2), dtype=np.int32)
        w.base.check_support(w.methods, stock, box, probability, coords)
        cap = min(8, area // 10, len(coords))
        dynamic, static = cfa.attraction.select(stock, coords, cap) if cap else ([], [])
        k = len(dynamic)
        require(len(static) == k <= cap and (pos in lookup) == bool(k), "Original budget or fallback differs")
        if not k:
            continue
        req = lookup[pos]
        require(req["k"] == k and req["native_rank"] == rank and req["query_id"] == info["query_id"]
                and req["score"] == det["score"] and req["segmentation"] == det["segmentation"]
                and np.array_equal(coords, req["coords"]), "Original request signed coordinates or mask differ")
        old = archive["changes"][str(pos)]
        require(cfa.write_mask(stock, coords, dynamic, k) == old["dynamic"]
                and cfa.write_mask(stock, coords, static, k) == old["static"], "Original classical RLE differs")
        changes[str(pos)] = edit_candidate(stock, coords, data[1], image_planes, old, k, counts)
        counts.update(positive_budget_candidates=1, swaps_per_arm=k, endpoint_pairs=len(coords))
    require(set(changes) == set(archive["changes"]), "Not all positive native slots were covered")
    return dict(image_id=image_id, changes=changes), counts


def run(a):
    started = perf_counter()
    require(sha256(a.protocol.read_bytes()).hexdigest() == a.protocol_sha256 == LOCK
            and not a.output.exists(), "Exact prelock and fresh output required")
    require(sys.flags.isolated and sys.dont_write_bytecode and sys.version_info[:3] == (3,10,13)
            and all(version(k) == v for k,v in RUNTIME.items()), "Qualified B5 CPU runtime required")
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    a.cohort, a.support_root, a.image_root = 74, None, a.stock_root / "conversion/images"
    builder = load_builder(a.code)
    rows, archive, images, chain = source_population(a, builder)
    a.output.mkdir()
    counts, ledger = Counter(), sha256()
    with (a.output / "EDITS_PRIVATE.jsonl").open("xb") as stream, ProcessPoolExecutor(
            max_workers=a.workers, initializer=initialize, initargs=(a, [r[0] for r in rows])) as pool:
        work = zip(rows, archive, images, strict=True)
        for ordinal, (entry, tally) in enumerate(pool.map(image_replay, work, chunksize=1), 1):
            require(entry["image_id"] == rows[ordinal-1][0]["image_id"], "Historical image order differs")
            raw = builder.encode(entry)
            require(stream.write(raw) == len(raw), "Incomplete edits ledger")
            ledger.update(raw)
            counts.update(tally)
            if ordinal in (1, 25, 50, 74):
                print(json.dumps(dict(phase="edits", images=ordinal, total=74)), flush=True)
    require(all(counts[k] == v for k,v in builder.TOTALS.items())
            and all(counts[f"{s}_original_exact"] == 4107 for s in SEEDS)
            and counts["point_head_forwards"] == 4107 * len(SEEDS) * len(SHIFTS),
            "Full closed population, budgets and all-seed parity required")
    ledger_sha = ledger.hexdigest()
    require(sha256((a.output / "EDITS_PRIVATE.jsonl").read_bytes()).hexdigest() == ledger_sha,
            "Completed written ledger readback differs")
    result = dict(status=STATUS, phase="edits", cohort=74, seeds=list(SEEDS), shifts=["minus", "plus"],
                  protocol_sha256=LOCK, worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  edits_sha256=ledger_sha, **chain, original_all_seeds_exact=True, per_candidate_signed_counts_exact=True,
                  non_image_features_exact=True, static_geometry_exact=True, counts=dict(counts), runtime=RUNTIME,
                  target_labels_opened=False, GT_opened=False, detector_forwards=0, optimizer_updates=0,
                  workers=a.workers, wall_seconds=perf_counter()-started)
    result_sha = builder.write(a.output / "RESULT_SAFE.json", builder.encode(result))
    print(json.dumps(dict(status=STATUS, result_sha256=result_sha)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ("code", "stock-root", "point-root", "cfa-verify", "cfa-terminal", "request",
                "request-terminal", "weights-dir", "protocol", "output"):
        p.add_argument("--"+key, required=True, type=Path)
    p.add_argument("--protocol-sha256", required=True)
    p.add_argument("--workers", type=int, default=4, choices=range(1,9))
    a = p.parse_args()
    for key, value in vars(a).items():
        if isinstance(value, Path):
            setattr(a, key, value.resolve(strict=key != "output"))
    run(a)


if __name__ == "__main__":
    main()
