"""Replay equal-count classical and frozen learned order on Instance-RSDD 74."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType


LOCK = "d826e08a75572cb643a5b4880f92529dadb27a78fe14696451ca7a3ae6738693"
INPUT_PATH = "POINTREND_SIGNED_QUOTA_SWAP_20261123/inputs.py"
INPUT_SHA = "2c3b84f852bb4f8234d935347708a2e71eba4466d0eb111471bd254aa2f45e58"
COMMON_PATH = "POINTREND_SIGNED_QUOTA_SWAP_20261123/replay_quota_swap.py"
COMMON_SHA = "0ccbbb90640990481f25b4aae5e91c877ef9634f05c826f6638a6b12e2ee4f9a"
ATTRACTION_PATH = "FIT109_CLASSICAL_ATTRACTION_20261121/spatial_attraction.py"
ATTRACTION_SHA = "8eb162a3bd544929843d7f2aeef174f1f1ba03a2c593b6bcda49289cd94ad3c6"
PREPARED = "INSTANCE_RSDD_CLASSICAL_FAIR_MATCH_PREPARED"
VERIFIED = "INSTANCE_RSDD_CLASSICAL_FAIR_MATCH_VERIFIED"
LEDGER = "EDITS_PRIVATE.jsonl"
HW = (512, 512)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def module(path, expected):
    raw = path.read_bytes()
    require(digest(raw) == expected, "Pinned source differs: " + path.name)
    obj = ModuleType(path.stem)
    obj.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), obj.__dict__)
    return obj


def initialize(args, items):
    global w, attraction, common
    common = module(args.code / COMMON_PATH, COMMON_SHA)
    common.initialize(args, items)
    w = common.w
    attraction = module(args.code / ATTRACTION_PATH, ATTRACTION_SHA)


def write_mask(stock, coords, order, count):
    np = w.np
    out = w.base.mask_edit(stock, coords, order)
    mask = w.codec.decode(out["rle"]).astype(bool)
    a, d = common.signed(stock, mask)
    same_add = np.array_equal(np.flatnonzero(mask & ~stock), np.sort(coords[[i for i, _ in order], 0]))
    same_drop = np.array_equal(np.flatnonzero(stock & ~mask), np.sort(coords[[j for _, j in order], 1]))
    require(len(order) == out["swaps"] == a == d == count and same_add and same_drop,
            "Mask cannot change its signed quotas or selected points")
    return out


def replay_image(rowset):
    np = w.np
    item, audit, pred, previous, unused = rowset
    image_id = item["image_id"]
    require(unused is None and image_id == audit["image_id"] == pred["image_id"] == previous["image_id"]
            and audit["original_hw"] == list(HW) and not audit["posterior_overrides"], "Closed source image identity")
    planes = w.base.load_array(w.a.stock_root / "capture", audit["logits_ref"])
    logits = dict(zip(audit["logit_ranks"], planes, strict=True))
    require(planes.shape == (len(logits), 144, 144), "Expected original query planes")
    image = w.methods.pair.image_features(w.base.load_rgb(w.a.image_root, audit))
    require(w.torch.get_float32_matmul_precision() == "high", "Original occupancy matmul precision required")
    changes = {}
    counts = Counter(images=1, mapped_candidates=len(pred["coco"]), eligible_planes=len(logits))
    for pos, (info, det) in enumerate(zip(audit["candidates"], pred["coco"], strict=True), 1):
        key = str(pos)
        rank, area = info["native_rank"], info["mask_area"]
        require((info["position"], rank, info["query_id"], info["score"]) ==
                (pos, audit["native_ranks"][pos - 1], audit["query_ids"][pos - 1], det["score"])
                and area == int(w.codec.area(det["segmentation"]))
                and (rank in logits) == (32 <= area < 1024), "Original candidate rank, geometry or area differs")
        old = previous["changes"].get(key, {}).get("learned_common")
        if rank not in logits:
            require(old is None, "Frozen out-of-area fallback")
            continue
        box = np.asarray(info["box_xyxy"], dtype=np.float32)
        if np.any(box[:2] == box[2:]):
            require(old is None, "Frozen degenerate fallback")
            counts["degenerate"] += 1
            continue
        counts["rendered"] += 1
        stock = w.codec.decode(det["segmentation"]).astype(bool)
        probability = w.base.native_probability(logits[rank], stock, None)
        data = w.pair.paired_points(stock, box, area, det["score"], probability, image, w.sampler)
        coords, features = data[:2] if data is not None else (np.empty((0, 2), np.int32), None)
        w.base.check_support(w.methods, stock, box, probability, coords)
        cap = min(8, area // 10, len(coords))
        if cap == 0:
            require(old is None, "Frozen no-support fallback")
            continue
        counts["nonempty_support"] += 1
        dynamic, static = attraction.select(stock, coords, cap)
        k = len(dynamic)
        require(len(static) == k <= cap, "Dynamic realized edit count must define both controls")
        ranked = {"dynamic": dynamic, "static": static}
        for seed, (head, center, scale) in w.models.items():
            if seed != 5217 and k == 0:
                continue
            q = w.step.predict(head, features, center, scale)
            require(q.shape == coords.shape and np.isfinite(q).all(), "Finite original occupancy scores required")
            counts["point_head_forwards"] += 1
            if seed == 5217:
                natural = w.methods.free.choose_free(q, cap)
                require((w.base.mask_edit(stock, coords, natural) if natural else None) == old,
                        "Archived seed-5217 natural ranks or mask differ")
            if k:
                chosen = w.methods.free.choose_free(q, cap, k)
                independent = common.select(coords, q, k, k)
                require(np.array_equal(coords[[i for i, _ in chosen], 0], independent[0])
                        and np.array_equal(coords[[j for _, j in chosen], 1], independent[1]),
                        "Exact independent signed-rank parity required")
                ranked[f"{seed}/learned"] = chosen
        if k:
            counts.update(positive_budget_candidates=1, swaps_per_arm=k)
            edits = {arm: write_mask(stock, coords, order, k) for arm, order in ranked.items()}
            for seed in w.models:
                variant = edits[f"{seed}/learned"]["rle"]
                for comparator in ("dynamic", "static"):
                    counts[f"{seed}_differs_from_{comparator}"] += int(variant != edits[comparator]["rle"])
            changes[key] = edits
    return dict(image_id=image_id, changes=changes), counts


def prior_edit_receipt(a, worker_sha):
    if a.phase == "prepare-edits":
        require(a.prepare is None and a.prepare_terminal is None, "Preparation has no intervening readback")
        return None, None, None
    require(a.prepare is not None and a.prepare_terminal is not None, "Observed closed preparation required")
    output = (a.prepare / "RESULT_SAFE.json").read_bytes()
    terminal = a.prepare_terminal.read_bytes()
    p, t = json.loads(output), json.loads(terminal)
    require(p["status"] == PREPARED and p["phase"] == t["phase"] == "prepare-edits"
            and p["protocol_sha256"] == LOCK and p["cohort"] == 74
            and p["target_labels_opened"] is False and p["model_forwards"] == p["optimizer_updates"] == 0
            and t["actual_exit"] == 0 and t["output_footer_verified"] is True
            and t["result_sha256"] == digest(output) and t["worker_sha256"] == p["worker_sha256"] == worker_sha,
            "Preparation must have an observed successful terminal and exact worker and protocol")
    return p, digest(output), digest(terminal)


def execute(a):
    clock = perf_counter()
    worker_sha = digest(Path(__file__).read_bytes())
    prior, prior_sha, terminal_sha = prior_edit_receipt(a, worker_sha)
    common_api = module(a.code / COMMON_PATH, COMMON_SHA)
    source = module(a.code / INPUT_PATH, INPUT_SHA)
    rows, lineage = source.load_population(a)
    require(len(rows) == lineage["images"] == 74, "Complete reserve roster")
    a.image_root = a.stock_root / "conversion/images"
    a.output.mkdir()
    counts, ledger = Counter(), sha256()
    items = [row[0] for row in rows]
    if prior is not None:
        require(prior["input_lineage"] == lineage and prior["workers"] != a.workers,
                "Independent execution must reuse exact inputs and change process partition")
    from contextlib import nullcontext
    compare = (a.prepare / LEDGER).open("rb") if prior else nullcontext()
    with compare as reference, (a.output / LEDGER).open("xb") as out, ProcessPoolExecutor(
            max_workers=a.workers, initializer=initialize, initargs=(a, items)) as pool:
        for ordinal, (row, tally) in enumerate(pool.map(replay_image, rows, chunksize=2), 1):
            require(row["image_id"] == items[ordinal - 1]["image_id"], "Original image order required")
            counts.update(tally)
            line = common_api.encode(row)
            if reference is not None:
                require(reference.readline() == line, "Independent replay row bytes differ")
            require(out.write(line) == len(line), "Complete output row")
            ledger.update(line)
            if ordinal in (1, 25, 50, 74):
                print(json.dumps(dict(phase=a.phase, images=ordinal, total=74)), flush=True)
        require(reference is None or reference.read(1) == b"", "Exact closed prepare ledger length")
    require(tuple(counts[k] for k in ("images", "mapped_candidates", "eligible_planes", "rendered", "nonempty_support")) ==
            (74, 7346, 4137, 4136, 4114), "Known complete 74-chip roster and support census")
    if prior is not None:
        require(prior["counts"] == dict(counts) and prior["edits_sha256"] == ledger.hexdigest(),
                "Complete repeated support and edit totals")
    result = dict(phase=a.phase, status=VERIFIED if prior else PREPARED, cohort=74, images=74,
                  seeds=list(common_api.HEADS), counts=dict(counts), input_lineage=lineage,
                  edits_sha256=ledger.hexdigest(), worker_sha256=worker_sha, protocol_sha256=LOCK,
                  input_source_sha256=INPUT_SHA, common_source_sha256=COMMON_SHA,
                  classical_source_sha256=ATTRACTION_SHA, frozen_head_sha256=common_api.HEADS,
                  prepare_result_sha256=prior_sha, prepare_terminal_sha256=terminal_sha,
                  exact_independent_edit_replay=prior is not None, native_changed_counts_verified=True,
                  original_seed_5217_natural_rank_verified=True, target_labels_opened=False, GT_opened=False,
                  model_forwards=0, detector_forwards=0, optimizer_updates=0,
                  workers=a.workers, runtime=a.runtime, seconds=perf_counter() - clock)
    value = common_api.encode(result)
    with (a.output / "RESULT_SAFE.json").open("xb") as stream:
        stream.write(value)
    print(json.dumps(dict(status=result["status"], result_sha256=digest(value))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("phase", choices=("prepare-edits", "verify-edits"))
    parser.add_argument("--cohort", required=True, type=int, choices=(74,))
    for name in ("code", "stock-root", "point-root", "weights-dir", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("prepare", "prepare-terminal"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    a = parser.parse_args()
    for key, value in vars(a).items():
        if isinstance(value, Path):
            setattr(a, key, value.resolve(strict=key != "output"))
    require(not a.output.exists() and a.protocol_sha256 == LOCK == digest(a.protocol.read_bytes()),
            "Unique output and exact published experimental protocol required")
    a.support_root = None
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    a.runtime = {name: version(name) for name in ("torch", "numpy", "Pillow", "pycocotools", "rfdetr")}
    require(sys.version_info[:3] == (3, 10, 13)
            and a.runtime == dict(torch="2.10.0+cu130", numpy="1.26.4", Pillow="12.3.0",
                                   pycocotools="2.0.11", rfdetr="1.12.0"), "Qualified B5 CPU runtime required")
    execute(a)


if __name__ == "__main__":
    main()
