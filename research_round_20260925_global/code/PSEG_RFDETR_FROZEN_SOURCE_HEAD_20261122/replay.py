"""Frozen HRSID RF point-head readout on a verified target-native RF capture."""
import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np
from pycocotools import mask as codec
import torch


ARMS = ("attraction_dynamic", "attraction_static_at_dynamic_count",
        "learned_at_attraction_count", "fixed", "free_matched_count")
FIT_SHA = "114ac529e2196fa4f60d8fe56e07381fb6814e0fdb09e7d5615626fd74a85f64"
FIT_ACCEPTED_SHA = "476b0f3fa748d1d3fd2393bdeb1af2d83e1532995d58b12625d57222e3f2638a"
BASE = "DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24"
RF = "RFDETR_MATCHED_POINT_FORMAL_20261120"
SPOT = "RFDETR_CORE36_MATCHED_POINT_SPOT_20261120"
PREPARED = "PSEG_TARGET_RF5201_PRELABEL_EDITS_PREPARED"
VERIFIED = "PSEG_TARGET_RF5201_PRELABEL_EDITS_VERIFIED"


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, expected):
    raw = path.read_bytes()
    need(digest(raw) == expected, "Source or artifact identity: " + path.name)
    return raw


def encoded(row):
    return (json.dumps(row, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n").encode()


def read_source(path, expected):
    module = ModuleType(path.stem)
    module.__file__ = str(path)
    exec(compile(pin(path, expected), str(path), "exec"), module.__dict__)
    return module


def methods(code):
    readout = read_source(code / RF / "run_dev_readout.py",
                          "543bc7ffae51cb0de5b247fef59eeae6309a155440021d3ed50a7b3716ec8e9f")
    native = read_source(code / "DFINE_HRSID_TO_SSDD_POINT_TRANSFER_20261120/native_points.py",
                         "475ecbdd932a2129f160e48bd1d838653b971df6cd69b9edda103c037ee3fcdc")
    pair, rank = native.load_old(code / BASE / "fit109_paired_point/pair_math.py",
                                 code / BASE / "fit109_probability_rank/rank_math.py")
    attraction = read_source(code / "FIT109_CLASSICAL_ATTRACTION_20261121/spatial_attraction.py",
                             "8eb162a3bd544929843d7f2aeef174f1f1ba03a2c593b6bcda49289cd94ad3c6")
    free = read_source(code / BASE / "fit109_free_pairing/free_pairing.py",
                       "d24b7050be95786fffdee386e1e8a6f87ddcb8df8b69c2171eb752e6b6a35fef")
    return SimpleNamespace(readout=readout, native=native, pair=pair, rank=rank, attraction=attraction, free=free)


def predictor(code, m, fit, fit_accepted, weights):
    source = json.loads(pin(fit, FIT_SHA))
    closure = json.loads(pin(fit_accepted, FIT_ACCEPTED_SHA))
    record = m.readout.validated_refits(source, closure, FIT_SHA)[5217]
    fit_step = read_source(code / SPOT / "fit_step.py", m.readout.PINS["fit_step.py"])
    plan = read_source(code / RF / "fit_plan.py", m.readout.PINS["fit_plan.py"])
    trainer = fit_step.load_trainer(code / BASE / "fit109_paired_point/train_point.py")
    need(weights.name == record["relative_weights"], "Locked 5217 all-source refit")
    heads, center, scale = plan.restore_bundle(weights, record["weights_sha256"], trainer, fit_step)
    return lambda features: fit_step.predict(heads["S"], features, center, scale)


def canvas(m, hw):
    # This original adapter changes only the four H/W assumptions in paired_points.
    # Its D-FINE coarse sampler is discarded. RF uses FP32 native posteriors throughout.
    pair, _ = m.native.make_native(m.pair, m.rank, *hw)
    def sample(p, coords, width):
        need(width == hw[1] and p.shape == hw, "Exact native rectangular dimensions")
        return p.ravel()[coords]
    return pair, SimpleNamespace(editable_pools=m.rank.editable_pools, sample_probability=sample)


def decisions(m, stock, coords, features, predict):
    q = predict(features)
    need(q.shape == coords.shape and q.dtype == np.float32 and np.isfinite(q).all()
         and np.all((q >= 0) & (q <= 1)), "Frozen source probabilities")
    cap = min(8, int(stock.sum()) // 10, len(coords))
    dynamic, static = m.attraction.select(stock, coords, cap)
    learned = m.free.choose_free(q, cap, fixed_count=len(dynamic))
    paired = m.pair.choose_pairs(q[:, 0] - q[:, 1], cap)
    constrained = [(i, i) for i in paired]
    independent = m.free.choose_free(q, cap, fixed_count=len(paired))
    return dict(zip(ARMS, (dynamic, static, learned, constrained, independent), strict=True))


def mask_edit(stock, coords, actions):
    result = stock.copy(order="C")
    for add, drop in actions:
        result.flat[coords[add, 0]] = True
        result.flat[coords[drop, 1]] = False
    rle = codec.encode(np.asfortranarray(result, dtype=np.uint8))
    return {"swaps": len(actions), "rle": {"size": rle["size"], "counts": rle["counts"].decode("ascii")}}


def check_edits(stock, coords, edits):
    need(set(edits) <= set(ARMS) and coords.ndim == 2 and coords.shape[1] == 2
         and coords.dtype.kind in "iu" and len(coords) <= 32, "Native endpoint geometry")
    need(np.all((coords >= 0) & (coords < stock.size)) and len(np.unique(coords)) == coords.size
         and not stock.flat[coords[:, 0]].any() and stock.flat[coords[:, 1]].all(), "Endpoint signs")
    counts = tuple(edits[name]["swaps"] if name in edits else 0 for name in ARMS)
    need(counts[0] == counts[1] == counts[2] and counts[3] == counts[4], "Exact candidate budgets")
    for e in edits.values():
        k = e["swaps"]
        need(type(k) is int and 0 < k <= min(len(coords), 8, int(stock.sum()) // 10)
             and e["rle"]["size"] == list(stock.shape), "Strict nonzero swap record and canvas")
        other = codec.decode(e["rle"]).astype(bool)
        adds, drops = np.flatnonzero(other & ~stock), np.flatnonzero(stock & ~other)
        need(len(adds) == len(drops) == k and np.isin(adds, coords[:, 0]).all()
             and np.isin(drops, coords[:, 1]).all(), "Exact support, XOR and area conservation")
    return counts


def expected_support(rank_math, stock, box, prob):
    """Independently reconstruct the complete per-side uncertainty-ranked support."""
    if box[2] == box[0] or box[3] == box[1]:
        return ([], [])
    pools, _ = rank_math.editable_pools(stock, box)
    sides = [np.concatenate([pools[s + str(depth)][0] for depth in (1, 2)]) for s in ("add", "drop")]
    n = min(32, len(sides[0]), len(sides[1]))
    return [np.sort(idx[np.lexsort((idx, np.abs(prob.ravel()[idx] - .5)))[:n]]) for idx in sides]


def check_support(m, stock, box, prob, coords):
    sides = expected_support(m.rank, stock, box, prob)
    for i in (0, 1):
        need(np.array_equal(np.sort(coords[:, i]), sides[i]), "Independent full posterior-selected support")
    if len(coords):
        need(np.array_equal(coords, m.pair.pair_assignment(*sides, stock.shape[1])), "Native nearest assignment")


def check_attraction(m, stock, coords, edits, area):
    """Recompute the two deterministic classical decisions independently of the prepared ledger."""
    cap = min(8, area // 10, len(coords))
    dynamic, static = m.attraction.select(stock, coords, cap)
    for arm, decisions_ in zip(ARMS[:2], (dynamic, static), strict=True):
        expected = mask_edit(stock, coords, decisions_) if decisions_ else None
        need(edits.get(arm) == expected, "Exact original dynamic and static attraction decisions")


def native_probability(plane, stock, override):
    tensor = torch.from_numpy(plane).reshape(1, 1, *plane.shape)
    native = torch.nn.functional.interpolate(tensor, size=stock.shape, mode="bilinear", align_corners=False)[0, 0]
    if override is None:
        need(np.array_equal((native > 0).numpy(), stock), "Strict rendered-logit stock parity")
        return native.sigmoid().numpy()
    need(not np.array_equal((native > 0).numpy(), stock) and override.shape == stock.shape
         and override.dtype == np.float32 and np.isfinite(override).all()
         and np.all((override >= 0) & (override <= 1)), "Verified native CUDA posterior for CPU sign disagreement")
    # Exact p=.5 ties can originate from either logit sign; capture checks the strict native sign.
    need(not np.any((override > .5) & ~stock) and not np.any((override < .5) & stock),
         "Original candidate and override probability agree off ties")
    return override


def load_array(root, ref):
    path = (root / ref["path"]).resolve(strict=True)
    need(path.is_relative_to(root.resolve(strict=True)), "Capture-local array")
    raw = pin(path, ref["sha256"])
    stream = BytesIO(raw)
    value = np.load(stream, allow_pickle=False)
    need(len(raw) == ref["bytes"] and not stream.read(1) and value.dtype == np.float32
         and np.isfinite(value).all(), "Exact bounded finite captured plane")
    return value


def load_rgb(images, row):
    from rfdetr.datasets.io_utils import decode_pil_image_bytes

    path = (images / row["source_image_name"]).resolve(strict=True)
    need(path.is_relative_to(images.resolve(strict=True)), "Bound original target image")
    raw = pin(path, row["image_sha256"])
    source, scale = decode_pil_image_bytes(raw, draft_size=None)
    rgb = np.asarray(source)
    need(source.mode == "RGB" and scale == (1., 1.) and rgb.dtype == np.uint8
         and list(rgb.shape) == [*row["original_hw"], 3] and digest(rgb.tobytes()) == row["rgb_sha256"],
         "Original decoded RGB canvas and content")
    return rgb


def replay_image(m, row, candidates, cap_root, counts, *, prepare, image_root, predict, stored):
    hw = tuple(row["original_hw"])
    need(len(hw) == 2 and all(type(x) is int and x > 0 for x in hw), "Original positive height and width")
    ranks, queries = row["native_ranks"], row["query_ids"]
    need(len(candidates) == len(row["native_boxes_xyxy"]) == len(ranks) == len(queries)
         and ranks == sorted(set(ranks)) and all(1 <= r <= 100 for r in ranks)
         and len(set(queries)) == len(queries) and all(0 <= q < 100 for q in queries), "Mapped Q100 ranks")
    pool = {}
    for cand, rank, query, box in zip(candidates, ranks, queries, row["native_boxes_xyxy"], strict=True):
        need(cand["image_id"] == row["image_id"] and cand["category_id"] == 1
             and cand["segmentation"]["size"] == list(hw), "Original mapped identity")
        area = int(codec.area(cand["segmentation"]))
        if 32 <= area < 1024:
            pool[rank] = (query, np.asarray(box, dtype=np.float32), cand, area)
    planes = load_array(cap_root, row["logits_ref"])
    need(list(pool) == row["logit_ranks"] and [x[0] for x in pool.values()] == row["logit_query_ids"]
         and planes.shape == (len(pool), 144, 144), "Exact original eligible stream")
    overrides = {r["native_rank"]: r for r in row["posterior_overrides"]}
    need(len(overrides) == len(row["posterior_overrides"]) and set(overrides) <= pool.keys(), "Override coverage")
    result = dict(image_id=row["image_id"], rgb_sha256=row["rgb_sha256"], points=[], changes={})
    if prepare:
        image = m.pair.image_features(load_rgb(image_root, row))
        pair, sampler = canvas(m, hw)
    else:
        need(set(stored) == set(result) and all(stored[k] == result[k] for k in ("image_id", "rgb_sha256"))
             and [p["rank"] for p in stored["points"]] == list(pool)
             and set(stored["changes"]) <= {str(rank) for rank in pool}, "Complete image, eligible ranks and edited ranks")
    for i, (rank, (query, box, cand, area)) in enumerate(pool.items()):
        stock = codec.decode(cand["segmentation"]).astype(bool)
        override = overrides.get(rank)
        need(override is None or override["query_id"] == query, "Override candidate identity")
        prob = native_probability(planes[i], stock, None if override is None else load_array(cap_root, override))
        need(box.shape == (4,) and np.isfinite(box).all() and np.all(box[2:] >= box[:2])
             and np.all(box[:2] >= 0) and box[2] <= hw[1] and box[3] <= hw[0], "Original clipped author box")
        if prepare:
            data = (pair.paired_points(stock, box, area, cand["score"], prob, image, sampler)
                    if np.all(box[2:] > box[:2]) else None)
            coords = np.empty((0, 2), np.int32) if data is None else data[0]
            result["points"].append(dict(rank=rank, query_id=query, coordinates=coords.tolist()))
            acts = {} if data is None else decisions(m, stock, coords, data[1], predict)
            edits = {arm: mask_edit(stock, coords, choices) for arm, choices in acts.items() if choices}
            if edits:
                result["changes"][str(rank)] = edits
        else:
            p = stored["points"][i]
            need(p["query_id"] == query, "Original query identity")
            coords = np.asarray(p["coordinates"])
            if coords.size == 0:
                need(p["coordinates"] == [], "Explicit unsupported no-op")
                coords = np.empty((0, 2), np.int32)
            check_support(m, stock, box, prob, coords)
            edits = stored["changes"].get(str(rank), {})
        budgets = check_edits(stock, coords, edits)
        if not prepare:
            check_attraction(m, stock, coords, edits, area)
        counts["supported"] += int(bool(len(coords)))
        for name, k in zip(ARMS, budgets, strict=True):
            counts[name + "_edited"] += int(k > 0)
            counts[name + "_swaps"] += k
    counts.update(images=1, mapped=len(candidates), eligible=len(pool))
    return result


STOCK_SHA = "c81599bf9c627e798a78fa47397a10afe11a0a50491330b5fc9ba583fd1b22c4"
CAPTURE_DONE = "PSEG_TARGET_RFDETR5201_DEV187_CAPTURE_COMPLETE_V1"
CAPTURE_ACCEPTED = "PSEG_TARGET_RFDETR5201_DEV187_CAPTURE_INDEPENDENTLY_VERIFIED_V1"


def payloads(a):
    verified = json.loads(pin(a.capture_accepted, a.capture_accepted_sha))
    captured = json.loads(pin(a.capture / "RESULT_SAFE.json", verified["capture_result_sha256"]))
    need(captured["status"] == CAPTURE_DONE and verified["status"] == CAPTURE_ACCEPTED,
         "Independently closed target center 5201 capture required")
    for receipt in (captured, verified):
        need(receipt["images"] == 187 and receipt["GT_opened"] is False
             and receipt["predictions_sha256"] == STOCK_SHA
             and receipt["image_manifest_sha256"] == "78390a66fbf01eea400f6528d699f40f3aa4acbf2129afbbd8d91d8224304bd4",
             "Original label-free target capture and stock")
    need(captured["detector_forwards"] == 187 and verified["detector_forwards"] == 0,
         "Capture once per image, independent verification without detector calls")
    stream_sha = verified["capture_stream_sha256"]
    need(stream_sha == captured["capture_stream_sha256"], "Independent capture stream identity")
    blocks = [json.loads(x) for x in pin(a.capture / "CAPTURE_PRIVATE.jsonl", stream_sha).splitlines()]
    stock = json.loads(pin(a.stock, STOCK_SHA))
    need(len(blocks) == len({r["image_id"] for r in blocks}) == 187 and len(stock) == 16161,
         "Complete stock carrier census")
    return blocks, stock, stream_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "verify"))
    for name in ("code", "capture", "capture-accepted", "stock", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--capture-accepted-sha", required=True)
    for name in ("image-root", "fit-result", "fit-accepted", "weights", "prepared"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--prepared-sha")
    a = parser.parse_args()
    prepare = a.phase == "prepare"
    need(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
         "Isolated CPU-only replay")
    for value in (a.image_root, a.fit_result, a.fit_accepted, a.weights):
        need((value is not None) == prepare, "RGB and source head only for generating predictions")
    need(all((v is None) == prepare for v in (a.prepared, a.prepared_sha)),
         "Verified stage must consume observed original prepared result")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    m = methods(a.code)
    blocks, stock, source_sha = payloads(a)
    worker_sha = digest(Path(__file__).read_bytes())
    args_common = dict(worker_sha256=worker_sha, capture_accepted_sha256=a.capture_accepted_sha,
                       capture_stream_sha256=source_sha, predictions_sha256=STOCK_SHA,
                       fit_result_sha256=FIT_SHA, fit_accepted_sha256=FIT_ACCEPTED_SHA,
                       source_seed=5217, source_head="S", target_seed=5201,
                       FIT=False, DEV=True, TEST=False, GT_opened=False, AP=False,
                       detector_forwards=0, optimizer_updates=0)
    stored = [None] * len(blocks)
    predict = None
    if prepare:
        predict = predictor(a.code, m, a.fit_result, a.fit_accepted, a.weights)
    else:
        pre = json.loads(pin(a.prepared / "RESULT_SAFE.json", a.prepared_sha))
        need(pre["status"] == PREPARED and all(pre[k] == v for k, v in args_common.items()),
             "Exact prior label-free replay")
        stored = [json.loads(x) for x in pin(a.prepared / "EDITS_PRIVATE.jsonl", pre["edits_sha256"]).splitlines()]
    a.output.mkdir()
    started = perf_counter()
    edit_hash, counts = sha256(), Counter()
    out = (a.output / "EDITS_PRIVATE.jsonl").open("xb") if prepare else None
    try:
        cursor = 0
        for row, prior in zip(blocks, stored, strict=True):
            n = row["stock_prediction_count"]
            need(row["stock_prediction_start"] == cursor and 0 <= n <= 100, "Complete contiguous stock slices")
            selected = stock[cursor:cursor+n]
            result = replay_image(m, row, selected, a.capture, counts, prepare=prepare,
                                  image_root=a.image_root, predict=predict, stored=prior)
            cursor += n
            if prepare:
                raw = encoded(result)
                need(out.write(raw) == len(raw), "Complete edit row")
                edit_hash.update(raw)
            if counts["images"] % 25 == 0:
                print(json.dumps(dict(phase=a.phase, images=counts["images"])), flush=True)
        need(cursor == counts["mapped"] == 16161 and counts["images"] == 187, "Complete original population")
        if out:
            out.flush()
            os.fsync(out.fileno())
    finally:
        if out:
            out.close()
    if not prepare:
        need(pre["counts"] == dict(counts), "Independent counts and all edit budgets")
    need(not torch.cuda.is_initialized(), "No detector context")
    summary = dict(**args_common, status=PREPARED if prepare else VERIFIED, arms=list(ARMS), counts=dict(counts),
                   prepared_sha256=a.prepared_sha, seconds=perf_counter() - started,
                   edits_sha256=edit_hash.hexdigest() if prepare else pre["edits_sha256"])
    raw = encoded(summary)
    with (a.output / "RESULT_SAFE.json").open("xb") as stream:
        need(stream.write(raw) == len(raw), "Durable result")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=summary["status"], result_sha256=digest(raw))), flush=True)


if __name__ == "__main__":
    main()
