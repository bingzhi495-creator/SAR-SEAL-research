"""Prepare immutable FIT109 stock candidates, endpoint features and fractional targets."""
import argparse
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

import numpy as np


STATUS = "RFDETR_FIT109_SIGNED_RANK_SOURCE_COMPLETE"
CAPTURE = "7f3f6dd112d287623360c8e3a174bae3a12a039c7f5505e4bf7c5d76b2e5ed6f"
FEATURES = "1cea85eb8c30b11bfa847292c55d2231fb4d16dd2a551c3da52ddffdd8d3830e"
PRELABEL = "33fc0e63fa5deff68aa384e03a3f00fd78430e63a7680b7a5a8025e33422acf4"
BASIS = "fc759d9cf1982f3970f3b75627f8eccbca0b474c97ecb141ae487ba7dacc3e94"
BASIS_RUN = "0a6112ad17d15a3846b6eca28a100455958dd28db4f195cd3ceeeac29a0c3657"
BASIS_TERMINAL = "ae28b31e9b16003d0c171d517768ab63659becb6ef53759f6324086515f21eef"
FIT_RUN = "114ac529e2196fa4f60d8fe56e07381fb6814e0fdb09e7d5615626fd74a85f64"
FIT_READBACK = "476b0f3fa748d1d3fd2393bdeb1af2d83e1532995d58b12625d57222e3f2638a"
FIT_TERMINAL = "16bdb45ae7252040760aff702aeceda8d9b68deb59ffd65fab8aa7c0740f99e3"
FORMAL = "RFDETR_MATCHED_POINT_FORMAL_20261120"
SPOT = "RFDETR_CORE36_MATCHED_POINT_SPOT_20261120"
PAIR = "DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/fit109_paired_point"
ROLES = {"unmatched": 0, "neutral": 1, "valid": 2}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, expected):
    raw = path.read_bytes()
    require(digest(raw) == expected, "Exact original artifact: " + path.name)
    return raw


def module(path, expected):
    loaded = ModuleType("_signed_rank_" + path.stem)
    loaded.__file__ = str(path)
    exec(compile(pin(path, expected), str(path), "exec"), vars(loaded))
    return loaded


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def acquire(a, cohort, spot, codec):
    a.acceptance_sha256 = CAPTURE
    roster, capture = cohort.read_population(a, spot)
    basis = json.loads(pin(a.basis / "INDEPENDENT_READBACK_SAFE.json", BASIS))
    terminal = json.loads(pin(a.basis / "ORIGINAL_TOOL_WAIT_PRIVATE.json", BASIS_TERMINAL))
    pin(a.basis / "RESULT_SAFE.json", BASIS_RUN)
    require(terminal["terminal"]["exit_code"] == 0 and terminal["result_sha256"] == basis["result_sha256"] == BASIS_RUN
            and basis["original_terminal_receipt_sha256"] == BASIS_TERMINAL
            and basis["status"] == "RFDETR_FIT109_QUERY_BASIS_INDEPENDENTLY_ACCEPTED"
            and basis["independent_readback_accepted"] is True and basis["old_features_reproduced"] is True
            and basis["GT_opened"] is basis["DEV"] is basis["TEST"] is False
            and basis["private_file_count"] == len(roster) == 109 and basis["accepted_capture_sha256"] == CAPTURE
            and basis["feature_sha256"] == FEATURES and basis["prelabel_sha256"] == PRELABEL,
            "Exact previously closed label-free FIT109 point basis")
    stocks = [json.loads(line) for line in pin(a.capture / "STOCK_NATIVE_PREDICTIONS_PRIVATE.jsonl",
                                              capture["stock_predictions_sha256"]).splitlines()]
    files, out = [], []
    feature_hash = sha256()
    counts = Counter()
    for (item, ordinal, block), stock in zip(roster, stocks, strict=True):
        image_id, group = item["image_id"], item["group"]
        candidates = spot.native_candidate_rows(block)
        coco = [dict(image_id=c["image_id"], category_id=c["category_id"], score=c["score"], segmentation=c["rle"])
                for c in candidates]
        require(stock["image_id"] == image_id and stock["coco"] == coco and
                stock["native_ranks"] == [c["rank"] for c in candidates], "Full original stock candidate and score order")
        for c in candidates:
            c["area"] = int(codec.area(c["rle"]))
        by_key = {(row["rank"], row["query_id"]): row["area"] for row in candidates}
        raw = (a.basis / f"NATIVE_ENDPOINT_BASIS_{ordinal:02d}_PRIVATE.npz").read_bytes()
        files.append(dict(ordinal=ordinal, sha256=digest(raw), bytes=len(raw)))
        with np.load(BytesIO(raw), allow_pickle=False) as z:
            array_id, ranks, qids, offset, coords, features = (
                z[key] for key in ("image_id", "ranks", "query_ids", "offsets", "coords", "old14"))
        require(int(array_id) == image_id and len(ranks) == len(qids) and offset.shape == (len(ranks) + 1,)
                and len(offset) >= 1 and offset[0] == 0 and offset[-1] == len(coords) == len(features)
                and (np.diff(ranks) > 0).all() and (np.diff(offset) > 0).all() and (np.diff(offset) <= 32).all()
                and coords.shape == (len(coords), 2) and coords.dtype == np.int32
                and features.shape == (len(coords), 2, 14) and features.dtype == np.float32
                and np.isfinite(features).all() and ((0 <= features[..., 0]) & (features[..., 0] <= 1)).all()
                and ((0 <= coords) & (coords < 800 * 800)).all(), "Exact original native feature shard")
        points = []
        for rank, qid, start, stop in zip(ranks.tolist(), qids.tolist(), offset[:-1], offset[1:], strict=True):
            require((rank, qid) in by_key and 32 <= by_key[(rank, qid)] < 1024,
                    "Same original supported area and rank/query identity")
            keys = dict(image_id=image_id, rank=rank, query_id=qid, group=group)
            c, x = coords[start:stop], features[start:stop]
            require(len(np.unique(c)) == c.size, "Unique addition/removal endpoints")
            feature_hash.update(spot.encode(keys))
            feature_hash.update(c.tobytes())
            feature_hash.update(x.tobytes())
            points.append(dict(keys, coordinates=c, features=x))
        counts.update(images=1, mapped=len(candidates), supported=len(points), endpoint_pairs=len(coords),
                      images_without_support=int(not points))
        out.append((item, candidates, points))
    require(digest(encode(files)) == basis["private_output_sha256"] and feature_hash.hexdigest() == FEATURES,
            "Every original private array digest and exact ordered 14-channel feature seal")
    require(counts == Counter(images=109, mapped=10488, supported=6421, endpoint_pairs=203903, images_without_support=1),
            "Whole original source geometry roster")
    return out, counts, capture


def archived_models(a, cohort, plan, step):
    receipt = json.loads(pin(a.heads / "INDEPENDENT_SAFE.json", FIT_READBACK))
    fit = json.loads(pin(a.heads / "RESULT_SAFE.json", FIT_RUN))
    tool = json.loads(pin(a.heads / "ORIGINAL_TOOL_WAIT_PRIVATE.json", FIT_TERMINAL))
    pre = json.loads(pin(a.heads / "PRELABEL_SAFE.json", PRELABEL))
    require(receipt["fit_result_sha256"] == FIT_RUN and receipt["run_natural_exit"] == 0
            and receipt["original_tool_wait_sha256"] == FIT_TERMINAL and tool["terminal"]["exit_code"] == 0
            and receipt["checkpoint_bundles_verified"] == len(fit["all_initializations"]) == 18
            and receipt["head_updates_verified"] == fit["head_updates"] == 7200
            and fit["detector_checkpoint_sha256"] == cohort.CHECKPOINT_SHA
            and fit["prelabel_sha256"] == PRELABEL and pre["feature_sha256"] == FEATURES
            and pre["accepted_capture_sha256"] == CAPTURE,
            "Original independent 3 seed, 5 fold plus all-source model admission")
    trainer = step.load_trainer(a.code / PAIR / "train_point.py")
    heads = {}
    for entry in fit["all_initializations"]:
        seed, fold = entry["seed"], entry["fold"]
        if fold is None:
            continue
        filename = f"{seed}_fold_{fold}.npz"
        require(entry["relative_weights"] == filename and (seed, fold) not in heads, "Original frozen held head identity")
        models, center, scale = plan.restore_bundle(
            a.heads / "POINT_HEADS_PRIVATE" / filename, entry["weights_sha256"], trainer, step)
        heads[seed, fold] = models["S"], center, scale
    require(set(heads) == {(s, f) for s in plan.SEEDS for f in range(5)}, "All original 15 held heads")
    return heads, fit


def fixed_budgets(grouped, heads, folds, plan, step, pair):
    rows = []
    for item, candidates, points in grouped:
        fold = folds[item["group"]]
        areas = {r["rank"]: r["area"] for r in candidates}
        for row in points:
            x = row["features"]
            cap = min(8, areas[row["rank"]] // 10, len(x))
            candidates_k = []
            for seed in plan.SEEDS:
                q = step.predict(heads[seed, fold][0], x, heads[seed, fold][1], heads[seed, fold][2])
                candidates_k.append(len(pair.choose_pairs(q[:, 0] - q[:, 1], cap)))
            rows.append(candidates_k)
    out = np.asarray(rows, dtype=np.int8)
    require(out.shape == (6421, 3) and np.count_nonzero(out[:, 0]) == 6310 and out[:, 0].sum() == 46581,
            "Original published 5217 held-fold positive distance-paired cardinality")
    return out


def pack_points(roster, flat, records, folds):
    stock = {(r["image_id"], r["rank"], r["query_id"]): i for i, r in enumerate(flat)}
    require(len(stock) == len(flat), "Original unique image/rank/query stock candidates")
    roles, ids, offsets, positions = [], [], [0], []
    target, truth_area, native_intersection = [], [], []
    for r in records:
        chosen = stock[r["image_id"], r["rank"], r["query_id"]]
        positions.append(chosen)
        roles.append(ROLES[r["role"]])
        ids.append(-1 if r["matched_gt_id"] is None else r["matched_gt_id"])
        offsets.append(offsets[-1] + len(r["coordinates"]))
        region = r["raw_continuous"]
        if r["role"] == "valid":
            q = r["truth"]
            require(region["predicted_area"] == flat[chosen]["area"] and q.shape == r["coordinates"].shape
                    and np.isfinite(q).all() and (q >= 0).all() and (q <= 1 + 1e-12).all(),
                    "Fixed original fractional geometry and source hull identity")
            target.append(q)
            truth_area.append(region["truth_area"])
            native_intersection.append(region["intersection"])
        else:
            require(r["truth"] is None and region is None, "No supervision for nonvalid matches")
            target.append(np.full(r["coordinates"].shape, np.nan, dtype=np.float64))
            truth_area.append(np.nan)
            native_intersection.append(np.nan)
    require(all(a < b for a, b in zip(positions, positions[1:])), "Original immutable candidate point order")
    return dict(image_id=np.asarray([r["image_id"] for r in records], dtype=np.int64),
                group=np.asarray([r["group"] for r in records]),
                fold=np.asarray([folds[r["group"]] for r in records], dtype=np.int8),
                rank=np.asarray([r["rank"] for r in records], dtype=np.int16),
                query_id=np.asarray([r["query_id"] for r in records], dtype=np.int16),
                candidate_index=np.asarray(positions, dtype=np.int32), offset=np.asarray(offsets, dtype=np.int32),
                coordinates=np.concatenate([r["coordinates"] for r in records]).astype(np.int32, copy=False),
                features=np.concatenate([r["features"] for r in records]).astype(np.float32, copy=False),
                truth=np.concatenate(target).astype(np.float64, copy=False), role=np.asarray(roles, dtype=np.uint8),
                matched_gt_id=np.asarray(ids, dtype=np.int64), native_area=np.asarray([flat[i]["area"] for i in positions], dtype=np.int32),
                truth_area=np.asarray(truth_area, dtype=np.float64), native_intersection=np.asarray(native_intersection, dtype=np.float64),
                roster_image_id=np.asarray([r["image_id"] for r in roster], dtype=np.int64),
                roster_group=np.asarray([r["group"] for r in roster]),
                roster_fold=np.asarray([folds[r["group"]] for r in roster], dtype=np.int8))


def run(a):
    started = perf_counter()
    require(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
            and np.__version__ == "1.26.4" and version("pycocotools") == "2.0.11" and not a.output.exists(),
            "Qualified fresh FIT-only CPU preparation")
    cohort = module(a.code / FORMAL / "run_fit109.py", "9f989c3a976642d9e790d65199dda43da4ae6b946d649fa13d1106258e99f84e")
    spot = module(a.code / SPOT / "run_spot.py", "75ff72a3177e99f06d212a878c8d8921de4db6da8f4249722fb762718083d0ee")
    plan = module(a.code / FORMAL / "fit_plan.py", "1fd0fe8f815b5ad4e01d0d8eeca0aa0e1480433292fc48f94fee17e5837e5c35")
    step = module(a.code / SPOT / "fit_step.py", "94e60f5b5557787687d1f228c33c249a1d7889172a225373b97bd789ec8b72b1")
    pair = module(a.code / PAIR / "pair_math.py", spot.PAIR_SHA)
    torch = step.torch
    require(torch.__version__ == "2.10.0+cu130" and not torch.cuda.is_initialized(), "Original qualified point runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    from pycocotools import mask as codec

    grouped, counts, capture = acquire(a, cohort, spot, codec)
    heads, fit = archived_models(a, cohort, plan, step)
    roster = [item for item, _, _ in grouped]
    folds = plan.source_folds([i["group"] for i in roster])
    archived_k = fixed_budgets(grouped, heads, folds, plan, step, pair)
    print(json.dumps(dict(stage="EXACT_PRELABEL_FEATURE_AND_FOLD_ADMISSION", **counts)), flush=True)
    records, labels = spot.attach(a, grouped, step)
    require(labels == fit["supervision"] and len(records) == counts["supported"] and
            len({r["group"] for r in records if r["role"] == "valid"}) == 92,
            "Original label geometry, all hulls, raster anchors and source supervision counts")
    flat = [row for _, candidates, _ in grouped for row in candidates]
    arrays = pack_points(roster, flat, records, folds)
    arrays.update(archived_f_k=archived_k, archived_seeds=np.asarray(plan.SEEDS, dtype=np.int16))
    a.output.mkdir()
    path = a.output / "POINTS_PRIVATE.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(path, allow_pickle=False) as readback:
        require(set(readback.files) == set(arrays) and all(
            digest(readback[k].tobytes()) == digest(value.tobytes()) for k, value in arrays.items()),
            "Exact authorized private table roundtrip")
    stock = b"".join(encode(dict(image=item, fold=folds[item["group"]], candidates=candidates))
                     for item, candidates, _ in grouped)
    with (a.output / "STOCK_PRIVATE.jsonl").open("xb") as stream:
        stream.write(stock)
    result = dict(status=STATUS, worker_sha256=digest(Path(__file__).read_bytes()), points_sha256=digest(path.read_bytes()),
                  stock_sha256=digest(stock), original_stock_sha256=capture["stock_predictions_sha256"],
                  feature_sha256=FEATURES, original_basis_readback_sha256=BASIS,
                  original_capture_acceptance_sha256=CAPTURE, original_fit_result_sha256=FIT_RUN,
                  original_fit_readback_sha256=FIT_READBACK, original_fit_terminal_sha256=FIT_TERMINAL,
                  original_detector_sha256=cohort.CHECKPOINT_SHA, source_cohort_sha256=cohort.SOURCE_COHORT_SHA,
                  author_fit_gt_sha256=spot.GT_SHA, original_fold_plan_sha256=digest((a.code / FORMAL / "fit_plan.py").read_bytes()),
                  dataset_counts=dict(counts), supervision_counts=labels,
                  fold_counts=[sum(f == n for f in folds.values()) for n in range(5)],
                  valid_group_count=92, archived_seeds=list(plan.SEEDS),
                  archived_f_positive=[int(x) for x in np.count_nonzero(archived_k, axis=0)],
                  archived_f_signed_exchanges=[int(x) for x in archived_k.sum(axis=0)],
                  FIT=True, DEV=False, TEST=False, SRNet=False, Umbra=False, AP=False,
                  author_fit_labels_opened=True, all_labels_source_fit_only=True, detector_forwards=0, optimizer_updates=0,
                  point_head_readout_candidates=int(archived_k.size), private_payload_readback_exact=True,
                  elapsed_seconds=perf_counter() - started)
    checksum = spot.write(a.output / "RESULT_SAFE.json", result)
    print(json.dumps(dict(status=STATUS, result_sha256=checksum)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code", "capture", "acceptance", "basis", "heads", "source-cohort", "fit-gt", "shapely-site", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

