"""Replace PointRend's native-endpoint positions while retaining its two signed counts."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from importlib.metadata import version
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace


HEADS = {5217: "3a60f658149f4262451cd9118902bbe52dc83345d9fbcd23973a11dc473fcdf1",
         5218: "08b39b9911f77c039b26f8bb9bc34e940f5a350390911586f47290b2c3d88869",
         5219: "93a3be4356816f9d29560d40d9bff8cc5d8730e9e600b29db07959612f504604"}
B5 = "45512c959eb914ae7713264ecff48eb437a2a52729810b34308abc52079148df"
EDIT = "d753f673dcd43f1a65b817be176b4d77a05332a66d01fe98c7833f86248b5feb"
STATUS = "POINTREND_SIGNED_QUOTA_SWAP_REPLAY_COMPLETE"
HW = (512, 512)


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


def stamp(raw):
    return sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()


def portable_module(code):
    path = code/"INSTANCE_RSDD_B5_REPRO_20261123/portable_eval.py"
    raw = path.read_bytes()
    require(stamp(raw) == B5, "Pinned detector package differs")
    module = ModuleType("_signed_quota_b5")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def initialize(a, items):
    global w
    import numpy as np
    import torch
    from pycocotools import mask as codec

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(os.environ["CUDA_VISIBLE_DEVICES"] == "-1" and not torch.cuda.is_initialized(), "CPU-only inference")
    portable = portable_module(a.code)
    native = portable.module(a.code/portable.NATIVE/"capture_instance_rsdd.py", portable.CAPTURE_SOURCE)
    native.install_guard(a.image_root.parent, items)
    edit = portable.module(a.code/portable.UEIKAP/"edit_rf.py", EDIT)
    base = edit.load_module(a.code/edit.BASE/"replay.py", edit.REPLAY_SHA)
    methods = base.methods(a.code)
    step = base.read_source(a.code/base.SPOT/"fit_step.py", methods.readout.PINS["fit_step.py"])
    plan = base.read_source(a.code/base.RF/"fit_plan.py", methods.readout.PINS["fit_plan.py"])
    trainer = step.load_trainer(a.code/base.BASE/"fit109_paired_point/train_point.py")
    models = {}
    for seed, sha in HEADS.items():
        heads, center, scale = plan.restore_bundle(a.weights_dir/f"{seed}_all_fit.npz", sha, trainer, step)
        models[seed] = (heads["S"], center, scale)
    pair, sampler = base.canvas(methods, HW)
    w = SimpleNamespace(np=np, torch=torch, codec=codec, base=base, a=a, methods=methods,
                        step=step, models=models, pair=pair, sampler=sampler)


def signed(stock, changed):
    return int(w.np.count_nonzero(changed & ~stock)), int(w.np.count_nonzero(stock & ~changed))


def encoded_mask(mask, counts):
    if counts == (0, 0):
        return None
    rle = w.codec.encode(w.np.asfortranarray(mask, dtype=w.np.uint8))
    return dict(rle=dict(size=rle["size"], counts=rle["counts"].decode("ascii")), adds=counts[0], drops=counts[1])


def select(coords, q, a, d):
    np = w.np
    # Reuse the B5 independent ranking. Its secondary key is the row of the original paired table.
    _, add_order, drop_order = w.methods.free.ranked_endpoints(q)
    chosen_add = coords[add_order[:a], 0]
    chosen_drop = coords[drop_order[:d], 1]
    ia = sorted(range(len(coords)), key=lambda i: (-float(q[i, 0]), i))[:a]
    idrop = sorted(range(len(coords)), key=lambda i: (float(q[i, 1]), i))[:d]
    require(np.array_equal(chosen_add, coords[ia, 0]) and np.array_equal(chosen_drop, coords[idrop, 1]),
            "Independent signed count and tie ranking differs")
    return chosen_add, chosen_drop


def swap_candidate(stock, coords, features, p_entry, old, historical_h10, has_h10, area, seencounts, totals):
    np = w.np
    p = w.codec.decode(p_entry["rle"]).astype(bool) if p_entry else stock
    p_counts = signed(stock, p)
    require(p_counts == ((p_entry["adds"], p_entry["drops"]) if p_entry else (0, 0)), "Original paste counts differ")
    add_idx, drop_idx = coords.T
    support = coords.reshape(-1)
    require(len(np.unique(support)) == len(support) and not stock.flat[add_idx].any() and stock.flat[drop_idx].all(),
            "Original signed rows must be disjoint subsets of background and foreground")
    a, d = int(p.flat[add_idx].sum()), int((~p.flat[drop_idx]).sum())
    quota = (a, d)
    p_local = stock.copy(order="C")
    p_local.flat[support] = p.flat[support]
    require(signed(stock, p_local) == quota, "PointRend localized signed counts differ")
    local_ref = encoded_mask(p_local, quota)
    if has_h10:
        require((local_ref["rle"] if local_ref else None) ==
                (historical_h10["rle"] if historical_h10 else None), "Exact historical 740 H10 RLE differs")
    seencounts.update(point_total_adds=p_counts[0], point_total_drops=p_counts[1], S_adds=a, S_drops=d,
        S_positive=int(a+d>0), S_unequal=int(a != d), S_above_eight=int(max(a,d)>8),
        S_has_alternative=int(0<a<len(coords) or 0<d<len(coords)), empty_paste=int(not p.any()))
    seen = {"p_local": local_ref} if local_ref else {}
    archived = old.get("learned_common")
    k = archived["swaps"] if archived else 0
    require((old.get("point_equal") or {}).get("swaps", 0) == k, "Old common budget correspondence differs")
    for seed, (head, center, scale) in w.models.items():
        if features is not None:
            q = w.step.predict(head, features, center, scale)
            require(q.shape == coords.shape and np.isfinite(q).all(), "Finite endpoint posteriors required")
            if seed == 5217:
                old_choices = w.methods.free.choose_free(q, min(8, area//10, len(coords)))
                ref = w.base.mask_edit(stock, coords, old_choices) if old_choices else None
                require(len(old_choices) == k and ref == archived, "Original natural-count seed-5217 rank/RLE differs")
            selected_add, selected_drop = select(coords, q, a, d)
        else:
            require(len(coords) == k == a == d == 0 and not archived, "No-support original fallback")
            selected_add, selected_drop = add_idx, drop_idx
        if local_ref:
            local = stock.copy(order="C")
            local.flat[selected_add] = True
            local.flat[selected_drop] = False
        else:
            local = stock
        full = p.copy(order="C")
        full.flat[support] = local.flat[support]
        # The same signed pixel substitutions must occur in both outside contexts.
        local_added, local_dropped = local & ~p_local, p_local & ~local
        require(np.array_equal(local_added, full & ~p) and np.array_equal(local_dropped, p & ~full),
                "Full and local rank interventions differ in location or sign")
        diff = int(np.count_nonzero(local_added))
        require(diff == int(np.count_nonzero(local_dropped)) and signed(stock, local) == quota
                and signed(stock, full) == p_counts, "Rank replacement changed native signed counts or area")
        if local_ref:
            seen[f"{seed}/local"] = encoded_mask(local, quota) if diff else local_ref
        if p_entry:
            seen[f"{seed}/full"] = encoded_mask(full, p_counts) if diff else p_entry
        else:
            require(diff == 0 and not local_ref, "A native identity paste has zero signed quotas")
        totals[seed].update(different_candidates=int(diff>0), swapped_pixel_positions=diff,
            symmetric_difference_pixels=2*diff, local_adds=a, local_drops=d, full_adds=p_counts[0],
            full_drops=p_counts[1], local_changed_candidates=int(bool(local_ref)), full_changed_candidates=int(bool(p_entry)))
    return seen


def replay_image(inputs):
    np = w.np
    item, row, native, previous, h10 = inputs
    require(item["image_id"] == row["image_id"] == native["image_id"] == previous["image_id"]
            and (h10 is None or h10["image_id"] == row["image_id"])
            and (h10 is not None) == (w.a.cohort == 740) and row["original_hw"] == list(HW)
            and not row["posterior_overrides"], "Ordered native cohort, H10 and original canvas required")
    planes = w.base.load_array(w.a.stock_root/"capture", row["logits_ref"])
    logits = dict(zip(row["logit_ranks"], planes, strict=True))
    require(planes.shape == (len(logits),144,144), "Original eligible coarse planes required")
    image = w.methods.pair.image_features(w.base.load_rgb(w.a.image_root, row))
    require(w.torch.get_float32_matmul_precision() == "high", "Original occupancy numerical precision required")
    original, prior_h10 = previous["changes"], h10["changes"] if h10 is not None else {}
    positions = {str(i) for i in range(1, len(native["coco"])+1)}
    require(original.keys() <= positions and prior_h10.keys() <= positions, "Archived compact positions outside native roster")
    changes, per_seed = {}, {seed: Counter() for seed in HEADS}
    counts = Counter(images=1, mapped_candidates=len(native["coco"]), eligible_planes=len(logits))
    for pos, (info, pred) in enumerate(zip(row["candidates"], native["coco"], strict=True), 1):
        key = str(pos)
        rank, area = info["native_rank"], info["mask_area"]
        require(rank == row["native_ranks"][pos-1] and info["query_id"] == row["query_ids"][pos-1]
                and info["position"] == pos and info["score"] == pred["score"]
                and area == int(w.codec.area(pred["segmentation"])) and (32<=area<1024) == (rank in logits),
                "Original candidate, score, area or ranking differs")
        if rank not in logits:
            require(key not in original and key not in prior_h10, "Outside-gate native fallback")
            continue
        box = np.asarray(info["box_xyxy"], dtype=np.float32)
        if np.any(box[:2] == box[2:]):
            require(key not in original and key not in prior_h10, "Degenerate native fallback")
            counts["degenerate"] += 1
            continue
        stock = w.codec.decode(pred["segmentation"]).astype(bool)
        probability = w.base.native_probability(logits[rank], stock, None)
        data = w.pair.paired_points(stock, box, area, pred["score"], probability, image, w.sampler)
        coords, features = data[:2] if data is not None else (np.empty((0,2), dtype=np.int32), None)
        w.base.check_support(w.methods, stock, box, probability, coords)
        old = original.get(key, {})
        edits = swap_candidate(stock, coords, features, old.get("point_paste"), old,
            prior_h10.get(key, {}).get("p_on_support"), h10 is not None, area, counts, per_seed)
        counts.update(rendered=1, support_per_side=len(coords), nonempty_support=int(len(coords)>0))
        if edits:
            changes[key] = edits
    return dict(image_id=row["image_id"], changes=changes), counts, per_seed


def main():
    # Isolated CPython excludes the script directory from sys.path. Resolve the sibling explicitly.
    path = Path(__file__).with_name("inputs.py")
    spec = spec_from_file_location("_signed_quota_inputs", path)
    inputs = module_from_spec(spec)
    sys.modules[spec.name] = inputs
    spec.loader.exec_module(inputs)
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    inputs.add_input_arguments(parser)
    for label in ("weights-dir", "protocol", "output"):
        parser.add_argument("--"+label, required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--workers", type=int, choices=range(1,9), default=4)
    a = parser.parse_args()
    for key, value in vars(a).items():
        if isinstance(value, Path):
            setattr(a, key, value.resolve(strict=key != "output"))
    require(not a.output.exists() and stamp(a.protocol.read_bytes()) == a.protocol_sha256,
            "Fresh output and named frozen protocol required")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    runtime = {key: version(key) for key in ("torch","numpy","Pillow","pycocotools","rfdetr")}
    require(sys.version_info[:3] == (3,10,13) and runtime == dict(torch="2.10.0+cu130",numpy="1.26.4",
        Pillow="12.3.0",pycocotools="2.0.11",rfdetr="1.12.0"), "Qualified reserve CPU runtime")
    start = perf_counter()
    rows, lineage = inputs.load_population(a)
    a.cohort = int(a.cohort)
    a.image_root = a.stock_root/("conversion" if a.cohort == 74 else "intake")/"images"
    require(len(rows) == a.cohort in (74,740), "Entire closed cohort required")
    a.output.mkdir()
    counts, per_seed, per_sensor, ledger = Counter(), {s: Counter() for s in HEADS}, defaultdict(Counter), sha256()
    items = [r[0] for r in rows]
    with (a.output/"EDITS_PRIVATE.jsonl").open("xb") as stream, ProcessPoolExecutor(
            max_workers=a.workers, initializer=initialize, initargs=(a, items)) as pool:
        for ordinal, (result, tally, by_seed) in enumerate(pool.map(replay_image, rows, chunksize=2), 1):
            item = items[ordinal-1]
            require(result["image_id"] == item["image_id"], "Exact candidate image order required")
            counts.update(tally)
            per_sensor[item["sensor"]].update(tally)
            for seed in HEADS:
                per_seed[seed].update(by_seed[seed])
            raw = encode(result)
            stream.write(raw)
            ledger.update(raw)
            if ordinal == 1 or ordinal%25 == 0 or ordinal == a.cohort:
                print(json.dumps(dict(phase="replay", images=ordinal, total=a.cohort)), flush=True)
    expected = (7346,4137,4136,4114,408586,444542,171) if a.cohort==74 else (73226,43252,43145,42843,3650117,4565170,1055)
    require(tuple(counts[k] for k in ("mapped_candidates","eligible_planes","rendered","nonempty_support",
            "point_total_adds","point_total_drops","empty_paste")) == expected
            and (a.cohort!=740 or (counts["S_adds"],counts["S_drops"],counts["S_positive"])==(538864,510848,42833)),
            "Entire historical PointRend census and native S counts required")
    safe = dict(status=STATUS, cohort=a.cohort, images=len(rows), seeds=list(HEADS), counts=dict(counts),
        counts_by_seed=per_seed, counts_by_sensor=per_sensor, input_lineage=lineage, weights_sha256=HEADS,
        edits_sha256=ledger.hexdigest(), protocol_sha256=a.protocol_sha256, exact_signed_quotas=True,
        exact_cross_context_xor=True, exact_native_support=True, archived_5217_rank_parity=True,
        historical_H10_exact=True if a.cohort==740 else None, point_full_preserved=True,
        GT_opened=False, target_labels_opened=False, AP=False, detector_forwards=0, pointrend_forwards=0,
        optimizer_updates=0, workers=a.workers, operator_versions=runtime, seconds=perf_counter()-start,
        worker_sha256=stamp(Path(__file__).read_bytes()), input_worker_sha256=stamp(path.read_bytes()))
    raw = encode(safe)
    with (a.output/"RESULT_SAFE.json").open("xb") as stream:
        stream.write(raw)
    print(json.dumps(dict(status=STATUS, result_sha256=stamp(raw))), flush=True)


if __name__ == "__main__":
    main()
