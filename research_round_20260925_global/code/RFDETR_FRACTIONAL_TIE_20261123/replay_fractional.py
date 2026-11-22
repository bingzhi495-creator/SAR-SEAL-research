"""Replay 12 frozen FIT109 occupancy heads on the same 74 native candidate lists."""

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
from types import ModuleType, SimpleNamespace


PORTABLE = "45512c959eb914ae7713264ecff48eb437a2a52729810b34308abc52079148df"
CAPTURE_RESULT = "f43fb04035045b1e30b7ae06ced778e66fcc4cd3e34077f83ebff6096dacc71d"
BUDGET_RESULT = "b31d4d1c718aa6cac9b73c61778aafb94404a897233928d8b78fa678d560f083"
BUDGET_EDITS = "353362e0efabc0374eaaf0b9c27fdabbcad195402a1090230c4ef56ca2a10398"
COHORT = "154d0d748b96df0d56ac249adcb8b6ae63275a09dfd7a0d7480839005e044338"
PROTOCOL_SHA = "270342431d38e48e48e011921c8791133af14746e7eba4d1622d186910cdeefa"
FRACTIONAL_HEADS = {5217: "3a60f658149f4262451cd9118902bbe52dc83345d9fbcd23973a11dc473fcdf1",
                    5218: "08b39b9911f77c039b26f8bb9bc34e940f5a350390911586f47290b2c3d88869",
                    5219: "93a3be4356816f9d29560d40d9bff8cc5d8730e9e600b29db07959612f504604"}
ARMS = ("F", "T", "H", "C")
KEYS = tuple(f"{s}/{arm}" for s in FRACTIONAL_HEADS for arm in ARMS)
STATUS = "RFDETR_FRACTIONAL_TIE_RESERVE74_REPLAY_COMPLETE"
HW = (512, 512)


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


def digest(raw):
    return sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()


def load_portable(code):
    path = code / "INSTANCE_RSDD_B5_REPRO_20261123/portable_eval.py"
    raw = path.read_bytes()
    require(digest(raw) == PORTABLE, "Published portable reserve loader differs")
    module = ModuleType("_portable_reserve")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def read_closed(result, terminal, expected=None):
    raw, event = result.read_bytes(), terminal.read_bytes()
    record, obs = json.loads(raw), json.loads(event)
    stamp = digest(raw)
    require(obs["actual_exit"] == 0 and obs["output_footer_verified"] is True and obs["result_sha256"] == stamp
            and obs["worker_sha256"] == record["worker_sha256"] and (expected is None or stamp == expected),
            "Successful external exit and matching output footer required: "+result.parent.name)
    return record, stamp, digest(event)


def source_heads(a):
    fit, stamp, event = read_closed(a.source/"RESULT_SAFE.json", a.source_terminal)
    require(fit["status"] == "RFDETR_FRACTIONAL_TIE_FIT109_COMPLETE" and fit["historical_fractional_exact"] is True
            and fit["FIT"] is True and fit["DEV"] is fit["TEST"] is fit["target_inputs_opened"] is False
            and fit["head_optimizer_updates"] == 4800 and fit["detector_forwards"] == 0
            and fit["protocol_sha256"] == PROTOCOL_SHA
            and set(fit["bundles"]) == set(KEYS), "Complete closed four-supervision three-seed FIT109 result required")
    weights = {}
    for name in KEYS:
        seed, arm = name.split("/")
        entry = fit["bundles"][name]
        relative = f"HEADS_PRIVATE/{seed}_{arm}.npz"
        require(entry["relative_path"] == relative and len(entry["sha256"]) == 64,
                "Source bundle arm, seed or relative path differs")
        path = (a.source/relative).resolve(strict=True)
        require(path.is_relative_to(a.source) and digest(path.read_bytes()) == entry["sha256"]
                and (arm != "F" or entry["sha256"] == FRACTIONAL_HEADS[int(seed)]),
                "Source result must bind the actual fitted weights and historical fractional heads")
        weights[name] = (path, entry["sha256"])
    return weights, stamp, event


def read_population(a, portable, native):
    items, manifest = portable.load_intake(a, native)
    native.install_guard(a.intake, items)
    cap, cap_sha, cap_event = read_closed(a.capture/portable.RESULT, a.stock_root/"observed_capture/TERMINAL.json",
                                           CAPTURE_RESULT)
    old, old_sha, old_event = read_closed(a.stock_root/"replay"/portable.RESULT,
                                          a.stock_root/"observed_replay/TERMINAL.json", BUDGET_RESULT)
    receipt, rows = portable.read_capture(a, items, manifest)
    fields = ("manifest_sha256", "roster_sha256", "predictions_sha256", "audit_sha256", "checkpoint_sha256")
    require(manifest == COHORT and cap == receipt and all(cap[key] == old[key] for key in fields)
            and cap["counts"] == dict(eligible=4137, mapped=7346, overrides=0, rank_gaps=54)
            and old["status"] == "B5_RESERVE74_REPLAY_COMPLETE" and old["edits_sha256"] == BUDGET_EDITS
            and old["point_head_sha256"] == FRACTIONAL_HEADS[5217] and old["target_labels_opened"] is False
            and old["independent_decisions_equal"] is True and cap["target_labels_opened"] is False
            and old["counts"]["fixed_candidates"] == old["counts"]["free_matched_count_candidates"] == 4092
            and old["counts"]["fixed_swaps"] == old["counts"]["free_matched_count_swaps"] == 30841,
            "Accepted native capture and paired baseline on the identical reserve required")
    ledger = portable.pin(a.stock_root/"replay"/portable.EDITS, BUDGET_EDITS)
    old_rows = [json.loads(line) for line in ledger.splitlines()]
    require(len(rows) == len(items) == len(old_rows) == 74, "Complete reserve census")
    return items, list(zip(rows, old_rows, strict=True)), dict(
        {k: cap[k] for k in fields}, capture_result_sha256=cap_sha, capture_terminal_sha256=cap_event,
        budget_result_sha256=old_sha, budget_terminal_sha256=old_event, budget_edits_sha256=BUDGET_EDITS)


def initialize(a, items, weights):
    global worker
    import numpy as np
    import torch
    from pycocotools import mask as codec

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(os.environ["CUDA_VISIBLE_DEVICES"] == "-1" and not torch.cuda.is_initialized(), "CPU-only replay")
    portable = load_portable(a.code)
    native = portable.module(a.code/portable.NATIVE/"capture_instance_rsdd.py", portable.CAPTURE_SOURCE)
    native.install_guard(a.intake, items)
    replay = portable.module(a.code/portable.RESERVE/"replay_reserve.py", portable.REPLAY_SOURCE)
    edit = portable.module(a.code/portable.UEIKAP/"edit_rf.py", replay.EDIT)
    base = edit.load_module(a.code/edit.BASE/"replay.py", edit.REPLAY_SHA)
    methods = base.methods(a.code)
    step = base.read_source(a.code/base.SPOT/"fit_step.py", methods.readout.PINS["fit_step.py"])
    plan = base.read_source(a.code/base.RF/"fit_plan.py", methods.readout.PINS["fit_plan.py"])
    trainer = step.load_trainer(a.code/base.BASE/"fit109_paired_point/train_point.py")
    models = {}
    for key, (path, stamp) in weights.items():
        heads, center, scale = plan.restore_bundle(path, stamp, trainer, step)
        if models:
            seed_scaler = models["5217/F"][1:]
            require(np.array_equal(center, seed_scaler[0]) and np.array_equal(scale, seed_scaler[1]),
                    "Common source feature normalization required for all interventions")
        models[key] = (heads["S"], center, scale)
    pair, sampler = base.canvas(methods, HW)
    worker = SimpleNamespace(np=np, codec=codec, base=base, methods=methods, pair=pair, sampler=sampler,
                             edit=edit, step=step, models=models, a=a)


def check_signed(stock, mask, k, coords, selection=None):
    np = worker.np
    edited = worker.codec.decode(mask["rle"]).astype(bool)
    require(edited.shape == stock.shape and type(mask["swaps"]) is int and mask["swaps"] == k,
            "Candidate budget or geometry differs")
    adds, drops = np.flatnonzero(edited & ~stock), np.flatnonzero(stock & ~edited)
    require(len(adds) == len(drops) == k and np.isin(adds, coords[:, 0]).all()
            and np.isin(drops, coords[:, 1]).all(), "Exact signed native support and unchanged candidate area required")
    if selection is not None:
        i, j = zip(*selection, strict=True)
        require(np.array_equal(adds, np.sort(coords[list(i), 0])) and np.array_equal(drops, np.sort(coords[list(j), 1])),
                "Independently ranked edits and encoded raster must agree")


def replay_image(input_row):
    (audit, native), previous = input_row
    w, np = worker, worker.np
    require(audit["image_id"] == native["image_id"] == previous["image_id"] and audit["original_hw"] == list(HW),
            "Native and budget stream image identity differs")
    planes = w.base.load_array(w.a.capture, audit["logits_ref"])
    by_rank = dict(zip(audit["logit_ranks"], planes, strict=True))
    require(planes.shape == (len(by_rank), 144, 144) and not audit["posterior_overrides"], "All native FP32 planes")
    image = w.methods.pair.image_features(w.base.load_rgb(w.a.intake/"images", audit))
    changes, counts = {}, Counter(images=1, mapped_candidates=len(native["coco"]), eligible_planes=len(by_rank))
    require(set(previous["changes"]) <= {str(k) for k in range(1, len(native["coco"])+1)}, "Budget key outside roster")
    for pos, (info, pred) in enumerate(zip(audit["candidates"], native["coco"], strict=True), 1):
        key = str(pos)
        old = previous["changes"].get(key, {})
        fixed, matched = old.get("fixed"), old.get("free_matched_count")
        k = fixed["swaps"] if fixed is not None else 0
        require(type(k) is int and ((k > 0) == (fixed is not None) == (matched is not None))
                and k == (matched["swaps"] if matched else 0), "Original exact realized fixed and matched budget")
        rank, area = info["native_rank"], info["mask_area"]
        require(info["position"] == pos and info["score"] == pred["score"] and (32 <= area < 1024) == (rank in by_rank),
                "Candidate identity, rank or gate differs")
        if rank not in by_rank:
            require(not old, "Out-of-gate candidate cannot receive edits")
            continue
        stock = w.codec.decode(pred["segmentation"]).astype(bool)
        probability = w.base.native_probability(by_rank[rank], stock, None)
        box = np.asarray(info["box_xyxy"], dtype=np.float32)
        samples = w.pair.paired_points(stock, box, area, pred["score"], probability, image, w.sampler)
        coords = samples[0] if samples is not None else np.empty((0, 2), dtype=np.int32)
        w.base.check_support(w.methods, stock, box, probability, coords)
        cap = min(8, area//10, len(coords))
        require(0 <= k <= cap, "Historical per-candidate budget must be attainable")
        counts["editable_supports"] += int(len(coords) > 0)
        if k == 0:
            continue
        check_signed(stock, fixed, k, coords)
        check_signed(stock, matched, k, coords)
        features = samples[1]
        actions = {}
        for arm, (head, center, scale) in w.models.items():
            q = w.step.predict(head, features, center, scale)
            choices = w.methods.free.choose_free(q, cap, fixed_count=k)
            independent = w.edit.ordered_pairs(q, cap, k)
            require(len(choices) == k and choices == independent, "Independent forced-count and tie ordering differ")
            out = w.base.mask_edit(stock, coords, choices)
            check_signed(stock, out, k, coords, independent)
            if arm == "5217/F":
                require(out == matched, "The historical seed-5217 matched-count fractional RLE must reproduce exactly")
            actions[arm] = out
        changes[key] = actions
        counts.update(budget_candidates=1, budget_swaps=k)
    return dict(image_id=audit["image_id"], changes=changes), counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code", "stock-root", "source", "source-terminal", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--protocol", type=Path, help="Frozen protocol; defaults to the published research directory")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    a = parser.parse_args()
    for key, path in vars(a).items():
        if isinstance(path, Path):
            setattr(a, key, path.resolve(strict=key != "output"))
    require(not a.output.exists(), "Supply a fresh output directory")
    protocol = a.protocol or a.code.parent/"RFDETR_FRACTIONAL_TIE_TRANSFER_PRELOCK_20261123.md"
    require(digest(protocol.read_bytes()) == PROTOCOL_SHA, "Exact frozen transfer protocol required")
    a.intake, a.roster, a.capture = a.stock_root/"conversion", a.stock_root/"reserve.csv", a.stock_root/"capture"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    runtime = {k: version(k) for k in ("torch", "numpy", "Pillow", "pycocotools", "rfdetr")}
    require(sys.version_info[:3] == (3,10,13) and runtime == {
        "torch":"2.10.0+cu130", "numpy":"1.26.4", "Pillow":"12.3.0", "pycocotools":"2.0.11", "rfdetr":"1.12.0"},
        "Original reserve CPU runtime required")
    start = perf_counter()
    portable = load_portable(a.code)
    native = portable.module(a.code/portable.NATIVE/"capture_instance_rsdd.py", portable.CAPTURE_SOURCE)
    items, rows, lineage = read_population(a, portable, native)
    weights, source_stamp, event_stamp = source_heads(a)
    a.output.mkdir()
    counts, sha = Counter(), sha256()
    per_arm = {arm: Counter(changed_candidates=0, adds=0, drops=0) for arm in KEYS}
    with (a.output/"EDITS_PRIVATE.jsonl").open("xb") as out, ProcessPoolExecutor(
            max_workers=a.workers, initializer=initialize, initargs=(a, items, weights)) as pool:
        for i, (row, totals) in enumerate(pool.map(replay_image, rows, chunksize=1), 1):
            require(row["image_id"] == items[i-1]["image_id"], "Preserve exact native order")
            for actions in row["changes"].values():
                require(set(actions) == set(KEYS), "Every head must edit the identical candidate")
                for arm, action in actions.items():
                    per_arm[arm].update(changed_candidates=1, adds=action["swaps"], drops=action["swaps"])
            raw = encode(row)
            out.write(raw)
            sha.update(raw)
            counts.update(totals)
            if i == 1 or i % 10 == 0 or i == 74:
                print(json.dumps(dict(phase="replay", images=i, positive_candidates=counts["budget_candidates"])), flush=True)
    require(counts == dict(images=74, mapped_candidates=7346, eligible_planes=4137, editable_supports=4114,
                            budget_candidates=4092, budget_swaps=30841)
            and all(c == dict(changed_candidates=4092, adds=30841, drops=30841) for c in per_arm.values()),
            "Entire fixed-count reserve census required")
    result = dict(status=STATUS, **lineage, **counts, protocol_sha256=PROTOCOL_SHA,
        source_result_sha256=source_stamp, source_terminal_sha256=event_stamp,
        seeds=list(FRACTIONAL_HEADS), arms=list(ARMS), weights_sha256={k: v[1] for k, v in weights.items()},
        counts_by_arm=per_arm, edits_sha256=sha.hexdigest(), historical_5217_F_exact=True,
        independent_support_exact=True, independent_ranking_exact=True, signed_xor_exact=True,
        GT_opened=False, target_labels_opened=False, model_forwards=0, optimizer_updates=0, AP=False,
        workers=a.workers, seconds=perf_counter()-start, operator_versions=runtime,
        worker_sha256=digest(Path(__file__).read_bytes()))
    data = encode(result)
    with (a.output/"RESULT_SAFE.json").open("xb") as out:
        out.write(data)
    print(json.dumps(dict(status=STATUS, result_sha256=digest(data))), flush=True)


if __name__ == "__main__":
    main()
