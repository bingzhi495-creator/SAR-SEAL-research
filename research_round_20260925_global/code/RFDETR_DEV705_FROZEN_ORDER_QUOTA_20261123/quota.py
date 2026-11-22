"""Read-only, fixed-geometry quota oracle on the closed HRSID DEV705 carrier."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace
from zipfile import ZipFile

import numpy as np


PROTOCOL = "51cadf37aea0f4aa2372afeb8808eb42836dd92cfcc954e0898b45d526b547a0"
DIAGNOSTIC = "0cec40581505fc8d5c3f0632d2a9b0fe55ecc334669d123ca60b4d4e2b2255aa"
STATUS = "RFDETR_DEV705_FROZEN_ORDER_QUOTA_ORACLE_COMPLETE"
SEEDS = (5217, 5218, 5219)
ARMS = ("native", "balanced", "quota", "truth_balanced", "truth_free")
FIELDS = ("iou", "fp_per_truth", "fn_per_truth")
CONTRASTS = (("quota", "balanced"), ("truth_balanced", "balanced"), ("truth_free", "truth_balanced"),
             ("truth_free", "quota"))
MIB = 1024**2
EPSILON = 1e-12


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def pin(path, expected, limit):
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    need(0 < len(raw) <= limit and sha256(raw).hexdigest() == expected, "Exact bounded input: " + path.name)
    return raw


def json_pin(path, expected, limit=MIB):
    return json.loads(pin(path, expected, limit))


def module(path, expected, name):
    loaded = ModuleType(name)
    loaded.__file__ = str(path)
    sys.modules[name] = loaded
    exec(compile(pin(path, expected, MIB), str(path), "exec"), loaded.__dict__)
    return loaded


def original_ranks(q):
    """Preserve pair-order tie resolution from the exact original S readout."""
    need(q.ndim == 2 and q.shape[1] == 2 and q.dtype == np.float32
         and np.isfinite(q).all() and np.all((q >= 0) & (q <= 1)), "Original finite FP32 posteriors")
    return np.lexsort((np.arange(len(q)), -q[:, 0])), np.lexsort((np.arange(len(q)), q[:, 1]))


def frozen_prefix(coords, add_order, drop_order, flips):
    k = len(flips) // 2
    ranked = np.column_stack((coords[add_order[:k], 0], coords[drop_order[:k], 1])).reshape(-1)
    need(len(flips) % 2 == 0 and k <= len(coords) and ranked.tolist() == list(flips),
         "Exact archived alternating SU endpoint order")


def candidate_quota(row, adds_order, drops_order, edits):
    """Optimize additions at fixed ranks and fixed realized flip count."""
    q = np.asarray(row["truth"], dtype=np.float64)
    need(q.ndim == 2 and q.shape[1] == 2 and np.isfinite(q).all()
         and np.all((q >= 0) & (q <= 1 + EPSILON)), "Exact original native-cell truth")
    size = len(q)
    need(type(edits) is int and 0 <= edits <= 2*min(8, size) and edits % 2 == 0
         and len(adds_order) == len(drops_order) == size
         and sorted(adds_order) == sorted(drops_order) == list(range(size)), "Admitted fixed ranks and budget")
    raw = row["raw_continuous"]
    I, P, T = (raw[k] for k in ("intersection", "predicted_area", "truth_area"))
    need(32 <= P < 1024 and edits <= 2*(int(P)//10), "Original native area and action cap")
    add_prefix = np.r_[0., np.cumsum(q[adds_order, 0], dtype=np.float64)]
    drop_prefix = np.r_[0., np.cumsum(q[drops_order, 1], dtype=np.float64)]

    def geometry(add, drop):
        iou_intersection = I + add_prefix[add] - drop_prefix[drop]
        area = P + add - drop
        union = area + T - iou_intersection
        need(union > 0 and -1e-8 <= iou_intersection <= min(area, T) + 1e-8, "Physical continuous geometry")
        return dict(I=iou_intersection, P=area, T=T, iou=iou_intersection/union, adds=add, drops=drop)

    choices = [geometry(add, edits-add) for add in range(max(0, edits-size), min(size, edits)+1)]
    balanced = geometry(edits//2, edits//2)
    quota = min(choices, key=lambda r: (-r["iou"], abs(r["adds"]-r["drops"]), r["adds"]))
    return balanced, quota


def original_inputs(a, d, original, spot, readout):
    score_root = a.original_score_root
    score = json_pin(score_root/"RESULT_SAFE.json", d.ORIGINAL_SCORE_SHA, 8*MIB)
    terminal = json_pin(score_root/"ORIGINAL_TOOL_WAIT_PRIVATE.json", d.ORIGINAL_RECEIPT_SHA)
    d.validate_score_closure(score, terminal, d.ORIGINAL_SCORE_SHA, d.ORIGINAL_RECEIPT_SHA)
    old = json_pin(a.readout_accepted, a.readout_accepted_sha)
    report = json_pin(a.readout_root/"RESULT_SAFE.json", old["readout_result_sha256"])
    need(old["status"] == "RFDETR_CENTER5202_DEV705_MATCHED_READOUT_INDEPENDENT_ACCEPTED"
         and old["original_exit"] == 0 and old["DEV"] is True and old["images"] == 705
         and old["capture_accepted_sha256"] == a.capture_accepted_sha
         and not any(old[k] or report[k] for k in ("FIT", "TEST", "AP", "GT_opened"))
         and all(old[k] for k in ("exact_candidate_identity", "exact_native_support", "exact_per_candidate_budgets",
                                  "actual_XOR_area_RLE_verified")) and old["edits_sha256"] == report["edits_sha256"]
         and old["probability_repair_sha256"] == report["probability_repair_sha256"]
         == score["probability_repair_sha256"] == a.probability_repair_sha, "Same independently verified native replay")
    original.verify_readout_pair(old, report, readout)
    d.validate_score_prelabel(json_pin(score_root/"PRELABEL_SAFE.json", score["prelabel_sha256"]), a, old, original)
    prior = json_pin(score_root/"SHAPE_MATCH_WITNESS_PRIVATE.json", score["private_shape_witness_sha256"], 32*MIB)
    witnesses = {seed: {} for seed in SEEDS}
    for row in prior:
        key = (row["image_id"], row["rank"], row["query_id"])
        slot = witnesses[row["seed"]]
        need(key not in slot, "Unique archived prior seed-match identity")
        slot[key] = row
    identities = set(witnesses[SEEDS[0]])
    need(len(identities) == 1461 and len(prior) == 3*1461
         and all(set(witnesses[s]) == identities for s in SEEDS), "All original valid matches in three seeds")
    anchors = defaultdict(dict)
    for image_id, rank, query_id in sorted(identities):
        anchors[image_id][rank] = query_id

    images, original_blocks, capture = readout.original_population(a, spot)
    sparse = [json.loads(line) for line in pin(a.readout_root/"SPARSE_EDITS_PRIVATE.jsonl",
                                              old["edits_sha256"], 512*MIB).splitlines()]
    need(len(sparse) == 705 and [r["image_id"] for r in sparse] == [r["image_id"] for r in images]
         and capture["raw_logit_rows"] == 42325, "Exact original 705-image streams and native parity scope")
    return score, report, images, original_blocks, sparse, anchors, witnesses


def build_rank_replay(a, inputs, spot, readout, fit_step, score_math, codec, torch):
    score, report, images, blocks, sparse, anchors, witnesses = inputs
    feat = readout.load(a.spot_source/"feature_adapter.py", readout.PINS["feature_adapter.py"], "_quota_features")
    plan = readout.load(a.code/"RFDETR_MATCHED_POINT_FORMAL_20261120/fit_plan.py",
                        readout.PINS["fit_plan.py"], "_quota_fit_plan")
    prior_fit = json_pin(a.fit_result, a.fit_result_sha, 8*MIB)
    prior_fit_accept = json_pin(a.fit_accepted, a.fit_accepted_sha)
    need(report["fit_result_sha256"] == a.fit_result_sha and report["fit_accepted_sha256"] == a.fit_accepted_sha,
         "Reuse the exact accepted source refits of the archived SU readout")
    bundles = readout.validated_refits(prior_fit, prior_fit_accept, a.fit_result_sha)
    base = a.code/"DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24"
    trainer = fit_step.load_trainer(base/"fit109_paired_point/train_point.py")
    refits = {s: plan.restore_bundle(a.head_root/r["relative_weights"], r["weights_sha256"], trainer, fit_step)
              for s, r in bundles.items()}
    need(set(refits) == set(SEEDS), "Exact small head for each archived seed")
    posterior = fit_step.load(a.code/"RFDETR_MATCHED_POINT_REFINEMENT_20261120/adapter.py",
                              spot.ADAPTER_SHA, "_quota_posterior")
    png = readout.load(base/"dev705/dfine_dev705.py", readout.PNG_READER, "_quota_png")
    pair = fit_step.load(base/"fit109_paired_point/pair_math.py", spot.PAIR_SHA, "_quota_pairs")
    ranker = fit_step.load(base/"fit109_probability_rank/rank_math.py", spot.RANK_SHA, "_quota_ranker")
    repair = readout.read_probability_repair(a)
    ranked, pending = {}, {}
    counts = Counter(source_images=705, full_candidates=67386, eligible_candidates=42325,
                     original_valid_matches=1461, skipped_non_anchor_eligible=42325-1461)
    with a.archive.open("rb") as file:
        need(a.archive.stat().st_size == png.ARCHIVE_BYTES, "Exact PNG archive length")
        file.seek(png.CENTRAL_OFFSET)
        need(sha256(file.read(png.CENTRAL_BYTES)).hexdigest() == png.CENTRAL_SHA, "Exact PNG directory")
        with ZipFile(file) as archive:
            for ordinal, (item, block, edits) in enumerate(zip(images, blocks, sparse, strict=True), 1):
                image_id = item["image_id"]
                if image_id not in anchors:
                    continue
                rgb, hw = png.read_png(archive, item)
                need(hw == (800, 800) and edits["original_hw"] == [800, 800], "Original native canvas")
                image_features = pair.image_features(rgb)
                pool = spot.native_candidate_rows(block)
                for candidate in pool:
                    candidate["area"] = int(codec.area(candidate["rle"]))
                boxes = spot.author_boxes(block["raw_boxes_cxcywh"], torch)
                eligible = [c for c in pool if 32 <= c["area"] < 1024]
                ref = block["logit_capture"]
                raw_path = a.capture/ref["relative_path"]
                need(ref["relative_path"] == f"RAW_LOGITS/{ordinal:03d}.npy", "Original raw plane order")
                data = pin(raw_path, ref["sha256"], 9*MIB)
                raw = np.load(BytesIO(data), allow_pickle=False)
                need(len(data) == ref["bytes"] and raw.dtype == np.dtype("<f4") and np.isfinite(raw).all()
                     and list(raw.shape) == ref["shape"] == [len(eligible), 144, 144], "Original 144-grid logits")
                points = {r["rank"]: r for r in edits["points"]}
                records = []
                for row, (candidate, proof, plane) in enumerate(zip(eligible, ref["candidates"], raw, strict=True)):
                    rank = candidate["rank"]
                    if rank not in anchors[image_id]:
                        continue
                    query = candidate["query_id"]
                    need(query == anchors[image_id][rank] == points[rank]["query_id"]
                         and (proof["logit_row"], proof["query_id"], proof["native_rank"], proof["score"],
                              proof["predicted_area"]) == (row, query, rank, candidate["score"], candidate["area"]),
                         "Original rank, query, score, area and prior match are unchanged")
                    key = (image_id, rank, query)
                    identity = dict(ordinal=ordinal, image_id=image_id, query_id=query, native_rank=rank, logit_row=row)
                    coords = np.asarray(points[rank]["coordinates"], dtype=np.int64).reshape(-1, 2)
                    record = dict(image_id=image_id, rank=rank, query_id=query, group=edits["group"], coordinates=coords)
                    records.append(record)
                    features = None
                    if len(coords):
                        stock = codec.decode(candidate["rle"]).astype(bool)
                        source = dict(relative_path=ref["relative_path"], path=str(raw_path.resolve()), bytes=ref["bytes"],
                                      sha256=ref["sha256"])
                        p = readout.probability_with_repair(np, torch, feat, posterior, plane, stock, candidate["rle"],
                                                           identity, source, repair)
                        fresh = feat.extract_candidate(np, pair, ranker, stock, boxes[query], candidate["score"],
                                                       p, image_features)
                        need(fresh is not None and np.array_equal(fresh["coordinates"], coords),
                             "Recomputed frozen geometry equals the archived endpoint coordinates")
                        features = fresh["features"]
                        counts["recomputed_native_planes"] += 1
                    else:
                        counts["anchors_without_endpoints"] += 1
                    ranked[key] = {}
                    for seed, (heads, center, scale) in refits.items():
                        orig = score_math.get_edit(edits, seed, rank, "SU")
                        if features is not None:
                            q = fit_step.predict(heads["S"], features, center, scale)
                            order = original_ranks(q)
                            counts["head_forward_calls"] += 1
                            counts["head_endpoint_values"] += q.size
                        else:
                            order = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
                        frozen_prefix(coords, *order, orig["flipped"])
                        frozen_prefix(coords, *order, score_math.get_edit(edits, seed, rank, "SU=F")["flipped"])
                        need(orig["edits"] == len(orig["flipped"]) and orig["edits"] <= 16,
                             "Exact realized archive budget")
                        archived_iou = witnesses[seed][key]["arms"]["SU"]
                        need(archived_iou["edits"] == orig["edits"], "Original SU score carries the same budget")
                        ranked[key][seed] = (order, orig)
                need(len(records) == len(anchors[image_id]), "Every requested fixed anchor and no extra candidate")
                pending[image_id] = (pool, records)
                counts["replayed_images"] += 1
    need(len(ranked) == 1461 and set(pending) == set(anchors), "No missing ranked anchor")
    applied = int(any((repair["image_id"], repair["native_rank"], repair["query_id"]) == key
                      and len(ranked[key][SEEDS[0]][0][0]) for key in ranked))
    need(repair["_consumed"] == applied, "Original single CUDA correction applied only to its recalculated candidate")
    counts["reapplied_single_cuda_correction"] = applied
    return ranked, pending, dict(counts)


def metric_values(iou_row):
    return (iou_row["iou"], (iou_row["P"]-iou_row["I"])/iou_row["T"],
            (iou_row["T"]-iou_row["I"])/iou_row["T"])


def score_quota(gt, ranked, pending, witness, fit_step, helpers, geometry, oracle):
    by_image = defaultdict(list)
    for row in gt["annotations"]:
        by_image[row["image_id"]].append(row)
    need(len(gt["images"]) == 705 and len(gt["annotations"]) == 1782, "Exact original DEV label roster")
    group_for, rows = {}, {}
    counts = {s: Counter() for s in SEEDS}
    transitions = {s: {a: Counter() for a in ARMS} for s in SEEDS}
    counts_ext = Counter()
    positive, positive_groups = set(), set()
    for image in gt["images"]:
        image_id = image["id"]
        if image_id not in pending:
            continue
        pool, records = pending[image_id]
        attached, _ = fit_step.attach_truth(image, by_image[image_id], pool, records, helpers)
        for row in attached:
            key = geometry.identity(row)
            group = row["group"]
            need(row["role"] == "valid" and key not in rows, "Same valid fixed match and no duplicates")
            if row["truth"] is None and not len(row["coordinates"]):
                row["truth"] = np.empty((0, 2), dtype=np.float64)
            group_for[key] = group
            rows[key] = {}
            for seed, (order, orig) in ranked[key].items():
                old = witness[seed][key]
                need(row["matched_gt_id"] == old["matched_gt_id"] and group == old["group"],
                     "Unchanged fixed native match identity and group")
                balanced, quota = candidate_quota(row, *order, orig["edits"])
                stock = row["raw_continuous"]
                coords, q = row["coordinates"], row["truth"]
                bounds = oracle.candidate_oracles(
                    q[:, 0].tolist(), q[:, 1].tolist(), stock["intersection"], stock["predicted_area"], stock["truth_area"],
                    orig["edits"], add_indices=coords[:, 0].tolist(), drop_indices=coords[:, 1].tolist())
                native, truth_balanced, truth_free = bounds["original"], bounds["oracles"]["O_U"], bounds["oracles"]["O_F"]
                old_n, old_su = old["arms"]["N"], old["arms"]["SU"]
                need(all(abs(old_n[f]-native[f]) <= EPSILON and abs(old_su[f]-balanced[f]) <= EPSILON
                         for f in ("I", "P", "T", "iou")), "Exact archived native and SU geometry, within FP64 rounding")
                need(quota["iou"] + EPSILON >= balanced["iou"] and truth_free["iou"] + EPSILON >= quota["iou"]
                     and truth_free["iou"] + EPSILON >= truth_balanced["iou"] >= balanced["iou"] - EPSILON,
                     "Nested quota and endpoint ordering bounds")
                arms = (native, balanced, quota, truth_balanced, truth_free)
                rows[key][seed] = np.asarray([metric_values(a) for a in arms], dtype=np.float64)
                b, z = balanced["iou"], quota["iou"]
                area_change = quota["adds"]-quota["drops"]
                counts[seed].update(actual_flips=orig["edits"], zero_budget=int(orig["edits"] == 0),
                                    quota_positive=int(z-b > EPSILON), quota_adds=quota["adds"],
                                    quota_drops=quota["drops"], quota_area_larger=int(area_change > 0),
                                    quota_area_smaller=int(area_change < 0),
                                    quota_correct_area_direction=int(area_change*(quota["T"]-balanced["P"]) > 0))
                if z-b > EPSILON:
                    positive.add(key)
                    positive_groups.add(group)
                for named, value in zip(ARMS, arms, strict=True):
                    for threshold in (.75, .90):
                        transitions[seed][named][f"strict_{threshold:.2f}"] += int(value["iou"] > threshold)
                        transitions[seed][named][f"new_from_SU_{threshold:.2f}"] += int(b <= threshold < value["iou"])
                        transitions[seed][named][f"lost_from_SU_{threshold:.2f}"] += int(value["iou"] <= threshold < b)
                counts_ext["exact_geometry_reproductions"] += 1
    need(set(rows) == set(ranked) and len(set(group_for.values())) == 27, "Complete predeclared diagnostic population")
    groups = sorted(set(group_for.values()))
    grouped_keys = {g: [k for k in rows if group_for[k] == g] for g in groups}
    def group_matrix(seed):
        return np.asarray([np.mean([rows[k][seed] for k in grouped_keys[g]], axis=0) for g in groups])
    matrices = {seed: group_matrix(seed) for seed in SEEDS}
    mean = np.mean(list(matrices.values()), axis=0)
    means = {name: dict(zip(FIELDS, (100*mean[:, i, :].mean(axis=0)).tolist(), strict=True))
             for i, name in enumerate(ARMS)}
    choices = np.random.Generator(np.random.PCG64(5217)).integers(len(groups), size=(4096, len(groups)))
    comparisons = {}
    for left, right in CONTRASTS:
        l, r = ARMS.index(left), ARMS.index(right)
        difference = 100*(mean[:, l, :]-mean[:, r, :])
        intervals = np.quantile(difference[choices].mean(axis=1), (.025, .975), axis=0)
        comparisons[left+" minus "+right] = {
            m: dict(delta=float(difference[:, i].mean()), interval95=intervals[:, i].tolist(),
                    per_seed={s: float(100*(a[:, l, i]-a[:, r, i]).mean()) for s, a in matrices.items()})
            for i, m in enumerate(FIELDS)
        }
    effective = comparisons["quota minus balanced"]["iou"]
    checks = dict(mean_at_least_0_20=effective["delta"] >= .20, positive_lower=effective["interval95"][0] > 0,
                  at_least_20_distinct_anchors=len(positive) >= 20, at_least_8_groups=len(positive_groups) >= 8)
    return dict(means_points=means, contrasts_points=comparisons, per_seed_counts=counts,
                strict_count_and_transitions=transitions, source_groups=len(groups), bootstrap_draws=4096,
                improved_distinct_anchors=len(positive), improvement_groups=len(positive_groups),
                improved_anchor_fraction=len(positive)/len(rows), improved_group_fraction=len(positive_groups)/len(groups),
                exact_geometry_reproductions=counts_ext["exact_geometry_reproductions"],
                headroom_checks=checks, headroom_gate=all(checks.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--authority-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {"protocol", "code", "capture", "capture_accepted", "image_manifest", "archive", "readout_root",
             "readout_accepted", "original_score_root", "original_dev_dir", "spot_source", "probability_repair",
             "fit_result", "fit_accepted", "head_root", "shapely_site"}
    hashes = {"capture_accepted_sha", "readout_accepted_sha", "probability_repair_sha", "fit_result_sha", "fit_accepted_sha"}
    raw = json_pin(args.authority, args.authority_sha)
    need(set(raw) == paths | hashes and all(Path(raw[k]).is_absolute() for k in paths)
         and args.output.is_absolute() and not args.output.exists() and args.output.parent.is_dir()
         and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and sys.flags.isolated and sys.dont_write_bytecode,
         "Complete isolated CPU-only authority and fresh result location")
    a = SimpleNamespace(**{k: Path(v).resolve(strict=True) if k in paths else v for k, v in raw.items()})
    pin(a.protocol, PROTOCOL, MIB)
    start = perf_counter()
    import torch
    from pycocotools import mask as codec
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    d = module(a.code/"RFDETR_OCCUPANCY_BOUND_ADAPTER_20261120/run_diagnostic.py", DIAGNOSTIC, "_quota_original_diagnostic")
    formal = a.code/"RFDETR_MATCHED_POINT_FORMAL_20261120"
    original = d.load_original(formal/"run_score_dev.py")
    fit_step = original.load_fit_step(a.spot_source/"fit_step.py")
    spot = fit_step.load(a.spot_source/"run_spot.py", original.SOURCE_SPOT, "_quota_spot")
    readout = fit_step.load(formal/"run_dev_readout.py", original.SOURCE_READOUT, "_quota_original_readout")
    score_math = fit_step.load(formal/"score_dev.py", original.MATHEMATICS, "_quota_original_geometry_base")
    geometry = fit_step.load(formal/"evaluate_geometry.py", original.GEOMETRY, "_quota_continuous_geometry")
    oracle = fit_step.load(a.code/"RFDETR_FIXED_BUDGET_GEOMETRY_BOUND_20261120/oracle_geometry.py",
                           d.ORACLE_SHA, "_quota_old_bounds")
    native = fit_step.load(a.code/"HRSID_COCO_E24_DEV705_NATIVE_EVAL_20261109/hrsid_native_dev705.py",
                           original.NATIVE_SOURCE, "_quota_dev")
    native.DEV = a.original_dev_dir
    inputs = original_inputs(a, d, original, spot, readout)
    ranked, pending, counts = build_rank_replay(a, inputs, spot, readout, fit_step, score_math, codec, torch)
    args.output.mkdir()
    rank_rows = [(key, {str(s): [x.tolist() for x in ranked[key][s][0]] for s in SEEDS}) for key in sorted(ranked)]
    rank_sha = sha256(json.dumps(rank_rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    prescore = spot.write(args.output/"PRELABEL_SAFE.json", dict(
        status="RFDETR_DEV705_FROZEN_ORDER_QUOTA_RANKED", protocol_sha256=PROTOCOL,
        authority_sha256=args.authority_sha, prior_score_sha256=d.ORIGINAL_SCORE_SHA,
        prior_shape_witness_sha256=inputs[0]["private_shape_witness_sha256"], same_rank_prefix=True,
        rank_digest=rank_sha, rank_is_fixed_before_dev_gt_opened=True, prior_match_roster_label_derived=True,
        DEV_GT_opened=False, counts=counts))
    label_started = perf_counter()
    gt, _ = d.load_dev_geometry_inputs(native)
    helpers = fit_step.load_geometry(a.code, a.shapely_site)
    summary = score_quota(gt, ranked, pending, inputs[-1], fit_step, helpers, geometry, oracle)
    need(not torch.cuda.is_initialized() and summary["exact_geometry_reproductions"] == 4383,
         "Host-only diagnostic verifies all 4,383 seed-specific fixed matches")
    output = dict(status=STATUS, **summary, protocol_sha256=PROTOCOL, authority_sha256=args.authority_sha,
                  prelabel_sha256=prescore, rank_digest=rank_sha, counts=counts, DEV_GT_opened=True,
                  DEV=True, FIT=False, TEST=False, AP=False, detector_forwards=0, optimizer_updates=0,
                  replay_seconds=label_started-start, geometry_seconds=perf_counter()-label_started,
                  worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    result_sha = spot.write(args.output/"RESULT_SAFE.json", output)
    print(json.dumps(dict(status=STATUS, result_sha256=result_sha)), flush=True)


if __name__ == "__main__":
    main()
