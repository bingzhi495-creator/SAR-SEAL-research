"""Materialize FIT109 quota features and continuous targets in separate shards."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np


PROTOCOL = "22048da5422cecb93ff825fd16f02932276c8a20e50a022e7bb2d212d56e732b"
FIT_LOADER = "9f989c3a976642d9e790d65199dda43da4ae6b946d649fa13d1106258e99f84e"
FIT_RESULT = "114ac529e2196fa4f60d8fe56e07381fb6814e0fdb09e7d5615626fd74a85f64"
FIT_PRELABEL = "33fc0e63fa5deff68aa384e03a3f00fd78430e63a7680b7a5a8025e33422acf4"
FIT_FEATURES = "1cea85eb8c30b11bfa847292c55d2231fb4d16dd2a551c3da52ddffdd8d3830e"
QUOTA_SOURCE = "cd6d1a732a01f8804d28991e4debf30c88f50057b0b2ad0754b7db44c46691a2"
ORIGINAL_LOADER = "3a77e660fd31570cf1f4b44e818b28cc6743d414591216a5f274a11139955b05"
STATUS = "RFDETR_FIT109_QUOTA_SOURCE_ACTIONS_COMPLETE"
SEEDS = (5217, 5218, 5219)
FEATURES = ("signed_area_delta_over_native", "abs_area_delta_over_native", "expected_add_mass_delta_over_native",
            "expected_drop_mass_delta_over_native", "ratio_change_proxy", "native_area", "native_score",
            "support_expected_signed_area_delta_over_native", "balanced_count")
MIB = 1024**2
TOL = 1.e-12


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def pin(path, expected, limit=MIB):
    with path.open("rb") as stream:
        raw = stream.read(limit+1)
    need(0 < len(raw) <= limit and sha256(raw).hexdigest() == expected, "Exact bounded input: "+path.name)
    return raw


def read_json(path, expected, limit=MIB):
    return json.loads(pin(path, expected, limit))


def load(path, expected, name):
    from types import ModuleType
    item = ModuleType(name)
    item.__file__ = str(path)
    sys.modules[name] = item
    exec(compile(pin(path, expected), str(path), "exec"), item.__dict__)
    return item


def prefixes(values, ranks):
    return np.r_[0., np.cumsum(values[ranks], dtype=np.float64)]


def actions_and_features(coords, q, ranks, area, score):
    """Encode all feasible same-count actions on an unchanged support and ranks."""
    need(coords.shape == q.shape and q.ndim == 2 and q.shape[1] == 2 and 0 < len(q) <= 32
         and np.unique(coords).size == coords.size and 32 <= area < 1024 and len(q) <= area
         and 0 <= score <= 1 and np.isfinite(q).all() and np.all((q >= 0) & (q <= 1)),
         "Original native eligibility and score, one finite posterior per endpoint")
    add_rank, drop_rank = ranks
    cap = min(8, int(area)//10, len(q))
    k = int(np.count_nonzero(q[add_rank[:cap], 0] > q[drop_rank[:cap], 1]))
    first, last = max(0, 2*k-len(q)), min(2*k, len(q))
    adds = np.arange(first, last+1, dtype=np.int16)
    drops = 2*k-adds
    mass_in = prefixes(q[:, 0], add_rank)
    mass_out = prefixes(q[:, 1], drop_rank)
    area_shift = (adds-drops)/area
    total_in = mass_in[-1]
    missed = len(q)-mass_out[-1]
    original_I = area-missed
    truth = original_I+total_in
    overlap = original_I+mass_in[adds]-mass_out[drops]
    union = area+adds-drops+truth-overlap
    values = overlap/union
    baseline = k-first
    features = np.column_stack((area_shift, np.abs(area_shift), (mass_in[adds]-mass_in[k])/area,
                                (mass_out[drops]-mass_out[k])/area, values-values[baseline],
                                np.full(len(adds), area), np.full(len(adds), score),
                                np.full(len(adds), (total_in-missed)/area), np.full(len(adds), k)))
    need(np.isfinite(features).all() and len(adds) <= 17
         and np.all(union > 0) and np.all((overlap >= -TOL) & (overlap <= truth+TOL))
         and np.all(features[baseline, :5] == 0), "Finite physical action features and balanced anchor")
    return k, np.column_stack((adds, drops)), features


def geometric_targets(q, rank, raw, actions, k):
    """Evaluate each action at true I/P/T, preserving the observed balanced count."""
    truth = np.asarray(q, dtype=np.float64)
    need(truth.ndim == 2 and truth.shape[1] == 2 and np.isfinite(truth).all()
         and np.all((truth >= 0) & (truth <= 1 + TOL)), "Physical continuous polygon endpoint values")
    i, p, t = (raw[v] for v in ("intersection", "predicted_area", "truth_area"))
    adds, drops = actions[:, 0], actions[:, 1]
    inc = prefixes(truth[:, 0], rank[0])[adds]
    dec = prefixes(truth[:, 1], rank[1])[drops]
    and_area = i+inc-dec
    new_area = p+adds-drops
    union = new_area+t-and_area
    need(np.all(adds+drops == 2*k) and np.all(union > 0) and np.all(and_area >= -1e-8)
         and np.all(and_area <= np.minimum(new_area, t)+1e-8), "Physical exact-edit truth union")
    ious = and_area/union
    balance = np.flatnonzero(adds == k)
    need(len(balance) == 1, "Exactly one unchanged balanced action")
    return ious-ious[balance[0]]


def checked_refits(a, source_plan, fit_step):
    accepted = read_json(a.fit_accepted, a.fit_accepted_sha)
    result = read_json(a.fit_result, FIT_RESULT)
    wait = read_json(a.fit_terminal, accepted["original_tool_wait_sha256"], 8*MIB)
    need(accepted["status"] == "RFDETR_MATCHED_POINTS_FIT109_INDEPENDENT_ACCEPTED"
         and accepted["fit_result_sha256"] == FIT_RESULT and accepted["run_natural_exit"] == 0
         and wait["terminal"]["exit_code"] == 0 and accepted["checkpoint_bundles_verified"] == 18
         and accepted["head_updates_verified"] == result["head_updates"] == 7200
         and accepted["exact_schedule"] and accepted["FIT"] and not any(accepted[k] for k in ("DEV", "TEST", "AP")),
         "Closed archived five-fold base heads")
    begin = read_json(a.fit_result.with_name("PRELABEL_SAFE.json"), FIT_PRELABEL)
    need(begin["feature_sha256"] == FIT_FEATURES and begin["accepted_capture_sha256"] == a.acceptance_sha256
         and begin["GT_opened"] is False, "Exact legacy FIT feature and capture fingerprints")
    rows = {(r["seed"], r["fold"]): r for r in result["all_initializations"] if r["fold"] is not None}
    need(set(rows) == {(s,f) for s in SEEDS for f in range(5)}, "All archived held-fold fits and their seeds")
    trainer = fit_step.load_trainer(a.code/"DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/fit109_paired_point/train_point.py")
    restored = {}
    for (seed, fold), r in rows.items():
        need(r["endpoint"] == f"fold_{fold}" and r["relative_weights"] == f"{seed}_fold_{fold}.npz",
             "Exact held-fold weights instead of full FIT weights")
        heads, center, scale = source_plan.restore_bundle(a.head_root/r["relative_weights"], r["weights_sha256"], trainer, fit_step)
        restored[seed, fold] = (heads["S"], center, scale)
    return restored, result


def save_npz(path, **arrays):
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return sha256(path.read_bytes()).hexdigest()


def precompute_shards(data, fold_by_group, chosen_heads, fit_step, quota, output):
    feature_dir = output/"FEATURE_SHARDS_PRIVATE"
    feature_dir.mkdir()
    manifests, contexts = [], []
    for ordinal, (item, candidates, records) in enumerate(data, 1):
        group = item["group"]
        fold = fold_by_group[group]
        candidate_by_rank = {c["rank"]: c for c in candidates}
        indices, coords_set, posts, ranks_set, support_counts = [], [], [], [], []
        actions, values, offsets, balanced_counts = [], [], [0], []
        for seed in SEEDS:
            model, center, scale = chosen_heads[seed, fold]
            for row in records:
                coords = row["coordinates"]
                posterior = fit_step.predict(model, row["features"], center, scale)
                ranks = quota.original_ranks(posterior)
                native = candidate_by_rank[row["rank"]]
                need(native["query_id"] == row["query_id"], "Exact native candidate identity")
                k, moves, features = actions_and_features(coords, posterior, ranks, native["area"], native["score"])
                offset = offsets[-1]
                offsets.append(offset+len(moves))
                indices.append((seed, row["image_id"], row["rank"], row["query_id"]))
                n = len(coords)
                coords_set.append(np.pad(coords, ((0,32-n),(0,0)), constant_values=-1))
                posts.append(np.pad(posterior, ((0,32-n),(0,0))))
                ranks_set.append(np.pad(np.column_stack(ranks), ((0,32-n),(0,0)), constant_values=-1))
                support_counts.append(n)
                balanced_counts.append(k)
                actions.append(moves)
                values.append(features)
        need(len(indices) == 3*len(records), "Every supported candidate has one row per archived seed")
        all_actions = np.concatenate(actions) if actions else np.empty((0,2), dtype=np.int16)
        features = np.concatenate(values) if values else np.empty((0,len(FEATURES)), dtype=np.float64)
        ids = np.asarray(indices, dtype=np.int64).reshape(-1,4)
        lengths = np.asarray(support_counts, dtype=np.int16)
        ordered = np.asarray(ranks_set, dtype=np.int16).reshape(-1,32,2)
        filename = f"{ordinal:03d}.npz"
        digest = save_npz(feature_dir/filename, identity=ids, actions=all_actions, features=features,
                          feature_names=np.asarray(FEATURES), endpoints=np.asarray(coords_set,dtype=np.int32).reshape(-1,32,2),
                          posterior=np.asarray(posts,dtype=np.float32).reshape(-1,32,2), rank_order=ordered,
                          endpoint_count=lengths, balanced_count=np.asarray(balanced_counts,dtype=np.int16),
                          action_offsets=np.asarray(offsets,dtype=np.int32))
        manifests.append(dict(file=filename, sha256=digest, image_id=item["image_id"], group=group,
                              fold=fold, candidate_rows=len(ids), actions=len(all_actions)))
        contexts.append((ids, all_actions, ordered, lengths, offsets, balanced_counts))
    return manifests, contexts


def attach_shards(records, feature_manifest, contexts, output):
    target_dir = output/"TARGET_SHARDS_PRIVATE"
    target_dir.mkdir()
    original = {(r["image_id"], r["rank"], r["query_id"]): r for r in records}
    need(len(original) == len(records) == 6421, "Complete original supported population")
    summaries = []
    counts = Counter()
    for feature_ref, context in zip(feature_manifest, contexts, strict=True):
        ids, moves, orders, lengths, offsets, quota_k = context
        mask = np.zeros(len(ids), dtype=bool)
        targets = np.full(len(moves), np.nan, dtype=np.float64)
        for i, identity in enumerate(ids):
            row = original[tuple(identity[1:])]
            need(feature_ref["image_id"] == row["image_id"] and feature_ref["group"] == row["group"],
                 "Same source group for features and labels")
            k = quota_k[i]
            counts["zero_budget_candidates"] += int(k == 0)
            if row["role"] != "valid":
                continue
            n = lengths[i]
            span = slice(offsets[i], offsets[i+1])
            ranks = orders[i,:n,0], orders[i,:n,1]
            values = geometric_targets(row["truth"], ranks, row["raw_continuous"], moves[span], k)
            targets[span] = values
            mask[i] = True
            counts.update(supervised_candidate_seed_rows=1, supervised_actions=len(values),
                          supervised_zero_budget_candidates=int(k == 0))
        result = save_npz(target_dir/feature_ref["file"], supervised_candidate=mask, delta_iou=targets)
        counts.update(all_actions=len(moves), all_candidate_seed_rows=len(ids))
        summaries.append(dict(feature_manifest=feature_ref, target_sha256=result,
                              supervised_candidate_seed_rows=int(mask.sum()),
                              supervised_actions=int(np.isfinite(targets).sum())))
    need(counts["supervised_candidate_seed_rows"] == 3*392 and counts["all_candidate_seed_rows"] == 3*6421,
         "Original matched and full candidate denominators")
    return summaries, dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--authority-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    cfg = parser.parse_args()
    paths = {"protocol", "code", "capture", "acceptance", "source_cohort", "fit_images", "fit_gt", "shapely_site",
             "fit_accepted", "fit_terminal", "fit_result", "head_root"}
    hashes = {"acceptance_sha256", "fit_accepted_sha"}
    authority = read_json(cfg.authority, cfg.authority_sha)
    need(set(authority) == paths | hashes and all(Path(authority[k]).is_absolute() for k in paths)
         and cfg.output.is_absolute() and not cfg.output.exists() and cfg.output.parent.is_dir()
         and sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
         "Fresh isolated CPU-only HRSID FIT invocation")
    a = SimpleNamespace(**{k: Path(v).resolve(strict=True) if k in paths else v for k, v in authority.items()})
    pin(a.protocol, PROTOCOL)
    started = perf_counter()
    import torch
    from pycocotools import mask as codec
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    formal = a.code/"RFDETR_MATCHED_POINT_FORMAL_20261120"
    old = load(formal/"run_score_dev.py", ORIGINAL_LOADER, "_fit_quota_archived_modules")
    fit_step = old.load_fit_step(a.code/"RFDETR_CORE36_MATCHED_POINT_SPOT_20261120/fit_step.py")
    spot = fit_step.load(a.code/"RFDETR_CORE36_MATCHED_POINT_SPOT_20261120/run_spot.py", old.SOURCE_SPOT, "_fit_quota_spot")
    readout = fit_step.load(formal/"run_dev_readout.py", old.SOURCE_READOUT, "_fit_quota_readout")
    previous_fit = load(formal/"run_fit109.py", FIT_LOADER, "_fit_quota_original_source_fit")
    source_plan = load(formal/"fit_plan.py", readout.PINS["fit_plan.py"], "_fit_quota_fold_plan")
    quota = load(a.code/"RFDETR_DEV705_FROZEN_ORDER_QUOTA_20261123/quota.py", QUOTA_SOURCE, "_fit_quota_ranks")
    feat = load(a.code/"RFDETR_CORE36_MATCHED_POINT_SPOT_20261120/feature_adapter.py",
                readout.PINS["feature_adapter.py"], "_fit_quota_features")
    posterior = fit_step.load(a.code/"RFDETR_MATCHED_POINT_REFINEMENT_20261120/adapter.py",
                              spot.ADAPTER_SHA, "_fit_quota_native_posterior")
    heads, old_fit = checked_refits(a, source_plan, fit_step)
    roster, accepted = previous_fit.read_population(a, spot)
    stock = pin(a.capture/"STOCK_NATIVE_PREDICTIONS_PRIVATE.jsonl", accepted["stock_predictions_sha256"], 128*MIB)
    need(len(stock.splitlines()) == 109, "Retain complete source stock predictions for future fixed evaluation")
    fold_by_group = source_plan.source_folds(item["group"] for item, _, _ in roster)
    need(len(fold_by_group) == 109, "Original five-fold population with one image per source")
    data, evidence, digest = spot.prepare(a, roster, fit_step, feat, posterior, np, torch, codec)
    need(digest == FIT_FEATURES and old_fit["preparation"] == evidence and accepted["same_device_raster_exact"] == 6738,
         "Reproduce exact archived FIT point features and original masks")
    cfg.output.mkdir()
    features, contexts = precompute_shards(data, fold_by_group, heads, fit_step, quota, cfg.output)
    prelabel = spot.write(cfg.output/"FEATURE_MANIFEST_PRIVATE.json", dict(
        groups=features, feature_names=list(FEATURES), GT_opened=False, protocol_sha256=PROTOCOL,
        original_stock_rle_sha256=accepted["stock_predictions_sha256"],
        original_candidate_roster_sha256=accepted["queries_sha256"],
        original_image_manifest_sha256=accepted["image_manifest_sha256"]))
    fit_target_started = perf_counter()
    records, labels = spot.attach(a, data, fit_step)
    need(labels == old_fit["supervision"], "Original HRSID FIT fractional polygon supervision")
    outputs, counts = attach_shards(records, features, contexts, cfg.output)
    target_index = spot.write(cfg.output/"TARGET_MANIFEST_PRIVATE.json", dict(groups=outputs, feature_manifest_sha256=prelabel))
    fold_summary = {f: Counter() for f in range(5)}
    for out in outputs:
        row = out["feature_manifest"]
        fold_summary[row["fold"]].update(groups=1, candidate_seed_rows=row["candidate_rows"], actions=row["actions"],
                                         supervised_seed_rows=out["supervised_candidate_seed_rows"],
                                         supervised_actions=out["supervised_actions"])
    need(not torch.cuda.is_initialized(), "CPU-only archived FIT action construction")
    report = dict(status=STATUS, protocol_sha256=PROTOCOL, authority_sha256=cfg.authority_sha,
                  original_feature_sha256=digest, source_head_refits=15, source_head_seeds=list(SEEDS),
                  original_stock_rle_sha256=accepted["stock_predictions_sha256"],
                  original_candidate_roster_sha256=accepted["queries_sha256"],
                  counts=counts, fold_counts=fold_summary, original_label_counts=labels,
                  supervision_geometry_identities=labels["valid"], independent_source_prefix_groups=109,
                  prelabel_feature_manifest_sha256=prelabel, target_manifest_sha256=target_index,
                  feature_seconds=fit_target_started-started, target_seconds=perf_counter()-fit_target_started,
                  original_detector_calls=0, head_forward_calls=counts["all_candidate_seed_rows"], optimizer_updates=0,
                  FIT=True, FIT_GT_opened=True, DEV=False, TEST=False, AP=False,
                  fully_nested_quota_cv=False, worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    digest = spot.write(cfg.output/"RESULT_SAFE.json", report)
    print(json.dumps(dict(status=STATUS, result_sha256=digest)), flush=True)


if __name__ == "__main__":
    main()
