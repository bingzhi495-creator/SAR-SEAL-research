"""Fit matched scalar occupancy heads with BCE and within-side rank supervision."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
import torch
from torch.nn import functional as F


STATUS = "FIT109_SIGNED_RANK_LOSS_FIT_COMPLETE"
PROTOCOL_SHA = "28257034c59461172cd26177c6ef89044d9cf41edd841d3aef9c8fc0ce8e0c95"
PRELOCK_COMMIT = "00a9f24452766a6d0953430e3e259ef49b06711f"
SOURCE_STATUS = "RFDETR_FIT109_SIGNED_RANK_SOURCE_COMPLETE"
BASE = "DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24"
STEP = "RFDETR_CORE36_MATCHED_POINT_SPOT_20261120/fit_step.py"
STEP_SHA = "94e60f5b5557787687d1f228c33c249a1d7889172a225373b97bd789ec8b72b1"
PLAN = "RFDETR_MATCHED_POINT_FORMAL_20261120/fit_plan.py"
PLAN_SHA = "1fd0fe8f815b5ad4e01d0d8eeca0aa0e1480433292fc48f94fee17e5837e5c35"
SEEDS = (5217, 5218, 5219)
ARMS = ("bce", "bce_rank")
STEPS = 200
PACKETS = 16
ROWS = 4
RANK_WEIGHT = .25


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(data):
    return sha256(data).hexdigest()


def pin(path, expected):
    data = path.read_bytes()
    require(digest(data) == expected, "Exact digest: " + path.name)
    return data


def load(path, expected):
    module = ModuleType(path.stem)
    module.__file__ = str(path)
    exec(compile(pin(path, expected), str(path), "exec"), module.__dict__)
    return module


def ranking_loss(logits, targets):
    """Average all unordered within-candidate, within-sign comparisons, including ties."""
    left, right = torch.triu_indices(ROWS, ROWS, offset=1)
    target_gap = targets[:, left, :] - targets[:, right, :]
    logit_gap = logits[:, left, :] - logits[:, right, :]
    return (target_gap.abs() * F.softplus(-target_gap.sign() * logit_gap)).mean()


def sample_packets(group_candidates, offset, seed, steps=STEPS):
    rng = np.random.Generator(np.random.PCG64(20261112 + seed - 5217))
    draws = rng.integers(len(group_candidates), size=(steps, PACKETS))
    batches = np.empty((steps, PACKETS, ROWS), dtype=np.int64)
    for update, groups in enumerate(draws):
        for packet, group in enumerate(groups):
            candidates = group_candidates[group]
            c = int(candidates[rng.integers(len(candidates))])
            batches[update, packet] = rng.integers(offset[c], offset[c + 1], size=ROWS)
    return batches


def open_source(c, plan):
    terminal = json.loads(pin(c.source / "observed_source/TERMINAL.json", c.source_terminal_sha))
    report = json.loads(pin(c.source / "actual/RESULT_SAFE.json", c.source_result_sha))
    require(terminal["phase"] == "prepare-edits" and terminal["actual_exit"] == 0
            and terminal["output_footer_verified"] is True and terminal["result_sha256"] == c.source_result_sha
            and terminal["worker_sha256"] == report["worker_sha256"] == c.source_worker_sha
            and report["status"] == SOURCE_STATUS and report["FIT"] is True
            and report["all_labels_source_fit_only"] is True and report["private_payload_readback_exact"] is True
            and all(report[key] is False for key in ("DEV", "TEST", "SRNet", "Umbra", "AP"))
            and report["detector_forwards"] == report["optimizer_updates"] == 0
            and report["original_fold_plan_sha256"] == PLAN_SHA
            and report["original_fit_result_sha256"] == "114ac529e2196fa4f60d8fe56e07381fb6814e0fdb09e7d5615626fd74a85f64"
            and report["original_fit_readback_sha256"] == "476b0f3fa748d1d3fd2393bdeb1af2d83e1532995d58b12625d57222e3f2638a",
            "Canonical source observed closure")
    points = c.source / "actual/POINTS_PRIVATE.npz"
    pin(points, report["points_sha256"])
    with np.load(points, allow_pickle=False) as npz:
        arrays = {key:npz[key] for key in ("group", "fold", "offset", "features", "truth", "role",
                                           "roster_image_id", "roster_group", "roster_fold", "image_id")}
    g, f, offsets, features, y, role, image_ids = (
        arrays[key] for key in ("group", "fold", "offset", "features", "truth", "role", "image_id"))
    gids, gset, folds = (arrays[key] for key in ("roster_image_id", "roster_group", "roster_fold"))
    assignments = plan.source_folds(gset.tolist())
    require(len(gids) == len(gset) == len(folds) == len(set(gids.tolist())) == len(assignments) == 109
            and all(assignments[str(group)] == fold for group, fold in zip(gset, folds, strict=True)),
            "Exact original held group and order")
    roster = dict(zip(gids.tolist(), gset.tolist(), strict=True))
    require(g.ndim == 1 and f.shape == image_ids.shape == role.shape == g.shape and len(g) + 1 == len(offsets)
            and all(roster[i] == group and assignments[group] == fold
                    for i, group, fold in zip(image_ids.tolist(), g.tolist(), f.tolist(), strict=True))
            and len(role) == report["dataset_counts"]["supported"] == 6421
            and int(np.count_nonzero(role == 2)) == report["supervision_counts"]["valid"]
            and offsets[0] == 0 and np.all(np.diff(offsets) >= 0) and offsets[-1] == len(features)
            and features.dtype == np.float32 and features.shape == (len(features), 2, 14)
            and y.dtype == np.float64 and y.shape == features.shape[:2]
            and np.isfinite(features).all() and np.all((features[:, :, 0] >= 0) & (features[:, :, 0] <= 1)),
            "Closed canonical point features, support and prefix membership")
    for i in range(len(g)):
        t = y[offsets[i]: offsets[i+1]]
        if role[i] == 2:
            require(np.isfinite(t).all() and np.all((t >= 0) & (t <= 1+1e-12)),
                    "Exact author fractional unit occupancy for validated candidate")
        else:
            require(role[i] in (0, 1) and np.isnan(t).all(), "Unsupervised original point candidates")
    return arrays, report


def fold_schedule(arrays, held, seed, trainer):
    groups = defaultdict(list)
    offsets = arrays["offset"]
    eligible = (arrays["role"] == 2) & (np.diff(offsets) > 0) & (arrays["fold"] != held)
    for c in np.flatnonzero(eligible):
        groups[arrays["group"][c]].append(int(c))
    ordered = [np.asarray(groups[k], dtype=np.int64) for k in sorted(groups)]
    require(ordered, "Nonempty training source prefixes")
    groups_rows = [np.concatenate([np.arange(offsets[c], offsets[c + 1], dtype=np.int64) for c in g])
                   for g in ordered]
    center, scale = trainer.group_moments(arrays["features"], groups_rows)
    schedule = sample_packets(ordered, offsets, seed)
    repeated = np.repeat(arrays["fold"], np.diff(offsets))
    require(np.all(repeated[schedule] != held)
            and np.all(np.isfinite(arrays["truth"][schedule])), "Training folds only in every optimizer input")
    return center, scale, schedule, dict(training_groups=len(ordered), training_candidates=int(eligible.sum()),
                                         training_endpoint_pairs=int(sum(map(len, groups_rows))))


def fit_model(trainer, step, arrays, center, scale, schedule, seed, arm):
    started = perf_counter()
    torch.manual_seed(seed)
    model = trainer.PointHead(14)
    initial = step.tensor_hash(model)
    require(sum(parameter.numel() for parameter in model.parameters()) == 1321
            and not torch.count_nonzero(model.network[-1].weight) and not torch.count_nonzero(model.network[-1].bias),
            "Exact original 1321 trainable parameters and zero final residual")
    opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
    feature = arrays["features"]
    # Only scheduled training rows enter the model or target tensors used by this optimizer.
    samples = feature[schedule]
    values = np.asarray((samples - center) / scale, dtype=np.float32)
    x = torch.from_numpy(values)
    p = torch.from_numpy(samples[..., 0].copy())
    y = torch.from_numpy(arrays["truth"][schedule].astype(np.float32))
    first, last = None, None
    for inputs, native, targets in zip(x, p, y, strict=True):
        opt.zero_grad(set_to_none=True)
        logits = model(inputs, inputs, native)
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        order = ranking_loss(logits, targets)
        loss = bce + RANK_WEIGHT * order if arm == "bce_rank" else bce
        require(bool(torch.isfinite(loss)), "Finite training objective")
        loss.backward()
        require(all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters()),
                "Finite trained scalar parameters and gradients")
        opt.step()
        last = dict(bce=float(bce.detach()), within_sign_rank=float(order.detach()), total=float(loss.detach()))
        if first is None:
            first = last
    model.eval()
    return model, dict(initial_sha256=initial, final_sha256=step.tensor_hash(model),
                       initial_loss=first, final_loss=last, steps=len(schedule),
                       fit_elapsed_seconds=perf_counter()-started)


def seal(path, arrays):
    require(all(value.dtype == np.float32 and np.isfinite(value).all() for value in arrays.values()),
            "Finite FP32 private checkpoint and held predictions")
    with path.open("xb") as stream:
        np.savez(stream, **arrays)
    with np.load(path, allow_pickle=False) as archived:
        require(set(archived.files) == set(arrays) and all(np.array_equal(archived[k], v) for k,v in arrays.items()),
                "Exact stored native tensors on readback")
    return digest(path.read_bytes())


def restore_head(path, sha, trainer):
    pin(path, sha)
    head = trainer.PointHead(14)
    parameters = head.state_dict()
    with np.load(path, allow_pickle=False) as stash:
        require(set(stash.files) == {"center", "scale", *("S."+name for name in parameters)}, "Scalar head bundle")
        center, scale = stash["center"], stash["scale"]
        for name in parameters:
            stored = stash["S." + name]
            require(stored.shape == tuple(parameters[name].shape) and stored.dtype == np.float32, "Exact scalar tensor")
            parameters[name] = torch.from_numpy(stored)
    require(center.shape == scale.shape == (14,) and center.dtype == scale.dtype == np.float32
            and np.isfinite(center).all() and np.isfinite(scale).all() and np.all(scale > 0), "Exact held scaler")
    head.load_state_dict(parameters, strict=True)
    head.eval()
    return head, center, scale


def fit_paired(arrays, trainer, step, plan, output):
    require(plan.SEEDS == SEEDS and plan.STEPS == STEPS and plan.BATCH == PACKETS*ROWS, "Original update budgets")
    weights_root = output / "POINT_HEADS_PRIVATE"
    held_root = output / "HELD_POSTERIORS_PRIVATE"
    output.mkdir()
    weights_root.mkdir()
    held_root.mkdir()
    weights, predictions = [], []
    for seed in SEEDS:
        held = {arm: np.full(arrays["features"].shape[:2], np.nan, dtype=np.float32) for arm in ARMS}
        for fold in range(5):
            center, scale, schedule, census = fold_schedule(arrays, fold, seed, trainer)
            common_batch_sha = digest(schedule.astype("<i8", copy=False).tobytes())
            initial = set()
            first_losses = set()
            scaler_sha = digest(center.tobytes() + scale.tobytes())
            for arm in ARMS:
                model, audit = fit_model(trainer, step, arrays, center, scale, schedule, seed, arm)
                initial.add(audit["initial_sha256"])
                first_losses.add((audit["initial_loss"]["bce"], audit["initial_loss"]["within_sign_rank"]))
                path = weights_root / f"{seed}_fold_{fold}_{arm}.npz"
                content = {"center": center, "scale": scale,
                           **{f"S.{key}": value.detach().numpy() for key, value in model.state_dict().items()}}
                checksum = seal(path, content)
                head, cen, sc = restore_head(path, checksum, trainer)
                require(step.tensor_hash(head) == audit["final_sha256"], "Exact independently restored trained tensors")
                for c in np.flatnonzero(arrays["fold"] == fold):
                    start, end = arrays["offset"][c:c+2]
                    require(np.isnan(held[arm][start:end]).all(), "No held row scored by two heads")
                    if end > start:
                        predictions_f = step.predict(head, arrays["features"][start:end], cen, sc)
                        held[arm][start:end] = predictions_f
                weights.append(dict(seed=seed, fold=fold, arm=arm, relative_weights=path.relative_to(output).as_posix(),
                                    weights_sha256=checksum, common_batch_sha256=common_batch_sha,
                                    scaler_sha256=scaler_sha, **audit, **census))
            require(len(initial) == len(first_losses) == 1,
                    "Both optimizer arms used the exact same initial tensors and first batch")
            print(json.dumps(dict(stage="OBSERVED_FOLD_PAIR", seed=seed, fold=fold, updates=2*STEPS)), flush=True)
        require(all(np.isfinite(x).all() and np.all((x >= 0.) & (x <= 1.)) for x in held.values()),
                "Each source point was predicted in its held fold once per arm")
        path = held_root / f"{seed}.npz"
        checksum = seal(path, held)
        predictions.append(dict(seed=seed, relative_predictions=path.relative_to(output).as_posix(),
                                sha256=checksum, source_shape=list(held[ARMS[0]].shape)))
    return weights, predictions


def run(c):
    started=perf_counter()
    require(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
            and torch.__version__ == "2.10.0+cu130" and np.__version__ == "1.26.4"
            and not torch.cuda.is_initialized() and not c.output.exists(), "Fresh admitted CPU fitting")
    require(c.prelock_commit == PRELOCK_COMMIT and c.protocol_sha == PROTOCOL_SHA, "Exact verified preregistration")
    pin(c.protocol, c.protocol_sha)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    plan = load(c.code / PLAN, PLAN_SHA)
    step = load(c.code / STEP, STEP_SHA)
    trainer = step.load_trainer(c.code / BASE / "fit109_paired_point/train_point.py")
    arrays, source = open_source(c, plan)
    weights, held = fit_paired(arrays, trainer, step, plan, c.output)
    total = sum(w["steps"] for w in weights)
    require(len(weights) == 30 and len(held) == 3 and total == 6000, "Complete matched optimizer and held schedule")
    note = dict(status=STATUS, protocol_sha256=c.protocol_sha, prelock_commit=c.prelock_commit,
                worker_sha256=digest(Path(__file__).read_bytes()), source_result_sha256=c.source_result_sha,
                source_terminal_sha256=c.source_terminal_sha, source_points_sha256=source["points_sha256"],
                source_worker_sha256=c.source_worker_sha, weights=weights, held_predictions=held,
                equal_initialization_and_rows=True, group_fold_train_only_standardization=True, seeds=list(SEEDS),
                n_parameters_per_arm=1321, folds=5, updates_per_arm_fold_seed=STEPS, optimizer_updates=total,
                training_pair_rows_per_update=PACKETS*ROWS, packet_candidates_per_update=PACKETS,
                paired_rows_per_candidate_packet=ROWS, rank_comparisons_per_update=192, rank_weight=RANK_WEIGHT,
                adamw_learning_rate=.002, adamw_weight_decay=.01, additional_refits=0,
                arm_total_fit_seconds={arm:sum(row["fit_elapsed_seconds"] for row in weights if row["arm"] == arm) for arm in ARMS},
                test_references_opened=0, detector_forwards=0, AP_computed=False, elapsed_seconds=perf_counter()-started)
    out = (json.dumps(note, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()
    (c.output / "RESULT_SAFE.json").write_bytes(out)
    print(json.dumps(dict(status=STATUS, result_sha256=digest(out))), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code", "protocol", "source", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("protocol-sha", "prelock-commit", "source-result-sha", "source-terminal-sha", "source-worker-sha"):
        p.add_argument("--" + name, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()

