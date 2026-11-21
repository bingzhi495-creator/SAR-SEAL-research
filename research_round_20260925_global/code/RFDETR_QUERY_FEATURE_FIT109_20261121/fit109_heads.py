"""Fixed, source-only OOF point-head experiment. All inputs are caller-admitted in memory."""
from collections import defaultdict
from contextlib import contextmanager
from hashlib import sha256

import numpy as np
import torch


ARMS = ("zero", "ordinary_centered", "query_conditioned_centered", "query_conditioned_permuted")
SEED, SAMPLE_SEED, STEPS, BATCH = 5217, 20261112, 200, 64


def need(condition, message):
    if not condition:
        raise ValueError(message)


@contextmanager
def _single_cpu_thread():
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        yield
    finally:
        torch.set_num_threads(previous)


def identity(row):
    key = tuple(row[name] for name in ("image_id", "rank", "query_id"))
    need(all(type(value) is int and value >= 0 for value in key) and key[1] > 0,
         "Original image/rank/query integer identity")
    return key


def _assignments(records, group_ids, old_by_fold, fit_plan):
    groups = tuple(group_ids)
    need(len(groups) == len(set(groups)) == 109 and all(isinstance(g, str) and g for g in groups),
         "Expected the original 109 unique source group IDs")
    need(len(records) == len({identity(r) for r in records}), "Duplicate candidate identity")
    need({r["group"] for r in records} <= set(groups), "Foreign source group")
    need(set(old_by_fold) == set(range(5)), "Five held-fold old S heads")
    for fold, old in old_by_fold.items():
        need(old["seed"] == SEED and old["fold"] == fold, "Old head belongs to another fold/seed")
        for key in ("center", "scale"):
            x = old[key]
            need(x.shape == (14,) and x.dtype == np.float32 and np.isfinite(x).all(),
                 "Old fold FP32 scaler")
        need(np.all(old["scale"] > 0), "Old fold positive scales")
    return fit_plan.source_folds(groups)


def _inputs(row):
    coords, old, features = row["coordinates"], row["old14"], row["features"]
    n = len(coords)
    need(coords.shape == (n, 2) and n <= 32 and coords.dtype.kind in "iu" and
         np.all(coords >= 0) and np.unique(coords).size == 2*n, "Original disjoint paired endpoints")
    need(type(row["cap"]) is int and 0 <= row["cap"] <= min(8, n), "Admitted area cap")
    if "area" in row:
        need(row["cap"] == min(8, int(row["area"]) // 10, n), "Cap differs from original mask area")
    need(old.shape == (n, 2, 14) and old.dtype == np.float32 and np.isfinite(old).all() and
         np.all((old[:, :, 0] >= 0) & (old[:, :, 0] <= 1)), "Finite native old14 and probability")
    need(set(features) == set(ARMS), "Exactly four predeclared arms")
    digest = sha256()
    digest.update(np.asarray(coords, dtype="<i8").tobytes())
    digest.update(old.astype("<f4", copy=False).tobytes())
    for name in ARMS:
        x = features[name]
        need(x.shape == (n, 2, 30) and x.dtype == np.float32 and np.isfinite(x).all() and
             np.array_equal(x[:, :, :14].view(np.uint32), old.view(np.uint32)),
             "Each aligned arm must preserve the original 14 features bitwise")
        digest.update(x[:, :, 14:].astype("<f4", copy=False).tobytes())
    need(not np.count_nonzero(features["zero"][:, :, 14:]), "Zero control residual")
    return digest.hexdigest()


def free_selection(probabilities, cap, count=None):
    """Same on-side ordinal tie break and positive FP32 marginals as readouts_sparse.SU."""
    q = probabilities
    n = len(q)
    need(q.dtype == np.float32 and q.shape == (n, 2) and np.isfinite(q).all() and
         np.all((q >= 0) & (q <= 1)) and type(cap) is int and 0 <= cap <= min(8, n),
         "Original aligned FP32 probabilities and action cap")
    ordinal = np.arange(n, dtype=np.int32)
    add = np.lexsort((ordinal, -q[:, 0]))[:cap]
    drop = np.lexsort((ordinal, q[:, 1]))[:cap]
    if count is None:
        count = int(np.count_nonzero(q[add, 0] - q[drop, 1] > 0))
    need(type(count) is int and 0 <= count <= cap, "Same prelabel k within native cap")
    return np.column_stack((add[:count], drop[:count])).astype(np.int32)


def seal_oof_budgets(records, group_ids, old_by_fold, fit_step, fit_plan):
    """Feature-only records in original order. Caller persists and verifies this map before GT access."""
    assignments = _assignments(records, group_ids, old_by_fold, fit_plan)
    budgets = {}
    need(all(not ({"truth", "matched_gt_id", "raw_continuous", "role"} & row.keys()) for row in records),
         "Label-derived fields forbidden during budget replay")
    with _single_cpu_thread():
        for row in records:
            witness = _inputs(row)
            fold = assignments[row["group"]]
            old = old_by_fold[fold]
            q = fit_step.predict(old["model"], row["old14"], old["center"], old["scale"])
            k = len(free_selection(q, row["cap"]))
            budgets[identity(row)] = dict(group=row["group"], fold=fold, coordinates=row["coordinates"].copy(),
                                          cap=row["cap"], k=k, old_q=q, inputs_sha256=witness)
    return budgets


def _schedule(groups):
    rng = np.random.Generator(np.random.PCG64(SAMPLE_SEED))
    draws = rng.integers(len(groups), size=(STEPS, BATCH))
    return np.asarray([[groups[g][rng.integers(len(groups[g]))] for g in row] for row in draws],
                      dtype=np.int64)


def _initial_model(trainer, fit_step, old_initial_hash):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(SEED)
        old = trainer.PointHead(14)
        need(fit_step.tensor_hash(old) == old_initial_hash, "Incorrect old S initial state")
        model = trainer.PointHead(30)
    with torch.no_grad():
        for target, source in zip(model.network[1:].parameters(), old.network[1:].parameters()):
            target.copy_(source)
        target, source = model.network[0], old.network[0]
        target.bias.copy_(source.bias)
        target.weight.zero_()
        target.weight[:, :14].copy_(source.weight[:, :14])
        target.weight[:, 30:44].copy_(source.weight[:, 14:])
    need(sum(v.numel() for v in model.parameters()) == 2089, "Fixed compact architecture")
    return model


def _fit_arm(model, features, targets, batches, trainer, groups, old, fit_step):
    center, scale = trainer.group_moments(features, groups)
    need(np.array_equal(center[:14], old["center"]) and np.array_equal(scale[:14], old["scale"]),
         "Original source-balanced scaler mismatch")
    need(np.isfinite(center).all() and np.isfinite(scale).all() and np.all(scale > 0),
         "Finite positive train-fold scales")
    x = torch.from_numpy(np.asarray((features - center) / scale, dtype=np.float32))
    p = torch.from_numpy(features[:, :, 0].copy())
    y = torch.from_numpy(targets)
    initial = fit_step.tensor_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
    losses = []
    for rows in batches:
        optimizer.zero_grad(set_to_none=True)
        logits = model(x[rows], x[rows], p[rows])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y[rows])
        need(bool(torch.isfinite(loss)), "Finite continuous occupancy BCE")
        loss.backward()
        need(all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters()),
             "Finite gradients")
        optimizer.step()
        losses.append(float(loss.detach()))
    need({int(v["step"].item()) for v in optimizer.state.values()} == {STEPS} and
         all(torch.isfinite(v).all() for v in model.parameters()), "Exact finite optimizer endpoint")
    model.eval()
    audit = dict(parameters=2089, optimizer_steps=STEPS, initial_tensor_sha256=initial,
                 final_tensor_sha256=fit_step.tensor_hash(model), loss_first=losses[0], loss_last=losses[-1])
    return center, scale, audit


def _validate_budgets(records, budgets, assignments):
    need(list(budgets) == [identity(r) for r in records], "Sealed prelabel candidate order changed")
    for row in records:
        b = budgets[identity(row)]
        need(b["inputs_sha256"] == _inputs(row) and b["group"] == row["group"] and
             b["fold"] == assignments[row["group"]] and b["cap"] == row["cap"] and
             np.array_equal(b["coordinates"], row["coordinates"]) and
             b["old_q"].shape == row["coordinates"].shape and
             b["k"] == len(free_selection(b["old_q"], row["cap"])), "Prelabel budget mismatch")
        q = row["truth"]
        if q is not None:
            need(q.shape == row["coordinates"].shape and np.isfinite(q).all() and
                 np.all((q >= 0) & (q <= 1 + 1e-12)), "Native continuous cell occupancy")


def fit_oof(records, group_ids, old_by_fold, trainer, fit_step, fit_plan, budgets):
    """Yield (fold, held rows, audit) in fold order; input rows retain original candidate order.

    Each held row carries four probability matrices and native flattened flip coordinates.
    Budgets must be read back from the caller's independently closed prelabel artifact.
    """
    assignments = _assignments(records, group_ids, old_by_fold, fit_plan)
    _validate_budgets(records, budgets, assignments)
    for fold in range(5):
        with _single_cpu_thread():
            held, audit = _fit_fold(records, budgets, assignments, fold, old_by_fold[fold], trainer, fit_step)
        yield fold, held, audit


def _fit_fold(records, budgets, assignments, fold, old, trainer, fit_step):
    selected = [r for r in records if assignments[r["group"]] != fold and
                r["truth"] is not None and len(r["old14"])]
    held = [r for r in records if assignments[r["group"]] == fold]
    need(selected, "Fold requires supervised source pairs")
    indices = defaultdict(list)
    offset = 0
    for r in selected:
        n = len(r["truth"])
        indices[r["group"]].extend(range(offset, offset+n))
        offset += n
    withheld = sorted(g for g, f in assignments.items() if f == fold)
    need(not set(indices).intersection(withheld), "Held source group entered training")
    groups = [np.asarray(indices[g]) for g in sorted(indices)]
    targets = np.concatenate([r["truth"] for r in selected]).astype(np.float32)
    batches = _schedule(groups)
    batch_hash = sha256(batches.astype("<i8", copy=False).tobytes()).hexdigest()
    need(batch_hash == old["common_batch_sha256"], "Old fold sampled pair indices differ")
    audit = dict(scope="FIT109 detector-exposed / OOF point head only", seed=SEED, fold=fold,
                 steps=STEPS, batch=BATCH, shared_pairs=offset, common_batch_sha256=batch_hash,
                 old_initial_tensor_sha256=old["initial_tensor_sha256"],
                 training_groups=sorted(indices), held_groups=withheld, held_candidates=len(held), heads={})
    predictions = []
    for r in held:
        b = budgets[identity(r)]
        predictions.append(dict(image_id=r["image_id"], rank=r["rank"], query_id=r["query_id"],
                                group=r["group"], fold=fold, coordinates=b["coordinates"], cap=b["cap"],
                                k=b["k"], old_q=b["old_q"], probabilities={}, flipped={}))
    for arm in ARMS:
        model = _initial_model(trainer, fit_step, old["initial_tensor_sha256"])
        features = np.concatenate([r["features"][arm] for r in selected])
        center, scale, stats = _fit_arm(model, features, targets, batches, trainer, groups, old, fit_step)
        del features
        stats["common_batch_sha256"] = batch_hash
        audit["heads"][arm] = stats
        for r, result in zip(held, predictions):
            q = fit_step.predict(model, r["features"][arm], center, scale)
            actions = free_selection(q, result["cap"], result["k"])
            coords = result["coordinates"]
            result["probabilities"][arm] = q
            result["flipped"][arm] = np.column_stack((coords[actions[:, 0], 0],
                                                      coords[actions[:, 1], 1])).reshape(-1)
    need(len({a["initial_tensor_sha256"] for a in audit["heads"].values()}) == 1,
         "Four identical initial compact heads")
    return predictions, audit
