"""FIT-only, source-held-out training for the frozen RF-DETR PointRend adaptation.

The caller supplies a previously fixed positive match roster and the five
source folds. No predictions on held sources, k-dependent sampling or early
stopping enter the training loop. Private schedules identify source RoIs and
matched annotations and must be stripped from public aggregate reports.
"""

from collections import defaultdict
from math import isfinite
from numbers import Integral

import torch


def _index_positives(rows, folds):
    if (not all(isinstance(group, str) and group for group in folds)
            or not all(isinstance(f, Integral) and not isinstance(f, bool) for f in folds.values())
            or set(folds.values()) != set(range(5))):
        raise ValueError("require five preassigned integer folds keyed by original source group")
    by_group = defaultdict(list)
    image_group, seen_ranks, seen_queries, seen_matches = {}, set(), set(), set()
    for row in rows:
        group = row["group"]
        if group not in folds:
            raise ValueError("positive record has no preassigned source fold")
        if any(not isinstance(row[key], Integral) or isinstance(row[key], bool)
               for key in ("image_id", "rank", "query_id", "matched_gt_id", "k")):
            raise ValueError("fixed candidate, match and budget identities must be integers")
        image_id, rank, query_id, gt_id, k = (int(row[key]) for key in
                                            ("image_id", "rank", "query_id", "matched_gt_id", "k"))
        if rank < 1 or query_id < 0 or not 0 <= k <= 8:
            raise ValueError("invalid native rank, original query id or sealed swap budget")
        if image_group.setdefault(image_id, group) != group:
            raise ValueError("the same image cannot belong to different source groups")
        if ((image_id, rank) in seen_ranks or (image_id, query_id) in seen_queries
                or (image_id, gt_id) in seen_matches):
            raise ValueError("fixed positive RoIs require unique within-image rank, query and GT match")
        seen_ranks.add((image_id, rank))
        seen_queries.add((image_id, query_id))
        seen_matches.add((image_id, gt_id))
        by_group[group].append(row)
    for group_rows in by_group.values():
        group_rows.sort(key=lambda row: (row["image_id"], row["rank"], row["query_id"]))
    return by_group


def _roi_tensors(payload, adapter):
    spatial, raw, box, truth = (torch.as_tensor(v).detach() for v in payload)
    if any(t.device.type != "cpu" for t in (spatial, raw, box, truth)):
        raise ValueError("this fit expects CPU capture tensors and native truth")
    if spatial.ndim == 3:
        spatial = spatial.unsqueeze(0)
    if (spatial.ndim != 4 or spatial.shape[:2] != (1, 256) or spatial.dtype != torch.float32
            or raw.ndim != 2 or raw.dtype != torch.float32 or raw.shape != spatial.shape[-2:]):
        raise ValueError("require one FP32 query-logit plane aligned with one 256-channel RF basis")
    if truth.ndim != 2 or truth.dtype != torch.bool or min(truth.shape) < 1:
        raise ValueError("fixed match requires a 2D native boolean instance mask")
    if box.shape != (4,) or not box.is_floating_point():
        raise ValueError("box must be a normalized floating [x0,y0,x1,y1]")
    height, width = truth.shape
    box = box.to(dtype=torch.float32)[None] * raw.new_tensor([width, height, width, height])
    coarse = adapter.coarse_roi_logits(raw[None, None], box, (height, width))
    return spatial, coarse, box, truth[None], (height, width)


def _fit_one_fold(fold, by_group, folds, load_roi, adapter, seed, steps):
    groups = sorted(g for g in by_group if folds[g] != fold)
    if not groups:
        raise ValueError(f"fold {fold} has no positive training source")
    # Match the initial tensor draw across folds without advancing caller RNGs.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        head = adapter.BinaryPointHead().train()
    parameters = list(head.parameters())
    initial = [p.detach().clone() for p in parameters]
    optimizer = torch.optim.AdamW(parameters, lr=0.002, weight_decay=0.01)
    schedule_seed, point_seed = 100_000 + 10 * seed + fold, 200_000 + 10 * seed + fold
    scheduler = torch.Generator(device="cpu").manual_seed(schedule_seed)
    points = torch.Generator(device="cpu").manual_seed(point_seed)
    losses, gradient_norms, schedule = [], [], []
    zero_budget_updates = 0
    for step in range(steps):
        group = groups[torch.randint(len(groups), (), generator=scheduler).item()]
        positive = by_group[group]
        row = positive[torch.randint(len(positive), (), generator=scheduler).item()]
        spatial, coarse, box, gt, size = _roi_tensors(load_roi(row), adapter)
        optimizer.zero_grad(set_to_none=True)
        loss = adapter.binary_point_training_loss(head, spatial, coarse, box, gt, size, points)
        loss_value = loss.detach().item()
        if not isfinite(loss_value):
            raise ValueError(f"fold {fold}, step {step + 1}: nonfinite training loss")
        loss.backward()
        if any(p.grad is None for p in parameters):
            raise ValueError("point-head parameter disconnected from BCE")
        grad_norm = torch.stack([p.grad.detach().norm() for p in parameters]).norm().item()
        if not isfinite(grad_norm):
            raise ValueError(f"fold {fold}, step {step + 1}: nonfinite gradient norm")
        optimizer.step()
        losses.append(loss_value)
        gradient_norms.append(grad_norm)
        zero_budget_updates += int(row["k"] == 0)
        schedule.append((group, int(row["image_id"]), int(row["rank"]),
                         int(row["query_id"]), int(row["matched_gt_id"])))
    with torch.no_grad():
        update = torch.stack([(p - saved).norm() for p, saved in zip(parameters, initial)]).norm().item()
    if not isfinite(update) or update <= 0:
        raise ValueError("no finite nonzero point-head update")
    head.eval()
    audit = dict(fold=fold, initialization_seed=seed, schedule_seed=schedule_seed,
                 point_seed=point_seed, updates=steps, supervised_points=196 * steps,
                 oversampled_uniform_points=588 * steps, fresh_uniform_points=49 * steps,
                 retained_uncertain_points=147 * steps, optimizer="AdamW", lr=0.002, weight_decay=0.01,
                 train_positive_sources=len(groups), train_positive_rows=sum(len(by_group[g]) for g in groups),
                 held_sources=sum(value == fold for value in folds.values()),
                 held_positive_rows=sum(len(rows) for g, rows in by_group.items() if folds[g] == fold),
                 used_sources=len({entry[0] for entry in schedule}), used_positive_rows=len(set(schedule)),
                 zero_budget_updates=zero_budget_updates, trainable_parameters=sum(p.numel() for p in parameters),
                 losses=losses, loss_first=losses[0], loss_last=losses[-1], loss_mean=sum(losses) / steps,
                 gradient_norm_min=min(gradient_norms), gradient_norm_max=max(gradient_norms),
                 parameter_update_l2=update, private_schedule=schedule)
    return head, audit


def train_oof_heads(positive_rows, fold_by_group, load_roi, *, adapter, seed=5217, steps=200, cpu_threads=1):
    """Return two dictionaries keyed by held fold: one trained head and its audit.

    Each positive row carries group, image_id, rank, query_id, matched_gt_id,
    k. load_roi(row) returns CPU (spatial, native_query_logits, clipped_xyxy01,
    matched_native_bitmask). The same held-fold head must be reused for all
    downstream comparison arms. FIT109 population identity is checked by the
    caller before using this interface.
    """
    if type(seed) is not int or seed < 0 or type(steps) is not int or steps < 1:
        raise ValueError("expect a nonnegative seed and a positive update count")
    if type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError("cpu_threads must be a positive integer")
    if torch.is_inference_mode_enabled():
        raise ValueError("invoke point-head training outside of inference mode")
    by_group = _index_positives(positive_rows, fold_by_group)
    previous_threads = torch.get_num_threads()
    heads, audits = {}, {}
    try:
        torch.set_num_threads(cpu_threads)
        with torch.enable_grad():
            for fold in range(5):
                heads[fold], audits[fold] = _fit_one_fold(
                    fold, by_group, fold_by_group, load_roi, adapter, seed, steps)
    finally:
        torch.set_num_threads(previous_threads)
    return heads, audits
