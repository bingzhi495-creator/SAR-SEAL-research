"""Matched FIT-only training using signed gains of frozen-budget native actions."""
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import operator
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F


SOURCE_SHA = "0f1537f12ed4890bb60f71957e5d81926aeaab4760c8b01291383e4b5c53ee0d"
SEED = 5297
STEPS = 1_000
BATCH = 64
BETA = 0.1
ACTION_WEIGHT = 1.0
MODES = ("bce", "bce_signed_action")


@dataclass(frozen=True)
class CandidatePacket:
    """One FIT candidate with label-blind, static-relative actions."""

    features: np.ndarray
    targets: np.ndarray
    action_weights: np.ndarray
    k: int
    group_id: str | int
    image_id: str | int


def signed_action_deltas(probabilities, weights, k):
    """Return signed intersection changes divided by 2k, in reference order.

    Probabilities end in (points, side, reference), weights in
    (actions, points, side). Weights are action-mask minus static-mask at
    each native endpoint, already signed for additions and deletions.
    """
    quota = torch.as_tensor(k, dtype=probabilities.dtype, device=probabilities.device)
    gains = weights.flatten(-2) @ probabilities.flatten(-3, -2)
    return gains / (2 * quota[..., None, None])


def _source(path):
    path = Path(path)
    data = path.read_bytes()
    if sha256(data).hexdigest() != SOURCE_SHA:
        raise ValueError("Expected the exact prior occupancy training source")
    core = {"__name__": "_verified_point_head", "__file__": str(path)}
    exec(compile(data, str(path), "exec"), core)
    if core["STEPS"] != STEPS or core["BATCH"] != BATCH:
        raise ValueError("Budget must equal the prior full training schedule")
    return core


def _packet_arrays(packet, feature_array):
    features = feature_array(packet.features)
    n = len(features)
    if isinstance(packet.k, (bool, np.bool_)):
        raise ValueError("Candidate quota must be an integer")
    quota = operator.index(packet.k)
    if not 0 < quota <= n:
        raise ValueError("Every FIT packet needs positive feasible original quota")
    targets = np.asarray(packet.targets)
    if (targets.shape != (n, 2, 2) or targets.dtype.kind not in "biuf" or
            not np.isin(targets, (0, 1)).all()):
        raise ValueError("Expected binary native occupancy with axes points, side, reference")
    weights = np.asarray(packet.action_weights)
    if (weights.ndim != 3 or weights.shape[1:] != (n, 2) or len(weights) == 0 or
            weights.dtype.kind not in "biuf" or not np.isin(weights, (-1, 0, 1)).all()):
        raise ValueError("Expected at least one signed static-relative action on the native endpoints")
    positive = weights > 0
    negative = weights < 0
    if np.any(positive.sum(axis=1) != negative.sum(axis=1)):
        raise ValueError("Every action must keep k additions and k deletions")
    used_positive, used_negative = positive.any(axis=0), negative.any(axis=0)
    if np.any(used_positive & used_negative):
        raise ValueError("All actions must use one common static incumbent")
    incumbent = (used_negative[:, 0].sum(), used_positive[:, 1].sum())
    outside = (used_positive[:, 0].sum(), used_negative[:, 1].sum())
    if max(incumbent) > quota or max(outside) > n - quota:
        raise ValueError("Action weights are infeasible for one static mask at the original quota")
    return features, targets.astype(np.float32), weights.astype(np.float32), quota


def _fit_moments(packets, groups):
    means = np.stack([f.mean(axis=(0, 1), dtype=np.float64) for f, _, _, _ in packets])
    squares = np.stack([np.square(f, dtype=np.float64).mean(axis=(0, 1)) for f, _, _, _ in packets])

    def balanced(values):
        image_means = [[values[indices].mean(axis=0) for indices in group] for group in groups]
        return np.mean([np.mean(members, axis=0) for members in image_means], axis=0)

    center = balanced(means)
    scale = np.sqrt(np.maximum(0.0, balanced(squares) - center ** 2))
    scale[scale < 1e-12] = 1.0
    return center.astype(np.float32), scale.astype(np.float32)


def _dense(packets):
    points = np.asarray([len(p[0]) for p in packets])
    actions = np.asarray([len(p[2]) for p in packets])
    p, a = int(points.max()), int(actions.max())
    feature_data = np.zeros((len(packets), p, 2, packets[0][0].shape[-1]), dtype=np.float32)
    target_data = np.zeros((len(packets), p, 2, 2), dtype=np.float32)
    weight_data = np.zeros((len(packets), a, p, 2), dtype=np.float32)
    quotas = np.empty(len(packets), dtype=np.float32)
    for i, (features, targets, weights, quota) in enumerate(packets):
        n, m = points[i], actions[i]
        feature_data[i, :n] = features
        target_data[i, :n] = targets
        weight_data[i, :m, :n] = weights
        quotas[i] = quota
    point_mask = np.arange(p)[None, :] < points[:, None]
    action_mask = np.arange(a)[None, :] < actions[:, None]
    return feature_data, tuple(map(torch.from_numpy, (target_data, weight_data, quotas, point_mask, action_mask, points, actions)))


def train_matched_action(packets, *, source_core, seed=SEED):
    """Run the fixed paired FIT-only loss comparison and return models, scaling, audit.

    Each CandidatePacket uses the original supplied endpoint order and positive
    quota. Features are (points,2,14) with native decoder probability in column
    zero. Targets are (points,2,2) in PSeg/SPSD order. Action weights are
    (actions,points,2), with add-side +(I_action-I_static) and drop-side
    -(I_action-I_static). Caller verifies source admission and label-blind
    action generation, then supplies FIT packs only. There is no reader for
    labeled inputs, model selection, or evaluation here.
    """
    if isinstance(seed, (bool, np.bool_)) or operator.index(seed) != SEED:
        raise ValueError("This comparison uses prelocked seed 5297")
    packets = tuple(packets)
    if not packets:
        raise ValueError("Provide nonempty FIT-only candidate packets")
    core = _source(source_core)
    own = [_packet_arrays(packet, core["_feature_array"]) for packet in packets]
    groups = core["_eligible_rows"]([p.group_id for p in packets], [p.image_id for p in packets], len(packets))
    center, scale = _fit_moments(own, groups)
    data, (targets, weights, quotas, valid_points, valid_actions, points, actions) = _dense(own)
    standardized, native_probabilities = core["_inputs"](data, center, scale)
    action_truth = signed_action_deltas(targets, weights, quotas)
    draws = core["_batches"](groups, SEED)
    draws_hash = sha256(draws.astype("<i8", copy=False).tobytes()).hexdigest()
    batches = torch.from_numpy(draws)
    endpoint_count = int(2 * points[batches].sum().item())
    action_count = int(actions[batches].sum().item())
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(SEED)
        initial = core["PointHead"]().train()
    parameters_hash = core["_parameter_digest"]
    initial_hash = parameters_hash(initial)

    models, audits = {}, {}
    for mode in MODES:
        started = perf_counter()
        model = deepcopy(initial)
        params = list(model.parameters())
        count = sum(p.numel() for p in params)
        if count != 1_346 or parameters_hash(model) != initial_hash:
            raise RuntimeError("Both heads require the same original initialized 1,346 parameters")
        optimizer = torch.optim.AdamW(params, lr=.002, weight_decay=.01)
        for idx in batches:
            optimizer.zero_grad(set_to_none=True)
            logits = model(standardized[idx], native_probabilities[idx])
            pixel_bce = F.binary_cross_entropy_with_logits(logits, targets[idx], reduction="none").mean(dim=(-1, -2))
            bce = ((pixel_bce * valid_points[idx]).sum(dim=-1) / points[idx]).mean()
            scored_logits = logits if mode == "bce_signed_action" else logits.detach()
            estimated = signed_action_deltas(scored_logits.sigmoid(), weights[idx], quotas[idx])
            gain_error = F.smooth_l1_loss(estimated, action_truth[idx], beta=BETA, reduction="none").mean(dim=-1)
            gain = ((gain_error * valid_actions[idx]).sum(dim=-1) / actions[idx]).mean()
            loss = bce + ACTION_WEIGHT * gain if mode == "bce_signed_action" else bce
            if not bool(torch.isfinite(loss) & torch.isfinite(gain)):
                raise ValueError("Non-finite training objective in " + mode)
            loss.backward()
            optimizer.step()
        updates = {int(state["step"].item()) for state in optimizer.state.values()}
        if len(optimizer.state) != len(params) or updates != {STEPS}:
            raise RuntimeError("Observed steps differ from the matched training schedule")
        models[mode] = model.eval()
        audits[mode] = dict(parameters=count, optimizer_updates=STEPS, observed_candidates=int(batches.numel()),
                            observed_endpoints=endpoint_count, observed_actions=action_count,
                            initial_tensor_sha256=initial_hash, final_tensor_sha256=parameters_hash(model),
                            batch_sha256=draws_hash, last_total_loss=float(loss.detach()), last_bce_loss=float(bce.detach()),
                            last_signed_action_loss=float(gain.detach()), elapsed_seconds=perf_counter() - started)

    return models, center, scale, dict(seed=SEED, batch_size=BATCH, update_count=STEPS,
                                       source_core_sha256=SOURCE_SHA, initial_tensor_sha256=initial_hash, batch_sha256=draws_hash,
                                       fit_groups=len(groups), fit_images=sum(map(len, groups)), fit_candidates=len(packets),
                                       fit_endpoints=int(2 * points.sum()), fit_actions=int(actions.sum()),
                                       smooth_l1_beta=BETA, signed_action_weight=ACTION_WEIGHT, heads=audits)
