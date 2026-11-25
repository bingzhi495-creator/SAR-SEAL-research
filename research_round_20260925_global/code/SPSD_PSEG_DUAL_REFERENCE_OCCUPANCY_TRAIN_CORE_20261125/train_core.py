"""Four matched CPU fits on already admitted dual-reference training endpoints."""
from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
import operator
from time import perf_counter

import numpy as np
import torch
from torch import nn


FEATURES = 14
SEEDS = (5297, 5298, 5299)
STEPS = 1_000
BATCH = 64
MODES = ("dual", "pseg_repeat", "spsd_repeat", "mean_repeat")


class PointHead(nn.Module):
    """Two native-cell occupancies using duplicate standardized features."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(28, 24), nn.ReLU(),
                                     nn.Linear(24, 24), nn.ReLU(), nn.Linear(24, 2))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, standardized, probabilities):
        duplicated = torch.cat((standardized, standardized), dim=-1)
        baseline = probabilities.clamp(1e-4, 1 - 1e-4).logit().unsqueeze(-1)
        return baseline + self.network(duplicated)


def _feature_array(features):
    raw = np.asarray(features)
    if raw.ndim != 3 or raw.shape[1:] != (2, FEATURES) or raw.dtype.kind not in "biuf":
        raise ValueError("Expected a real feature array shaped (pairs, 2, 14)")
    table = np.asarray(raw, dtype=np.float32)
    probability = table[..., 0]
    if not np.isfinite(table).all() or not np.all((probability >= 0.0) & (probability <= 1.0)):
        raise ValueError("Features must be finite and original probabilities in [0, 1]")
    return table


def _keys(values, size, name):
    raw = np.asarray(values, dtype=object)
    if raw.shape != (size,):
        raise ValueError(f"Expected one {name} identity per pair")
    if all(isinstance(v, str) and v for v in raw):
        return raw.tolist()
    if all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
           and 0 <= v for v in raw):
        return [int(v) for v in raw]
    raise ValueError(f"Use uniform strings or nonnegative integer {name} identities")


def _eligible_rows(group_ids, image_ids, size):
    group_keys = _keys(group_ids, size, "group")
    image_keys = _keys(image_ids, size, "image")
    rows = defaultdict(lambda: defaultdict(list))
    image_group = {}
    for index, (group, image) in enumerate(zip(group_keys, image_keys, strict=True)):
        if image_group.setdefault(image, group) != group:
            raise ValueError("An eligible image can belong to only one exact RGB group")
        rows[group][image].append(index)
    return tuple(tuple(np.asarray(rows[g][i], dtype=np.int64) for i in sorted(rows[g]))
                 for g in sorted(rows))


def _fit_moments(features, row_groups):
    """Group- and image-balanced moments across FIT pairs and their two endpoints."""
    first = features.mean(axis=1, dtype=np.float64)
    second = np.square(features, dtype=np.float64).mean(axis=1)
    moments = []
    for values in (first, second):
        group = [np.mean([values[rows].mean(axis=0) for rows in images], axis=0)
                 for images in row_groups]
        moments.append(np.mean(group, axis=0))
    center, second_moment = moments
    scale = np.sqrt(np.maximum(0.0, second_moment - center ** 2))
    scale[scale < 1e-12] = 1.0
    return center.astype(np.float32), scale.astype(np.float32)


def _batches(row_groups, seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    groups = rng.integers(len(row_groups), size=(STEPS, BATCH))
    batches = np.empty_like(groups, dtype=np.int64)
    for pos in np.ndindex(groups.shape):
        images = row_groups[groups[pos]]
        rows = images[rng.integers(len(images))]
        batches[pos] = rows[rng.integers(len(rows))]
    return batches


def _parameter_digest(model):
    return sha256(b"".join(p.detach().numpy().astype("<f4", copy=False).tobytes()
                           for p in model.parameters())).hexdigest()


def _inputs(features, center, scale):
    standardized = np.ascontiguousarray((features - center) / scale, dtype=np.float32)
    return torch.from_numpy(standardized), torch.from_numpy(features[..., 0].copy())


def train_matched(features, targets, group_ids, image_ids, *, seed=5297):
    """Fit four heads on admitted FIT-only pairs and return models, center, scale, audit.

    Features have shape (N,2,14), with original decoder probabilities in
    feature column zero. The two endpoints are the caller's original paired
    order. Targets have shape (N,2,2); the last axis is binary native author
    raster occupancy in PSeg then SPSD order. Supply one FIT exact decoded
    identity group and image key per pair. Each key array is uniformly
    nonnegative integer or string. Caller verifies source admission and keeps
    gate and historical test pairs outside this API.

    Statistics, NumPy draws, initial Torch tensors and the 1,000-update
    AdamW schedule are shared across all supervision arms. This CPU routine
    performs no model selection and returns final models in evaluation mode.
    """
    if isinstance(seed, (bool, np.bool_)) or operator.index(seed) not in SEEDS:
        raise ValueError("Use a prelocked seed: 5297, 5298 or 5299")
    seed = int(seed)
    own = _feature_array(features)
    q = np.asarray(targets)
    if (len(own) == 0 or q.shape != (len(own), 2, 2) or q.dtype.kind not in "biuf"
            or not np.all((q == 0) | (q == 1))):
        raise ValueError("At least one FIT pair with two exact binary endpoints per reference is required")
    row_groups = _eligible_rows(group_ids, image_ids, len(own))
    center, scale = _fit_moments(own, row_groups)
    own_t, probabilities = _inputs(own, center, scale)
    y = torch.from_numpy(np.ascontiguousarray(q, dtype=np.float32))
    modes = {
        "dual": y,
        "pseg_repeat": y[..., :1].expand_as(y),
        "spsd_repeat": y[..., 1:2].expand_as(y),
        "mean_repeat": y.mean(dim=-1, keepdim=True).expand_as(y),
    }
    draw = _batches(row_groups, seed)
    batch_sha = sha256(draw.astype("<i8", copy=False).tobytes()).hexdigest()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        initial = PointHead().train()
    initial_sha = _parameter_digest(initial)
    models, audits = {}, {}
    for name in MODES:
        started = perf_counter()
        model = deepcopy(initial)
        arm_sha = _parameter_digest(model)
        parameters = list(model.parameters())
        parameter_count = sum(t.numel() for t in parameters)
        if arm_sha != initial_sha or parameter_count != 1_346:
            raise RuntimeError("All heads require the exact same 1,346-parameter initialization")
        optimizer = torch.optim.AdamW(parameters, lr=.002, weight_decay=.01)
        targets_t = modes[name]
        loss_fn = nn.BCEWithLogitsLoss()
        for indices in draw:
            optimizer.zero_grad(set_to_none=True)
            logits = model(own_t[indices], probabilities[indices])
            loss = loss_fn(logits, targets_t[indices])
            if not bool(torch.isfinite(loss)):
                raise ValueError(f"Non-finite loss in {name}")
            loss.backward()
            optimizer.step()
        steps = {int(state["step"].item()) for state in optimizer.state.values()}
        if len(optimizer.state) != len(parameters) or steps != {STEPS}:
            raise RuntimeError("Observed optimizer steps do not match the fixed schedule")
        audits[name] = dict(parameters=parameter_count, optimizer_updates=STEPS,
                            observed_pair_examples=STEPS * BATCH,
                            observed_point_examples=2 * STEPS * BATCH,
                            initial_tensor_sha256=arm_sha, final_tensor_sha256=_parameter_digest(model),
                            batch_sha256=batch_sha, last_batch_loss=float(loss.detach()),
                            elapsed_seconds=perf_counter() - started)
        models[name] = model.eval()
    return models, center, scale, dict(seed=seed, batch_size=BATCH, update_count=STEPS,
                                       fit_groups=len(row_groups), fit_images=sum(map(len, row_groups)),
                                       fit_pairs=len(own), batch_sha256=batch_sha,
                                       initial_tensor_sha256=initial_sha, heads=audits)


def predict(model, features, center, scale):
    """Return N x 2 x 2 probabilities in the same endpoint and reference order."""
    own = _feature_array(features)
    center = np.asarray(center, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    if (center.shape != (FEATURES,) or scale.shape != (FEATURES,)
            or not np.isfinite(center).all() or not np.isfinite(scale).all() or not np.all(scale > 0)):
        raise ValueError("Provide the 14-element fitted center and positive fitted scale")
    with torch.inference_mode():
        return model(*_inputs(own, center, scale)).sigmoid().numpy()

