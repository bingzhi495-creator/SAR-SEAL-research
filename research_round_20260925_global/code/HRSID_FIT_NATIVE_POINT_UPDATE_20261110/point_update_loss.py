"""Freeze P252 point banks and evaluate their equal-row update objective."""

from functools import lru_cache
from hashlib import sha256
from importlib.machinery import PathFinder
from math import isfinite
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

import numpy as np
import torch
from torch.nn import functional as F


P252_POLICY_SHA256 = "de4012b69e4f3027ba2af7cf755050b2f23e30c892aeed6467375caa9b70ddff"
ARMS = ("phase", "uncertainty", "random")


class FrozenPointBank(NamedTuple):
    indices: np.ndarray
    targets: np.ndarray


@lru_cache(maxsize=1)
def _policy():
    spec = PathFinder.find_spec("point_bank_analysis")
    if spec is None or spec.origin is None:
        raise ValueError("Place the exact P252 point-bank source on the Python path")
    path = Path(spec.origin).resolve(strict=True)
    if path.name != "point_bank_analysis.py" or not 0 < path.stat().st_size <= 64 * 1024:
        raise ValueError("Expected the bounded original P252 point-bank source")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != P252_POLICY_SHA256:
        raise ValueError("Original P252 point-bank source digest differs")
    module = ModuleType("_p253_pinned_p252_bank")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def freeze_row_banks(hard_c, hard_b, raster, reference_logits, strata, seed):
    """Choose each arm's points once from the same frozen pre-update bank.

    Input positions define a common row's candidate order. Raster occupancy is
    gathered only after selection. Both outputs are backed by immutable bytes.
    """
    policy = _policy()
    logits = policy._vector(reference_logits, "reference_logits").astype(np.float64)
    if not np.isfinite(logits).all():
        raise ValueError("Frozen pre-update reference logits must be finite")
    strata = policy._vector(strata, "strata", len(logits))
    if strata.dtype.kind not in "iu" or not np.isin(strata, range(4)).all():
        raise ValueError("Expected integer strata 0 through 3")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("Expected nonnegative row seed")
    c = policy._hard(hard_c, "hard_c", len(logits))
    b = policy._hard(hard_b, "hard_b", len(logits))
    selected, _, quotas, _, _ = policy._banks(strata, b != c, logits, int(seed))
    target = policy._occupancy(raster, "raster", len(logits))
    if tuple(selected) != ARMS:
        raise ValueError("Exact P252 three-arm selector required")

    banks = {}
    for arm in ARMS:
        indices = selected[arm]
        if (indices.dtype != np.int64 or not np.all(np.diff(indices) > 0)
                or np.bincount(strata[indices].astype(np.int64), minlength=4).tolist() != quotas):
            raise ValueError("Exact ordered unique points and common quotas required")
        banks[arm] = FrozenPointBank(np.frombuffer(indices.tobytes(), dtype=np.int64),
                                    np.frombuffer(target[indices].tobytes(), dtype=np.float64))
    return banks


def paired_point_loss(rows, *, bce_weight=1., dice_weight=1.):
    """Return a differentiable scalar with equal weights for supplied rows.

    Each row is (current selected logits, fixed occupancy). The caller groups
    rows by source image when enforcing image-equal update weighting.
    """
    weights = (bce_weight, dice_weight)
    if not all(type(w) in (int, float) and isfinite(w) and w >= 0 for w in weights) or not any(weights):
        raise ValueError("Finite nonnegative BCE and Dice weights required")
    losses = []
    backend = None
    for logits, occupancy in rows:
        if (not isinstance(logits, torch.Tensor) or not logits.is_floating_point() or logits.ndim != 1
                or not logits.numel() or not torch.isfinite(logits).all()):
            raise ValueError("Expected finite, nonempty selected logit vector")
        current_backend = logits.dtype, logits.device
        if backend is None:
            backend = current_backend
        if backend != current_backend:
            raise ValueError("Every row must use the same logit dtype and device")
        if isinstance(occupancy, torch.Tensor):
            if occupancy.requires_grad:
                raise ValueError("Raster targets must be frozen")
            targets = occupancy.to(dtype=logits.dtype, device=logits.device)
        else:
            targets = torch.as_tensor(np.array(occupancy, copy=True), dtype=logits.dtype, device=logits.device)
        if targets.shape != logits.shape or not torch.isfinite(targets).all() or ((targets < 0) | (targets > 1)).any():
            raise ValueError("Aligned fixed raster occupancies in [0, 1] required")

        loss = bce_weight * F.binary_cross_entropy_with_logits(logits, targets)
        if dice_weight:
            probability = logits.sigmoid()
            dice = 1. - (2. * (probability * targets).sum() + 1.) / (probability.sum() + targets.sum() + 1.)
            loss = loss + dice_weight * dice
        losses.append(loss)
    if not losses:
        raise ValueError("At least one fixed point row required")
    return torch.stack(losses).mean()
