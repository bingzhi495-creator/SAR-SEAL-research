"""Same-width use of frozen mask-branch input cues at admitted native endpoints."""
from hashlib import sha256
import operator

import numpy as np
import torch


CUE_WIDTH = 6
SALT = "SPSD-SPATIAL-CUE-SCRAMBLE-v1|5297"
MODES = ("regression", "spatial", "scrambled")


def input_array(features):
    raw = np.asarray(features, dtype=np.float32)
    if (raw.ndim != 3 or raw.shape[1:] != (2, 14) or not np.isfinite(raw).all()
            or not np.all((raw[..., 0] >= 0) & (raw[..., 0] <= 1))):
        raise ValueError("Original (endpoints, signed side, 14) finite input features")
    return raw


def scrambled_indices(count, ordinal, rank, side):
    ordinal, rank, side = map(operator.index,(ordinal, rank, side))
    if ordinal <= 0 or not 1 <= rank <= 100 or side not in (0, 1) or count < 0:
        raise ValueError("Original positive archive ordinal, one-based native rank and sign")
    salt = f"{SALT}|{ordinal}|{rank}|{side}".encode("ascii")
    entropy = int.from_bytes(sha256(salt).digest()[:16], "big")
    order = np.random.Generator(np.random.PCG64(entropy)).permutation(count)
    assign = np.empty_like(order)
    assign[order] = np.roll(order, 1)
    return assign


def transform_candidate(features, projected, arm, ordinal, rank):
    """Keep all native features and place the original or same-sign scrambled cue."""
    raw = input_array(features)
    if arm == "regression":
        return np.concatenate((raw, raw), axis=-1)
    cue = np.asarray(projected, dtype=np.float32)
    if cue.shape != (len(raw), 2, CUE_WIDTH) or not np.isfinite(cue).all():
        raise ValueError("Exact original per-endpoint frozen cue shape")
    if arm == "scrambled":
        cue = np.stack([cue[scrambled_indices(len(cue), ordinal, rank, side), side, :]
                        for side in (0,1)], axis=1)
    elif arm != "spatial":
        raise ValueError("Unknown cue arm")
    return np.concatenate((raw, raw[..., :14-CUE_WIDTH], cue), axis=-1)


def forward_cues(model, standardized, native_probabilities):
    baseline = native_probabilities.clamp(1e-4, 1 - 1e-4).logit().unsqueeze(-1)
    return baseline + model.network(standardized)


def predict(model, raw, projected, center, scale, arm, ordinal, rank):
    features = input_array(raw)
    center = np.asarray(center, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    width = 14 if arm == "regression" else 28
    if (center.shape != (width,) or scale.shape != (width,) or not np.isfinite(center).all()
            or not np.isfinite(scale).all() or not np.all(scale > 0)):
        raise ValueError("Exact finite arm-specific FIT scaling")
    with torch.inference_mode():
        native = torch.from_numpy(features[..., 0].copy())
        values = features if arm == "regression" else transform_candidate(features, projected, arm, ordinal, rank)
        z = torch.from_numpy(np.ascontiguousarray((values - center) / scale, dtype=np.float32))
        logits = model(z, native) if arm == "regression" else forward_cues(model, z, native)
        return logits.sigmoid().numpy()
