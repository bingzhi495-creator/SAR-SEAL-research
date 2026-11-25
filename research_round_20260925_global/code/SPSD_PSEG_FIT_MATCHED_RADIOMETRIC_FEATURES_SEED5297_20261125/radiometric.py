"""Deterministic per-candidate radiometric context for the original 28-input head."""
import numpy as np
import torch


MODES = ("regression", "joint", "per_side")


def input_array(features):
    x = np.asarray(features, dtype=np.float32)
    if (x.ndim != 3 or x.shape[1:] != (2, 14) or not np.isfinite(x).all()
            or not np.all((x[..., 0] >= 0) & (x[..., 0] <= 1))):
        raise ValueError("Original (endpoints, signed side, 14) finite input features")
    return x


def midrank(values):
    path = np.asarray(values, dtype=np.float64)
    if not len(path):
        return path.copy()
    _, inverse, counts = np.unique(path, return_inverse=True, return_counts=True)
    lower = np.cumsum(counts) - counts
    return 2 * ((lower + .5 * counts) / len(path))[inverse] - 1


def robust_z(values):
    path = np.asarray(values, dtype=np.float64)
    if not len(path):
        return path.copy()
    low, center, high = np.quantile(path, (.25, .5, .75), method="linear")
    scale = high - low
    if scale == 0:
        scale = np.ptp(path)
        if scale == 0:
            return np.zeros_like(path)
    return np.clip((path - center) / scale, -4, 4)


def cue_columns(table):
    gray = table[..., 9].astype(np.float64)
    mean3 = table[..., 10].astype(np.float64)
    residual = gray - table[..., 11]
    return np.stack((midrank(gray), midrank(mean3), midrank(residual),
                     robust_z(gray), robust_z(residual)), axis=-1)


def transform_candidate(features, arm):
    """Retain native order and replace only five redundant appearance inputs."""
    x = input_array(features)
    n = len(x)
    if arm == "joint":
        cues = cue_columns(x.reshape(-1, 14)).reshape(n, 2, 5)
    elif arm == "per_side":
        cues = np.stack([cue_columns(x[:, side, :]) for side in (0, 1)], axis=1)
    elif arm == "regression":
        return np.concatenate((x, x), axis=-1)
    else:
        raise ValueError("Unknown radiometric arm")
    return np.concatenate((x, x[..., :9], cues.astype(np.float32)), axis=-1)


def forward_cues(model, standardized, native_probabilities):
    """Same network and frozen native probability residual as PointHead.forward."""
    baseline = native_probabilities.clamp(1e-4, 1 - 1e-4).logit().unsqueeze(-1)
    return baseline + model.network(standardized)


def predict(model, features, center, scale, arm):
    x = input_array(features)
    center = np.asarray(center, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    width = 14 if arm == "regression" else 28
    if (center.shape != (width,) or scale.shape != (width,) or not np.isfinite(center).all()
            or not np.isfinite(scale).all() or not np.all(scale > 0)):
        raise ValueError("Exact finite arm-specific FIT scaling")
    with torch.inference_mode():
        native = torch.from_numpy(x[..., 0].copy())
        features = x if arm == "regression" else transform_candidate(x, arm)
        z = torch.from_numpy(np.ascontiguousarray((features - center) / scale, dtype=np.float32))
        logits = model(z, native) if arm == "regression" else forward_cues(model, z, native)
        return logits.sigmoid().numpy()
