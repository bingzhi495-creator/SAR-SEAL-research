"""Fixed five source folds and three seeds for the matched RF-DETR point heads."""

from hashlib import sha256
import numpy as np

SALT = "DFINE-FIT109-trained-pair-v1|"
SEEDS = (5217, 5218, 5219)
STEPS = 200
BATCH = 64


def source_folds(groups):
    ordered = sorted(set(groups), key=lambda group: sha256((SALT + group).encode()).digest())
    if len(ordered) < 5:
        raise ValueError("At least one original source group per fold")
    return {group: index % 5 for index, group in enumerate(ordered)}


def frozen_arrays(fitted):
    arrays = {"center": fitted["center"], "scale": fitted["scale"]}
    for head, model in fitted["heads"].items():
        arrays.update({f"{head}.{name}": value.detach().numpy() for name, value in model.state_dict().items()})
    if not all(array.dtype == np.float32 and np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("Only finite FP32 parameters and source standardization")
    return arrays


def seal_bundle(path, fitted):
    arrays = frozen_arrays(fitted)
    with path.open("xb") as stream:
        np.savez(stream, **arrays)
    with np.load(path, allow_pickle=False) as verified:
        if set(verified.files) != set(arrays) or any(not np.array_equal(verified[key], value) for key, value in arrays.items()):
            raise ValueError("Frozen point-head tensors and scaler differ on readback")
    return sha256(path.read_bytes()).hexdigest()


def restore_bundle(path, digest, trainer, fit_step):
    if sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("Expected frozen S/L bundle digest")
    import torch

    heads = {"S": trainer.PointHead(14), "L": fit_step.LargeHead(14)}
    with np.load(path, allow_pickle=False) as bundle:
        scaler = [bundle[key] for key in ("center", "scale")]
        expected = {"center", "scale"}
        for head, model in heads.items():
            state = model.state_dict()
            keys = {f"{head}.{name}" for name in state}
            expected.update(keys)
            for name, original in state.items():
                val = bundle[f"{head}.{name}"]
                if val.dtype != np.float32 or tuple(val.shape) != tuple(original.shape) or not np.isfinite(val).all():
                    raise ValueError("Exact S/L checkpoint tensor dimensions and dtype")
                state[name] = torch.from_numpy(val)
            model.load_state_dict(state, strict=True)
            model.eval()
        if set(bundle.files) != expected or any(v.shape != (14,) or v.dtype != np.float32 or not np.isfinite(v).all() for v in scaler):
            raise ValueError("Exact frozen arrays and standardization dimensions")
        if np.any(scaler[1] <= 0):
            raise ValueError("Positive point feature scales")
    return heads, *scaler


def fit_fixed_schedule(records, group_ids, output, fit_step, trainer):
    """Fit once per head/seed/fold and refit on all source groups. No DEV inputs."""
    if len(set(group_ids)) != 109:
        raise ValueError("Exact 109 independent FIT prefix groups in the locked roster")
    assignments = source_folds(group_ids)
    unknown_groups = {row["group"] for row in records} - assignments.keys()
    if unknown_groups:
        raise ValueError("Source group outside the fixed FIT cohort")
    ready = [row for row in records if row["truth"] is not None and row["features"].size]
    audit = []
    for seed in SEEDS:
        for fold in (*range(5), None):
            training = [row for row in ready if fold is None or assignments[row["group"]] != fold]
            train_groups = {row["group"] for row in training}
            if fold is not None and train_groups.intersection(group for group in assignments if assignments[group] == fold):
                raise ValueError("A held source entered its point-head fit")
            fitted = fit_step.fit_heads(training, trainer, steps=STEPS, batch=BATCH,
                                        seed=seed, sample_seed=20261112 + seed - SEEDS[0])
            endpoint = "all_fit" if fold is None else f"fold_{fold}"
            path = output / f"{seed}_{endpoint}.npz"
            digest = seal_bundle(path, fitted)
            restored, center, scale = restore_bundle(path, digest, trainer, fit_step)
            sample = training[0]["features"]
            for name in ("S", "L"):
                want = fit_step.predict(fitted["heads"][name], sample, fitted["center"], fitted["scale"])
                got = fit_step.predict(restored[name], sample, center, scale)
                if not np.array_equal(want, got):
                    raise ValueError("Exact restored point probabilities")
            report = fitted["audit"]
            for stats in report["heads"].values():
                losses = stats.pop("losses")
                norms = stats.pop("gradient_norms")
                stats["loss_first"] = losses[0]
                stats["loss_last"] = losses[-1]
                stats["gradient_norm_range"] = [min(norms), max(norms)]
            audit.append(dict(seed=seed, fold=fold, endpoint=endpoint, relative_weights=path.name,
                              weights_sha256=digest, **report))
            print(f"seed {seed} {endpoint}: {STEPS} steps per head", flush=True)
    return audit
