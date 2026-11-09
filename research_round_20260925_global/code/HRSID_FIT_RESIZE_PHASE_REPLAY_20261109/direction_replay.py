"""Replay the native and two vertical-only sampling arms on one FIT view."""

from collections import Counter, defaultdict
from contextlib import contextmanager, nullcontext
from unittest.mock import patch

import numpy as np
import torch

from common_mask_supervision import ROW_ID, common_targets
from reference_resize import center_reference, inverse_scales
from replay_fit import mask_centers, masks_count, normalized_trace, rng_state
from trace_resize import capture_transform
from vertical_phase_resize import resize_masks_vertical


MODES = ("native", "center_y", "reflected_y")


@contextmanager
def capture_direction(native_module, mode):
    if mode not in MODES:
        raise ValueError("Unknown vertical phase mode")
    resize_call = native_module.Resize.__call__
    if getattr(resize_call, "_phase_trace", False):
        raise RuntimeError("Nested geometry capture is unsupported")

    def resize(self, image, target=None):
        def resize_masks(masks, _):
            return resize_masks_vertical(masks, self.size, mode)
        with patch.object(native_module, "_apply_to_masks", resize_masks):
            return resize_call(self, image, target)

    context = nullcontext() if mode == "native" else patch.object(native_module.Resize, "__call__", resize)
    with context, capture_transform(native_module, mode="native") as trace:
        yield trace


def direction_metrics(targets, raw, steps):
    ids = targets[0][ROW_ID]
    masks = [target["masks"].numpy() for target in targets]
    areas = [masks_count(mask) for mask in masks]
    y_scale = abs(inverse_scales(steps)[0])
    centers = [mask_centers(mask, area)[:, 0] * y_scale for mask, area in zip(masks, areas, strict=True)]
    reference = center_reference(raw["masks"][ids].numpy(), steps)
    ref_areas = masks_count(reference)
    ref_ok = ref_areas > 0
    small = raw["area"][ids].numpy() < 1024
    values = {}
    for label, chosen in (("small", small), ("non_small", ~small)):
        supported = chosen & ref_ok
        ref_y = mask_centers(reference[supported], ref_areas[supported])[:, 0] * y_scale
        entry = dict(
            count=int(chosen.sum()),
            reference_count=int(supported.sum()),
            native_minus_center_y=float((centers[0][chosen] - centers[1][chosen]).sum()),
            reflected_minus_center_y=float((centers[2][chosen] - centers[1][chosen]).sum()),
        )
        for mode, mask, area, center in zip(MODES, masks, areas, centers, strict=True):
            error = center[supported] - ref_y
            overlap = masks_count(mask[supported] & reference[supported])
            union = area[supported] + ref_areas[supported] - overlap
            entry.update({
                f"{mode}_area": int(area[chosen].sum()),
                f"{mode}_reference_y": float(error.sum()),
                f"{mode}_reference_abs_y": float(np.abs(error).sum()),
                f"{mode}_reference_iou": float((overlap / union).sum()),
            })
        values[label] = entry
    return values


def replay_direction_view(native_module, transform, set_seed, image, raw, seed):
    """Measure phase offsets on one common set of nonempty source instances."""
    expected_ids = torch.arange(len(raw["labels"]), dtype=torch.int64)
    if not torch.equal(raw[ROW_ID], expected_ids):
        raise ValueError("Expected original row IDs in annotation order")
    baseline = None
    targets = []
    membership = {"raw_rows": len(expected_ids)}
    native_rows = set()
    for mode in MODES:
        set_seed(seed)
        target_input = {key: value.clone() for key, value in raw.items()}
        with capture_direction(native_module, mode) as trace:
            transformed_image, target = transform(image.copy(), target_input)
        if (
            not isinstance(transformed_image, torch.Tensor)
            or transformed_image.dtype != torch.float32
            or transformed_image.ndim != 3
            or transformed_image.shape[0] != 3
            or list(transformed_image.shape[-2:]) != target["size"].tolist()
            or not torch.equal(target["image_id"], raw["image_id"])
            or not torch.equal(target["orig_size"], raw["orig_size"])
        ):
            raise ValueError("Expected normalized image and original image identity")
        state = rng_state()
        if baseline is None:
            baseline = transformed_image, trace, state
        elif not torch.equal(transformed_image, baseline[0]) or trace != baseline[1] or state != baseline[2]:
            raise ValueError("Image, geometry and random stream must match across arms")
        ids = target[ROW_ID]
        if (ids < 0).any() or (ids >= len(expected_ids)).any():
            raise ValueError("Augmentation produced an unknown original row ID")
        live = set(ids[target["masks"].flatten(1).any(dim=1)].tolist())
        if mode == "native":
            native_rows = live
        membership.update({
            f"{mode}_present": len(ids),
            f"{mode}_nonempty": len(live),
            f"{mode}_empty": len(ids) - len(live),
            f"{mode}_native_lost": len(native_rows - live),
            f"{mode}_added": len(live - native_rows),
        })
        targets.append(target)
    shared = common_targets(*targets)
    membership["common_nonempty"] = len(shared[0][ROW_ID])
    steps = normalized_trace(baseline[1])
    path = "crop" if any(step["kind"] == "crop" for step in steps) else "direct"
    return dict(
        membership=membership, path=path, flip=steps[-1]["kind"] == "hflip",
        steps=steps, **direction_metrics(shared, raw, steps)
    )

def _group_area(views, area_key):
    grouped = defaultdict(Counter)
    for view in views:
        grouped[view["group"]].update(view[area_key])
    groups = [values for _, values in sorted(grouped.items()) if values["count"]]
    reference_groups = [values for values in groups if values["reference_count"]]
    diffs = ("native_minus_center_y", "reflected_minus_center_y")
    paired = np.asarray([[row[k] / row["count"] for k in diffs] for row in groups]).reshape(-1, len(diffs))
    result = dict(
        groups=len(groups), rows=sum(row["count"] for row in groups),
        reference_groups=len(reference_groups), reference_rows=sum(row["reference_count"] for row in groups),
    )
    if groups:
        result.update(zip(diffs, paired.mean(axis=0).tolist(), strict=True))
        for mode in MODES:
            name = f"{mode}_area"
            result[name] = float(np.mean([row[name] / row["count"] for row in groups]))
    if reference_groups:
        for mode in MODES:
            for suffix in ("y", "abs_y", "iou"):
                name = f"{mode}_reference_{suffix}"
                result[name] = float(np.mean([row[name] / row["reference_count"] for row in reference_groups]))
    return result, paired


def summarize_direction(views, *, primary):
    """Average within each content group, then weight content groups equally."""
    membership = Counter()
    for view in views:
        membership.update(view["membership"])
    small, paired = _group_area(views, "small")
    result = dict(
        views=len(views), groups=len({view["group"] for view in views}), membership=dict(membership),
        small=small, non_small=_group_area(views, "non_small")[0], by_path={},
    )
    for path in ("direct", "crop"):
        for flip in (False, True):
            subset = [view for view in views if view["path"] == path and view["flip"] is flip]
            result["by_path"][f"{path}_{'flipped' if flip else 'plain'}"] = dict(
                views=len(subset), small=_group_area(subset, "small")[0],
                non_small=_group_area(subset, "non_small")[0],
            )
    if primary:
        result["decision"] = "INSUFFICIENT_COMMON_SMALL_GROUPS"
        if len(paired) >= 30:
            groups = np.random.default_rng(5219).integers(0, len(paired), size=(4096, len(paired)))
            distribution = paired[groups].mean(axis=1)
            native, reflected = np.quantile(distribution, [.025, .975], axis=0, method="linear").T.tolist()
            supports = (
                small["native_minus_center_y"] >= 0.25 and small["reflected_minus_center_y"] <= -0.25
                and native[0] > 0.0 and reflected[1] < 0.0
            )
            result.update(
                native_minus_center_y_interval=native, reflected_minus_center_y_interval=reflected,
                decision="SUPPORTS_DIRECTIONAL_LABEL_EFFECT" if supports else "DIRECTIONAL_LABEL_GATE_NOT_MET",
            )
    return result
