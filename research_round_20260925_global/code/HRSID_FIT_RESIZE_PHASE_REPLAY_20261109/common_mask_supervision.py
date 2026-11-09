"""Project augmented RF-DETR targets onto the same nonempty source rows."""

import torch


ROW_ID = "_p198_row_id"
ROW_FIELDS = (ROW_ID, "boxes", "labels", "area", "iscrowd", "masks")
IMAGE_FIELDS = ("image_id", "size", "orig_size")
TARGET_FIELDS = set(ROW_FIELDS + IMAGE_FIELDS)


def _same(a, b):
    return a.dtype == b.dtype and torch.equal(a, b)


def _live_rows(target):
    if target.keys() != TARGET_FIELDS or any(
        not isinstance(value, torch.Tensor) or value.device.type != "cpu"
        for value in target.values()
    ):
        raise ValueError("Expected the fixed CPU instance target schema")
    ids = target[ROW_ID]
    masks = target["masks"]
    if ids.ndim != 1 or ids.dtype != torch.int64 or (ids < 0).any() or ids.unique().numel() != len(ids):
        raise ValueError("Expected unique nonnegative original row IDs")
    if (
        masks.ndim != 3
        or masks.dtype != torch.bool
        or min(masks.shape[1:]) < 1
        or list(masks.shape[1:]) != target["size"].tolist()
        or target["boxes"].shape != (len(ids), 4)
        or any(target[key].shape != ids.shape for key in ("labels", "area", "iscrowd"))
        or len(masks) != len(ids)
    ):
        raise ValueError("Instance fields and mask canvas must match original row IDs")
    return ids, masks.flatten(1).any(dim=1)


def common_targets(native, center, *other_arms):
    """Keep the same ordered, nonempty annotation IDs across all arm outputs.

    Each output preserves its own mask bytes and shares all other selected
    supervision fields. Original row IDs remain available to the caller for
    auditing and can be removed before handing the target to the model.
    """
    targets = (native, center, *other_arms)
    live_rows = []
    for target in targets:
        ids, nonempty = _live_rows(target)
        if any(not _same(native[key], target[key]) for key in IMAGE_FIELDS):
            raise ValueError("The paired targets describe different images or canvases")
        live_rows.append(ids[nonempty])
    shared = live_rows[0]
    for rows in live_rows[1:]:
        shared = shared[torch.isin(shared, rows)]
    projected = []
    for target in targets:
        keep = torch.isin(target[ROW_ID], shared)
        selected = {**target, **{key: target[key][keep] for key in ROW_FIELDS}}
        if projected and any(not _same(projected[0][key], selected[key])
                             for key in ROW_FIELDS if key != "masks"):
            raise ValueError("Common supervision IDs, order or nonmask fields differ")
        projected.append(selected)
    return tuple(projected)
