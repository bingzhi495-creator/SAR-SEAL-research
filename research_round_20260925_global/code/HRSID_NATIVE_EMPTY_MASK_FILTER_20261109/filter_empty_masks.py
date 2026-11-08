"""Drop only empty instance rows after the native FIT augmentation."""
from torch import bool as torch_bool


def drop_empty_augmented_masks(target):
    masks = target["masks"]
    if masks.ndim != 3 or masks.dtype != torch_bool:
        raise ValueError("Expected native N×H×W boolean masks")
    count = masks.shape[0]
    fields = tuple(key for key in ("boxes", "labels", "area", "iscrowd", "keypoints", "masks") if key in target)
    if "boxes" not in target or "labels" not in target or any(
        target[key].ndim == 0 or target[key].shape[0] != count for key in fields
    ):
        raise ValueError("Instance field rows must align with masks")
    keep = masks.flatten(1).any(1)
    removed = count-int(keep.sum())
    if removed:
        retained = {key: target[key][keep] for key in fields}
        target.update(retained)
    return removed
