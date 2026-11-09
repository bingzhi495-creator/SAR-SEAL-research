"""CPU Boolean-mask resizing with three vertical phases and one fixed horizontal phase."""

import torch
from torch.nn import functional as F
from torchvision import tv_tensors


MODES = ("native", "center_y", "reflected_y")


def resize_masks_vertical(masks, size, mode):
    """Sample [N,H,W] masks while holding horizontal sampling at legacy nearest.

    The vertical reflection control flips the mask before legacy resizing and
    flips the resized result back. This operation changes sampling phase
    without reflecting the output geometry or the surrounding augmentation.
    """
    if mode not in MODES:
        raise ValueError("Expected native, center_y or reflected_y")
    if (
        not isinstance(masks, torch.Tensor) or masks.ndim != 3
        or masks.dtype != torch.bool or masks.device.type != "cpu"
        or min(masks.shape[-2:]) <= 0
    ):
        raise ValueError("Expected CPU Boolean instance masks shaped N,H,W")
    if not isinstance(size, (tuple, list)) or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
        raise ValueError("Expected positive integer output height and width")

    out_h, out_w = size
    plain = masks.as_subclass(torch.Tensor)
    old_h, old_w = plain.shape[-2:]
    if (out_h, out_w) == (old_h, old_w):
        result = plain
    elif plain.shape[0] == 0:
        result = plain.reshape(0, out_h, out_w)
    else:
        sampled = plain.unsqueeze(0).float()
        if out_h != old_h:
            reflected = mode == "reflected_y"
            if reflected:
                sampled = sampled.flip((-2,))
            y_mode = "nearest-exact" if mode == "center_y" else "nearest"
            sampled = F.interpolate(sampled, size=(out_h, old_w), mode=y_mode)
            if reflected:
                sampled = sampled.flip((-2,))
        if out_w != old_w:
            sampled = F.interpolate(sampled, size=(out_h, out_w), mode="nearest")
        result = sampled.squeeze(0).bool()
    return tv_tensors.wrap(result, like=masks) if isinstance(masks, tv_tensors.Mask) else result
