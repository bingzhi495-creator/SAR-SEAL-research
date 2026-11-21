"""Binary inference on a fixed spatial extent using a caller-supplied point head.

Based on the subdivision algorithm in Detectron2 projects/PointRend at
247691a1c292654867deb35cabb27f18d75a1363 (mask_head.py and point_features.py).
"""

from dataclasses import dataclass
from typing import Callable, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F


GridSize = Tuple[int, int]


@dataclass(frozen=True)
class StageCount:
    size: GridSize
    points_per_mask: int
    point_evaluations: int


@dataclass(frozen=True)
class SubdivisionResult:
    logits: Tensor
    stages: Tuple[StageCount, ...]

    @property
    def point_evaluations(self) -> int:
        return sum(stage.point_evaluations for stage in self.stages)


@torch.no_grad()
def subdivide_binary_mask(
    coarse_logits: Tensor,
    predict_points: Callable[[Tensor, Tensor], Tensor],
    *,
    initial_size: GridSize,
    native_size: GridSize,
    points_per_step: int,
) -> SubdivisionResult:
    """Render final [N,1,H,W] logits at native_size for N aligned query masks.

    coarse_logits is the fixed [N,1,Hc,Wc] source over the same spatial extent.
    predict_points(xy, coarse_samples) receives [N,P,2] normalized (x,y)
    centers and [N,1,P] samples. It returns fresh [N,1,P] logits on the input
    device and dtype. It owns the trained head and fine-feature sampling,
    including the mapping from this extent into any feature crop/padding.
    The head must be in evaluation mode and produce pointwise predictions.
    """
    if coarse_logits.ndim != 4 or coarse_logits.shape[1] != 1 or min(coarse_logits.shape[-2:]) < 1:
        raise ValueError("coarse_logits must have shape [N,1,Hc,Wc] with positive Hc,Wc")
    if coarse_logits.dtype not in (torch.float32, torch.float64):
        raise ValueError("coarse_logits must use float32 or float64 for coordinate sampling")
    for size in (initial_size, native_size):
        if len(size) != 2 or any(type(axis) is not int or axis < 1 for axis in size):
            raise ValueError("grid sizes must contain two positive integers")
    if any(start > target for start, target in zip(initial_size, native_size)):
        raise ValueError("initial_size must fit inside native_size")
    if type(points_per_step) is not int or points_per_step < 1:
        raise ValueError("points_per_step must be a positive integer")

    batch = coarse_logits.shape[0]
    native_h, native_w = native_size
    if batch == 0:
        return SubdivisionResult(coarse_logits.new_empty((0, 1, native_h, native_w)), ())

    h, w = initial_size
    sizes = [(h, w)]
    while (h, w) != (native_h, native_w):
        h, w = min(2 * h, native_h), min(2 * w, native_w)
        # A fully evaluated next grid overwrites every earlier prediction.
        if h * w <= points_per_step:
            sizes.clear()
        sizes.append((h, w))

    current = None
    counts = []
    for h, w in sizes:
        if current is None:
            transform = coarse_logits.new_tensor([[[0.5, 0, 0.5], [0, 0.5, 0.5]]])
            centers = F.affine_grid(transform, (1, 1, h, w), align_corners=False)
            xy = centers.reshape(1, h * w, 2).expand(batch, -1, -1)
        else:
            current = F.interpolate(current, size=(h, w), mode="bilinear", align_corners=False)
            uncertainty = -current.flatten(1).abs()
            indices = uncertainty.topk(min(points_per_step, h * w), dim=1).indices
            # Preserve the multiply-then-add order of the reference FP32 grid.
            dx, dy = 1 / w, 1 / h
            x = indices.remainder(w).to(coarse_logits.dtype) * dx + dx / 2
            y = indices.div(w, rounding_mode="floor").to(coarse_logits.dtype) * dy + dy / 2
            xy = torch.stack((x, y), dim=-1)
        grid = (2 * xy - 1).unsqueeze(2)
        samples = F.grid_sample(coarse_logits, grid, mode="bilinear",
                                padding_mode="zeros", align_corners=False).squeeze(-1)
        fresh = predict_points(xy, samples)
        if fresh.shape != samples.shape or fresh.device != samples.device or fresh.dtype != samples.dtype:
            raise ValueError("predict_points must preserve coarse_samples shape, device and dtype")
        if current is None:
            current = fresh.reshape(batch, 1, h, w)
        else:
            current = current.flatten(2).scatter(2, indices.unsqueeze(1), fresh).reshape(batch, 1, h, w)
        counts.append(StageCount((h, w), fresh.shape[-1], fresh.numel()))
    return SubdivisionResult(current, tuple(counts))
