"""Native pixel-center admission and two PointRend writeback samplers."""

import operator

import numpy as np
import torch
from torch.nn import functional as F


def _indices(values):
    values = np.asarray(values)
    if values.ndim != 1 or (values.size and values.dtype.kind not in "iu"):
        raise ValueError("supply a vector of integer row-major pixel indices")
    if values.size and (values.min() < 0 or values.max() > np.iinfo(np.int64).max):
        raise ValueError("native pixel index exceeds int64 range")
    return np.ascontiguousarray(values, dtype=np.int64)


def eligible(indices, width, box):
    """Admission on a clipped, continuous XYXY RoI, inclusive left/top only."""
    width = operator.index(width)
    if width <= 0:
        raise ValueError("image width must be positive")
    edges = np.asarray(box, dtype=np.float64)
    if edges.shape != (4,) or not np.isfinite(edges).all() or (edges[:2] >= edges[2:]).any():
        raise ValueError("supply four finite, nondegenerate XYXY edges")
    y, x = np.divmod(_indices(indices), width)
    x, y = x + 0.5, y + 0.5
    return (edges[0] <= x) & (x < edges[2]) & (edges[1] <= y) & (y < edges[3])


@torch.no_grad()
def border_and_zero(logits, native_box_tensor, side_indices, hw):
    """Return (zeros, border, exposed_masks), each in (add, drop) input order.

    Scores are NumPy FP32 vectors. Exposure means a positive bilinear weight
    falls outside the 224-by-224 grid, based on the sampler's FP32 coordinates.
    Callers choose the support on which to sample.
    """
    height, width = map(operator.index, hw)
    if min(height, width) <= 0 or len(side_indices) != 2:
        raise ValueError("supply positive image dimensions and two pixel vectors")
    box = native_box_tensor
    if (logits.shape != (1, 224, 224) or box.shape != (1, 4)
            or logits.dtype != torch.float32 or box.dtype != torch.float32
            or logits.device != box.device):
        raise ValueError("expected aligned float32 logits [1,224,224] and box [1,4]")
    left_top, right_bottom = box[:, :2], box[:, 2:]
    if (not torch.isfinite(box).all() or (left_top < 0).any()
            or (right_bottom <= left_top).any()
            or (right_bottom > box.new_tensor([width, height])).any()):
        raise ValueError("box must be finite, clipped and nondegenerate")
    sides = tuple(_indices(values) for values in side_indices)
    all_indices = np.concatenate(sides)
    if all_indices.size and all_indices.max() >= height * width:
        raise ValueError("native pixel index outside image")
    if not all_indices.size:
        empty = tuple(np.empty(0, np.float32) for _ in sides)
        return empty, tuple(row.copy() for row in empty), tuple(np.empty(0, bool) for _ in sides)

    pixels = torch.as_tensor(all_indices, device=box.device)
    xy = torch.stack((pixels % width, pixels // width), dim=-1)[None]
    unit = (xy.to(box.dtype) + 0.5 - left_top[:, None]) / (right_bottom - left_top)[:, None]
    grid = (2 * unit - 1).unsqueeze(2)
    probabilities = logits[None].sigmoid()
    zero, border = (
        F.grid_sample(probabilities, grid, mode="bilinear", padding_mode=mode,
                      align_corners=False)[0, 0, :, 0]
        for mode in ("zeros", "border")
    )
    if not (torch.isfinite(zero).all() and torch.isfinite(border).all()):
        raise ValueError("nonfinite readback probabilities")
    # align_corners=False maps normalized -1 to -0.5 and +1 to 223.5.
    # A sample exactly on the first/last lattice center needs no padding.
    source_xy = (grid[0, :, 0] + 1) * 112 - 0.5
    edge = ((source_xy < 0) | (source_xy > 223)).any(dim=1)
    if not torch.equal(zero[~edge].view(torch.int32), border[~edge].view(torch.int32)):
        raise ValueError("padding modes differ on a fully supported footprint")
    n = len(sides[0])
    zero, border, edge = zero.cpu().numpy(), border.cpu().numpy(), edge.cpu().numpy()
    return (zero[:n], zero[n:]), (border[:n], border[n:]), (edge[:n], edge[n:])
