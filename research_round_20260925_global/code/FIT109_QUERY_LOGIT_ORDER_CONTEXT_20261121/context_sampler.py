"""Sample ordered query-logit rings at original HRSID pixel centers."""

import torch
from torch.nn import functional as F


NATIVE_SIZE = 800
QUERY_SIZE = 144
RING_OFFSETS_XY = tuple(
    (dx, dy)
    for radius in (1, 3)
    for dy in range(-radius, radius + 1)
    for dx in range(-radius, radius + 1)
    if max(abs(dx), abs(dy)) == radius
)
_INDEX_TYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def sample_context(planes, coords, lengths):
    """Return [R,M,2,32] FP32 samples in unchanged row, endpoint and side order.

    planes contains the signed 144x144 query logits for each row, optionally
    with a singleton channel. coords stores flat row-major 800x800 indices.
    Only the first lengths[r] pairs are used. Padding returns zero even if its
    coordinates are invalid or a row with no support has a nonfinite plane.

    Within each ring entries run top to bottom and left to right. Offsets
    refer to query-grid cells. Native pixel indices refer to pixel centers.
    """
    planes = torch.as_tensor(planes)
    if planes.ndim == 3:
        planes = planes.unsqueeze(1)
    if (planes.ndim != 4 or planes.shape[1:] != (1, QUERY_SIZE, QUERY_SIZE)
            or not planes.is_floating_point()):
        raise ValueError("planes must be floating [R,144,144] or [R,1,144,144]")
    planes = planes.to(torch.float32)
    rows = planes.size(0)
    coords = torch.as_tensor(coords, device=planes.device)
    if (coords.ndim != 3 or coords.size(0) != rows or coords.size(2) != 2
            or coords.dtype not in _INDEX_TYPES):
        raise ValueError("coords must be integer [R,M,2] native flat indices")
    width = coords.size(1)
    lengths = torch.as_tensor(lengths, device=planes.device)
    if lengths.shape != (rows,) or (rows and lengths.dtype not in _INDEX_TYPES):
        raise ValueError("lengths must contain one integer per row")
    lengths = lengths.long()
    if ((lengths < 0) | (lengths > width)).any():
        raise ValueError("lengths must be between zero and M")
    shape = (rows, width, 2, len(RING_OFFSETS_XY))
    if not rows or not width:
        return planes.new_zeros(shape) + planes.reshape(-1)[:0].sum()

    valid = torch.arange(width, device=planes.device)[None, :] < lengths[:, None]
    indices = coords.long().masked_fill(~valid[:, :, None], 0)
    if ((indices < 0) | (indices >= NATIVE_SIZE * NATIVE_SIZE)).any():
        raise ValueError("a supported native flat index lies outside the 800x800 image")
    planes = planes.masked_fill((lengths == 0)[:, None, None, None], 0)
    if not torch.isfinite(planes).all():
        raise ValueError("a query plane with valid endpoints contains nonfinite logits")

    xy = torch.stack((indices % NATIVE_SIZE, indices // NATIVE_SIZE), dim=-1).float()
    # With align_corners=False a pixel center is (index + 0.5)/size. One
    # 144-grid step is therefore 2/144 in grid_sample's [-1,1] coordinates.
    centers = 2 * (xy + 0.5) / NATIVE_SIZE - 1
    offsets = planes.new_tensor(RING_OFFSETS_XY) * (2.0 / QUERY_SIZE)
    grid = (centers.unsqueeze(-2) + offsets).reshape(rows, width * 2, -1, 2)
    samples = F.grid_sample(planes, grid, mode="bilinear", padding_mode="border",
                            align_corners=False)
    return samples.reshape(shape).masked_fill(~valid[:, :, None, None], 0)


def without_direction(context):
    """Discard angular order within each ring and retain each ring's multiset."""
    context = torch.as_tensor(context)
    if (context.ndim != 4 or context.shape[-2:] != (2, 32)
            or not context.is_floating_point()):
        raise ValueError("context must be floating [R,M,2,32]")
    context = context.float()
    return torch.cat((context[..., :8].sort(dim=-1).values,
                      context[..., 8:].sort(dim=-1).values), dim=-1)
