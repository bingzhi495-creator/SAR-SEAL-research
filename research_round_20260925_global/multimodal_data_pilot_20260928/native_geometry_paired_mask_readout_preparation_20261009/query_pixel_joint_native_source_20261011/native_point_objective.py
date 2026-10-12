"""Native matched-point BCE control; no readout, sampling, matching, or I/O.

Caller supplies live original-raster native logits and fixed one-to-one GT rows.
Duplicates in 12544 original flat pixel indices are retained. Reduction is
pixel mean, then matched-GT mean per image; accumulation scaling is external.
This pure loss does not qualify a native model or any training/evaluation run.
"""

import torch
from torch.nn import functional as F

POINTS_PER_GT = 12544


def matched_native_point_loss(native_logits, query_ids, gt_masks, point_indices):
    """[1,100,H,W], int64[G], bool[G,H,W], int64[G,12544] -> scalar.

    No unmatched-query loss or bias/scale is added. Empty G returns a zero
    connected to native_logits, with exact-zero gradient when it is live.
    Float32/float64 dense finite logits are accepted on a shared device.
    """
    if not isinstance(native_logits, torch.Tensor) or native_logits.layout != torch.strided:
        raise TypeError('native_logits must be a dense strided tensor')
    if native_logits.dtype not in (torch.float32, torch.float64) or native_logits.ndim != 4:
        raise ValueError('native_logits must be FP32/FP64 [1,100,H,W]')
    if native_logits.shape[:2] != (1, 100) or any(v <= 0 for v in native_logits.shape):
        raise ValueError('native_logits must have positive shape [1,100,H,W]')
    if not bool(torch.isfinite(native_logits).all()):
        raise ValueError('native_logits must be finite')
    if not isinstance(query_ids, torch.Tensor) or query_ids.layout != torch.strided or             query_ids.dtype != torch.int64 or query_ids.ndim != 1 or query_ids.device != native_logits.device:
        raise ValueError('query_ids must be shared-device int64[G]')
    count = query_ids.numel()
    if count > 100 or bool(((query_ids < 0) | (query_ids >= 100)).any()) or             torch.unique(query_ids).numel() != count:
        raise ValueError('matched query ordinals must be unique and within [0,99]')
    height, width = native_logits.shape[-2:]
    if not isinstance(gt_masks, torch.Tensor) or gt_masks.layout != torch.strided or             gt_masks.dtype != torch.bool or gt_masks.device != native_logits.device or             tuple(gt_masks.shape) != (count, height, width):
        raise ValueError('gt_masks must be aligned shared-device bool[G,H,W]')
    if not isinstance(point_indices, torch.Tensor) or point_indices.layout != torch.strided or             point_indices.dtype != torch.int64 or point_indices.device != native_logits.device or             tuple(point_indices.shape) != (count, POINTS_PER_GT):
        raise ValueError('point_indices must be shared-device int64[G,12544]')
    if bool(((point_indices < 0) | (point_indices >= height * width)).any()):
        raise ValueError('point_indices must index the complete original raster')
    if count == 0:
        return native_logits.reshape(-1)[:0].sum()
    logits = native_logits[0, query_ids].flatten(1).gather(1, point_indices)
    targets = gt_masks.flatten(1).gather(1, point_indices).to(native_logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, targets, reduction='none').mean(1).mean()
