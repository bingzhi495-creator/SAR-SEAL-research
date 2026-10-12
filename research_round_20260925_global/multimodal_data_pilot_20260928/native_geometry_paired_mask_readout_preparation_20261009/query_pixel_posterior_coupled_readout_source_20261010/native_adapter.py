"""Pure native-tensor glue; no model, matching, sampling, or data/file I/O.

Run project_native_fields after the detached teacher forward. Its two learned
projections and feature renderer follow ambient autograd; inference may use
no_grad. renderer(masks, identity) is caller-bound, for example
functools.partial(qualified_diagnostic.render_original, torch). It must implement
native 200 -> bilinear800 -> image crop -> bilinear original, align_corners=False,
and return [channels,Horig,Worig] without detaching learned projected features.

Matched rows and raw original-raster bool masks are supplied by the caller.
This module never generates an assignment, semantic union, GT ranking, or RNG.
SOURCE preparation does not qualify native integration or method efficacy.
"""

import torch
from torch.nn import functional as F

POINTS_PER_GT = 12544
FIELD_NAMES = {'projected_pixel', 'projected_query', 'native_logits', 'contrast'}


def _tensor(name, value, shape, reference=None):
    if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
        raise TypeError(name + ' must be a dense strided tensor')
    if value.dtype != torch.float32 or tuple(value.shape) != tuple(shape):
        raise ValueError(name + ' must have the specified FP32 shape')
    if reference is not None and value.device != reference.device:
        raise ValueError(name + ' must share the native tensor device')
    if not bool(torch.isfinite(value).all()):
        raise ValueError(name + ' must be finite')


def _shape2(name, value, upper=None):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(name + ' must contain exactly two integer dimensions')
    if any(type(v) is not int or v <= 0 or (upper is not None and v > upper) for v in value):
        raise ValueError(name + ' has invalid dimensions')
    return tuple(value)


def project_native_fields(readout, F256, e0, M200, actual_img_shape,
                          original_shape, renderer):
    """Return live projected fields and detached native original-raster fields.

    Input teacher shapes are [1,256,200,200], [1,100,256], [1,100,200,200].
    Returned shapes are [1,16,H,W], [1,100,16], and two [1,100,H,W] fields.
    All are FP32 on the teacher device. No operand is modified in place.
    """
    _tensor('F256', F256, (1, 256, 200, 200))
    _tensor('e0', e0, (1, 100, 256), F256)
    _tensor('M200', M200, (1, 100, 200, 200), F256)
    image_shape = _shape2('actual_img_shape', actual_img_shape, upper=800)
    original = _shape2('original_shape', original_shape)
    if not callable(renderer):
        raise TypeError('renderer must be a caller-bound two-argument callable')
    identity = dict(actual_img_shape=list(image_shape), original_shape=list(original))
    projected = readout.project_features(F256.detach())
    _tensor('projected feature', projected, (1, 16, 200, 200), F256)
    pixel = renderer(projected, identity)
    _tensor('rendered projected pixel', pixel, (16,) + original, F256)
    query = readout.project_queries(e0.detach())
    _tensor('projected query', query, (1, 100, 16), F256)
    native = renderer(M200.detach(), identity)
    _tensor('rendered native logits', native, (100,) + original, F256)
    native = native.unsqueeze(0).detach()
    probability = native.sigmoid()
    local_mean = F.avg_pool2d(probability[0, :, None], kernel_size=3,
                             stride=1, padding=1, count_include_pad=False)[:, 0]
    contrast = (probability - local_mean.unsqueeze(0)).detach()
    return dict(projected_pixel=pixel.unsqueeze(0), projected_query=query,
                native_logits=native, contrast=contrast)


def matched_point_loss(readout, fields, query_ids, gt_masks, point_indices):
    """Matched per-GT point BCE, reduced pixel mean then image mean.

    query_ids: unique int64[G] actual query ordinals, aligned with bool[G,H,W]
    raw GT rows. point_indices: int64[G,12544] original-raster flat indices;
    replacement/duplicates are preserved, and no sampling occurs here.
    G=0 returns a finite connected zero for every trainable readout parameter.
    No unmatched-query mask loss or accumulation scaling is added.
    """
    if not torch.is_grad_enabled():
        raise ValueError('matched_point_loss requires ambient autograd')
    if not isinstance(fields, dict) or set(fields) != FIELD_NAMES:
        raise ValueError('fields must be the exact project_native_fields packet')
    pixel, query = fields['projected_pixel'], fields['projected_query']
    native, contrast = fields['native_logits'], fields['contrast']
    if not isinstance(pixel, torch.Tensor) or pixel.ndim != 4:
        raise ValueError('projected_pixel must be [1,16,H,W]')
    height, width = pixel.shape[-2:]
    if height <= 0 or width <= 0:
        raise ValueError('original raster dimensions must be positive')
    _tensor('projected_pixel', pixel, (1, 16, height, width))
    _tensor('projected_query', query, (1, 100, 16), pixel)
    _tensor('native_logits', native, (1, 100, height, width), pixel)
    _tensor('contrast', contrast, (1, 100, height, width), pixel)
    if native.requires_grad or contrast.requires_grad:
        raise ValueError('native original logits and contrast must be detached')
    parameters = tuple(readout.parameters())
    if not parameters or not all(p.requires_grad for p in parameters):
        raise ValueError('all readout parameters must remain trainable')
    if not pixel.requires_grad or not query.requires_grad:
        raise ValueError('projected fields must retain their learned projection graph')
    if not isinstance(query_ids, torch.Tensor) or query_ids.layout != torch.strided or \
            query_ids.dtype != torch.int64 or query_ids.ndim != 1 or query_ids.device != pixel.device:
        raise ValueError('query_ids must be int64[G] on the readout device')
    count = query_ids.numel()
    if count > 100 or bool(((query_ids < 0) | (query_ids >= 100)).any()) or \
            torch.unique(query_ids).numel() != count:
        raise ValueError('matched queries must be unique actual ordinals in [0,99]')
    if not isinstance(gt_masks, torch.Tensor) or gt_masks.layout != torch.strided or \
            gt_masks.dtype != torch.bool or gt_masks.device != pixel.device or \
            tuple(gt_masks.shape) != (count, height, width):
        raise ValueError('gt_masks must be aligned bool[G,H,W] on the readout device')
    if not isinstance(point_indices, torch.Tensor) or point_indices.layout != torch.strided or \
            point_indices.dtype != torch.int64 or point_indices.device != pixel.device or \
            tuple(point_indices.shape) != (count, POINTS_PER_GT):
        raise ValueError('point_indices must be int64[G,12544] on the readout device')
    if bool(((point_indices < 0) | (point_indices >= height * width)).any()):
        raise ValueError('point_indices must index the complete original raster')
    if count == 0:
        return sum((parameter.reshape(-1)[:0].sum() for parameter in parameters), pixel.new_zeros(()))
    samples = POINTS_PER_GT
    flat_indices = point_indices.reshape(-1)
    pixel_points = pixel[0].flatten(1).t()[flat_indices]
    query_points = query[0, query_ids][:, None].expand(count, samples, 16).reshape(-1, 16)
    native_points = native[0, query_ids].reshape(count, height * width).gather(1, point_indices)
    contrast_points = contrast[0, query_ids].reshape(count, height * width).gather(1, point_indices)
    targets = gt_masks.reshape(count, height * width).gather(1, point_indices).to(pixel.dtype)
    result = readout.point_outputs(pixel_points, query_points,
                                   native_points.reshape(-1), contrast_points.reshape(-1))
    logits = result['posterior_logits'].reshape(count, samples)
    return F.binary_cross_entropy_with_logits(logits, targets, reduction='none').mean(1).mean()
