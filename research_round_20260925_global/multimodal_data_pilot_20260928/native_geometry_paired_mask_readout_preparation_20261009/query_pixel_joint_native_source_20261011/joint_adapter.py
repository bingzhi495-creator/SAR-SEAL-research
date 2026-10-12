"""Live native-tensor glue for a caller-bound 9441-parameter readout.

Unlike the frozen adapter, all three native inputs and all four returned fields
retain ambient autograd. The caller supplies the exact native renderer, current
one-to-one assignments and original-raster GT/point indices. No model, matching,
sampling, RNG, optimizer, file I/O or native training qualification is provided.
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


def _live(name, value):
    if not value.requires_grad:
        raise ValueError(name + ' must retain its training autograd graph')


def _shape2(name, value, upper=None):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(name + ' must contain exactly two integer dimensions')
    if any(type(v) is not int or v <= 0 or (upper is not None and v > upper) for v in value):
        raise ValueError(name + ' has invalid dimensions')
    return tuple(value)


def project_live_native_fields(readout, F256, e0, M200, actual_img_shape,
                               original_shape, renderer):
    """Return the four live fields, without detaching or modifying any input.

    Native FP32 shapes: F256[1,256,200,200], e0[1,100,256], M200[1,100,200,200].
    renderer(masks, identity) must apply bilinear200->800, crop actual_img_shape,
    then bilinear original_shape, align_corners=False, returning [C,H,W].
    It must preserve autograd for both projected F16 and native M200.
    """
    if not torch.is_grad_enabled():
        raise ValueError('project_live_native_fields requires ambient autograd')
    _tensor('F256', F256, (1, 256, 200, 200))
    _tensor('e0', e0, (1, 100, 256), F256)
    _tensor('M200', M200, (1, 100, 200, 200), F256)
    for name, value in (('F256', F256), ('e0', e0), ('M200', M200)):
        _live(name, value)
    image_shape = _shape2('actual_img_shape', actual_img_shape, upper=800)
    original = _shape2('original_shape', original_shape)
    if not callable(renderer):
        raise TypeError('renderer must be a caller-bound two-argument callable')
    identity = dict(actual_img_shape=list(image_shape), original_shape=list(original))
    projected = readout.project_features(F256)
    _tensor('projected feature', projected, (1, 16, 200, 200), F256)
    _live('projected feature', projected)
    pixel = renderer(projected, identity)
    _tensor('rendered projected pixel', pixel, (16,) + original, F256)
    _live('rendered projected pixel', pixel)
    query = readout.project_queries(e0)
    _tensor('projected query', query, (1, 100, 16), F256)
    _live('projected query', query)
    native = renderer(M200, identity)
    _tensor('rendered native logits', native, (100,) + original, F256)
    _live('rendered native logits', native)
    native = native.unsqueeze(0)
    probability = native.sigmoid()
    local_mean = F.avg_pool2d(probability[0, :, None], kernel_size=3,
                              stride=1, padding=1, count_include_pad=False)[:, 0]
    contrast = probability - local_mean.unsqueeze(0)
    _tensor('contrast', contrast, (1, 100) + original, F256)
    _live('contrast', contrast)
    return dict(projected_pixel=pixel.unsqueeze(0), projected_query=query,
                native_logits=native, contrast=contrast)


def matched_joint_point_loss(readout, fields, query_ids, gt_masks, point_indices):
    """Stable BCE(L+u): pixel mean, then matched-GT mean; no extra scaling.

    All four fields must be live FP32 tensors on one device. Unique int64[G]
    query ordinals align with bool[G,H,W] GT and int64[G,12544] original flat
    indices. Repeated indices retain their multiplicity. No assignments or
    unmatched-query losses are generated. G=0 connects every field and all eight
    readout parameters with an exact finite zero, without summing full rasters.
    """
    if not torch.is_grad_enabled():
        raise ValueError('matched_joint_point_loss requires ambient autograd')
    if not isinstance(fields, dict) or set(fields) != FIELD_NAMES:
        raise ValueError('fields must be the exact project_live_native_fields packet')
    pixel, query = fields['projected_pixel'], fields['projected_query']
    native, contrast = fields['native_logits'], fields['contrast']
    if not isinstance(pixel, torch.Tensor) or pixel.ndim != 4:
        raise ValueError('projected_pixel must be [1,16,H,W]')
    height, width = pixel.shape[-2:]
    if height <= 0 or width <= 0:
        raise ValueError('original raster dimensions must be positive')
    for name, value, shape in (
        ('projected_pixel', pixel, (1, 16, height, width)),
        ('projected_query', query, (1, 100, 16)),
        ('native_logits', native, (1, 100, height, width)),
        ('contrast', contrast, (1, 100, height, width)),
    ):
        _tensor(name, value, shape, pixel)
        _live(name, value)
    parameters = tuple(readout.parameters())
    if len(parameters) != 8 or sum(p.numel() for p in parameters) != 9441:
        raise ValueError('caller-bound readout must have eight tensors / 9441 scalars')
    for parameter in parameters:
        _tensor('readout parameter', parameter, tuple(parameter.shape), pixel)
        if parameter.ndim == 0 or not parameter.requires_grad:
            raise ValueError('all eight readout parameter tensors must remain trainable')
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
        connected = (*fields.values(), *parameters)
        return sum((value[..., :0].sum() for value in connected), pixel.new_zeros(()))
    samples = POINTS_PER_GT
    pixel_points = pixel[0].flatten(1).t()[point_indices.reshape(-1)]
    query_points = query[0, query_ids][:, None].expand(count, samples, 16).reshape(-1, 16)
    native_points = native[0, query_ids].reshape(count, height * width).gather(1, point_indices)
    contrast_points = contrast[0, query_ids].reshape(count, height * width).gather(1, point_indices)
    targets = gt_masks.reshape(count, height * width).gather(1, point_indices).to(pixel.dtype)
    result = readout.point_outputs(pixel_points, query_points,
                                   native_points.reshape(-1), contrast_points.reshape(-1))
    logits = result['posterior_logits'].reshape(count, samples)
    _tensor('posterior_logits', logits, (count, samples), pixel)
    _live('posterior_logits', logits)
    return F.binary_cross_entropy_with_logits(logits, targets, reduction='none').mean(1).mean()
