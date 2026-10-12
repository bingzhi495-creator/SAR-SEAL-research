"""CPU scalar geometry/GT-position proxies; no I/O, model, RNG or dependencies.

Caller supplies a binary two-dimensional CPU mask and feature-index (x, y)
points. ``measure_occupancy`` validates the whole mask once per batch of points.
``nearest_800`` is the native nearest-resized, right/bottom-zero-padded GT;
``original_proxy`` samples the original GT directly. They are different measures:
the second does not reproduce nearest resize or the native two-resize renderer.

Source conventions: native_ordinal_two_mode_mask_adapter_learning_preparation_
20261008/train.py:246-301 (img_shape, ori_shape, fixed800, GT pad); parent
preparation's native_hooks.py:114-146 (4x4-cell metadata valid-area fractions). align_corners=
False gives u=(x_F+.5)/200 and x_800=800*u-.5. Original-image pixel-centre
coordinates are (x_800+.5)*ori_width/img_width-.5, and likewise for y.

General bilinear sampling accepts any finite coordinate with zero padding.
Paired's centre-domain test is separately 0<=x_F,y_F<=199, NOT [-.5,199.5].
Direct800 pixel-support fractions and sampled F200 cell-valid-area fractions
are separately named; neither alone reconstructs paired eligibility. No E,
donor permutation or GT/query correspondence is inferred. Caller retains E0,
unmatched queries and all failures. Occupancy is a GT-position proxy, not
feature purity, actual sea background, or causal evidence. Python scalar
arithmetic is not a bitwise reconstruction of CUDA/FP32 grid_sample.
"""

from math import floor, fsum, isfinite
from numbers import Integral


def _hw(shape, name):
    if not isinstance(shape, (list, tuple)) or len(shape) not in (2, 3):
        raise ValueError(name + ' must be (height,width), optionally with channels')
    if any(isinstance(v, bool) or not isinstance(v, Integral) or v <= 0
           for v in shape):
        raise ValueError(name + ' entries must be positive integers')
    return int(shape[0]), int(shape[1])


def _finite(value, name):
    if isinstance(value, (bool, str, bytes, bytearray)):
        raise ValueError(name + ' must be a finite numeric coordinate, not Boolean/text')
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(name + ' must be a finite scalar')
    if not isfinite(value):
        raise ValueError(name + ' must be finite')
    return value


def _metadata(img_shape, ori_shape, batch_shape, feature_shape):
    image = _hw(img_shape, 'img_shape')
    original = _hw(ori_shape, 'ori_shape')
    batch = _hw(batch_shape, 'batch_shape')
    feature = _hw(feature_shape, 'feature_shape')
    if batch != (800, 800) or feature != (200, 200):
        raise ValueError('This source is restricted to batch800 and F200')
    if image[0] > batch[0] or image[1] > batch[1]:
        raise ValueError('Resized valid image must fit the top-left batch800 crop')
    ih, iw = image
    return dict(img_hw=image, ori_hw=original, batch_hw=batch, feature_hw=feature,
                valid_pixels=ih * iw, padded_pixels=800 * 800 - ih * iw,
                bottom_pad_rows=800 - ih, right_pad_columns=800 - iw)


def map_feature_point(x_feature, y_feature, img_shape, ori_shape,
                      batch_shape=(800, 800), feature_shape=(200, 200)):
    """Map one F200 index to800/original centre indices; never filter a point.

    ``paired_canvas_legal`` is only paired's geometric centre-domain predicate.
    ``normalized_canvas_legal`` describes coordinates within grid [-1,1]; it
    does not replace that predicate, E, or metadata valid-area sampling.
    """
    meta = _metadata(img_shape, ori_shape, batch_shape, feature_shape)
    x = _finite(x_feature, 'x_feature')
    y = _finite(y_feature, 'y_feature')
    x800, y800 = 4.0 * (x + .5) - .5, 4.0 * (y + .5) - .5
    ih, iw = meta['img_hw']
    oh, ow = meta['ori_hw']
    xo, yo = (x800 + .5) * ow / iw - .5, (y800 + .5) * oh / ih - .5
    if not all(isfinite(v) for v in (x800, y800, xo, yo)):
        raise ValueError('Mapped centre coordinate overflowed')
    return dict(feature_xy=(x, y), canvas800_xy=(x800, y800),
                original_xy=(xo, yo),
                paired_canvas_legal=(0.0 <= x <= 199.0 and 0.0 <= y <= 199.0),
                normalized_canvas_legal=(-.5 <= x <= 199.5 and -.5 <= y <= 199.5))


def bilinear_stencil(x_index, y_index, shape):
    """Four (row,column,weight,in_bounds) entries, with zero-padding semantics.

    Indices outside the array are retained in the stencil and contribute zero,
    including the half-cell border. At grid nodes some weights are zero.
    """
    h, w = _hw(shape, 'shape')
    x, y = _finite(x_index, 'x_index'), _finite(y_index, 'y_index')
    left, top = floor(x), floor(y)
    tx, ty = x - left, y - top
    return [dict(row=row, column=col, weight=weight,
                 in_bounds=(0 <= row < h and 0 <= col < w))
            for row, col, weight in
            ((top, left, (1.0 - tx) * (1.0 - ty)),
             (top, left + 1, tx * (1.0 - ty)),
             (top + 1, left, (1.0 - tx) * ty),
             (top + 1, left + 1, tx * ty))]


def _binary(value):
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError('Mask entries must be numeric Boolean or exact 0/1, not text')
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('Mask entries must be scalar Boolean or exact 0/1')
    if not isfinite(value) or value not in (0.0, 1.0):
        raise ValueError('Mask entries must be finite exact 0/1')
    return value


def _validate_mask(mask, shape, valid_crop=None):
    h, w = shape
    declared = getattr(mask, 'shape', None)
    if declared is not None and tuple(declared) != shape:
        raise ValueError('Mask declared shape differs from requested GT measure')
    if len(mask) != h:
        raise ValueError('Mask row count differs from requested GT measure')
    foreground = 0
    for row in range(h):
        current = mask[row]
        if len(current) != w:
            raise ValueError('Mask must be rectangular with the expected column count')
        for col in range(w):
            value = _binary(current[col])
            if valid_crop is not None and (row >= valid_crop[0] or col >= valid_crop[1]):
                if value != 0.0:
                    raise ValueError('nearest_800 GT must be zero in every padded pixel')
            foreground += int(value)
    return dict(height=h, width=w, pixels=h * w, foreground_pixels=foreground,
                padded_foreground_pixels=0 if valid_crop is not None else None)


def _canvas_fractions(stencil, img_hw):
    ih, iw = img_hw
    valid, padding, outside = [], [], []
    for item in stencil:
        if not item['in_bounds']:
            outside.append(item['weight'])
        elif item['row'] < ih and item['column'] < iw:
            valid.append(item['weight'])
        else:
            padding.append(item['weight'])
    return dict(valid_image_pixel_support_fraction=fsum(valid),
                right_bottom_padding_pixel_support_fraction=fsum(padding),
                outside_canvas_zero_padding_fraction=fsum(outside),
                canvas_pixel_support_fraction=fsum(valid + padding))


def _metadata_fraction(feature_stencil, img_hw):
    """Sample the metadata-only F200 4x4-cell-area map, not an800 rectangle."""
    ih, iw = img_hw
    terms = []
    for item in feature_stencil:
        if item['in_bounds']:
            y_area = min(4.0, max(0.0, ih - 4.0 * item['row'])) / 4.0
            x_area = min(4.0, max(0.0, iw - 4.0 * item['column'])) / 4.0
            terms.append(item['weight'] * y_area * x_area)
    return fsum(terms)


def measure_occupancy(mask, sample_xy, img_shape, ori_shape, mask_space,
                      batch_shape=(800, 800), feature_shape=(200, 200)):
    """Validate one mask, then return all point proxies in caller order.

    mask_space is exactly 'nearest_800' or 'original_proxy'. sample_xy is an
    iterable of (x_feature,y_feature), empty allowed. No E/donor/filtering is
    applied. nearest_800 requires shape800 and zero outside img_shape; original
    requires ori_shape. The full original/canvas counts and right/bottom pads
    are explicit. This validates layout and padding, not the caller's resize
    provenance, annotation identity, E predicates or arbitrary dataset counts.
    """
    meta = _metadata(img_shape, ori_shape, batch_shape, feature_shape)
    if mask_space == 'nearest_800':
        shape, crop = meta['batch_hw'], meta['img_hw']
    elif mask_space == 'original_proxy':
        shape, crop = meta['ori_hw'], None
    else:
        raise ValueError('mask_space must be nearest_800 or original_proxy')
    population = _validate_mask(mask, shape, crop)
    points = []
    for xy in sample_xy:
        if len(xy) != 2:
            raise ValueError('Each sample must contain x_feature,y_feature')
        mapped = map_feature_point(xy[0], xy[1], img_shape, ori_shape,
                                   batch_shape, feature_shape)
        feature_stencil = bilinear_stencil(*mapped['feature_xy'], shape=meta['feature_hw'])
        canvas_stencil = bilinear_stencil(*mapped['canvas800_xy'], shape=meta['batch_hw'])
        if mask_space == 'nearest_800':
            gt_stencil = canvas_stencil
        else:
            gt_stencil = bilinear_stencil(*mapped['original_xy'], shape=meta['ori_hw'])
        occupancy = fsum(item['weight'] * _binary(mask[item['row']][item['column']])
                         for item in gt_stencil if item['in_bounds'])
        points.append(dict(coordinates=mapped, feature_stencil=feature_stencil,
                           canvas800_stencil=canvas_stencil, GT_stencil=gt_stencil,
                           GT_bilinear_occupancy=occupancy,
                           GT_pixel_support_fraction=fsum(item['weight'] for item in gt_stencil
                                                          if item['in_bounds']),
                           direct800_domain=_canvas_fractions(canvas_stencil, meta['img_hw']),
                           sampled_feature_cell_valid_area_fraction=
                           _metadata_fraction(feature_stencil, meta['img_hw'])))
    return dict(mask_space=mask_space, metadata=meta, mask_population=population,
                point_count=len(points), points=points)
