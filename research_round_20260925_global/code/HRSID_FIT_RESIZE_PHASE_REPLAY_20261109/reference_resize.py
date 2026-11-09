"""Exact one-pass center sampling for separable image geometry."""
from fractions import Fraction
from math import lcm
from operator import index

import numpy as np


def _size(value):
    h, w = map(index, value)
    if h <= 0 or w <= 0:
        raise ValueError("Raster dimensions must be positive.")
    return h, w


def _compose(operations, source_shape=None):
    initial = None if source_shape is None else _size(source_shape)
    current = initial
    scale = [Fraction(1), Fraction(1)]
    offset = [Fraction(0), Fraction(0)]
    for operation in operations:
        before, after = _size(operation["before"]), _size(operation["after"])
        if current is None:
            initial = current = before
        if current != before:
            raise ValueError("Operation trace has inconsistent raster sizes.")
        kind = operation["kind"]
        if kind == "resize":
            for axis in range(2):
                scale[axis] *= Fraction(before[axis], after[axis])
        elif kind == "crop":
            start = index(operation["top"]), index(operation["left"])
            if any(pos < 0 or pos + out > size
                   for pos, out, size in zip(start, after, before)):
                raise ValueError("Crop must remain within its input raster.")
            for axis in range(2):
                offset[axis] += scale[axis] * start[axis]
        elif kind == "hflip":
            if before != after:
                raise ValueError("A horizontal flip must preserve raster size.")
            offset[1] += scale[1] * before[1]
            scale[1] = -scale[1]
        else:
            raise ValueError(f"Unsupported operation {kind!r}.")
        current = after
    return initial, current, scale, offset


def _axis_indices(count, scale, offset):
    # Coordinates are measured from pixel edges. At a tie exactly on an
    # interior source edge, floor chooses the pixel on the larger-index side.
    start = scale / 2 + offset
    divisor = lcm(start.denominator, scale.denominator)
    first = start.numerator * (divisor // start.denominator)
    step = scale.numerator * (divisor // scale.denominator)
    # Python integer arithmetic avoids overflow in composed rational maps.
    return np.fromiter(((first + i * step) // divisor for i in range(count)),
                       dtype=np.intp, count=count)


def source_indices(operations, source_shape=None):
    """Return the original row and column index vectors for the output canvas."""
    initial, final, scale, offset = _compose(operations, source_shape)
    if initial is None:
        raise ValueError("An empty trace needs an explicit source shape.")
    indices = tuple(_axis_indices(size, slope, shift)
                    for size, slope, shift in zip(final, scale, offset))
    if any(values.min() < 0 or values.max() >= size for values, size in zip(indices, initial)):
        raise ValueError("Composite map samples outside the source raster.")
    return indices


def center_reference(raw_masks, operations):
    """Rasterize original Boolean masks after composing all operations."""
    raw = np.asarray(raw_masks)
    if raw.ndim != 3 or raw.dtype != np.bool_:
        raise ValueError("Expected Boolean masks with shape (N, H, W).")
    rows, cols = source_indices(operations, raw.shape[-2:])
    return raw[:, rows[:, None], cols]


def inverse_scales(operations):
    """Return signed (dy, dx) scales in original pixels per output pixel."""
    _, _, scale, _ = _compose(operations)
    return tuple(float(value) for value in scale)
