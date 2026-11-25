"""Pixel-cell averages for an edge-clamped, align_corners=False bilinear field."""
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=8192)
def _axis_average_weights(source_length, native_length, native_index):
    """Normalize source hat integrals across one destination cell."""
    # Represent source coordinate u as integer numerator / (2 * native_length).
    # The native cell's two edges map to x * source_length / native_length - .5
    # and (x + 1) * source_length / native_length - .5.
    unit = 2 * native_length
    left = 2 * native_index * source_length - native_length
    right = left + 2 * source_length
    first_knot = max(0, left // unit + 1)
    last_knot = min(source_length - 1, (right - 1) // unit)
    cuts = [left]
    cuts.extend(knot * unit for knot in range(first_knot, last_knot + 1))
    cuts.append(right)

    # The midpoint integrates each affine segment exactly. Use integer
    # numerators even at the midpoint, rounding only the normalized weights.
    midpoint_unit = 2 * unit
    last_midpoint = (source_length - 1) * midpoint_unit
    masses = {}
    for start, end in zip(cuts, cuts[1:]):
        midpoint = min(max(start + end, 0), last_midpoint)
        first, fraction = divmod(midpoint, midpoint_unit)
        length = end - start
        masses[first] = masses.get(first, 0) + length * (midpoint_unit - fraction)
        if fraction:
            next_index = first + 1
            masses[next_index] = masses.get(next_index, 0) + length * fraction
    indices = sorted(masses)
    denominator = (right - left) * midpoint_unit
    weights = np.array([masses[i] / denominator for i in indices], dtype=np.float64)
    return np.array(indices, dtype=np.intp), weights


def pixel_cell_average(grid, native_hw, flat_ordinals):
    """Integrate an FP32 source field over indexed native pixel footprints.

    grid is a finite two-dimensional FP32 array. native_hw is the destination
    (height, width). flat_ordinals are zero-based integer indices in native
    row-major order. The result retains the shape of flat_ordinals and is FP32.

    Source sample centers are integer coordinates. Native center x projects
    to (x + .5) * source_width / native_width - .5, and y analogously.
    The interval has half-native-cell shifts about that center. Interpolation
    clamps to the boundary source samples. We integrate on each affine span,
    use separable float64 weighted sums, and cast once to FP32.
    """
    field = np.asarray(grid)
    if (field.ndim != 2 or field.dtype != np.float32 or not field.size
            or not np.isfinite(field).all()):
        raise ValueError("grid must be a nonempty, finite, two-dimensional FP32 array")
    try:
        height, width = native_hw
    except (TypeError, ValueError) as error:
        raise ValueError("native_hw must contain two positive integers") from error
    if any(not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
           or value < 1 for value in (height, width)):
        raise ValueError("native_hw must contain two positive integers")
    height, width = int(height), int(width)
    ordinals = np.asarray(flat_ordinals)
    if ordinals.dtype.kind not in "iu":
        raise ValueError("flat_ordinals must have integer dtype")
    if ordinals.size and (int(ordinals.min()) < 0 or int(ordinals.max()) >= height * width):
        raise ValueError("flat_ordinals are outside the native grid")

    source_height, source_width = field.shape
    output = np.empty(ordinals.size, dtype=np.float32)
    for i, ordinal in enumerate(ordinals.flat):
        row, col = divmod(int(ordinal), width)
        source_y, weight_y = _axis_average_weights(source_height, height, row)
        source_x, weight_x = _axis_average_weights(source_width, width, col)
        patch = field[np.ix_(source_y, source_x)].astype(np.float64)
        output[i] = (weight_y @ patch) @ weight_x
    return output.reshape(ordinals.shape)

