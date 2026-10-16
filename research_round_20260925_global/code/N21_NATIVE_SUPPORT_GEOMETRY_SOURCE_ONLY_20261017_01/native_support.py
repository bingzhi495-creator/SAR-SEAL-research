"""UNBOUND exact geometry; UNIT_FIXTURE only, no current scientific path verified.

Only this contract is supported: ordinary two-tap bilinear, align_corners=False,
native HxW -> canvas -> top-left crop -> image.  Edge handling clamps each
resize's input coordinate.  No antialias filter, padding transform or floating
point implementation is modeled.  Shapes must always be supplied explicitly.
"""
from fractions import Fraction


def _shape(value, name):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(name + " must contain exactly two positive integers")
    if any(type(x) is not int or x < 1 for x in value):
        raise ValueError(name + " must contain exactly two positive integers")
    return tuple(value)


def _contract(native_shape, canvas_shape, crop_shape, image_shape, point):
    native = _shape(native_shape, "native_shape")
    canvas = _shape(canvas_shape, "canvas_shape")
    crop = _shape(crop_shape, "crop_shape")
    image = _shape(image_shape, "image_shape")
    if any(a > b for a, b in zip(crop, canvas)):
        raise ValueError("The top-left crop cannot exceed the canvas")
    if (not isinstance(point, (tuple, list)) or len(point) != 2
            or any(type(x) is not int for x in point)
            or any(not 0 <= x < size for x, size in zip(point, image))):
        raise ValueError("point must be an integer pixel inside image_shape")
    return native, canvas, crop, image, tuple(point)


def _linear_weights(input_size, output_size, output_index):
    raw = Fraction((2 * output_index + 1) * input_size, 2 * output_size) - Fraction(1, 2)
    coordinate = min(max(raw, Fraction(0)), Fraction(input_size - 1))
    lower = coordinate.numerator // coordinate.denominator
    delta = coordinate - lower
    weights = {lower: 1 - delta}
    if delta:
        weights[lower + 1] = delta
    return weights, raw != coordinate


def _compose_axis(native_size, canvas_size, crop_size, image_size, index):
    crop_weights, second_clamped = _linear_weights(crop_size, image_size, index)
    result = {}
    first_clamped = False
    for canvas_index, outer_weight in crop_weights.items():
        inner, clamped = _linear_weights(native_size, canvas_size, canvas_index)
        first_clamped = first_clamped or clamped
        for node, inner_weight in inner.items():
            result[node] = result.get(node, Fraction(0)) + outer_weight * inner_weight
    return result, first_clamped, second_clamped


def _point_trace(native, canvas, crop, image, point):
    ys, yc1, yc2 = _compose_axis(native[0], canvas[0], crop[0], image[0], point[0])
    xs, xc1, xc2 = _compose_axis(native[1], canvas[1], crop[1], image[1], point[1])
    sparse = {(y, x): yw * xw for y, yw in ys.items() for x, xw in xs.items()}
    return sparse, {"native_resize": yc1 or xc1, "image_resize": yc2 or xc2}


def point_support(native_shape, canvas_shape, crop_shape, image_shape, point):
    """Return positive exact sparse {(native_y, native_x): Fraction} weights."""
    arguments = _contract(native_shape, canvas_shape, crop_shape, image_shape, point)
    return _point_trace(*arguments)[0]


def matrix_rank(rows):
    """Exact Gaussian rank of a rectangular int/Fraction matrix; no tolerance."""
    rows = [list(row) for row in rows]
    width = len(rows[0]) if rows else 0
    if any(len(row) != width for row in rows):
        raise ValueError("Coefficient matrix must be rectangular")
    if any(type(v) not in (int, Fraction) for row in rows for v in row):
        raise ValueError("Only exact integer or Fraction coefficients are supported")
    work = [[Fraction(v) for v in row] for row in rows]
    rank = 0
    for column in range(width):
        pivot = next((i for i in range(rank, len(work)) if work[i][column]), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        scale = work[rank][column]
        work[rank] = [v / scale for v in work[rank]]
        for i in range(rank + 1, len(work)):
            factor = work[i][column]
            if factor:
                work[i] = [a - factor * b for a, b in zip(work[i], work[rank])]
        rank += 1
        if rank == len(work):
            break
    return rank


def trace_neighborhood(native_shape, canvas_shape, crop_shape, image_shape, center):
    """Trace the row-major nine integer neighbors, clamped in image coordinates.

    `rank` belongs to the 9 x union-support coefficient matrix, not any measured
    feature tensor. `same_native_2x2_cell` means all positive support nodes fit
    one adjacent native cell; a single row/column or 1x1 grid may be degenerate.
    All outputs concern the stated ideal geometry, not current N20 execution.
    """
    native, canvas, crop, image, center = _contract(
        native_shape, canvas_shape, crop_shape, image_shape, center
    )
    points, supports, clamp_flags = [], [], []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            requested = (center[0] + dy, center[1] + dx)
            point = tuple(min(max(v, 0), size - 1) for v, size in zip(requested, image))
            sparse, flags = _point_trace(native, canvas, crop, image, point)
            flags["neighborhood"] = point != requested
            points.append(point)
            supports.append(sparse)
            clamp_flags.append(flags)
    columns = sorted({node for sparse in supports for node in sparse})
    matrix = [[sparse.get(node, Fraction(0)) for node in columns] for sparse in supports]
    axes = [{node[i] for node in columns} for i in (0, 1)]
    same_cell = all(len(axis) <= 2 and max(axis) - min(axis) <= 1 for axis in axes)
    return {
        "points": points,
        "supports": supports,
        "union_support": columns,
        "union_support_count": len(columns),
        "coefficient_matrix": matrix,
        "rank": matrix_rank(matrix),
        "same_native_2x2_cell": same_cell,
        "clamp_flags": clamp_flags,
        "any_clamped": any(any(flags.values()) for flags in clamp_flags),
        "scope": "UNBOUND_GEOMETRY_ONLY_NOT_CURRENT_PATH_QUALIFICATION",
    }
