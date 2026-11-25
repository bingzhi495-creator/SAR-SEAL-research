"""Native-grid inward rings following Cheng et al.'s square-erosion Boundary IoU API."""

from operator import index

import numpy as np


def _binary_mask(mask):
    array = np.asarray(mask)
    if array.ndim != 2 or not all(array.shape):
        raise ValueError("Expected a nonempty two-dimensional canvas")
    if array.dtype == np.bool_:
        return array
    if not np.issubdtype(array.dtype, np.integer) or not np.all((array == 0) | (array == 1)):
        raise ValueError("Expected a binary raster mask")
    return array.astype(bool)


def boundary_ring(mask, width):
    """Return the inward ring at an integer pixel width with background beyond the image."""
    source = _binary_mask(mask)
    radius = index(width)
    if radius < 1:
        raise ValueError("Boundary width must be positive")
    rows, cols = source.shape
    side = 2 * radius + 1
    if side > min(rows, cols):
        return source.copy()

    padded = np.pad(source, radius, constant_values=False)
    row_eroded = padded[:, :cols].copy()
    for offset in range(1, side):
        row_eroded &= padded[:, offset:offset + cols]
    eroded = row_eroded[:rows].copy()
    for offset in range(1, side):
        eroded &= row_eroded[offset:offset + rows]
    return source & ~eroded


def ring_iou(prediction_ring, reference_ring):
    """Return inward-ring IoU, undefined (NaN) if both rings are empty."""
    prediction = _binary_mask(prediction_ring)
    reference = _binary_mask(reference_ring)
    if prediction.shape != reference.shape:
        raise ValueError("Prediction and reference must share the native canvas")
    union = np.count_nonzero(prediction | reference)
    return np.count_nonzero(prediction & reference) / union if union else float("nan")


def boundary_iou(prediction, reference, width):
    """Return Boundary IoU at the stated width on a common native raster grid."""
    return ring_iou(boundary_ring(prediction, width), boundary_ring(reference, width))
