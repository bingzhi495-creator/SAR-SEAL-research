"""Reciprocal mask-IoU matching with an ambiguity margin at both endpoints."""
import numpy as np


def reciprocal_matches(iou, min_iou=.50, margin=.10):
    values = np.asarray(iou, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Mask IoUs must be a finite matrix in [0,1]")
    anchors, donors = values.shape
    if not anchors or not donors:
        return {}, ('no_donor',) * anchors
    row_order = np.argsort(-values, axis=1, kind="stable")
    column_order = np.argsort(-values, axis=0, kind="stable")
    matches = {}
    status = []
    for anchor, order in enumerate(row_order):
        donor = int(order[0])
        best = values[anchor, donor]
        if best < min_iou:
            status.append("below_iou")
            continue
        if column_order[0, donor] != anchor:
            status.append("not_reciprocal")
            continue
        row_second = values[anchor, order[1]] if donors > 1 else 0.
        column_second = values[column_order[1, donor], donor] if anchors > 1 else 0.
        if best - max(row_second, column_second) + 1e-12 < margin:
            status.append("ambiguous")
            continue
        matches[anchor] = donor
        status.append("accepted")
    return matches, tuple(status)
