"""Exact overlap with the scaled union of pre-resize foreground pixel cells."""

import numpy as np


def _overlap_axis(first, last, input_size, output_size):
    scale = output_size / input_size
    lo = first * output_size // input_size
    hi = (last * output_size + input_size - 1) // input_size
    edges = np.arange(first, last + 1, dtype=np.float64) * scale
    cells = np.arange(lo, hi, dtype=np.float64)
    overlap = np.minimum(cells[:, None] + 1., edges[None, 1:]) - np.maximum(cells[:, None], edges[None, :-1])
    return slice(lo, hi), np.maximum(overlap, 0.)


def exact_footprint_scores(source, candidates):
    source = np.asarray(source, dtype=bool)
    native, center = (np.asarray(item, dtype=bool) for item in candidates)
    if source.ndim != 2 or native.ndim != 2 or native.shape != center.shape or min(native.shape) < 1:
        raise ValueError("Expected a source mask and two matched output masks")
    yy, xx = np.nonzero(source)
    if not len(yy):
        raise ValueError("Continuous footprint needs nonempty source support")
    h, w = source.shape
    out_h, out_w = native.shape
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    ys, wy = _overlap_axis(y0, y1, h, out_h)
    xs, wx = _overlap_axis(x0, x1, w, out_w)
    target_area = float(len(yy) * out_h * out_w / (h * w))
    ref = wy @ source[y0:y1, x0:x1].astype(np.float64) @ wx.T
    if not np.isclose(ref.sum(), target_area, rtol=1e-12, atol=1e-10):
        raise ValueError("The footprint must conserve the continuous reference area")
    rows = {"reference_area": target_area}
    for name, candidate in (("native", native), ("center", center)):
        cy, cx = np.nonzero(candidate)
        n = len(cy)
        intersection = float(ref[candidate[ys, xs]].sum())
        rows[name] = dict(
            iou=intersection / (target_area + n - intersection),
            signed_y_px=float((cy.mean() + .5) * h / out_h - (yy.mean() + .5)) if n else None,
            signed_x_px=float((cx.mean() + .5) * w / out_w - (xx.mean() + .5)) if n else None,
        )
    return rows

