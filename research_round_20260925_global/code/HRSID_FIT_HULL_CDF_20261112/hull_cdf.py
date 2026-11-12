"""Compare matched polygon translation rules using projected foreground mass."""

import math
from statistics import median

import numpy as np
from shapely import area, box, intersection
from shapely.affinity import affine_transform


MENU = (-1.5, -1.0, -.5, 0.0, .5, 1.0, 1.5)
QUANTILES = np.array([.25, .50, .75])
TIE_ATOL = 1e-12


def _best_shift(costs):
    best = min(costs)
    tied = (shift for shift, cost in zip(MENU, costs, strict=True)
            if math.isclose(cost, best, rel_tol=0, abs_tol=TIE_ATOL))
    return min(tied, key=lambda shift: (abs(shift), shift))


def _nearest_shift(delta):
    return _best_shift(tuple(abs(shift - delta) for shift in MENU))


def _cdf(polygon, x):
    """Exact half-plane mass at each grid location in aligned coordinates."""
    x = np.asarray(x, dtype=np.float64)
    x_min, y_min, x_max, y_max = polygon.bounds
    cdf = (x >= x_max).astype(np.float64)
    internal = (x > x_min) & (x < x_max)
    if np.any(internal):
        boundaries = x[internal]
        clips = box(x_min, y_min, boundaries, y_max)
        cdf[internal] = np.clip(area(intersection(polygon, clips)) / polygon.area, 0.0, 1.0)
    return cdf


def _quantiles(polygon):
    low = np.full(QUANTILES.shape, polygon.bounds[0], dtype=np.float64)
    high = np.full(QUANTILES.shape, polygon.bounds[2], dtype=np.float64)
    for _ in range(56):
        midpoint = low + (high - low) / 2
        below = _cdf(polygon, midpoint) < QUANTILES
        low = np.where(below, midpoint, low)
        high = np.where(below, high, midpoint)
    return high


def projected_choices(gt, pred, axis_xy, step=.125):
    """Choose signed native-pixel shifts from fixed area-preserving translations.

    GT and prediction are valid positive-area Shapely polygonal footprints.
    Moving the prediction by returned shift times unit axis aligns it toward
    the GT. Projected CDF values use exact planar polygon intersections.
    The Wasserstein distance integral is approximated by grid trapezoids.
    """
    axis = np.asarray(axis_xy, dtype=np.float64)
    if axis.shape != (2,) or not np.isfinite(axis).all() or not np.any(axis):
        raise ValueError("Expected a finite nonzero planar axis")
    if not math.isfinite(step) or step <= 0:
        raise ValueError("Expected a positive finite integration step")
    for geom in (gt, pred):
        if (geom.geom_type not in ("Polygon", "MultiPolygon") or not geom.is_valid
                or not math.isfinite(geom.area) or geom.area <= 0):
            raise ValueError("Expected valid positive-area polygonal footprints")

    ux, uy = axis / math.hypot(*axis)
    matrix = [ux, uy, -uy, ux, 0.0, 0.0]
    aligned_gt = affine_transform(gt, matrix)
    aligned_pred = affine_transform(pred, matrix)
    x_start = min(aligned_gt.bounds[0], aligned_pred.bounds[0] + MENU[0])
    x_stop = max(aligned_gt.bounds[2], aligned_pred.bounds[2] + MENU[-1])
    grid = np.arange(math.floor(x_start / step), math.ceil(x_stop / step) + 1, dtype=np.float64) * step

    gt_mass = _cdf(aligned_gt, grid)
    shifted_boundaries = grid[None, :] - np.asarray(MENU)[:, None]
    unique, backmap = np.unique(shifted_boundaries, return_inverse=True)
    pred_mass = _cdf(aligned_pred, unique)[backmap].reshape(shifted_boundaries.shape)
    costs = tuple(map(float, np.trapz(np.abs(pred_mass - gt_mass[None, :]), x=grid, axis=1)))

    quantile_deltas = tuple(map(float, _quantiles(aligned_gt) - _quantiles(aligned_pred)))
    mean_delta = aligned_gt.centroid.x - aligned_pred.centroid.x
    median_delta = quantile_deltas[1]
    q3_mean_delta = math.fsum(quantile_deltas) / len(quantile_deltas)
    q3_median_delta = median(quantile_deltas)
    return {
        "menu": MENU,
        "step": step,
        "w1_costs": costs,
        "w1_shift": _best_shift(costs),
        "mean_shift": _nearest_shift(mean_delta),
        "median_shift": _nearest_shift(median_delta),
        "q3_mean_shift": _nearest_shift(q3_mean_delta),
        "q3_median_shift": _nearest_shift(q3_median_delta),
        "mean_delta": mean_delta,
        "median_delta": median_delta,
        "q3_mean_delta": q3_mean_delta,
        "q3_median_delta": q3_median_delta,
        "quantile_deltas": quantile_deltas,
    }
