"""Constrain signed allocation using image-plane transverse widening."""
import numpy as np


NEW_ARMS = ("transverse_iou", "area_iou", "longitudinal_iou")
REFERENCE_ARMS = ("fixed_iou", "variable_iou")
TALLY_ARMS = (*REFERENCE_ARMS, *NEW_ARMS)


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def one_component(points):
    unseen = set(map(tuple, points.tolist()))
    queue = [unseen.pop()]
    while queue:
        y, x = queue.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                neighbor = (y + dy, x + dx)
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
    return not unseen


def native_axes(points):
    center = points.mean(axis=0)
    centered = points - center
    eigenvalues, axes = np.linalg.eigh(centered.T @ centered / len(points))
    short, long = np.maximum(eigenvalues, 0)
    aspect = float(np.sqrt(long / short)) if short > 0 else None
    if long <= 0 or long < 4.0 * short:
        return centered, center, axes, aspect, "aspect"
    if not one_component(points):
        return centered, center, axes, aspect, "disconnected"
    return centered, center, axes, aspect, "eligible"


def quota_winners(writer, posterior, incumbent, k, area, quotas):
    """Recover the one original maximin winner for each feasible allocation."""
    p = np.asarray(posterior, dtype=np.float64)
    add, drop = p[:, 0], p[:, 1]
    outside = area - len(p)
    estimated_area = outside + p.sum(axis=(0, 1))
    initial_hit = outside + drop.sum(axis=0)
    static_hit = (
        initial_hit
        + writer.side_sum(add, (incumbent[0],))[0]
        - writer.side_sum(drop, (incumbent[1],))[0]
    )
    static_iou = writer.iou(static_hit, area, estimated_area)
    add_order = writer.ranking(add, True)
    drop_order = writer.ranking(drop, False)
    winners = {}
    for a in quotas:
        d = 2 * k - a
        adds = writer.side_sets(add_order, a, incumbent[0])
        drops = writer.side_sets(drop_order, d, incumbent[1])
        hits = (initial_hit
                + writer.side_sum(add, adds)[:, None, :]
                - writer.side_sum(drop, drops)[None, :, :])
        change = writer.iou(hits, area + a - d, estimated_area) - static_iou
        score = change.min(axis=-1)
        j, h = np.unravel_index(int(score.argmax()), score.shape)
        winners[a] = dict(
            action=(adds[j], drops[h]), score=float(score[j, h]),
            iou_gain=change[j, h].tolist(),
            intersection_gain=(hits[j, h] - static_hit).tolist(),
        )
    return winners


def projected_variances(stock_projection, points_projection, action):
    """Variance about the edited mask's own centroid on each fixed axis."""
    additions, removals = (np.asarray(side, dtype=np.intp) for side in action)
    add = points_projection[additions, 0]
    drop = points_projection[removals, 1]
    count, total, total_square = stock_projection
    count_after = count + len(add) - len(drop)
    mean = (total + add.sum(axis=0) - drop.sum(axis=0)) / count_after
    second = (total_square + np.square(add).sum(axis=0) - np.square(drop).sum(axis=0)) / count_after
    variance = second - np.square(mean)
    need(np.isfinite(variance).all() and (variance >= -1e-12).all(),
         "Finite central projected moment on the fixed native axes")
    return np.maximum(variance, 0)


def within_width_cap(value, baseline):
    tolerance = 1e-12 * max(1.0, abs(baseline), abs(value))
    return value <= baseline + tolerance


def select(writer, posterior, incumbent, k, native_plane, coordinates):
    """Keep endpoint predictions and total flips fixed across the quota controls."""
    stock = np.asarray(native_plane)
    coords = np.asarray(coordinates)
    area = int(stock.sum())
    n = len(posterior)
    need(stock.ndim == 2 and stock.dtype == bool and coords.shape == (n, 2)
         and coords.dtype.kind in "iu" and len(np.unique(coords)) == coords.size
         and np.all((0 <= coords) & (coords < stock.size))
         and not stock.ravel()[coords[:, 0]].any() and stock.ravel()[coords[:, 1]].all(),
         "Original signed native endpoint support")
    incumbent = tuple(tuple(sorted(int(x) for x in side)) for side in incumbent)
    original = writer.select(posterior, incumbent, k, area)
    quotas = sorted(range(max(0, 2*k-n), min(n, 2*k)+1), key=lambda a: (abs(a-k), a))
    points = np.column_stack(stock.nonzero())
    centered, center, axes, aspect, disposition = native_axes(points)
    eligible = disposition == "eligible"
    audit = dict(eligible=eligible, disposition=disposition, axis_ratio=aspect,
                 feasible_quotas=len(quotas))
    if not eligible:
        audit.update(admitted_counts=dict.fromkeys(NEW_ARMS, len(quotas)),
                     admitted_quotas=dict.fromkeys(NEW_ARMS, quotas),
                     width_by_arm={}, longitudinal_variance_by_arm={})
        return dict(original, **dict.fromkeys(NEW_ARMS, original["variable_iou"])), audit

    winners = quota_winners(writer, posterior, incumbent, k, area, quotas)
    free = max(winners.values(), key=lambda result: result["score"])
    need(winners[k] == original["fixed_iou"] and free == original["variable_iou"],
         "Every fixed and free solution of the per-quota reconstruction matches the source rule")
    projected = centered @ axes
    support = np.stack(np.unravel_index(coords, stock.shape), axis=-1)
    point_projection = (support - center) @ axes
    sufficient = area, projected.sum(axis=0), np.square(projected).sum(axis=0)
    spreads = {a: projected_variances(sufficient, point_projection, value["action"])
               for a, value in winners.items()}
    transverse = [a for a in quotas if within_width_cap(spreads[a][0], spreads[k][0])]
    size = len(transverse)
    area_control = quotas[:size]
    longitudinal = [k, *sorted((a for a in quotas if a != k),
                               key=lambda a: spreads[a][1])[:size-1]]
    admitted = dict(zip(NEW_ARMS, map(set, (transverse, area_control, longitudinal)), strict=True))
    decisions = dict(original)
    for arm, options in admitted.items():
        need(len(options) == size and k in options, "Control retains B and identical menu size")
        decisions[arm] = max((winners[a] for a in quotas if a in options),
                             key=lambda result: result["score"])
        need(sum(map(len, decisions[arm]["action"])) == 2*k,
             "Exact common sum of added and removed native pixels")
    chosen = {a: len(decisions[a]["action"][0]) for a in TALLY_ARMS}
    audit.update(
        admitted_counts={arm: len(options) for arm, options in admitted.items()},
        admitted_quotas={arm: [a for a in quotas if a in options] for arm, options in admitted.items()},
        width_by_arm={arm: float(spreads[a][0]) for arm, a in chosen.items()},
        longitudinal_variance_by_arm={arm: float(spreads[a][1]) for arm, a in chosen.items()},
    )
    return decisions, audit

