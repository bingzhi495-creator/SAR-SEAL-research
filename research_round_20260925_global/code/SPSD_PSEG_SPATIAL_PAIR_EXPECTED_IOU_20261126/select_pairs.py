"""Select the exact joint expected-IoU winner on the original signed menu."""

import numpy as np


ARMS = ("local_exact", "local_fixed", "nonspatial_exact", "nonspatial_fixed")


def flags(n, actions):
    selected = np.zeros((len(actions), n, 2), dtype=bool)
    selected[:, :, 1] = True
    for row, (a, d) in zip(selected, actions, strict=True):
        row[np.asarray(a, dtype=np.intp), 0] = True
        row[np.asarray(d, dtype=np.intp), 1] = False
    return selected.reshape(len(actions), 2 * n)


def choose(writer, independent, prepare, posterior, static, k, area, graphs, kappas):
    p = np.asarray(posterior, dtype=np.float64)
    n = len(p)
    writer.need(p.shape == (n, 2, 2) and 0 < k <= n <= area and area > 2*k
                and set(graphs) == set(kappas) == {"local", "nonspatial"}, "Frozen pair-menu inputs")
    add, drop = p[:, 0], p[:, 1]
    var_add, var_drop = add*(1-add), drop*(1-drop)
    outside = area - n
    base_i = outside + drop.sum(axis=0)
    static_i = base_i + writer.side_sum(add, (static[0],))[0] - writer.side_sum(drop, (static[1],))[0]
    ref_area = outside + p.sum(axis=(0, 1))
    evals = {graph: prepare(p.reshape(-1, 2), pairs, kappas[graph], outside) for graph, pairs in graphs.items()}
    held = flags(n, (static,))[0]
    static_u = {graph: e.utility(held, area) for graph, e in evals.items()}
    rank_add, rank_drop = writer.ranking(add, True), writer.ranking(drop, False)
    quotas = sorted(range(max(0, 2*k-n), min(n, 2*k)+1), key=lambda a: (abs(a-k), a))
    options = []
    greatest_lower = {graph: -np.inf for graph in evals}
    fixed_lower = {}
    for add_count in quotas:
        drop_count = 2*k - add_count
        edited = area + add_count - drop_count
        adds = writer.side_sets(rank_add, add_count, static[0])
        drops = writer.side_sets(rank_drop, drop_count, static[1])
        actions = tuple((a, d) for a in adds for d in drops)
        selected = flags(n, actions)
        i = (base_i + writer.side_sum(add, adds)[:, None, :] -
             writer.side_sum(drop, drops)[None, :, :]).reshape(-1, 2)
        mean_union = edited + ref_area - i
        variance = (var_add.sum(axis=0) - writer.side_sum(var_add, adds)[:, None, :] +
                    writer.side_sum(var_drop, drops)[None, :, :]).reshape(-1, 2)
        lo, hi, _ = independent.moment_screen(i, np.maximum(variance, 0), mean_union, edited,
                                                n-add_count+drop_count)
        envelopes = {}
        for graph, evaluator in evals.items():
            gap = evaluator.bound(selected, edited)
            lower = (lo - gap - static_u[graph]).min(axis=-1)
            upper = (hi + gap - static_u[graph]).min(axis=-1)
            envelopes[graph] = lower, upper
            greatest_lower[graph] = max(greatest_lower[graph], float(lower.max()))
            if add_count == k:
                fixed_lower[graph] = float(lower.max())
        options.append((add_count, edited, actions, selected, i, envelopes))

    winners = {}
    count = sum(len(option[2]) for option in options)
    counts = {graph: dict(evaluated=0, menu_actions=count) for graph in evals}
    for add_count, edited, actions, selected, intersection, envelopes in options:
        for graph, (lower, upper) in envelopes.items():
            could_win = upper >= greatest_lower[graph] - independent.TOL
            could_win_fixed = upper >= fixed_lower[graph] - independent.TOL if add_count == k else np.zeros_like(could_win)
            for ordinal in np.flatnonzero(could_win | could_win_fixed):
                gain = evals[graph].utility(selected[ordinal], edited) - static_u[graph]
                score = float(gain.min())
                writer.need(lower[ordinal]-independent.TOL <= score <= upper[ordinal]+independent.TOL,
                            "Joint utility lies in the certified moment/covariance interval")
                for arm, eligible in zip((f"{graph}_exact", f"{graph}_fixed"),
                                         (could_win[ordinal], could_win_fixed[ordinal]), strict=True):
                    if eligible and (arm not in winners or score > winners[arm]["score"]):
                        winners[arm] = dict(action=actions[ordinal], score=score, iou_gain=gain.tolist(),
                                            intersection_gain=(intersection[ordinal]-static_i).tolist())
                counts[graph]["evaluated"] += 1
    writer.need(set(winners) == set(ARMS), "Joint search covers both nested budgets")
    for graph in evals:
        unrestricted = winners[f"{graph}_exact"]["score"]
        fixed = winners[f"{graph}_fixed"]["score"]
        writer.need(unrestricted + independent.TOL >= fixed >= -independent.TOL,
                    "Nested joint menu dominates incumbent at the same total")
        counts[graph]["pruned"] = count - counts[graph]["evaluated"]
    return winners, counts
