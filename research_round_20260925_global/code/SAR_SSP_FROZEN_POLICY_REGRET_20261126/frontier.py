"""Exact signed allocation frontiers with label access restricted to audit."""
import numpy as np


def audit(policy, posterior, incumbent, k, area, truth_area, native_hit,
          true_add, true_drop, observed):
    p = np.asarray(posterior, dtype=np.float64)
    n = len(p)
    add, drop = p[:, 0], p[:, 1]
    actual = policy.select(p, incumbent, k, area)
    for arm, observed_action in observed.items():
        if actual[arm]["action"] != observed_action:
            raise ValueError("Original target action failed exact replay")

    def index_hit(labels, sets):
        return labels[np.asarray(sets, dtype=np.intp)].sum(axis=1)

    def observed_iou(action):
        a, d = action
        hit = native_hit + np.sum(true_add[list(a)]) - np.sum(true_drop[list(d)])
        return float(hit / (area + len(a) - len(d) + truth_area - hit))

    static_hit = native_hit + np.sum(true_add[list(incumbent[0])]) - np.sum(true_drop[list(incumbent[1])])
    estimated_g = area - n + p.sum(axis=(0, 1))
    base_i = area - n + drop.sum(axis=0)
    estimated_static = base_i + policy.side_sum(add, (incumbent[0],))[0] - policy.side_sum(drop, (incumbent[1],))[0]
    static_iou = policy.iou(estimated_static, area, estimated_g)
    true_static_iou = static_hit / (area + truth_area - static_hit)
    orders_add = policy.ranking(add, True)
    orders_drop = policy.ranking(drop, False)
    quotas = sorted(range(max(0, 2*k-n), min(n, 2*k)+1), key=lambda a: (abs(a-k), a))
    chosen_a = len(observed["variable_iou"][0])
    objectives = {key: -np.inf for key in ("allocation_only", "full_menu", "full_support", "exact_geometry")}
    true_at_exact_geometry = None
    recovered = None
    per_quota_winners = 0
    selected_exact_geometry_a = None

    for a in quotas:
        d = 2*k-a
        adds = policy.side_sets(orders_add, a, incumbent[0])
        drops = policy.side_sets(orders_drop, d, incumbent[1])
        predicted_i = base_i + policy.side_sum(add, adds)[:, None] - policy.side_sum(drop, drops)[None, :]
        scores = (policy.iou(predicted_i, area+a-d, estimated_g) - static_iou).min(axis=-1)
        best_position = np.unravel_index(int(scores.argmax()), scores.shape)
        true_i = native_hit + index_hit(true_add, adds)[:, None] - index_hit(true_drop, drops)[None, :]
        oracle = true_i / (area + a-d + truth_area - true_i)
        objectives["allocation_only"] = max(objectives["allocation_only"], float(oracle[best_position]))
        objectives["full_menu"] = max(objectives["full_menu"], float(oracle.max()))
        if a == chosen_a:
            recovered = (adds[best_position[0]], drops[best_position[1]])
            rank_only = float(oracle.max())
        maximum_hit = native_hit + min(a, int(true_add.sum())) - max(0, d-(n-int(true_drop.sum())))
        objectives["full_support"] = max(
            objectives["full_support"], float(maximum_hit / (area + a-d + truth_area - maximum_hit)))
        proxy_i = static_hit + (predicted_i - estimated_static)
        proxy_union = area + a - d + truth_area - proxy_i
        if np.any(proxy_union <= 0):
            raise ValueError("Nonpositive union in the exact-geometry substitution")
        proxy_objective = (proxy_i / proxy_union - true_static_iou).min(axis=-1)
        proxy_position = np.unravel_index(int(proxy_objective.argmax()), proxy_objective.shape)
        if proxy_objective[proxy_position] > objectives["exact_geometry"]:
            objectives["exact_geometry"] = float(proxy_objective[proxy_position])
            true_at_exact_geometry = float(oracle[proxy_position])
            selected_exact_geometry_a = a
        per_quota_winners += 1

    fixed = observed_iou(observed["fixed_iou"])
    variable = observed_iou(observed["variable_iou"])
    if recovered != observed["variable_iou"] or not (
        objectives["full_support"] + 1e-12 >= objectives["full_menu"] >= max(rank_only, objectives["allocation_only"])
        and min(rank_only, objectives["allocation_only"]) + 1e-12 >= variable
    ):
        raise ValueError("Frontier constraints or per-quota target parity failed")
    objectives.update(exact_geometry=true_at_exact_geometry, rank_only=rank_only,
                      fixed=fixed, variable=variable, feasible_quotas=per_quota_winners,
                      chosen_adds=chosen_a, exact_geometry_adds=selected_exact_geometry_a)
    return objectives
