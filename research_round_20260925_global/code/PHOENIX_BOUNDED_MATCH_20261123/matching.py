"""Exact, distance-capped assignment on the frozen native endpoint support."""
from dataclasses import dataclass
from heapq import heappop, heappush

import numpy as np
from scipy.optimize import linear_sum_assignment


BINARY32_SCALE = 1 << 149


@dataclass(slots=True)
class Edge:
    node: int
    reverse: int
    remaining: bool
    cost: int


def connect(graph, source, target, cost):
    arc = Edge(target, len(graph[target]), True, cost)
    graph[target].append(Edge(source, len(graph[source]), False, -cost))
    graph[source].append(arc)
    return arc


def augment(graph, source, sink, potential):
    """One unit of flow using shortest paths in reduced integer costs."""
    distance = [None] * len(graph)
    predecessor = [None] * len(graph)
    distance[source] = 0
    frontier = [(0, source)]
    while frontier:
        cost, tail = heappop(frontier)
        if cost != distance[tail]:
            continue
        for ordinal, arc in enumerate(graph[tail]):
            if not arc.remaining:
                continue
            head = arc.node
            candidate = cost + arc.cost + potential[tail] - potential[head]
            if distance[head] is None or candidate < distance[head]:
                distance[head] = candidate
                predecessor[head] = (tail, ordinal)
                heappush(frontier, (candidate, head))
    if predecessor[sink] is None:
        raise ValueError("Historical fixed pairs should make the exact quota feasible")
    for node, increment in enumerate(distance):
        if increment is not None:
            potential[node] += increment
    node = sink
    while node != source:
        tail, ordinal = predecessor[node]
        arc = graph[tail][ordinal]
        arc.remaining = False
        graph[node][arc.reverse].remaining = True
        node = tail


def best_matching(distances, cap, count, add, drop, add_rank, drop_rank):
    size = len(add)
    source, sink = size * 2, size * 2 + 1
    graph = [[] for _ in range(sink + 1)]
    for i in range(size):
        connect(graph, source, i, 0)
        connect(graph, size + i, sink, 0)

    # One pair supplies one bit per side. Disjointness makes the sum
    # a lexicographic endpoint-set code in historical probability order.
    tie_scale = 1 << (2 * size)
    displacement_scale = count * cap + 1
    add_bits = [1 << (2 * size - 1 - int(rank)) for rank in add_rank]
    drop_bits = [1 << (size - 1 - int(rank)) for rank in drop_rank]
    candidate_edges = []
    for i, row in enumerate(distances):
        for j in np.flatnonzero(row <= cap):
            j = int(j)
            posterior_cost = BINARY32_SCALE - add[i] + drop[j]
            set_cost = tie_scale - add_bits[i] - drop_bits[j]
            cost = (tie_scale * posterior_cost + set_cost) * displacement_scale + int(row[j])
            edge = connect(graph, i, size + j, cost)
            candidate_edges.append((i, j, edge))

    potential = [0] * len(graph)
    for _ in range(count):
        augment(graph, source, sink, potential)
    pairs = [(i, j) for i, j, edge in candidate_edges if not edge.remaining]
    return np.asarray(pairs, dtype=np.int64).T


def bounded_endpoints(coords, probabilities, k, width, paired_add, paired_drop, ia, idrop):
    """Preserve k native writes per sign and the fixed pairing's largest displacement."""
    paired_add, paired_drop, ia, idrop = map(np.asarray, (paired_add, paired_drop, ia, idrop))
    size = len(coords)
    if any(values.shape != (k,) or np.unique(values).size != k
           or np.any((values < 0) | (values >= size)) for values in (paired_add, paired_drop, ia, idrop)):
        raise ValueError("Endpoint ordinals must be distinct and satisfy the inherited quota")
    if not (0 < k <= size and width > 0
            and np.array_equal(probabilities, probabilities.astype(np.float32).astype(np.float64))
            and np.all((0 <= probabilities) & (probabilities <= 1))):
        raise ValueError("Expected original binary32 probabilities and a positive feasible quota")

    ay, ax = np.divmod(coords[:, 0].astype(np.int64), width)
    dy, dx = np.divmod(coords[:, 1].astype(np.int64), width)
    distances = (ay[:, None] - dy[None, :]) ** 2 + (ax[:, None] - dx[None, :]) ** 2
    fixed_distances = distances[paired_add, paired_drop]
    cap = int(fixed_distances.max())
    adds = np.argsort(-probabilities[:, 0], kind="stable")
    drops = np.argsort(probabilities[:, 1], kind="stable")
    if set(ia) != set(adds[:k]) or set(idrop) != set(drops[:k]):
        raise ValueError("Independent ranks differ from their historical sets")

    independent_distances = distances[np.ix_(ia, idrop)]
    try:
        free_rows, free_cols = linear_sum_assignment(np.where(independent_distances <= cap, independent_distances, np.inf))
    except ValueError:
        independent_feasible = False
    else:
        independent_feasible = True

    add = [int(float(p) * BINARY32_SCALE) for p in probabilities[:, 0]]
    drop = [int(float(p) * BINARY32_SCALE) for p in probabilities[:, 1]]
    if independent_feasible:
        chosen_add, chosen_drop = ia[free_rows], idrop[free_cols]
    else:
        chosen_add, chosen_drop = best_matching(distances, cap, k, add, drop, np.argsort(adds), np.argsort(drops))

    lengths = distances[chosen_add, chosen_drop]
    if (len(lengths) != k or np.unique(chosen_add).size != k or np.unique(chosen_drop).size != k
            or np.any(lengths > cap)):
        raise ValueError("The computed assignment violates its fixed cardinality or distance cap")
    fixed_gain = sum(add[a] - drop[d] for a, d in zip(paired_add, paired_drop, strict=True))
    bounded_gain = sum(add[a] - drop[d] for a, d in zip(chosen_add, chosen_drop, strict=True))
    independent_gain = sum(add[a] - drop[d] for a, d in zip(ia, idrop, strict=True))
    if not fixed_gain <= bounded_gain <= independent_gain:
        raise ValueError("Expected exact posterior objective fixed <= bounded <= independent")
    rows, cols = linear_sum_assignment(independent_distances)
    stats = dict(
        independent_feasible=int(independent_feasible),
        bounded_gain_gt_fixed=int(bounded_gain > fixed_gain),
        independent_gain_gt_bounded=int(independent_gain > bounded_gain),
        cap_squared_sum=cap,
        fixed_transport_squared_sum=int(fixed_distances.sum()),
        bounded_transport_squared_sum=int(lengths.sum()),
        independent_min_transport_squared_sum=int(independent_distances[rows, cols].sum()),
    )
    return chosen_add, chosen_drop, stats
