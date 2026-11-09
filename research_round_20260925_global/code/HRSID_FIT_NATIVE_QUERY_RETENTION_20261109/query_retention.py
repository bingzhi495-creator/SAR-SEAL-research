"""Compare candidate retention policies on one fixed query-to-GT overlap matrix."""
from pathlib import Path
import runpy

import numpy as np


# The earlier DEV diagnostic defines the same matching and threshold convention.
_COVERAGE_PATH = (
    Path(__file__).resolve().parent.parent
    / "HRSID_DEV705_NATIVE_CANDIDATE_COVERAGE_20261109"
    / "candidate_coverage.py"
)
_COVERAGE = runpy.run_path(str(_COVERAGE_PATH))
_GRAPH_COUNTS = _COVERAGE["graph_counts"]
_THRESHOLDS = _COVERAGE["THRESHOLDS"]


def analyze_image(overlap, ship_scores, native_ids, empty):
    """Describe coverage from frozen masks and fixed, GT-free query policies."""
    overlap = np.asarray(overlap, dtype=np.float64)
    scores = np.asarray(ship_scores, dtype=np.float64)
    native = np.asarray(native_ids)
    empty = np.asarray(empty)
    if overlap.ndim != 2:
        raise ValueError("Expected query-to-GT overlap matrix")
    query_count, gt_count = overlap.shape
    if scores.shape != (query_count,) or empty.shape != (query_count,) or empty.dtype.kind != "b":
        raise ValueError("Expected one finite ship score and one Boolean empty flag per query")
    if not np.isfinite(overlap).all() or (overlap < 0).any() or (overlap > 1).any():
        raise ValueError("Expected finite mask IoU in [0, 1]")
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("Expected finite ship scores in [0, 1]")
    if np.any(overlap[empty] != 0):
        raise ValueError("Empty predicted masks must have zero IoU")
    if native.ndim != 1 or (native.size and native.dtype.kind not in "iu"):
        raise ValueError("Expected one-dimensional integer native query IDs")
    if (native < 0).any() or (native >= query_count).any() or np.unique(native).size != native.size:
        raise ValueError("Expected unique native query IDs within the query matrix")
    native = native.astype(np.intp, copy=False)

    ship_order = np.argsort(-scores, kind="stable")
    empty_in_ship_order = empty[ship_order]
    nonempty_order = np.concatenate((ship_order[~empty_in_ship_order], ship_order[empty_in_ship_order]))
    policies = {
        "native": native,
        "ship_same_count": ship_order[:native.size],
        "nonempty_same_count": nonempty_order[:native.size],
        "all_queries": ship_order,
    }
    native_best = overlap[native].max(axis=0, initial=0)
    near_strict = (native_best >= .75) & (native_best < .90)
    report = {}
    for name, selected in policies.items():
        selected_overlap = overlap[selected]
        by_threshold = {}
        for threshold in _THRESHOLDS:
            independent, maximum, greedy = _GRAPH_COUNTS(selected_overlap, threshold)
            by_threshold[f"{threshold:.2f}"] = {
                "independent": independent,
                "maximum_one_to_one": maximum,
                "native_greedy": greedy,
            }
        report[name] = {
            "candidates": int(selected.size),
            "empty_candidates": int(empty[selected].sum()),
            "thresholds": by_threshold,
            "strict_reach": (selected_overlap >= .90).any(axis=0).tolist(),
        }

    return {
        "eligible_gt": gt_count,
        "native_near_strict": near_strict.tolist(),
        "policies": report,
    }
