"""UNBOUND_DRAFT: pure fixed-candidate, exact-K binary action diagnostics.

Scope: ONE fixed, already matched instance with nonempty Boolean ground truth.
This module is not a runtime/model adapter and does not read any scientific data.
The caller supplies every array, including a candidate C fixed independently of
selection. That provenance is a caller obligation; arrays alone cannot certify it.
Thresholding and its strict/inclusive comparison convention belong to the caller;
this pure Boolean module neither sets nor changes a threshold convention.
No COCO AP claim, cross-instance/global upper bound, or task efficacy claim is
made. An oracle here is local to this fixed instance and the supplied eligibility.
Synthetic self-tests do not qualify this file for scientific execution.

Ranking uses the r=0.5 category order repair FN (+1), repair FP (+0.5),
no-op (0), add FP (-0.5), delete TP (-1). Scores are action-ordering scores,
not numerical IoU changes. Within one category ties use ascending flat index.
For binary replacements this order maximizes IoU among all EXACT-K eligible
subsets: an exchange of lower-ranked for higher-ranked actions cannot lower
IoU, taking account of the FN/FP that make the exchange feasible. Enumerated
synthetic tests independently verify this claim for the tested inputs.

Exact-K includes no-op fillers and can eventually force harmful actions when
C differs from G. Thus fixed-C oracle IoU need not increase with K. The controlled
GT-action oracle cannot harm pixels, so its IoU is nondecreasing with K.
Only Python standard-library modules and NumPy are used. No files are read or
written by this module, and no model, dataset, or runtime is loaded.
"""

from __future__ import annotations

import argparse
import itertools
import json
from typing import Any

import numpy as np

STATUS = "UNBOUND_DRAFT"
SYNTHETIC_LABEL = "SYNTHETIC_ONLY"
SCORE_R = 0.5
ACTION_NAMES = ("repair_fn", "repair_fp", "noop", "add_fp", "delete_tp")
ACTION_SCORES = (1.0, SCORE_R, 0.0, -SCORE_R, -1.0)
SCOPE = (
    "one fixed matched instance; nonempty GT; supplied Boolean masks; "
    "fixed candidate independent of selection; no COCO AP or global upper-bound claim"
)


def _validate_arrays(**arrays: np.ndarray) -> tuple[int, ...]:
    shape = None
    for name, array in arrays.items():
        if not isinstance(array, np.ndarray) or array.dtype != np.dtype(np.bool_):
            raise TypeError(f"{name} must be a NumPy Boolean array; no implicit casts")
        if array.ndim == 0:
            raise ValueError(f"{name} must have at least one axis")
        if shape is None:
            shape = array.shape
        elif array.shape != shape:
            raise ValueError(f"{name} shape {array.shape} differs from {shape}")
    assert shape is not None
    return shape


def _validate_budget(K: int, eligible_count: int) -> int:
    if isinstance(K, (bool, np.bool_)) or not isinstance(K, (int, np.integer)):
        raise TypeError("K must be a scalar integer, excluding Boolean values")
    K = int(K)
    if K < 0 or K > eligible_count:
        raise ValueError("K must satisfy 0 <= K <= number of eligible pixels")
    return K


def _validate_problem(
    A: np.ndarray, C: np.ndarray, G: np.ndarray, Omega: np.ndarray, K: int
) -> int:
    _validate_arrays(A=A, C=C, G=G, Omega=Omega)
    if not np.any(G):
        raise ValueError("G must be nonempty for this single-instance IoU diagnostic")
    return _validate_budget(K, int(np.count_nonzero(Omega)))


def _action_masks(
    A: np.ndarray, C: np.ndarray, G: np.ndarray, selected: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        "repair_fn": selected & ~A & C & G,
        "repair_fp": selected & A & ~C & ~G,
        "noop": selected & (A == C),
        "add_fp": selected & ~A & C & ~G,
        "delete_tp": selected & A & ~C & G,
    }


def apply_fixed_candidate(
    A: np.ndarray, C: np.ndarray, selected: np.ndarray
) -> np.ndarray:
    """Return A outside selected and C inside selected, without input mutation."""
    _validate_arrays(A=A, C=C, selected=selected)
    output = A.copy()
    output[selected] = C[selected]
    return output


def exact_k_action_oracle(
    A: np.ndarray, C: np.ndarray, G: np.ndarray, Omega: np.ndarray, K: int
) -> np.ndarray:
    """Select K distinct eligible actions using known GT and fixed candidate C.

    The returned mask preserves input shape. K=0 and an empty native A are valid.
    Eligible no-ops are included when required to meet the exact budget.
    """
    K = _validate_problem(A, C, G, Omega, K)
    selected = np.zeros(A.shape, dtype=np.bool_)
    if K == 0:
        return selected
    indices = np.flatnonzero(Omega)
    scores = np.zeros(A.size, dtype=np.float64)
    actions = _action_masks(A, C, G, Omega)
    for name, score in zip(ACTION_NAMES, ACTION_SCORES):
        scores[actions[name].ravel()] = score
    # flatnonzero is ascending; stable sorting preserves flat-index score ties.
    order = np.argsort(-scores[indices], kind="stable")
    selected.ravel()[indices[order[:K]]] = True
    return selected


def at_most_k_repair_oracle(
    A: np.ndarray, C: np.ndarray, G: np.ndarray, Omega: np.ndarray, K: int
) -> np.ndarray:
    """Select up to K correct repairs; a constraint diagnosis, not exact-K control.

    No-op fillers and harmful actions are omitted. Repair FP can leave IoU
    unchanged when intersection is zero, but it still corrects a native error.
    """
    K = _validate_problem(A, C, G, Omega, K)
    actions = _action_masks(A, C, G, Omega)
    repair_count = int(np.count_nonzero(actions["repair_fn"])) + int(
        np.count_nonzero(actions["repair_fp"])
    )
    return exact_k_action_oracle(A, C, G, Omega, min(K, repair_count))


def _branch_metrics(
    A: np.ndarray, candidate: np.ndarray, G: np.ndarray, selected: np.ndarray
) -> dict[str, Any]:
    output = apply_fixed_candidate(A, candidate, selected)
    intersection = int(np.count_nonzero(output & G))
    union = int(np.count_nonzero(output | G))
    counts = {
        name: int(np.count_nonzero(mask))
        for name, mask in _action_masks(A, candidate, G, selected).items()
    }
    return {
        "iou": intersection / union,
        "intersection": intersection,
        "union": union,
        "fp": int(np.count_nonzero(output & ~G)),
        "fn": int(np.count_nonzero(~output & G)),
        "selected_count": int(np.count_nonzero(selected)),
        "selected_repair_fn": counts["repair_fn"],
        "selected_repair_fp": counts["repair_fp"],
        "selected_repairs": counts["repair_fn"] + counts["repair_fp"],
        "selected_add_fp": counts["add_fp"],
        "selected_delete_tp": counts["delete_tp"],
        "selected_harms": counts["add_fp"] + counts["delete_tp"],
        "selected_noops": counts["noop"],
        "selected_mask": selected.copy(),
        "output_mask": output,
    }


def diagnose_fixed_candidate(
    A: np.ndarray,
    C: np.ndarray,
    G: np.ndarray,
    Omega: np.ndarray,
    S: np.ndarray,
    K: int,
) -> dict[str, Any]:
    """Compare original selection and known-GT local oracle branches.

    A: native output. C: independently supplied, fixed candidate output.
    G: nonempty ground truth. Omega: eligible points. S: original selection.
    S must be a subset of Omega with exactly K true entries.
    Counts cover the entire instance; action counts cover each selected region.
    Error coverage is undefined (None) if the native output has zero errors.
    The at-most-K branch diagnoses exact-budget constraints and can select <K;
    it must not be described as a matched exact-K control.
    """
    K = _validate_problem(A, C, G, Omega, K)
    _validate_arrays(A=A, S=S)
    if np.any(S & ~Omega):
        raise ValueError("S must be a subset of Omega")
    if int(np.count_nonzero(S)) != K:
        raise ValueError("S must contain exactly K selected pixels")

    fixed_selection = exact_k_action_oracle(A, C, G, Omega, K)
    gt_selection = exact_k_action_oracle(A, G, G, Omega, K)
    at_most_selection = at_most_k_repair_oracle(A, C, G, Omega, K)
    errors = A != G
    total_errors = int(np.count_nonzero(errors))
    eligible_errors = int(np.count_nonzero(errors & Omega))
    eligible_actions = _action_masks(A, C, G, Omega)
    fixed_repair_fn = int(np.count_nonzero(eligible_actions["repair_fn"]))
    fixed_repair_fp = int(np.count_nonzero(eligible_actions["repair_fp"]))
    eligible_noops = int(np.count_nonzero(eligible_actions["noop"]))
    fixed_branch = _branch_metrics(A, C, G, fixed_selection)

    return {
        "status": STATUS,
        "scope": SCOPE,
        "ranking_r": SCORE_R,
        "budget_k": K,
        "eligible_count": int(np.count_nonzero(Omega)),
        "total_error_coverage": {
            "native_total_errors": total_errors,
            "eligible_native_errors": eligible_errors,
            "eligible_fraction_of_native_errors": (
                eligible_errors / total_errors if total_errors else None
            ),
        },
        "fixed_candidate_correct_action_eligibility": {
            "repair_fn": fixed_repair_fn,
            "repair_fp": fixed_repair_fp,
            "total_repairs_available": fixed_repair_fn + fixed_repair_fp,
            "eligible_native_errors_without_fixed_candidate_repair": (
                eligible_errors - fixed_repair_fn - fixed_repair_fp
            ),
            "eligible_noops": eligible_noops,
            "eligible_add_fp": int(np.count_nonzero(eligible_actions["add_fp"])),
            "eligible_delete_tp": int(np.count_nonzero(eligible_actions["delete_tp"])),
        },
        "exact_K_harm_constraint": {
            "eligible_nonharmful_actions": fixed_repair_fn + fixed_repair_fp + eligible_noops,
            "minimum_unavoidable_harmful_selections": max(
                0, K - fixed_repair_fn - fixed_repair_fp - eligible_noops
            ),
            "fixed_candidate_oracle_actual_selected_harms": fixed_branch["selected_harms"],
        },
        "at_most_K_constraint_diagnosis": "repair actions only; may use fewer than K; not an exact-K control",
        "native": _branch_metrics(A, A, G, np.zeros(A.shape, dtype=np.bool_)),
        "branches": {
            "original_selection_fixed_candidate": _branch_metrics(A, C, G, S),
            "exact_k_fixed_candidate_oracle": fixed_branch,
            "exact_k_gt_action_oracle": _branch_metrics(A, G, G, gt_selection),
            "unlimited_eligible_gt_oracle": _branch_metrics(A, G, G, Omega),
            "at_most_k_fixed_candidate_repair_oracle": _branch_metrics(A, C, G, at_most_selection),
        },
    }


# The helpers below are synthetic validation only, with independent direct
# enumeration of actual binary outputs. No dataset or execution binding exists.
def _exhaustive_best_fraction(
    A: np.ndarray, C: np.ndarray, G: np.ndarray, Omega: np.ndarray, K: int
) -> tuple[int, int, int]:
    if A.size > 9:
        raise ValueError("Synthetic exhaustive enumeration is limited to <=9 pixels")
    a, c, g = A.ravel().tolist(), C.ravel().tolist(), G.ravel().tolist()
    eligible = np.flatnonzero(Omega).tolist()
    best = None
    enumerated = 0
    for combination in itertools.combinations(eligible, K):
        selected = set(combination)
        output = [c[i] if i in selected else a[i] for i in range(len(a))]
        intersection = sum(x and y for x, y in zip(output, g))
        union = sum(x or y for x, y in zip(output, g))
        if best is None or intersection * best[1] > best[0] * union:
            best = (intersection, union)
        enumerated += 1
    assert best is not None
    return best[0], best[1], enumerated


def _fraction_leq(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left["intersection"] * right["union"] <= right["intersection"] * left["union"]


def _check_case(
    A: np.ndarray, C: np.ndarray, G: np.ndarray, Omega: np.ndarray
) -> tuple[int, int]:
    before = tuple(array.copy() for array in (A, C, G, Omega))
    eligible = np.flatnonzero(Omega)
    budget_checks = subset_checks = 0
    previous_gt = None
    best_at_most_fraction = None
    for K in range(len(eligible) + 1):
        S = np.zeros(A.shape, dtype=np.bool_)
        S.ravel()[eligible[:K]] = True
        result = diagnose_fixed_candidate(A, C, G, Omega, S, K)
        branches = result["branches"]
        fixed = branches["exact_k_fixed_candidate_oracle"]
        gt = branches["exact_k_gt_action_oracle"]
        for branch, candidate in ((fixed, C), (gt, G)):
            best_i, best_u, enumerated = _exhaustive_best_fraction(A, candidate, G, Omega, K)
            assert branch["intersection"] * best_u == best_i * branch["union"]
            subset_checks += enumerated
            if branch is fixed and (
                best_at_most_fraction is None
                or best_i * best_at_most_fraction[1] > best_at_most_fraction[0] * best_u
            ):
                best_at_most_fraction = (best_i, best_u)
        assert _fraction_leq(branches["original_selection_fixed_candidate"], fixed)
        assert _fraction_leq(fixed, gt)
        assert _fraction_leq(gt, branches["unlimited_eligible_gt_oracle"])
        assert _fraction_leq(result["native"], gt)
        at_most = branches["at_most_k_fixed_candidate_repair_oracle"]
        assert best_at_most_fraction is not None
        assert at_most["intersection"] * best_at_most_fraction[1] == best_at_most_fraction[0] * at_most["union"]
        assert _fraction_leq(fixed, at_most)
        assert _fraction_leq(at_most, gt)
        assert at_most["selected_harms"] == 0 and at_most["selected_noops"] == 0
        assert fixed["selected_harms"] == result["exact_K_harm_constraint"]["minimum_unavoidable_harmful_selections"]
        if previous_gt is not None:
            assert _fraction_leq(previous_gt, gt)
        previous_gt = gt
        for name, branch in branches.items():
            selection = branch["selected_mask"]
            assert not np.any(selection & ~Omega)
            if name.startswith("at_most"):
                assert branch["selected_count"] <= K
            else:
                assert branch["selected_count"] == (len(eligible) if name.startswith("unlimited") else K)
            assert branch["selected_count"] == (
                branch["selected_repairs"] + branch["selected_harms"] + branch["selected_noops"]
            )
            assert branch["fn"] == result["native"]["fn"] - branch["selected_repair_fn"] + branch["selected_delete_tp"]
            assert branch["fp"] == result["native"]["fp"] - branch["selected_repair_fp"] + branch["selected_add_fp"]
        budget_checks += 1
    assert all(np.array_equal(array, saved) for array, saved in zip((A, C, G, Omega), before))
    return budget_checks, subset_checks


def _bits(code: int, n: int) -> np.ndarray:
    return np.array([bool(code & (1 << i)) for i in range(n)], dtype=np.bool_)


def _expect_rejection(call: Any, error_type: type[Exception]) -> None:
    try:
        call()
    except error_type:
        return
    raise AssertionError(f"Expected {error_type.__name__}")


def _validation_checks() -> int:
    A = np.array([False, True], dtype=np.bool_)
    C = np.array([True, False], dtype=np.bool_)
    G = np.array([True, False], dtype=np.bool_)
    Omega = np.ones(2, dtype=np.bool_)
    S = np.array([True, False], dtype=np.bool_)
    calls = [
        (lambda: diagnose_fixed_candidate(A, C, G, Omega, S, -1), ValueError),
        (lambda: diagnose_fixed_candidate(A, C, G, Omega, S, 3), ValueError),
        (lambda: diagnose_fixed_candidate(A, C, G, Omega, S, 1.0), TypeError),
        (lambda: diagnose_fixed_candidate(A, C, G, Omega, S, True), TypeError),
        (lambda: diagnose_fixed_candidate(A, C, G, Omega, S, np.array(1)), TypeError),
        (lambda: diagnose_fixed_candidate(A, C[:1], G, Omega, S, 1), ValueError),
        (lambda: diagnose_fixed_candidate(A, C, G, Omega, S.reshape(1, 2), 1), ValueError),
        (lambda: diagnose_fixed_candidate(A, C, np.zeros(2, dtype=np.bool_), Omega, S, 1), ValueError),
        (lambda: diagnose_fixed_candidate(A, C, G, np.array([False, True]), S, 1), ValueError),
        (lambda: diagnose_fixed_candidate(A, C, G, Omega, S, 0), ValueError),
        (lambda: diagnose_fixed_candidate(A.astype(np.uint8), C, G, Omega, S, 1), TypeError),
        (lambda: diagnose_fixed_candidate(A, C.astype(float), G, Omega, S, 1), TypeError),
        (lambda: diagnose_fixed_candidate(A, C, G.astype(int), Omega, S, 1), TypeError),
        (lambda: diagnose_fixed_candidate(A, C, G, Omega.astype(int), S, 1), TypeError),
        (lambda: diagnose_fixed_candidate(A, C, G, Omega, S.astype(int), 1), TypeError),
        (lambda: diagnose_fixed_candidate(A.tolist(), C, G, Omega, S, 1), TypeError),
        (lambda: diagnose_fixed_candidate(*(np.array(True) for _ in range(5)), 1), ValueError),
    ]
    for call, error_type in calls:
        _expect_rejection(call, error_type)
    # A NumPy integer is valid; no implicit array or float budget conversion is used.
    diagnose_fixed_candidate(A, C, G, Omega, S, np.int64(1))
    return len(calls)


def synthetic_self_test(random_cases: int = 256, seed: int = 20261014) -> dict[str, Any]:
    """Run exhaustive tiny-array checks; returns SYNTHETIC_ONLY evidence."""
    if not __debug__:
        raise RuntimeError(
            "SYNTHETIC_ONLY verification requires enabled assertions; "
            "optimized Python (-O or PYTHONOPTIMIZE) is rejected"
        )
    if isinstance(random_cases, bool) or not isinstance(random_cases, int) or random_cases < 0:
        raise ValueError("random_cases must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    budget_checks = subset_checks = 0
    configurations = 0
    # Complete Boolean input space for three pixels, excluding empty GT.
    for a, c, g, omega in itertools.product(range(8), range(8), range(1, 8), range(8)):
        b, s = _check_case(_bits(a, 3), _bits(c, 3), _bits(g, 3), _bits(omega, 3))
        budget_checks += b
        subset_checks += s
        configurations += 1
    exact_edges = [
        ("empty_native", [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 1]),
        ("empty_eligibility", [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0]),
        ("noops_only", [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]),
        ("mandatory_noop_fillers", [1, 0, 0, 0, 0], [1, 0, 1, 0, 0], [1, 0, 1, 0, 0], [1, 1, 1, 1, 1]),
        ("repair_category_order", [0, 1, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 0, 0]),
        ("harm_and_noop_fillers", [1, 1, 0, 0], [1, 0, 0, 1], [1, 1, 0, 0], [0, 1, 1, 1]),
        ("forced_harms", [1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 1, 1]),
        ("zero_fixed_repair", [0, 0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0, 0, 0], [1] * 9),
    ]
    for _, a, c, g, omega in exact_edges:
        b, s = _check_case(*(np.array(x, dtype=np.bool_) for x in (a, c, g, omega)))
        budget_checks += b
        subset_checks += s
    # Shape preservation is checked in two and three dimensions.
    for shape in ((2, 2), (1, 2, 2)):
        b, s = _check_case(*(_bits(code, 4).reshape(shape) for code in (3, 12, 5, 15)))
        budget_checks += b
        subset_checks += s
    for _ in range(random_cases):
        n = int(rng.integers(1, 10))
        arrays = [rng.integers(0, 2, n).astype(np.bool_) for _ in range(4)]
        if not np.any(arrays[2]):
            arrays[2][int(rng.integers(n))] = True
        b, s = _check_case(*arrays)
        budget_checks += b
        subset_checks += s

    # Explicit equal-score tie and exact-K filler checks.
    tie_A = np.zeros(5, dtype=np.bool_)
    tie_C = np.ones(5, dtype=np.bool_)
    tie_G = np.ones(5, dtype=np.bool_)
    tie_Omega = np.array([False, True, True, True, True])
    assert np.flatnonzero(exact_k_action_oracle(tie_A, tie_C, tie_G, tie_Omega, 2)).tolist() == [1, 2]
    assert np.flatnonzero(exact_k_action_oracle(tie_A, tie_A, tie_G, tie_Omega, 3)).tolist() == [1, 2, 3]
    # A fixed-C exact-K branch may decline with K when harms become compulsory.
    harm_A, harm_C, harm_G, harm_Omega = (_bits(x, 4) for x in (3, 12, 3, 15))
    zero = _branch_metrics(harm_A, harm_C, harm_G, exact_k_action_oracle(harm_A, harm_C, harm_G, harm_Omega, 0))
    one = _branch_metrics(harm_A, harm_C, harm_G, exact_k_action_oracle(harm_A, harm_C, harm_G, harm_Omega, 1))
    assert one["iou"] < zero["iou"]

    a, c, g, omega = (np.array(x, dtype=np.bool_) for x in exact_edges[-1][1:])
    original = np.array([True] * 6 + [False] * 3, dtype=np.bool_)
    example = diagnose_fixed_candidate(a, c, g, omega, original, 6)
    assert example["total_error_coverage"]["eligible_native_errors"] == 9
    assert example["fixed_candidate_correct_action_eligibility"]["total_repairs_available"] == 0
    example_summary = {
        "eligible_native_errors_GT_can_repair": 9,
        "fixed_candidate_repairs_available": 0,
        "K": 6,
        "native_iou": example["native"]["iou"],
        "fixed_candidate_oracle_iou": example["branches"]["exact_k_fixed_candidate_oracle"]["iou"],
        "same_K_GT_action_oracle_iou": example["branches"]["exact_k_gt_action_oracle"]["iou"],
        "unlimited_eligible_GT_oracle_iou": example["branches"]["unlimited_eligible_gt_oracle"]["iou"],
    }
    validation_rejections = _validation_checks()
    return {
        "label": SYNTHETIC_LABEL,
        "status": STATUS,
        "result": "PASS",
        "scope": SCOPE,
        "max_pixels": 9,
        "complete_three_pixel_configurations": configurations,
        "named_exact_edge_cases": len(exact_edges),
        "multidimensional_edge_cases": 2,
        "random_configurations": random_cases,
        "seed": seed,
        "exact_K_budget_checks": budget_checks,
        "enumerated_subsets_fixed_C_and_GT": subset_checks,
        "invalid_input_rejections": validation_rejections,
        "stable_flat_index_ties": "PASS",
        "ordering_original_le_fixed_C_le_same_K_GT_le_unlimited_GT": "PASS",
        "same_K_GT_monotonic_with_K": "PASS",
        "fixed_C_exact_K_le_at_most_K_constraint_diagnosis": "PASS",
        "at_most_K_fixed_C_exhaustive_optimality": "PASS",
        "unavoidable_harm_count_checks": "PASS",
        "forced_harm_fixed_C_decrease_demonstrated": True,
        "large_GT_potential_zero_fixed_C_potential_example": example_summary,
        "task_efficacy": "NOT_TESTED",
        "execution_qualification": "NOT_REGISTERED",
        "publication": "NOT_PUBLISHED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run SYNTHETIC_ONLY enumeration")
    parser.add_argument("--random-cases", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    if not args.self_test:
        parser.print_help()
        return 0
    print(json.dumps(synthetic_self_test(args.random_cases, args.seed), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
