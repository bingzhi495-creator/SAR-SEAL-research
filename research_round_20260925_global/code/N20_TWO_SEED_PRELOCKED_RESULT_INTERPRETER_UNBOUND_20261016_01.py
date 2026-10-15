"""Unbound, pure two-seed N20 continuation rule; no I/O or qualification."""

import math

STRONG_AP = 68.09938494745545
SEED_201 = 202610150201
SEED_202 = 202610150202
SEED_203 = 202610150203
MAIN_KEYS = {"seed", "epoch", "mode", "AP_unit", "LOCAL_AP", "CENTER_AP", "strong_AP"}


def interpret_two_seed_main(results):
    """Interpret two E24-atmost AP-percent summaries under the locked rule."""
    if type(results) not in (list, tuple) or len(results) != 2:
        raise ValueError("Exactly two main-result rows are required")
    if any(type(row) is not dict or type(row.get("seed")) is not int for row in results):
        raise ValueError("Each row needs an integer seed")
    if {row["seed"] for row in results} != {SEED_201, SEED_202}:
        raise ValueError("Exactly the two prelocked distinct seeds are required")
    for row in results:
        if set(row) != MAIN_KEYS:
            raise ValueError("Exact main-result keys required; no secondary endpoints")
        if (type(row["epoch"]) is not int or row["epoch"] != 24
                or row["mode"] != "atmost" or row["AP_unit"] != "percent"
                or type(row["strong_AP"]) not in (int, float)
                or row["strong_AP"] != STRONG_AP):
            raise ValueError("Only E24-atmost AP percent and the fixed strong reference are accepted")
        for name in ("LOCAL_AP", "CENTER_AP"):
            value = row[name]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100:
                raise ValueError("Both AP values must be defined finite percentages in [0, 100]")
    rows = sorted(results, key=lambda row: row["seed"])
    deltas = {
        row["seed"]: row["LOCAL_AP"] - max(row["CENTER_AP"], STRONG_AP)
        for row in rows
    }
    paired = {row["seed"]: row["LOCAL_AP"] - row["CENTER_AP"] for row in rows}
    mean_delta = sum(deltas.values()) / 2
    mean_local = sum(row["LOCAL_AP"] for row in rows) / 2
    mean_center = sum(row["CENTER_AP"] for row in rows) / 2
    diagnostic_aggregate = mean_local - max(mean_center, STRONG_AP)
    allow = all(gain > 0 for gain in paired.values()) and mean_delta >= 1
    return {
        "schema": "N20-two-seed-prelocked-main-result-decision-unbound-v1",
        "main_epoch": 24,
        "main_mode": "atmost",
        "AP_unit": "percent",
        "strong_AP_percent": STRONG_AP,
        "main_results": [dict(row) for row in rows],
        "decision": "ALLOW_SEED203" if allow else "STOP_CURRENT_RECIPE",
        "allowed_next_seeds": [SEED_203] if allow else [],
        "seeds": [row["seed"] for row in rows],
        "paired_LOCAL_minus_CENTER_AP_points_by_seed": paired,
        "Delta_AP_points_by_seed": deltas,
        "mean_Delta_AP_points": mean_delta,
        "mean_Delta_formula": "mean_i[LOCAL_i-max(CENTER_i,strong_R)]",
        "mean_LOCAL_minus_max_meanCENTER_strong_AP_points": diagnostic_aggregate,
        "diagnostic_aggregate_used_for_decision": False,
        "source_authentication_performed": False,
        "qualification_performed": False,
        "task_efficacy_established": False,
        "innovation_established": False,
        "TGRS_readiness_established": False,
        "additional_budget_or_fusion_allowed": False,
        "secondary_results_may_replace_main": False,
    }
