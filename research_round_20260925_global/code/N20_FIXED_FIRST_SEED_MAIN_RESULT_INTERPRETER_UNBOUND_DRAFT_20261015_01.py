"""UNBOUND pure first-seed continuation rule; no IO or qualification.

Accept only [one seed201 E24-atmost summary] in explicit AP percent units.
Repeat permission is not scientific acceptance, novelty or publication success.
The caller must first qualify complete full187 evidence and protocol Source.
"""

import math

STRONG_AP = 68.09938494745545
SEEDS = (202610150201, 202610150202, 202610150203)
KEYS = {'seed', 'epoch', 'mode', 'AP_unit', 'LOCAL_AP', 'CENTER_AP', 'strong_AP'}
PROTOCOL_SCHEMA = 'N20-true-feature-neighborhood-fixed24-protocol-v1'
DECISION_LITERAL = {
    'Delta': 'AP_LOCAL-max(AP_CENTER,strong_R)',
    'nonpositive': 'STOP current recipe, no extra seeds/budget/fusion',
    'positive_below1': 'at most seed202; permit203 only if two individually beat CENTER and meanDelta>=1',
    'at_least1': 'replicate202 and203',
    'secondary_not_rescue': ['E8', 'exact']}


def interpret_fixed_main(results):
    """Only first201; no fabricated interpretation/acceptance of future repeats."""
    if type(results) not in (list, tuple) or len(results) != 1:
        raise ValueError('Exactly one fixed first-seed201 main summary required')
    row = results[0]
    if type(row) is not dict or set(row) != KEYS:
        raise ValueError('Exact main-only summary keys required')
    if (type(row['seed']) is not int or row['seed'] != SEEDS[0]
            or type(row['epoch']) is not int or row['epoch'] != 24
            or row['mode'] != 'atmost' or row['AP_unit'] != 'percent'):
        raise ValueError('Locked seed201/E24/atmost/explicit AP percent only')
    for name in ('LOCAL_AP', 'CENTER_AP', 'strong_AP'):
        value = row[name]
        if type(value) not in (int, float) or not math.isfinite(value) or not (value == -1 or 0 <= value <= 100):
            raise ValueError('Finite AP percent0..100 or -1 undefined; bool forbidden')
    if row['strong_AP'] != STRONG_AP:
        raise ValueError('Exact unchanged strong E24 reference required')
    delta, allowed, decision = None, [], 'RULE_NOT_EVALUABLE'
    if row['LOCAL_AP'] != -1 and row['CENTER_AP'] != -1:
        delta = row['LOCAL_AP'] - max(row['CENTER_AP'], STRONG_AP)
        if delta <= 0:
            decision = 'STOP'
        elif delta < 1:
            decision, allowed = 'ONE_EXTRA_SEED', [SEEDS[1]]
        else:
            decision, allowed = 'TWO_EXTRA_SEEDS', list(SEEDS[1:])
    return dict(schema='N20-first-seed-fixed-main-result-interpretation-v1',
        main_epoch=24, main_mode='atmost', AP_unit='percent', seeds=[SEEDS[0]],
        main_results=[dict(row)], LOCAL_AP_percent=row['LOCAL_AP'],
        CENTER_AP_percent=row['CENTER_AP'], strong_AP_percent=STRONG_AP,
        Delta_AP_points=delta, aggregate_formula='LOCAL-max(CENTER,strong)',
        paired_LOCAL_above_CENTER=(None if delta is None else row['LOCAL_AP'] > row['CENTER_AP']),
        decision=decision, allowed_next_seeds=allowed,
        conditional_seed203_rule=DECISION_LITERAL['positive_below1'],
        additional_budget_or_fusion_allowed=False, secondary_results_may_replace_main=False,
        repeated_seed_results_interpreted=False, multi_seed_acceptance_evaluated=False,
        qualification_performed=False, task_efficacy_established=False,
        innovation_established=False, TGRS_readiness_established=False,
        evidence_scope='First private initialization under fixed teacher/PR/data/24-epoch budget only; continuation rule, not publication acceptance')
