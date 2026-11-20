"""Post-closure, fixed-match occupancy bounds on caller-admitted records.

No file, label, model, RLE or COCO loader. This computes geometry only and does
not authenticate the caller's closure evidence or grant an execution permit.
"""
from types import SimpleNamespace

import numpy as np

ARMS = ('N', 'SF', 'SU=F', 'SU', 'LF=S', 'LU=S', 'LW=S', 'NP=S')
SEEDS = (5217, 5218, 5219)
LABELS = ARMS + ('O_U', 'O_F')
DRAW_COUNT = 4096
BOOTSTRAP_SEED = 5217
PRIMARY = (('O_U_minus_SU', 'O_U', 'SU'),
           ('O_F_minus_O_U', 'O_F', 'O_U'),
           ('O_F_minus_LW=S', 'O_F', 'LW=S'))
COMPARISONS = (PRIMARY
               + tuple(('O_U_minus_' + arm, 'O_U', arm) for arm in ('LF=S', 'LU=S', 'NP=S'))
               + tuple(('SU_minus_' + arm, 'SU', arm) for arm in ARMS if arm != 'SU'))


def need(ok, message):
    if not ok:
        raise ValueError(message)


def require_trigger(eligibility):
    keys = {'original_score_closed', 'lw_ap_blocker', 'lw_geometry_blocker'}
    need(type(eligibility) is dict and set(eligibility) == keys
         and all(type(value) is bool for value in eligibility.values()),
         'Explicit caller-verified score closure and LW blocker flags')
    need(eligibility['original_score_closed']
         and (eligibility['lw_ap_blocker'] or eligibility['lw_geometry_blocker']),
         'Closed original score and existing LW AP or geometry blocker required')


def from_prepared(np, gt, prepared, *, eligibility, score_module, fit_step,
                  geometry_helpers, geometry_module, oracle_module):
    """Compose the released stock matcher with this geometry-only evaluator.

    gt/prepared are already admitted in-memory objects. The caller retains the
    original reconstruction, readback, score-closure and label-opening checks.
    No original module is patched. fixed_match_geometry's public evaluator
    argument supplies the existing attach_truth records directly to this module.
    """
    require_trigger(eligibility)
    evaluator = SimpleNamespace(evaluate_geometry=lambda records, edits: evaluate_admitted(
        records, edits, eligibility=eligibility, geometry_module=geometry_module,
        oracle_module=oracle_module))
    return score_module.fixed_match_geometry(np, gt, prepared, fit_step, geometry_helpers, evaluator)


def evaluate_admitted(records, edits_by_seed, *, eligibility, geometry_module, oracle_module):
    """Use unchanged attach_truth records and the complete three-seed/eight-arm map.

    The eligibility flags report the caller's already established reason for this
    diagnostic. They are not a substitute for original input/terminal admission.
    The original evaluator validates identities, fixed matches and all budgets.
    Oracle budgets use each seed's actual SU edits, including zero-budget rows.
    """
    require_trigger(eligibility)
    need(tuple(geometry_module.ARMS) == ARMS and tuple(geometry_module.SEEDS) == SEEDS,
         'Original eight arms and three seeds')
    observed = geometry_module.evaluate_geometry(records, edits_by_seed)
    valid = [row for row in records if row['role'] == 'valid']
    groups = sorted({row['group'] for row in valid})
    group_indices = [np.array([i for i, row in enumerate(valid) if row['group'] == group], dtype=np.intp)
                     for group in groups]
    observed_rows = {(row['seed'], row['image_id'], row['rank'], row['query_id']): row
                     for row in observed['private_witness']}
    values = np.empty((len(SEEDS), len(valid), len(LABELS)), dtype=np.float64)
    private = []
    per_seed = {}
    for si, seed in enumerate(SEEDS):
        counters = dict(valid_candidates=len(valid), k_zero_candidates=0,
                        realized_edits_total=0, add_support_total=0, drop_support_total=0)
        for ri, row in enumerate(valid):
            identity = geometry_module.identity(row)
            original = row['raw_continuous']
            actual = observed_rows[(seed,) + identity]['arms']
            coords = geometry_module.point_geometry(row)
            q = (np.empty((0, 2), dtype=np.float64) if row['truth'] is None and len(coords) == 0
                 else np.asarray(row['truth'], dtype=np.float64))
            oracle = oracle_module.candidate_oracles(
                q[:, 0].tolist(), q[:, 1].tolist(),
                original['intersection'], original['predicted_area'], original['truth_area'],
                actual['SU']['edits'], add_indices=coords[:, 0].tolist(), drop_indices=coords[:, 1].tolist())
            for ai, arm in enumerate(ARMS):
                values[si, ri, ai] = actual[arm]['iou']
            for arm in ('O_U', 'O_F'):
                values[si, ri, LABELS.index(arm)] = oracle['oracles'][arm]['iou']
            budget = oracle['budget']
            counters['k_zero_candidates'] += int(budget['k_zero'])
            counters['realized_edits_total'] += budget['edits']
            counters['add_support_total'] += budget['add_support_count']
            counters['drop_support_total'] += budget['drop_support_count']
            private.append(dict(seed=seed, image_id=identity[0], rank=identity[1], query_id=identity[2],
                                group=row['group'], matched_gt_id=row['matched_gt_id'],
                                observed=actual, oracle=oracle))
        per_seed[seed] = dict(counts=counters, means={
            label: dict(group_equal_iou_points=(float(np.mean([
                values[si, index, ai].mean() for index in group_indices])) * 100 if groups else None))
            for ai, label in enumerate(LABELS)})
    seed_average = values.mean(axis=0)
    comparisons = {}
    if groups:
        draws = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(
            len(groups), size=(DRAW_COUNT, len(groups)))
    for name, left, right in COMPARISONS:
        if groups:
            effects = seed_average[:, LABELS.index(left)] - seed_average[:, LABELS.index(right)]
            grouped = np.array([effects[index].mean() for index in group_indices]) * 100
            bootstrap = grouped[draws].mean(axis=1)
            result = dict(delta_points=float(grouped.mean()),
                          interval95_points=np.quantile(bootstrap, [.025, .975]).tolist())
        else:
            result = dict(delta_points=None, interval95_points=None)
        comparisons[name] = result
    numerical = {}
    for name, left, right in PRIMARY:
        gaps = values[:, :, LABELS.index(left)] - values[:, :, LABELS.index(right)]
        numerical[name] = dict(minimum_unrounded_gap=float(gaps.min()) if gaps.size else None,
                               negative_gap_count=int((gaps < 0).sum()))
    summary = dict(schema='RFDETR-post-closure-fixed-match-occupancy-bounds-v1',
                   seeds=list(SEEDS), observed_arms=list(ARMS), oracle_labels=['O_U', 'O_F'],
                   counts=observed['summary']['counts'], source_groups=len(groups),
                   primary_comparisons=[name for name, _, _ in PRIMARY],
                   bootstrap_draws=DRAW_COUNT, bootstrap_seed=BOOTSTRAP_SEED,
                   bootstrap_unit='fixed source prefix; seed-averaged paired effects',
                   means={label: dict(group_equal_iou_points=(float(np.mean([
                       per_seed[seed]['means'][label]['group_equal_iou_points'] for seed in SEEDS]))
                       if groups else None)) for label in LABELS},
                   per_seed=per_seed, comparisons=comparisons, numerical_ordering=numerical,
                   caller_reported_trigger=dict(eligibility),
                   oracle_budget='per-seed realized SU total edits, equal to LF/LU/LW/NP',
                   geometry_math='original observed arithmetic and released oracle arithmetic, no clamping')
    return dict(summary=summary, observed_geometry_summary=observed['summary'], private_witness=private,
                AP_computed=False, model_executed=False, input_admission_verified_here=False,
                interpretation='Fixed-support geometry bounds only, not AP or a deployable policy')
