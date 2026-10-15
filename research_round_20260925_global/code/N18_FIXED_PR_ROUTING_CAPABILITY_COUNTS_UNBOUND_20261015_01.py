# UNBOUND_DRAFT
"""Candidate-local binary counts only. No I/O, model, GT loading, AP or oracle."""

SCHEMA = 'N18-fixed-PR-routing-capability-counts-v1'
POLICIES = ('E24_LOCAL_atmost', 'E24_CENTER_atmost', 'Omega_strong_R')


def _bool_vector(value, name):
    if not isinstance(value, (list, tuple)):
        if (getattr(value, 'ndim', None) != 1 or
                getattr(getattr(value, 'dtype', None), 'kind', None) != 'b'):
            raise ValueError(name + ' must be a 1D boolean array or boolean list/tuple')
        value = value.tolist()
    if not all(type(x) is bool for x in value):
        raise ValueError(name + ' must contain only booleans')
    return value


def _ratio(numerator, denominator):
    return None if denominator is None or denominator == 0 else numerator / denominator


def count_query(P_C, GT_C, A_C, K, selections):
    """All vectors share the caller-verified ascending C order; S are supports.

    A_C is the actual common-C PR after-bit, never a GT replacement. K=0
    requires A_C=None and empty supports: native E remains observable but
    PR capability is unavailable. The caller supplies only fixed matched GT.
    """
    p, g = _bool_vector(P_C, 'P_C'), _bool_vector(GT_C, 'GT_C')
    n = len(p)
    if len(g) != n or type(K) is not int or not 0 <= K <= n:
        raise ValueError('GT_C length or integer K outside [0, |C|]')
    if type(selections) is not dict or set(selections) != set(POLICIES):
        raise ValueError('Exactly the three fixed policy support names are required')
    supports = {name: _bool_vector(selections[name], name) for name in POLICIES}
    for name, support in supports.items():
        if len(support) != n or sum(support) > K:
            raise ValueError(name + ' support length or budget is invalid')
    if sum(supports['Omega_strong_R']) != K:
        raise ValueError('Omega_strong_R must select the physical |Omega|=K support')
    e = [x != y for x, y in zip(p, g)]
    observed = K > 0
    if not observed:
        if A_C is not None:
            raise ValueError('K=0 has no observed PR action: A_C must be None')
        b = h = delta = None
    else:
        a = _bool_vector(A_C, 'A_C')
        if len(a) != n:
            raise ValueError('A_C length differs from C')
        delta = [x != y for x, y in zip(a, p)]
        b = [err and changed for err, changed in zip(e, delta)]
        h = [not err and changed for err, changed in zip(e, delta)]
    E = sum(e)
    B, H = (sum(b), sum(h)) if observed else (None, None)
    budget_denominator = min(K, B) if observed else None
    result = dict(schema=SCHEMA, candidate_points=n, K=K, PR_action_observed=observed,
                  E=E, B=B, H=H, actual_action_flips=sum(delta) if observed else None,
                  PR_capability_E_denominator=E if observed else None,
                  PR_repairable_fraction=_ratio(B, E) if observed else None,
                  budget_repairable_denominator=budget_denominator, selections={})
    for name, support in supports.items():
        chosen = sum(support)
        helpful = sum(s and v for s, v in zip(support, b)) if observed else 0
        harmful = sum(s and v for s, v in zip(support, h)) if observed else 0
        flips = helpful + harmful
        result['selections'][name] = dict(
            selected_support=chosen, helpful_selected=helpful, harmful_selected=harmful,
            actual_flips=flips, selected_noop=chosen - flips,
            helpful_over_budget_repairable=_ratio(helpful, budget_denominator))
    return result


def aggregate_counts(rows):
    """Micro sums of count_query outputs; do not average per-query fractions.

    K=0 native errors contribute to E_all_queries, never to the capability
    denominator. Unmatched queries are not rows and must remain outer counts.
    """
    rows = list(rows)
    if any(type(r) is not dict or r.get('schema') != SCHEMA for r in rows):
        raise ValueError('aggregate_counts requires count_query outputs')
    observed = [r for r in rows if r['PR_action_observed']]
    B = sum(r['B'] for r in observed) if observed else None
    H = sum(r['H'] for r in observed) if observed else None
    error_denominator = sum(r['E'] for r in observed) if observed else None
    budget_denominator = (sum(r['budget_repairable_denominator'] for r in observed)
                          if observed else None)
    result = dict(schema=SCHEMA, aggregation='micro', matched_queries=len(rows),
                  PR_observed_queries=len(observed), K0_queries=len(rows) - len(observed),
                  candidate_points=sum(r['candidate_points'] for r in rows),
                  E_all_queries=sum(r['E'] for r in rows), B=B, H=H,
                  actual_action_flips=sum(r['actual_action_flips'] for r in observed)
                  if observed else None,
                  PR_capability_E_denominator=error_denominator,
                  PR_repairable_fraction=_ratio(B, error_denominator),
                  budget_repairable_denominator=budget_denominator, selections={})
    for name in POLICIES:
        values = [r['selections'][name] for r in rows]
        counts = {key: sum(v[key] for v in values) for key in
                  ('selected_support', 'helpful_selected', 'harmful_selected',
                   'actual_flips', 'selected_noop')}
        counts['helpful_over_budget_repairable'] = _ratio(
            counts['helpful_selected'], budget_denominator)
        result['selections'][name] = counts
    return result
