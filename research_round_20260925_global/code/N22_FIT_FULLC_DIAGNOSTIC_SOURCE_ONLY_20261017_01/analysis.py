"""UNBOUND N22 descriptive fixed-pair action accounting, using exact integers.

No I/O, tensors, labels, models, execution gate or AP computation. The caller
injects the reviewed oracle module and supplies counts for every original
matcher pair with a real nonempty GT, including native IoU zero. Counts do not
authenticate pair identities, all-100-query capture, distinct C membership,
FP32 fixed-PR signs or selected coordinates; those are caller obligations.
"""
from fractions import Fraction


SCHEMA = 'N22_FIXED_PAIR_ACTION_ANALYSIS_V1'
METHODS = ('native', 'router_R', 'fixed_PR_beneficial_oracle', 'ideal_C_error_oracle')
IDENTITY_KEYS = ('pair_id', 'image_id', 'query_index', 'gt_index')
COUNT_KEYS = ('I', 'U', 'native_fn', 'native_fp',
              'candidate_fn', 'candidate_fp', 'candidate_tp', 'candidate_tn',
              'pr_beneficial_fn', 'pr_beneficial_fp',
              'b_FN', 'b_FP', 'h_TP', 'h_TN', 'K', 'selected_count')
AUDIT_KEYS = ('image_count', 'collected_queries', 'unassigned_queries',
              'ignored_queries', 'unmatched_gt', 'total_real_gt')
QUERY_COUNT_PER_IMAGE = 100


def _count(name, value):
    if type(value) is not int or value < 0:
        raise ValueError(name + ' must be a nonnegative nonboolean integer')
    return value


def _exact_keys(value, keys, name):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(name + ' has missing or unexpected keys')


def _json_exact(value):
    if type(value) is Fraction:
        return dict(numerator=value.numerator, denominator=value.denominator)
    if isinstance(value, dict):
        return {key: _json_exact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_exact(item) for item in value]
    return value


def _mean(values):
    return sum(values, Fraction(0)) / len(values) if values else None


def _transitions(before, after, thresholds):
    if len(before) != len(after):
        raise ValueError('Threshold comparisons need the same fixed population')
    return [dict(threshold=t, pair_denominator=len(before),
                 before_ge=sum(x >= t for x in before),
                 after_ge=sum(x >= t for x in after),
                 up=sum(x < t <= y for x, y in zip(before, after)),
                 down=sum(y < t <= x for x, y in zip(before, after)))
            for t in thresholds]


def analyze_instances(records, audit_counts, *, oracle):
    """Return JSON-safe exact summaries, never choose a research direction.

    R selects exactly K distinct C coordinates in this N22 route. Unchanged
    selected slots still count toward K; the two beneficial oracles use at most
    K flips. No pair is selected or dropped using its native IoU. Audits count
    unassigned/ignored queries and unmatched real GT, without invented masks.
    Every output mean and threshold rate uses the full original paired count.
    """
    if type(records) not in (list, tuple):
        raise ValueError('records must be an explicit list/tuple of real GT pairs')
    _exact_keys(audit_counts, AUDIT_KEYS, 'audit_counts')
    for key in AUDIT_KEYS:
        _count(key, audit_counts[key])
    pair_count = len(records)
    if (audit_counts['collected_queries'] != QUERY_COUNT_PER_IMAGE * audit_counts['image_count']
            or audit_counts['collected_queries'] != pair_count
            + audit_counts['unassigned_queries'] + audit_counts['ignored_queries']
            or audit_counts['total_real_gt'] != pair_count + audit_counts['unmatched_gt']):
        raise ValueError('All-query or real-GT population counts do not reconcile')

    seen_pairs, seen_queries, seen_gt, image_ids = set(), set(), set(), set()
    instances, exact_values = [], {method: [] for method in METHODS}
    native_zero_count = 0
    totals = {key: 0 for key in COUNT_KEYS}
    thresholds = tuple(oracle.IOU_THRESHOLDS)
    expected_thresholds = tuple(Fraction(n, 20) for n in range(10, 20))
    if thresholds != expected_thresholds:
        raise ValueError('N22 requires exactly the ten fixed rational thresholds')

    for record in records:
        _exact_keys(record, IDENTITY_KEYS + COUNT_KEYS, 'paired record')
        if type(record['pair_id']) is not str or not record['pair_id'].strip():
            raise ValueError('pair_id must be a nonempty string')
        image_id = record['image_id']
        if type(image_id) not in (str, int) or (type(image_id) is str and not image_id.strip()):
            raise ValueError('image_id must be a nonempty string or nonboolean integer')
        for key in ('query_index', 'gt_index') + COUNT_KEYS:
            _count(key, record[key])
        if record['query_index'] >= QUERY_COUNT_PER_IMAGE:
            raise ValueError('query_index exceeds the original all-100-query population')
        image_key = (type(image_id).__name__, image_id)
        query_key = (image_key, record['query_index'])
        gt_key = (image_key, record['gt_index'])
        if (record['pair_id'] in seen_pairs or query_key in seen_queries or gt_key in seen_gt):
            raise ValueError('Duplicate original pair, query or real-GT assignment')
        seen_pairs.add(record['pair_id'])
        seen_queries.add(query_key)
        seen_gt.add(gt_key)
        image_ids.add(image_key)
        if len(image_ids) > audit_counts['image_count']:
            raise ValueError('More paired image identities than audited images')
        if record['I'] + record['native_fn'] == 0:
            raise ValueError('Only original matcher pairs with real nonempty GT enter this table')
        if (record['selected_count'] != record['K']
                or record['pr_beneficial_fn'] > record['candidate_fn']
                or record['pr_beneficial_fp'] > record['candidate_fp']
                or record['b_FN'] > record['pr_beneficial_fn']
                or record['b_FP'] > record['pr_beneficial_fp']):
            raise ValueError('R selection or fixed-PR beneficial action counts are inconsistent')

        i, u, k = record['I'], record['U'], record['K']
        native_args = dict(native_fn=record['native_fn'], native_fp=record['native_fp'])
        results = dict(native=oracle.beneficial_oracle(i, u, 0, 0, k, **native_args))
        results['router_R'] = oracle.router_outcome(i, u, k, **native_args,
            **{key: record[key] for key in ('candidate_fn', 'candidate_fp', 'candidate_tp',
                                           'candidate_tn', 'b_FN', 'b_FP', 'h_TP', 'h_TN',
                                           'selected_count')})
        results['fixed_PR_beneficial_oracle'] = oracle.beneficial_oracle(
            i, u, record['pr_beneficial_fn'], record['pr_beneficial_fp'], k, **native_args)
        results['ideal_C_error_oracle'] = oracle.ideal_candidate_oracle(
            i, u, record['candidate_fn'], record['candidate_fp'], k, **native_args)
        if (results['router_R']['iou'] > results['fixed_PR_beneficial_oracle']['iou']
                or results['fixed_PR_beneficial_oracle']['iou'] > results['ideal_C_error_oracle']['iou']):
            raise ValueError('Conditional nested action sets produced inconsistent pair IoUs')
        native_iou = results['native']['iou']
        native_zero_count += int(native_iou == 0)
        crossings = {method: oracle.threshold_crossings(native_iou, results[method]['iou'])
                     for method in METHODS[1:]}
        instances.append(dict(identity={key: record[key] for key in IDENTITY_KEYS},
                              counts={key: record[key] for key in COUNT_KEYS},
                              methods=results, crossings_from_native=crossings))
        for method in METHODS:
            exact_values[method].append(results[method]['iou'])
        for key in COUNT_KEYS:
            totals[key] += record[key]

    native_values = exact_values['native']
    method_table = [dict(method=method, pair_denominator=pair_count,
                         mean_iou=_mean(exact_values[method]),
                         mean_iou_delta_from_native=_mean([y-x for x, y in
                             zip(native_values, exact_values[method])]),
                         threshold_table=_transitions(native_values, exact_values[method], thresholds))
                    for method in METHODS]
    directions = []
    for direction, before_name, after_name in (
        ('router_selection', 'router_R', 'fixed_PR_beneficial_oracle'),
        ('fixed_PR_action_availability', 'fixed_PR_beneficial_oracle', 'ideal_C_error_oracle')):
        before, after = exact_values[before_name], exact_values[after_name]
        gaps = [y-x for x, y in zip(before, after)]
        directions.append(dict(direction=direction, before=before_name, after=after_name,
                               pair_denominator=pair_count, mean_iou_gap=_mean(gaps),
                               pairs_positive_gap=sum(gap > 0 for gap in gaps),
                               pairs_zero_gap=sum(gap == 0 for gap in gaps),
                               threshold_table=_transitions(before, after, thresholds)))
    coverage = dict(native_FN=totals['native_fn'], native_FP=totals['native_fp'],
                    C_FN=totals['candidate_fn'], C_FP=totals['candidate_fp'],
                    outside_C_FN=totals['native_fn']-totals['candidate_fn'],
                    outside_C_FP=totals['native_fp']-totals['candidate_fp'])
    return _json_exact(dict(schema=SCHEMA,
        population=dict(audit_counts, paired_instances=pair_count,
                        native_iou_zero_pairs=native_zero_count,
                        query_count_per_image=QUERY_COUNT_PER_IMAGE,
                        pair_population='ALL_ORIGINAL_MATCHER_REAL_GT_PAIRS_INCLUDING_NATIVE_IOU_ZERO'),
        instances=instances, method_table=method_table,
        descriptive_direction_table=directions, count_totals=totals,
        native_error_candidate_coverage=coverage,
        claims=dict(is_AP_bound=False, is_AP=False, causal_attribution=False,
                    research_direction_selected=False, actual_training_point_distribution=False,
                    count_checks_authenticate_coordinate_or_pair_provenance=False),
        scope='Conditional fixed original-matcher pair IoU; candidate C and fixed FP32 PR actions; descriptive only'))
