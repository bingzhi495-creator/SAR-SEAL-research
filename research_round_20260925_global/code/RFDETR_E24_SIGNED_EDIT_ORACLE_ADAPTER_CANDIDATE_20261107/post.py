"""Independently verify the signed-edit result and match closed raster witnesses.

The caller owns artifact byte verification and the actual terminal/closure checks.
This module reads no files and performs no mask decode, matching, or AP.
"""
import json
import re


VARIANTS = ('INTEGER_PRIMARY', 'LOCAL_CANDIDATE_FLOAT')
SIZES = ('small', 'medium', 'large')
OUTCOMES = ('integer', 'float', 'dual')
REGIONS = ('band', 'outside')
MODES = ('fn_only', 'fp_only', 'joint')
BUDGETS = (0, 1, 2, 4, 8, 16, 32)
SCOPE = dict(images=187, GT_instances=380, fixed_pairs=377, cohort_pairs=224, cohort_images=121,
             other_fixed_pairs_not_evaluated=153, original_unpaired_GT=3)
CAPACITIES = dict(band=dict(FN=14242, FP=10791), outside=dict(FN=7775, FP=4204))
WITNESS_FIELDS = ('image_id', 'GT_annotation_id', 'selected_raw_query_id', 'raster_sha256')
ROW_FIELDS = WITNESS_FIELDS + ('assigned_native_rank', 'assigned_query_id', 'original_hw',
                              'integer_annotation_area', 'area_group', 'overlap_counts',
                              'baseline_pass', 'capacities', 'regions')
AREA_PARTITION = ('Original integer COCO annotation area: [0,1024), [1024,9216), [9216,1e10]. '
                  'Exact shared COCO boundaries are assigned to the larger group for disjoint totals.')
COUNTS = ('Newly reaches threshold within the measured cohort. The fixed denominator includes '
          'unevaluated pairs, so these counts are not a whole-population edited result.')
SHA256 = re.compile(r'[0-9a-f]{64}')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def keys(value, names, message):
    require(type(value) is dict and set(value) == set(names), message)


def equal(value, expected, message):
    # Serialization normalizes in-memory tuples and JSON lists while separating bool from int.
    encode = lambda item: json.dumps(item, sort_keys=True, separators=(',', ':'), allow_nan=False)
    require(encode(value) == encode(expected), message)


def reaches(pairs):
    integer, floating = (10 * intersection >= 9 * union for intersection, union in pairs)
    return dict(integer=integer, float=floating, dual=integer and floating)


def minimum_edits(pairs, additions, removals):
    """Search monotone budget feasibility without using the adapter's inverse formula."""
    deficits = tuple(9 * union - 10 * intersection for intersection, union in pairs)
    minima = {}
    for name, deficit in zip(OUTCOMES, (*deficits, max(deficits))):
        low, high = 0, additions + removals
        if 10 * additions + 9 * removals < deficit:
            minima[name] = None
            continue
        while low < high:
            mid = (low + high) // 2
            # An addition contributes 10 and any remaining edit contributes 9.
            if 9 * mid + min(mid, additions) >= deficit:
                high = mid
            else:
                low = mid + 1
        minima[name] = low
    return minima


def check_region(region, pairs, capacities):
    keys(region, ('threshold', 'minimum_edits', 'curve'), 'Exact regional result fields')
    equal(region['threshold'], [9, 10], 'Regional threshold differs')
    curve = region['curve']
    require(type(curve) is list and len(curve) == len(BUDGETS), 'All seven regional budgets required')
    a, b = capacities['FN'], capacities['FP']
    limits = dict(fn_only=(a, 0), fp_only=(0, b), joint=(a, b))
    outcomes = {mode: dict(curve=[], unlimited=None) for mode in MODES}
    for budget, point in zip(BUDGETS, curve):
        keys(point, ('budget', *MODES), 'Exact per-budget fields')
        equal(point['budget'], budget, 'Regional budgets must be in fixed order')
        for mode, (add_limit, remove_limit) in limits.items():
            recorded = point[mode]
            keys(recorded, ('additions', 'removals', 'integer_iou', 'float_iou'), 'Exact edit fields')
            additions, removals = recorded['additions'], recorded['removals']
            require(type(additions) is type(removals) is int and 0 <= additions <= add_limit
                    and 0 <= removals <= remove_limit and additions + removals <= budget,
                    'Reported edits violate region or budget')
            selected = tuple((i + additions, u - removals) for i, u in pairs)
            equal([recorded['integer_iou'], recorded['float_iou']], selected, 'Reported edited I/U differs')
            possible = dict.fromkeys(OUTCOMES, False)
            # For each possible addition count, maximum legal removals dominate all smaller counts.
            for candidate_a in range(min(add_limit, budget) + 1):
                candidate_b = min(remove_limit, budget - candidate_a)
                candidate = tuple((i + candidate_a, u - candidate_b) for i, u in pairs)
                require(all(si * cu >= ci * su for (si, su), (ci, cu) in zip(selected, candidate)),
                        'Reported edits fail to maximize both exact IoUs')
                for name, crossed in reaches(candidate).items():
                    possible[name] |= crossed
            outcomes[mode]['curve'].append(dict(budget=budget, **possible))
    expected_minima = {}
    for mode, (add_limit, remove_limit) in limits.items():
        saturated = tuple((i + add_limit, u - remove_limit) for i, u in pairs)
        outcomes[mode]['unlimited'] = reaches(saturated)
        expected_minima[mode] = minimum_edits(pairs, add_limit, remove_limit)
    equal(region['minimum_edits'], expected_minima, 'Regional minimum edit distances differ')
    return outcomes


def verify(summary, rows, raster_witnesses):
    """Return only aggregate confirmation for the exact 224-pair accepted cohort."""
    require(type(rows) is type(raster_witnesses) is list
            and len(rows) == len(raster_witnesses) == SCOPE['cohort_pairs'],
            'Exact fixed rows and raster witnesses required')
    require(type(summary) is dict and 'groups' in summary, 'Exact summary object required')
    supplied_groups = summary['groups']
    group_names = ('all', *SIZES)
    keys(supplied_groups, group_names, 'Exact size groups required')
    groups = {}
    for name in group_names:
        group = supplied_groups[name]
        keys(group, ('cohort_pairs', 'fixed_pairs_denominator', 'baseline_pass', 'regions'),
             'Exact size group fields required')
        fixed = group['fixed_pairs_denominator']
        require(type(fixed) is int and 0 <= fixed <= SCOPE['fixed_pairs'], 'Invalid fixed pair denominator')
        groups[name] = dict(cohort_pairs=0, fixed_pairs_denominator=fixed,
                            baseline_pass=dict.fromkeys(OUTCOMES, 0), regions={
            region: {mode: dict(curve=[dict(budget=k, **dict.fromkeys(OUTCOMES, 0)) for k in BUDGETS],
                                unlimited=dict.fromkeys(OUTCOMES, 0)) for mode in MODES} for region in REGIONS})
    require(groups['all']['fixed_pairs_denominator'] == SCOPE['fixed_pairs']
            == sum(groups[name]['fixed_pairs_denominator'] for name in SIZES), 'Fixed denominators do not reconcile')

    seen_ann, seen_image = set(), set()
    totals = {region: dict(FN=0, FP=0) for region in REGIONS}
    for row, witness in zip(rows, raster_witnesses):
        keys(row, ROW_FIELDS, 'Exact private pair row required')
        keys(witness, WITNESS_FIELDS, 'Exact reference raster witness required')
        equal({field: row[field] for field in WITNESS_FIELDS}, witness, 'Ordered raster witness mismatch')
        fingerprints = row['raster_sha256']
        keys(fingerprints, ('prediction', *VARIANTS), 'Exact three mask fingerprints required')
        require(all(type(digest) is str and SHA256.fullmatch(digest) for digest in fingerprints.values()),
                'Invalid raster fingerprint')
        image, annotation, query = (row[field] for field in WITNESS_FIELDS[:3])
        rank, assigned = row['assigned_native_rank'], row['assigned_query_id']
        require(all(type(value) is int for value in (image, annotation, query, rank, assigned))
                and annotation not in seen_ann and 0 <= query < 100 and 0 <= assigned < 100 and 1 <= rank <= 100,
                'Invalid or duplicate fixed pair identity')
        seen_ann.add(annotation)
        seen_image.add(image)

        hw = row['original_hw']
        require(type(hw) in (list, tuple) and len(hw) == 2 and all(type(v) is int and v > 0 for v in hw),
                'Invalid original mask grid')
        overlaps = row['overlap_counts']
        keys(overlaps, VARIANTS, 'Exact integer and float GT overlap counts')
        pairs = tuple(overlaps[name] for name in VARIANTS)
        require(all(type(pair) in (list, tuple) and len(pair) == 2 and all(type(v) is int for v in pair)
                    and 0 <= pair[0] <= pair[1] <= hw[0] * hw[1] and pair[1] > 0 for pair in pairs),
                'Invalid original I/U counts')
        baseline = reaches(pairs)
        equal(row['baseline_pass'], baseline, 'Selected-query baseline differs')
        require(not any(baseline.values()), 'Pair is outside the fixed dual-miss cohort')

        area = row['integer_annotation_area']
        require(type(area) in (int, float) and 0 <= area <= 100000**2, 'Invalid original annotation area')
        size = SIZES[int(area >= 1024) + int(area >= 9216)]
        require(row['area_group'] == size, 'Area group differs from the disjoint area convention')
        target_groups = (groups['all'], groups[size])
        for group in target_groups:
            group['cohort_pairs'] += 1
        capacities = row['capacities']
        keys(capacities, REGIONS, 'Exact capacities regions required')
        keys(row['regions'], REGIONS, 'Exact calculated regions required')
        for capacity in capacities.values():
            keys(capacity, ('FN', 'FP'), 'Exact signed capacities required')
            require(all(type(count) is int and count >= 0 for count in capacity.values()),
                    'Invalid signed capacities')
        fn, fp = (sum(capacities[r][sign] for r in REGIONS) for sign in ('FN', 'FP'))
        require(max(i for i, _ in pairs) + fp <= min(min(u - fn, u + i - 1) for i, u in pairs),
                'Combined regional capacities are incompatible with the same nonempty GT masks')
        for region in REGIONS:
            for sign in ('FN', 'FP'):
                totals[region][sign] += capacities[region][sign]
            computed = check_region(row['regions'][region], pairs, capacities[region])
            for group in target_groups:
                for mode, result in computed.items():
                    aggregate = group['regions'][region][mode]
                    for dst, src in zip((*aggregate['curve'], aggregate['unlimited']),
                                        (*result['curve'], result['unlimited'])):
                        for outcome in OUTCOMES:
                            dst[outcome] += src[outcome]

    require(len(seen_image) == SCOPE['cohort_images'], 'Exact number of cohort images required')
    equal(totals, CAPACITIES, 'Accepted aggregate capacities differ')
    require(all(g['cohort_pairs'] <= g['fixed_pairs_denominator'] for g in groups.values()),
            'Measured counts exceed a fixed size denominator')
    expected = dict(schema='Root-RFDETR-E24-dual-GT-signed-edit-oracle-v1',
                    status='SIGNED_EDIT_ORACLE_COMPLETE_PENDING_ROOT_ACCEPTANCE', qualified=False,
                    TEST=False, AP_recomputed=False, COCO_matching_recomputed=False, new_training=False,
                    scope=SCOPE, threshold=[9, 10], budgets=list(BUDGETS), radius_pixels=2,
                    area_partition=AREA_PARTITION, counts=COUNTS, input_IoU_and_localization_exact=True,
                    no_new_query_rank_search=True, capacities=totals, groups=groups)
    equal(summary, expected, 'Recomputed summary differs')
    return dict(status='INDEPENDENT_SIGNED_EDIT_CONTENT_VERIFIED', verified_pairs=len(rows),
                verified_raster_fingerprints=3 * len(rows), cohort_images=len(seen_image),
                checked_budgets=list(BUDGETS), unlimited_checked=True, region_mode_size_counts_checked=True,
                fixed_size_denominators_reconciled=True, original_terminal_checked=False)
