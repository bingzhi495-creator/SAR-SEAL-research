"""Independent stdlib content checks for the closed production-support oracle."""
from collections import Counter
from itertools import combinations
import json
import re


BUDGETS = (0, 1, 2, 4, 8, 16, 32, None)
OUTCOMES = ('integer', 'float', 'dual')
VARIANTS = ('INTEGER_PRIMARY', 'LOCAL_CANDIDATE_FLOAT')
SUPPORTS = ('gt_band', 'predicted_band')
MODES = ('fn_only', 'fp_only', 'joint')
SIZES = ('all', 'small', 'medium', 'large')
CELL_ORDER = tuple(role + '_' + support for role in ('assigned', 'best') for support in SUPPORTS)
WITNESS_FIELDS = ('image_id', 'GT_annotation_id', 'assigned_native_rank', 'assigned', 'best')
SUPPORT_FIELDS = (*SUPPORTS, 'predicted_band_agreement', 'intersection')
SHA = re.compile(r'[0-9a-f]{64}')
SCOPE = dict(images=187, GT_instances=380, fixed_pairs=377, paired_224_pairs=224,
             paired_224_images=121, other_fixed_pairs=153, original_unpaired_GT=3)
BEST_GT_DUAL = dict(fn_only=(0, 0, 0, 3, 16, 30, 54, 85),
                    fp_only=(0, 0, 0, 2, 6, 17, 27, 53),
                    joint=(0, 0, 0, 4, 23, 53, 103, 173))
ASSIGNED_BASELINE = (60, 86)
BEST_GT_CAPACITIES = dict(FN=14242, FP=10791)
BEST_SIZES = (169, 54, 1)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def keys(value, fields, message):
    require(type(value) is dict and set(value) == set(fields), message)


def equal(actual, expected, message):
    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    require(canonical(actual) == canonical(expected), message)


def outcomes(pairs):
    integer, floating = (10 * i >= 9 * u for i, u in pairs)
    return dict(integer=integer, float=floating, dual=integer and floating)


def edited(pairs, additions, removals):
    return tuple((i + additions, u - removals) for i, u in pairs)


def minimum(pairs, fn, fp):
    """Binary search the budget feasibility condition, independent of the producer."""
    deficits = [9 * u - 10 * i for i, u in pairs]
    result = {}
    for name, deficit in zip(OUTCOMES, (*deficits, max(deficits))):
        left, right = 0, fn + fp
        if deficit > 10 * fn + 9 * fp:
            result[name] = None
            continue
        while left < right:
            mid = (left + right) // 2
            if 9 * mid + min(mid, fn) >= deficit:
                right = mid
            else:
                left = mid + 1
        result[name] = left
    return result


def replay_cell(pairs, fn, fp, recorded=None):
    """Brute-enumerate additions for finite budgets and verify the stored edit witness."""
    curve = {None: outcomes(edited(pairs, fn, fp))}
    for index, k in enumerate(BUDGETS[:-1]):
        possible = dict.fromkeys(OUTCOMES, False)
        if recorded is not None:
            point = recorded[index]
            keys(point, ('additions', 'removals', 'integer_iou', 'float_iou'), 'Exact edit witness fields')
            a, b = point['additions'], point['removals']
            require(type(a) is type(b) is int and 0 <= a <= fn and 0 <= b <= fp and a + b <= k,
                    'Stored edit exceeds support or budget')
            stored = edited(pairs, a, b)
            equal([point['integer_iou'], point['float_iou']], stored, 'Stored edited overlap differs')
        # For any addition count, using the maximum remaining deletions cannot lower either IoU.
        for a in range(min(fn, k) + 1):
            candidate = edited(pairs, a, min(fp, k - a))
            if recorded is not None:
                require(all(si * cu >= ci * su for (si, su), (ci, cu) in zip(stored, candidate)),
                        'Stored edit fails to attain the best paired IoUs')
            for name, crossed in outcomes(candidate).items():
                possible[name] |= crossed
        curve[k] = possible
    return dict(baseline=outcomes(pairs), curve=curve, minimum_edits=minimum(pairs, fn, fp))



def witness_projection(row):
    roles = {role: None if row[role] is None else
             {name: row[role][name] for name in ('query_id', 'raster_sha256')}
             for role in ('assigned', 'best')}
    return {**{name: row[name] for name in WITNESS_FIELDS[:3]}, **roles}


def replay_query(query, pixels):
    keys(query, ('query_id', 'overlap_counts', 'baseline_pass', 'raster_sha256',
                 'support_pixels', 'capacities', 'regions'), 'Exact query fields')
    require(type(query['query_id']) is int and 0 <= query['query_id'] < 100, 'Invalid natural query')
    fingerprints = query['raster_sha256']
    keys(fingerprints, ('prediction', *VARIANTS), 'Exact three raster fingerprints')
    require(all(type(v) is str and SHA.fullmatch(v) for v in fingerprints.values()), 'Invalid raster fingerprint')
    keys(query['overlap_counts'], VARIANTS, 'Exact overlap variants')
    pairs = tuple(query['overlap_counts'][v] for v in VARIANTS)
    require(all(type(pair) in (list, tuple) and len(pair) == 2 and all(type(v) is int for v in pair)
                and 0 <= pair[0] <= pair[1] <= pixels and pair[1] > 0 for pair in pairs),
            'Invalid original intersection and union')
    baseline = outcomes(pairs)
    equal(query['baseline_pass'], baseline, 'Baseline threshold count differs')
    areas = query['support_pixels']
    keys(areas, SUPPORT_FIELDS, 'Exact support area fields')
    require(all(type(v) is int and 0 <= v <= pixels for v in areas.values())
            and areas['predicted_band_agreement'] <= areas['predicted_band']
            and areas['intersection'] <= min(areas['gt_band'], areas['predicted_band_agreement'])
            and areas['gt_band'] + areas['predicted_band'] - areas['intersection'] <= pixels,
            'Invalid support area or intersection')
    keys(query['capacities'], SUPPORTS, 'Exact capacities by support')
    keys(query['regions'], SUPPORTS, 'Exact outcomes by support')
    result = {}
    for support in SUPPORTS:
        cap, region = query['capacities'][support], query['regions'][support]
        keys(cap, ('FN', 'FP'), 'Exact signed capacities')
        fn, fp = cap['FN'], cap['FP']
        require(type(fn) is type(fp) is int and fn >= 0 and fp >= 0
                and fn + fp <= areas[support if support == 'gt_band' else 'predicted_band_agreement']
                and max(i for i, _ in pairs) + fp <= min(min(u - fn, u + i - 1) for i, u in pairs),
                'Signed capacity incompatible with support or original overlap')
        keys(region, ('threshold', 'minimum_edits', 'curve'), 'Exact regional oracle fields')
        equal(region['threshold'], [9, 10], 'Regional threshold differs')
        curve = region['curve']
        require(type(curve) is list and len(curve) == 7, 'Missing regional budget points')
        for k, point in zip(BUDGETS[:-1], curve):
            keys(point, ('budget', *MODES), 'Exact budget point fields')
            equal(point['budget'], k, 'Regional budget order differs')
        modes = {mode: replay_cell(pairs, a, b, [point[mode] for point in curve])
                 for mode, a, b in (('fn_only', fn, 0), ('fp_only', 0, fp), ('joint', fn, fp))}
        equal(region['minimum_edits'], {mode: value['minimum_edits'] for mode, value in modes.items()},
              'Stored minimum edit distances differ')
        result[support] = dict(baseline=baseline, support_pixels=areas[support], capacities=cap, modes=modes)
    return result


def is_new(cell, mode, budget, outcome):
    return not cell['baseline'][outcome] and cell['modes'][mode]['curve'][budget][outcome]


def aggregate(items):
    """Items hold private image identities solely for counting distinct affected images."""
    baseline = {name: sum(cell['baseline'][name] for _, cell in items) for name in OUTCOMES}
    modes = {}
    for mode in MODES:
        points = []
        for budget in BUDGETS:
            new = {name: sum(is_new(cell, mode, budget, name) for _, cell in items) for name in OUTCOMES}
            images = {image for image, cell in items if is_new(cell, mode, budget, 'dual')}
            points.append(dict(budget=budget, new_crossings=new, images_with_new_dual=len(images),
                               total_pass={name: baseline[name] + new[name] for name in OUTCOMES}))
        histogram = Counter(cell['modes'][mode]['minimum_edits']['dual'] for _, cell in items
                            if not cell['baseline']['dual'])
        unreachable = histogram.pop(None, 0)
        modes[mode] = dict(curve=points[:-1], unlimited=points[-1],
                           minimum_dual_edits=dict(histogram=[dict(edits=k, pairs=histogram[k])
                                                            for k in sorted(histogram)], unreachable=unreachable))
    return dict(pairs=len(items), baseline_pass=baseline,
                eligible_failures={name: len(items) - baseline[name] for name in OUTCOMES},
                support_pixels=sum(cell['support_pixels'] for _, cell in items),
                capacities={sign: sum(cell['capacities'][sign] for _, cell in items) for sign in ('FN', 'FP')},
                modes=modes)


def stratify(records, cell_name):
    return {size: aggregate([(r['image'], r['cells'][cell_name]) for r in records
                             if size == 'all' or r['size'] == size]) for size in SIZES}


def recompute_rescues(cohort):
    curve = []
    for budget in BUDGETS:
        bits = [tuple(is_new(r['cells'][cell], 'joint', budget, 'dual') for cell in CELL_ORDER) for r in cohort]
        patterns = Counter(''.join(str(int(flag)) for flag in flags) for flags in bits)
        pairwise = {}
        for a, b in combinations(range(len(CELL_ORDER)), 2):
            frequency = Counter((flags[a], flags[b]) for flags in bits)
            pairwise[CELL_ORDER[a] + '__' + CELL_ORDER[b]] = dict(
                both=frequency[True, True], left_only=frequency[True, False],
                right_only=frequency[False, True], neither=frequency[False, False])
        curve.append(dict(budget=budget, pairwise=pairwise, patterns=dict(sorted(patterns.items()))))
    return curve


def verify(summary, rows, witnesses):
    """Verify actual content supplied by Root, without file IO, model code, or terminal assertions."""
    require(type(rows) is type(witnesses) is list and len(rows) == len(witnesses) == SCOPE['fixed_pairs'],
            'Exact fixed-pair and reference-raster populations required')
    row_fields = (*WITNESS_FIELDS, 'integer_annotation_area', 'area_group', 'original_hw')
    all_records, cohort, annotations = [], [], set()
    for row, witness in zip(rows, witnesses):
        keys(row, row_fields, 'Exact private pair row')
        keys(witness, WITNESS_FIELDS, 'Exact reference raster witness')
        equal(witness_projection(row), witness, 'Ordered reference raster witness differs')
        image, annotation, rank = (row[field] for field in WITNESS_FIELDS[:3])
        require(type(image) is type(annotation) is type(rank) is int and annotation not in annotations
                and 1 <= rank <= 100, 'Invalid or duplicate fixed pair')
        annotations.add(annotation)
        hw = row['original_hw']
        require(type(hw) in (list, tuple) and len(hw) == 2 and all(type(v) is int and v > 0 for v in hw),
                'Invalid original grid')
        area = row['integer_annotation_area']
        require(type(area) in (int, float) and 0 <= area <= 100000**2, 'Invalid original annotation area')
        size = SIZES[1 + int(area >= 1024) + int(area >= 9216)]
        equal(row['area_group'], size, 'Area group differs')
        record = dict(image=image, size=size, cells={}, support={})
        for role in ('assigned', 'best'):
            query = row[role]
            if role == 'best' and query is None:
                continue
            cells = replay_query(query, hw[0] * hw[1])
            record['cells'].update({role + '_' + support: cell for support, cell in cells.items()})
            record['support'][role] = query['support_pixels']
        if row['best'] is not None:
            assigned, best = row['assigned'], row['best']
            require(not any(assigned['baseline_pass'].values()) and not any(best['baseline_pass'].values()),
                    'The fixed paired subset must be a dual raw miss for both query roles')
            require(all(assigned['raster_sha256'][v] == best['raster_sha256'][v] for v in VARIANTS),
                    'Paired query roles do not use the same two GT rasters')
            if assigned['query_id'] == best['query_id']:
                equal(assigned, best, 'Same natural query produced different paired contents')
            cohort.append(record)
        all_records.append(record)

    require(len(cohort) == SCOPE['paired_224_pairs']
            and len({record['image'] for record in cohort}) == SCOPE['paired_224_images'],
            'Fixed paired subset does not reconcile')
    production = {support: stratify(all_records, 'assigned_' + support) for support in SUPPORTS}
    paired = {name: stratify(cohort, name) for name in CELL_ORDER}
    overlaps = {role: {name: sum(record['support'][role][name] for record in cohort) for name in SUPPORT_FIELDS}
                for role in ('assigned', 'best')}
    production_baseline = production['predicted_band']['all']['baseline_pass']
    require((production_baseline['integer'], production_baseline['float']) == ASSIGNED_BASELINE,
            'Original assigned single-GT baseline anchors differ')
    best_gt = paired['best_gt_band']['all']
    equal(best_gt['capacities'], BEST_GT_CAPACITIES, 'Accepted best/GT signed capacities differ')
    equal(tuple(paired['best_gt_band'][size]['pairs'] for size in SIZES[1:]), BEST_SIZES,
          'Accepted best/GT area populations differ')
    for mode, expected in BEST_GT_DUAL.items():
        cell = best_gt['modes'][mode]
        actual = tuple(point['new_crossings']['dual'] for point in (*cell['curve'], cell['unlimited']))
        equal(actual, expected, 'Previously accepted best/GT budget anchors differ')

    count = production['predicted_band']['all']['modes']['joint']['curve'][BUDGETS.index(16)]['new_crossings']['dual']
    gate = dict(budget=16, min_new_dual=20, observed_new_dual=count, passed=count >= 20)
    return check_summary(summary, production, paired, overlaps, recompute_rescues(cohort), gate)


def check_summary(summary, production, paired, overlaps, rescues, gate):
    expected = dict(
        schema='Root-RFDETR-E24-production-boundary-oracle-v1',
        status='PRODUCTION_BOUNDARY_ORACLE_COMPLETE_PENDING_ROOT_ACCEPTANCE',
        qualified=False, TEST=False, AP_recomputed=False, COCO_matching_recomputed=False, new_training=False,
        scope=SCOPE, threshold=[9, 10], budgets=list(BUDGETS[:-1]), radius_pixels=2,
        area_partition='Original integer COCO annotation area: [0,1024), [1024,9216), [9216,1e10]. '
                       'Exact shared COCO boundaries are assigned to the larger group for disjoint totals.',
        gate=gate, production_assigned=production,
        paired_224=dict(cells=paired, support_overlaps=overlaps, rescue_intersections=dict(
            cell_order=list(CELL_ORDER), scope='New dual passes at matched budget, joint edits only', curve=rescues)),
    )
    equal(summary, expected, 'Independent aggregate or gate differs')
    return dict(status='INDEPENDENT_PRODUCTION_BOUNDARY_CONTENT_VERIFIED',
                verified_assigned_pairs=SCOPE['fixed_pairs'], verified_best_pairs=SCOPE['paired_224_pairs'],
                verified_raster_fingerprints=3 * (SCOPE['fixed_pairs'] + SCOPE['paired_224_pairs']),
                paired_quartet_recomputed=True, gate=gate, original_terminal_checked=False)
