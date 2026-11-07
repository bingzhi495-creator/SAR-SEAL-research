"""Unbound E24 production-assigned predicted-boundary ceiling."""
from collections import Counter
from copy import deepcopy
import hashlib
from itertools import combinations


INTEGER = 'INTEGER_PRIMARY'
FLOAT = 'LOCAL_CANDIDATE_FLOAT'
VARIANTS = (INTEGER, FLOAT)
OUTCOMES = ('integer', 'float', 'dual')
SUPPORTS = ('gt_band', 'predicted_band')
CELLS = tuple((query, support) for query in ('assigned', 'best') for support in SUPPORTS)
SIZES = ('small', 'medium', 'large')
MODES = ('fn_only', 'fp_only', 'joint')
BUDGETS = (0, 1, 2, 4, 8, 16, 32)
GATE_BUDGET = 16
GATE_MINIMUM = 20
DEV_IMAGES = 187
GT_INSTANCES = 380
FIXED_PAIRS = 377
BEST_PAIRS = 224
BEST_IMAGES = 121


def need(condition, message):
    if not condition:
        raise ValueError(message)


def exact(value, keys, role):
    need(type(value) is dict and set(value) == set(keys), 'Exact ' + role + ' fields')


def support_masks(np, integer, floating, prediction, dilate):
    """Return original clipped GT support and GT-blind predicted support."""
    gt_band = dilate(np, integer | floating) & dilate(np, ~(integer & floating)) & ~(integer ^ floating)
    predicted_band = dilate(np, prediction) & dilate(np, ~prediction)
    return dict(gt_band=gt_band, predicted_band=predicted_band)


def _new_crossing(record, support, mode, outcome, budget):
    edits = record['regions'][support]['minimum_edits'][mode][outcome]
    return (not record['baseline_pass'][outcome] and edits is not None
            and (budget is None or edits <= budget))


def _cell_group(records, support):
    baseline = {outcome: sum(row['baseline_pass'][outcome] for _, row in records) for outcome in OUTCOMES}

    def stats(mode, budget):
        new = {outcome: sum(_new_crossing(row, support, mode, outcome, budget) for _, row in records)
               for outcome in OUTCOMES}
        images = {image for image, row in records if _new_crossing(row, support, mode, 'dual', budget)}
        return dict(budget=budget, new_crossings=new,
                    total_pass={outcome: baseline[outcome] + new[outcome] for outcome in OUTCOMES},
                    images_with_new_dual=len(images))

    modes = {}
    for mode in MODES:
        distances = [row['regions'][support]['minimum_edits'][mode]['dual']
                     for _, row in records if not row['baseline_pass']['dual']]
        histogram = Counter(value for value in distances if value is not None)
        modes[mode] = dict(curve=[stats(mode, budget) for budget in BUDGETS], unlimited=stats(mode, None),
                           minimum_dual_edits=dict(histogram=[dict(edits=key, pairs=histogram[key])
                                                            for key in sorted(histogram)],
                                                   unreachable=distances.count(None)))
    return dict(pairs=len(records), baseline_pass=baseline,
                eligible_failures={outcome: len(records) - baseline[outcome] for outcome in OUTCOMES},
                support_pixels=sum(row['support_pixels'][support] for _, row in records),
                capacities={sign: sum(row['capacities'][support][sign] for _, row in records) for sign in ('FN', 'FP')},
                modes=modes)


def summarize_cell(rows, query, support):
    groups = {'all': rows, **{size: [row for row in rows if row['area_group'] == size] for size in SIZES}}
    return {name: _cell_group([(row['image_id'], row[query]) for row in subset], support)
            for name, subset in groups.items()}


def rescue_intersections(rows):
    """Tabulate common, exclusive and neither for each cell pair, within the same cohort."""
    labels = [query + '_' + support for query, support in CELLS]
    output = []
    for budget in (*BUDGETS, None):
        patterns = Counter(tuple(_new_crossing(row[query], support, 'joint', 'dual', budget)
                                 for query, support in CELLS) for row in rows)
        pairs = {}
        for left, right in combinations(range(len(CELLS)), 2):
            pair_counts = Counter()
            for pattern, count in patterns.items():
                pair_counts[(pattern[left], pattern[right])] += count
            pairs[labels[left] + '__' + labels[right]] = dict(
                both=pair_counts[(True, True)], left_only=pair_counts[(True, False)],
                right_only=pair_counts[(False, True)], neither=pair_counts[(False, False)],
            )
        output.append(dict(budget=budget, pairwise=pairs,
                           patterns={''.join('1' if flag else '0' for flag in pattern): count
                                     for pattern, count in sorted(patterns.items())}))
    return dict(cell_order=labels, scope='New dual passes at matched budget, joint edits only', curve=output)


def summarize_production(rows):
    assigned = {support: summarize_cell(rows, 'assigned', support) for support in SUPPORTS}
    primary = assigned['predicted_band']['all']['modes']['joint']['curve'][BUDGETS.index(GATE_BUDGET)]
    observed = primary['new_crossings']['dual']
    paired = [row for row in rows if row['best'] is not None]
    cells = {query + '_' + support: summarize_cell(paired, query, support) for query, support in CELLS}
    support_overlaps = {query: {field: sum(row[query]['support_pixels'][field] for row in population)
                                for field in ('gt_band', 'predicted_band', 'predicted_band_agreement', 'intersection')}
                        for query, population in (('assigned', paired), ('best', paired))}
    return dict(gate=dict(budget=GATE_BUDGET, min_new_dual=GATE_MINIMUM, observed_new_dual=observed,
                          passed=observed >= GATE_MINIMUM),
                production_assigned=assigned,
                paired_224=dict(cells=cells, support_overlaps=support_overlaps,
                                rescue_intersections=rescue_intersections(paired)))


def _raster_sha(np, mask):
    return hashlib.sha256(np.ascontiguousarray(mask, dtype=np.uint8).tobytes()).hexdigest()


def _measure_query(np, query, raw, gt_masks, gt_sha, modules):
    localization = modules['localization']
    mask_api = modules['mask_api']
    math_api = modules['oracle_math']
    need(type(query) is int and 0 <= query < 100, 'Valid accepted natural raw query ID')
    hw = raw['original_hw']
    rle = raw['raw_masks'][query]
    exact(rle, ('size', 'counts'), 'selected compressed raw mask')
    need(rle['size'] == hw and type(rle['counts']) is str and rle['counts'] and rle['counts'].isascii(),
         'Selected original-grid compressed raw query mask')
    prediction = localization.binary_mask(np, mask_api.decode(dict(size=hw, counts=rle['counts'].encode('ascii'))))
    need(prediction.shape == tuple(hw), 'Selected prediction matches original grid')
    gi, gf = (gt_masks[variant] for variant in VARIANTS)
    count = lambda pixels: int(np.count_nonzero(pixels))
    overlaps = {variant: (count(prediction & mask), count(prediction | mask))
                for variant, mask in gt_masks.items()}
    baseline = {name: 10 * overlaps[variant][0] >= 9 * overlaps[variant][1]
                for name, variant in zip(OUTCOMES, VARIANTS)}
    baseline['dual'] = baseline['integer'] and baseline['float']

    supports = support_masks(np, gi, gf, prediction, localization.dilate_chebyshev)
    shared_miss = gi & gf & ~prediction
    shared_spill = prediction & ~(gi | gf)
    capacities = {support: dict(FN=count(shared_miss & band), FP=count(shared_spill & band))
                  for support, band in supports.items()}
    regions = {support: math_api.signed_edit_oracle(overlaps[INTEGER], overlaps[FLOAT], cap['FN'], cap['FP'])
               for support, cap in capacities.items()}
    support_pixels = {support: count(band) for support, band in supports.items()}
    support_pixels['predicted_band_agreement'] = count(supports['predicted_band'] & ~(gi ^ gf))
    support_pixels['intersection'] = count(supports['predicted_band'] & supports['gt_band'])
    return dict(query_id=query, overlap_counts=overlaps, baseline_pass=baseline,
                raster_sha256=dict(prediction=_raster_sha(np, prediction), **gt_sha),
                support_pixels=support_pixels, capacities=capacities, regions=regions)


def build_pair(np, anchor, previous_best, raw, cocos, modules):
    """Rebuild only already accepted production and optional archived best masks."""
    localization = modules['localization']
    mask_api = modules['mask_api']
    image, aid, q, rank = (anchor[key] for key in ('image_id', 'GT_annotation_id', 'assigned_query_id',
                                                 'assigned_native_rank'))
    need(image == raw['image_id'] and type(q) is type(rank) is int and 0 <= q < 100 and 1 <= rank <= 100
         and raw['slot0_native_ranks'][q] == rank and raw['retained'].count(dict(native_rank=rank, query_id=q)) == 1,
         'Exact accepted retained production query and native rank')
    gt_masks = {}
    for variant in VARIANTS:
        coco = cocos[variant]
        annotation = coco.anns[aid]
        need(annotation['image_id'] == image and annotation['category_id'] == 1
             and annotation.get('iscrowd', 0) == 0, 'Same original noncrowd GT identity')
        mask = localization.binary_mask(np, mask_api.decode(coco.annToRLE(annotation)))
        need(mask.shape == tuple(raw['original_hw']) and bool(mask.any()), 'Nonempty original GT mask grid')
        gt_masks[variant] = mask
    gt_sha = {variant: _raster_sha(np, mask) for variant, mask in gt_masks.items()}
    assigned = _measure_query(np, q, raw, gt_masks, gt_sha, modules)
    need(all((intersection / union).hex() == float(anchor['by_GT'][variant]['assigned']).hex()
             for variant, (intersection, union) in assigned['overlap_counts'].items()),
         'Original production assigned mask overlaps differ from exact accepted values')
    area = cocos[INTEGER].anns[aid]['area']
    group = modules['signed_adapter'].area_group(area)

    best = None
    if previous_best is not None:
        chosen = previous_best['selected_raw_query_id']
        need(type(chosen) is int and 0 <= chosen < 100 and previous_best['image_id'] == image
             and previous_best['GT_annotation_id'] == aid and previous_best['assigned_query_id'] == q
             and previous_best['assigned_native_rank'] == rank and previous_best['original_hw'] == raw['original_hw']
             and previous_best['area_group'] == group and previous_best['integer_annotation_area'] == area,
             'Same accepted archived best pair identity and natural query')
        best = assigned if chosen == q else _measure_query(np, chosen, raw, gt_masks, gt_sha, modules)
        ii, ui = best['overlap_counts'][INTEGER]
        iff, uf = best['overlap_counts'][FLOAT]
        same = lambda left, right: localization.signature(left) == localization.signature(right)
        need((ii / ui).hex() == float(anchor['by_GT'][INTEGER]['raw']).hex()
             and iff / uf <= anchor['by_GT'][FLOAT]['raw'] and not any(best['baseline_pass'].values())
             and same(best['overlap_counts'], previous_best['overlap_counts'])
             and best['raster_sha256'] == previous_best['raster_sha256']
             and best['capacities']['gt_band'] == previous_best['capacities']['band']
             and same(best['regions']['gt_band'], previous_best['regions']['band']),
             'Reproduced accepted archived best exact pixels, overlaps and GT band oracle')
    return dict(image_id=image, GT_annotation_id=aid, assigned_native_rank=rank, original_hw=raw['original_hw'],
                integer_annotation_area=area, area_group=group, assigned=assigned, best=best)


def evaluate_production_boundary_oracle(np, control, *, integer_gt, floating_gt, accepted_raw_private_pairs,
                                        accepted_best_private_rows, raw_records, modules):
    """Evaluate the predeclared E24 fixed production masks and archived 224 subset."""
    roles = ('localization', 'raw_geometry', 'gt_sensitivity', 'coco_loader', 'mask_api', 'oracle_math', 'signed_adapter')
    exact(modules, roles, 'selected production adapter dependency')
    localization = modules['localization']
    raw_geometry = modules['raw_geometry']
    math_api = modules['oracle_math']
    need(localization.RADIUS == 2 and math_api.THRESHOLD == (9, 10) and math_api.BUDGETS == BUDGETS,
         'Exact predeclared radius, threshold and budgets')
    need(type(accepted_raw_private_pairs) is list and len(accepted_raw_private_pairs) == FIXED_PAIRS
         and type(accepted_best_private_rows) is list and len(accepted_best_private_rows) == BEST_PAIRS,
         'Original full fixed pairs and already accepted best-query subset')
    control.check()
    modules['gt_sensitivity'].validate_GT_pair(integer_gt, floating_gt)
    images = integer_gt['images']
    ids = [image['id'] for image in images]
    need(len(ids) == len(set(ids)) == DEV_IMAGES and all(type(image) is int for image in ids)
         and len(integer_gt['annotations']) == GT_INSTANCES and type(raw_records) is list
         and len(raw_records) == DEV_IMAGES, 'Same complete original DEV187 and 380 GT')
    raw_by_image = {}
    for image, raw in zip(images, raw_records):
        exact(raw, raw_geometry.RAW_KEYS, 'accepted raw image record')
        hw = [image['height'], image['width']]
        need(type(raw['image_id']) is int and raw['image_id'] == image['id']
             and raw['original_hw'] == hw and all(type(n) is int and n > 0 for n in hw)
             and all(type(raw[key]) is list and len(raw[key]) == 100
                     for key in ('raw_masks', 'slot0_native_ranks'))
             and type(raw['retained']) is list, 'Same original-grid ordered raw records and retained witnesses')
        raw_by_image[image['id']] = raw

    cocos = {}
    for variant, document in ((INTEGER, integer_gt), (FLOAT, floating_gt)):
        coco = modules['coco_loader'].COCO()
        coco.dataset = deepcopy(document)
        coco.createIndex()
        need(list(coco.imgs) == ids and len(coco.anns) == GT_INSTANCES and set(coco.cats) == {1},
             'Original image order, annotation population and category')
        cocos[variant] = coco
    fixed_ids, assigned_ranks, cohort = set(), set(), []
    for anchor in accepted_raw_private_pairs:
        exact(anchor, ('image_id', 'GT_annotation_id', 'assigned_native_rank', 'assigned_query_id', 'by_GT'),
              'accepted raw fixed pair')
        image, aid, rank = (anchor[key] for key in ('image_id', 'GT_annotation_id', 'assigned_native_rank'))
        need(type(image) is type(aid) is type(rank) is int and image in raw_by_image
             and aid in cocos[INTEGER].anns and aid not in fixed_ids and (image, rank) not in assigned_ranks
             and cocos[INTEGER].anns[aid]['image_id'] == image,
             'Unique accepted GT and production assigned row')
        fixed_ids.add(aid)
        assigned_ranks.add((image, rank))
        exact(anchor['by_GT'], VARIANTS, 'accepted overlap versions')
        for overlap in anchor['by_GT'].values():
            exact(overlap, ('raw', 'retained', 'assigned'), 'accepted raw/retained/assigned overlaps')
        need(anchor['by_GT'][INTEGER]['assigned'] >= .50, 'Original nonignored integer 0.50 assignment')
        if all(anchor['by_GT'][variant]['raw'] < localization.THRESHOLD for variant in VARIANTS):
            cohort.append(anchor)

    totals = raw_geometry.decompose(accepted_raw_private_pairs)
    need(all(tuple(totals[variant][threshold][role] for role in ('raw', 'retained', 'assigned')) == counts
             for variant, thresholds in localization.ACCEPTED_COUNTS.items() for threshold, counts in thresholds.items()),
         'Exact accepted raw/retained/assigned fixed-pair totals')
    need(len(cohort) == BEST_PAIRS and len({row['image_id'] for row in cohort}) == BEST_IMAGES,
         'Exact archived dual-raw-miss subset')
    best_by_id = {}
    for anchor, previous in zip(cohort, accepted_best_private_rows):
        need(previous['image_id'] == anchor['image_id'] and previous['GT_annotation_id'] == anchor['GT_annotation_id'],
             'Same exact ordered accepted best-query subset')
        best_by_id[anchor['GT_annotation_id']] = previous

    rows = []
    for anchor in accepted_raw_private_pairs:
        control.check()
        rows.append(build_pair(np, anchor, best_by_id.get(anchor['GT_annotation_id']), raw_by_image[anchor['image_id']],
                               cocos, modules))
    production = summarize_production(rows)
    baseline = production['production_assigned']['predicted_band']['all']['baseline_pass']
    need(all(baseline[name] == localization.ACCEPTED_COUNTS[variant]['0.90'][2]
             for name, variant in zip(OUTCOMES, VARIANTS)), 'Exact accepted assigned 0.90 counts by variant')
    control.check()
    summary = dict(schema='Root-RFDETR-E24-production-boundary-oracle-v1',
                   status='PRODUCTION_BOUNDARY_ORACLE_COMPLETE_PENDING_ROOT_ACCEPTANCE', qualified=False,
                   TEST=False, AP_recomputed=False, COCO_matching_recomputed=False, new_training=False,
                   scope=dict(images=DEV_IMAGES, GT_instances=GT_INSTANCES, fixed_pairs=FIXED_PAIRS,
                              paired_224_pairs=BEST_PAIRS, paired_224_images=BEST_IMAGES,
                              other_fixed_pairs=FIXED_PAIRS-BEST_PAIRS, original_unpaired_GT=GT_INSTANCES-FIXED_PAIRS),
                   threshold=[9, 10], budgets=list(BUDGETS), radius_pixels=localization.RADIUS,
                   area_partition='Original integer COCO annotation area: [0,1024), [1024,9216), [9216,1e10]. '
                                  'Exact shared COCO boundaries are assigned to the larger group for disjoint totals.',
                   **production)
    return dict(summary=summary, private_pair_records=rows)
