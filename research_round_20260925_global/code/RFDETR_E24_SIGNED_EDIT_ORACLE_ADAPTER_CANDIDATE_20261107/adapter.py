"""Unbound exact signed-edit upper bounds for the accepted E24 fixed pairs."""
from copy import deepcopy
import hashlib


INTEGER = 'INTEGER_PRIMARY'
FLOAT = 'LOCAL_CANDIDATE_FLOAT'
VARIANTS = (INTEGER, FLOAT)
SIZES = ('small', 'medium', 'large')
OUTCOMES = ('integer', 'float', 'dual')
MODES = ('fn_only', 'fp_only', 'joint')
REGIONS = ('band', 'outside')
DEV_IMAGES = 187
GT_INSTANCES = 380
FIXED_PAIRS = 377
COHORT_PAIRS = 224
COHORT_IMAGES = 121
ACCEPTED_CAPACITIES = {'band': (14242, 10791), 'outside': (7775, 4204)}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def exact(value, keys, role):
    need(type(value) is dict and set(value) == set(keys), 'Exact ' + role + ' fields')


def area_group(area):
    """Use original integer annotation area and put shared endpoints in the larger group."""
    need(type(area) in (int, float) and 0 <= area <= 100000**2, 'Valid original integer COCO area')
    if area < 32**2:
        return 'small'
    if area < 96**2:
        return 'medium'
    return 'large'


def summarize(rows, fixed_sizes, budgets):
    """Count new crosses on the measured subset; expose only denominator aggregates."""
    groups = {'all': rows, **{size: [row for row in rows if row['area_group'] == size] for size in SIZES}}
    totals = {'all': sum(fixed_sizes.values()), **fixed_sizes}
    result = {}
    for name, subset in groups.items():
        regions = {}
        for region in REGIONS:
            regions[region] = {}
            for mode in MODES:
                minima = [(row['baseline_pass'], row['regions'][region]['minimum_edits'][mode]) for row in subset]

                def crossings(budget):
                    return {outcome: sum(not baseline[outcome] and distances[outcome] is not None
                                         and (budget is None or distances[outcome] <= budget)
                                         for baseline, distances in minima)
                            for outcome in OUTCOMES}

                regions[region][mode] = dict(
                    curve=[dict(budget=budget, **crossings(budget)) for budget in budgets],
                    unlimited=crossings(None),
                )
        result[name] = dict(cohort_pairs=len(subset), fixed_pairs_denominator=totals[name],
                            baseline_pass={key: sum(row['baseline_pass'][key] for row in subset)
                                           for key in OUTCOMES}, regions=regions)
    return result


def rebuild_pair(np, anchor, row, raw, cocos, modules):
    """Decode only the already accepted query and the two GT masks for one pair."""
    localization = modules['localization']
    mask_api = modules['mask_api']
    oracle = modules['oracle_math']
    exact(row, ('image_id', 'GT_annotation_id', 'selected_raw_query_id', 'prior_overlaps',
                'selected_IoU_by_GT', 'areas', 'errors'), 'accepted localization row')
    image, aid, chosen = (row[key] for key in ('image_id', 'GT_annotation_id', 'selected_raw_query_id'))
    need(type(image) is type(aid) is type(chosen) is int and 0 <= chosen < 100
         and image == anchor['image_id'] == raw['image_id'] and aid == anchor['GT_annotation_id']
         and localization.signature(row['prior_overlaps']) == localization.signature(anchor['by_GT']),
         'Exact accepted pair identity, query and archived coverage')
    expected = row['selected_IoU_by_GT']
    exact(expected, VARIANTS, 'selected IoU')
    need(all(type(value) in (int, float) and 0 <= value <= 1 for value in expected.values())
         and float(expected[INTEGER]).hex() == float(anchor['by_GT'][INTEGER]['raw']).hex()
         and expected[FLOAT] <= anchor['by_GT'][FLOAT]['raw'],
         'Selected query agrees with exact archived raw overlap anchors')

    hw = tuple(raw['original_hw'])
    rle = raw['raw_masks'][chosen]
    exact(rle, ('size', 'counts'), 'selected compressed raw mask')
    need(rle['size'] == list(hw) and type(rle['counts']) is str
         and rle['counts'] and rle['counts'].isascii(), 'Selected raw mask RLE matches original grid')
    prediction = localization.binary_mask(np, mask_api.decode(dict(size=list(hw), counts=rle['counts'].encode('ascii'))))
    need(prediction.shape == hw, 'Selected prediction matches original image dimensions')

    gt_masks, overlaps = {}, {}
    for variant in VARIANTS:
        coco = cocos[variant]
        annotation = coco.anns[aid]
        need(annotation['image_id'] == image and annotation['category_id'] == 1
             and annotation.get('iscrowd', 0) == 0, 'Same accepted noncrowd GT identity in both versions')
        mask = localization.binary_mask(np, mask_api.decode(coco.annToRLE(annotation)))
        need(mask.shape == hw and bool(mask.any()), 'Nonempty original GT on the same mask grid')
        intersection = int(np.count_nonzero(prediction & mask))
        union = int(np.count_nonzero(prediction | mask))
        need((intersection / union).hex() == float(expected[variant]).hex(),
             'Decoded original-grid pixels differ from exact archived selected overlap')
        gt_masks[variant] = mask
        overlaps[variant] = (intersection, union)

    rebuilt = localization.localize_pair(np, gt_masks[INTEGER], gt_masks[FLOAT], prediction)
    need(localization.signature(rebuilt) == localization.signature(dict(areas=row['areas'], errors=row['errors'])),
         'Exact archived localization counts differ')
    numerator, denominator = oracle.THRESHOLD
    baseline = {name: denominator * overlaps[variant][0] >= numerator * overlaps[variant][1]
                for name, variant in zip(OUTCOMES, VARIANTS)}
    baseline['dual'] = baseline['integer'] and baseline['float']
    need(not any(baseline.values()), 'Accepted dual raw-miss cohort remains below exact 9/10')
    errors = rebuilt['errors']
    capacities = {region: dict(FN=errors['FN'][region], FP=errors['FP'][region]) for region in REGIONS}
    regions = {region: oracle.signed_edit_oracle(overlaps[INTEGER], overlaps[FLOAT], counts['FN'], counts['FP'])
               for region, counts in capacities.items()}
    fingerprints = {name: hashlib.sha256(np.ascontiguousarray(mask, dtype=np.uint8).tobytes()).hexdigest()
                    for name, mask in (('prediction', prediction), *gt_masks.items())}
    area = cocos[INTEGER].anns[aid]['area']
    return dict(image_id=image, GT_annotation_id=aid, selected_raw_query_id=chosen,
                assigned_native_rank=anchor['assigned_native_rank'], assigned_query_id=anchor['assigned_query_id'],
                original_hw=list(hw), integer_annotation_area=area, area_group=area_group(area),
                overlap_counts=overlaps, baseline_pass=baseline, capacities=capacities,
                raster_sha256=fingerprints, regions=regions)


def evaluate_signed_edit_oracle(np, control, *, integer_gt, floating_gt, accepted_raw_private_pairs,
                                accepted_localization_rows, raw_records, modules):
    """Compute private exact per-pair oracles and identity-free summary from admitted inputs."""
    roles = ('localization', 'raw_geometry', 'gt_sensitivity', 'coco_loader', 'mask_api', 'oracle_math')
    exact(modules, roles, 'selected adapter dependency')
    localization = modules['localization']
    raw_geometry = modules['raw_geometry']
    oracle = modules['oracle_math']
    need(oracle.THRESHOLD == (9, 10) and oracle.BUDGETS == (0, 1, 2, 4, 8, 16, 32),
         'Exact 9/10 oracle and seven predeclared edit budgets')
    need(type(accepted_raw_private_pairs) is list and len(accepted_raw_private_pairs) == FIXED_PAIRS
         and type(accepted_localization_rows) is list and len(accepted_localization_rows) == COHORT_PAIRS,
         'Accepted fixed and dual-miss cohort populations')
    control.check()

    modules['gt_sensitivity'].validate_GT_pair(integer_gt, floating_gt)
    ordered_images = integer_gt['images']
    ids = [image['id'] for image in ordered_images]
    need(len(ids) == len(set(ids)) == DEV_IMAGES and all(type(image) is int for image in ids)
         and len(integer_gt['annotations']) == GT_INSTANCES and type(raw_records) is list
         and len(raw_records) == DEV_IMAGES, 'Exact original DEV187 image and GT population')
    raw_by_image = {}
    for original, raw in zip(ordered_images, raw_records):
        exact(raw, raw_geometry.RAW_KEYS, 'accepted raw image')
        image = original['id']
        hw = [original['height'], original['width']]
        need(type(raw['image_id']) is int and raw['image_id'] == image and raw['original_hw'] == hw
             and all(type(value) is int and value > 0 for value in hw)
             and type(raw['raw_masks']) is list and len(raw['raw_masks']) == 100,
             'Accepted raw records preserve original grid, population and natural query count')
        raw_by_image[image] = raw

    cocos = {}
    for variant, document in ((INTEGER, integer_gt), (FLOAT, floating_gt)):
        coco = modules['coco_loader'].COCO()
        coco.dataset = deepcopy(document)
        coco.createIndex()
        need(list(coco.imgs) == ids and len(coco.anns) == GT_INSTANCES and set(coco.cats) == {1},
             'Exact ordered DEV187, all GT identities and original category')
        cocos[variant] = coco

    cohort, fixed_ids, fixed_sizes = [], set(), dict.fromkeys(SIZES, 0)
    for anchor in accepted_raw_private_pairs:
        exact(anchor, ('image_id', 'GT_annotation_id', 'assigned_native_rank', 'assigned_query_id', 'by_GT'),
              'accepted raw fixed-pair anchor')
        image, aid = anchor['image_id'], anchor['GT_annotation_id']
        integer_ann = cocos[INTEGER].anns[aid]
        need(type(image) is type(aid) is type(anchor['assigned_native_rank']) is type(anchor['assigned_query_id']) is int
             and aid not in fixed_ids and image in raw_by_image and integer_ann['image_id'] == image
             and 1 <= anchor['assigned_native_rank'] <= 100 and 0 <= anchor['assigned_query_id'] < 100,
             'Unique accepted fixed-pair identity and assigned query')
        fixed_ids.add(aid)
        fixed_sizes[area_group(integer_ann['area'])] += 1
        exact(anchor['by_GT'], VARIANTS, 'accepted coverage by GT')
        for overlaps in anchor['by_GT'].values():
            exact(overlaps, ('raw', 'retained', 'assigned'), 'accepted raw coverage')
        need(anchor['by_GT'][INTEGER]['assigned'] >= .50, 'Accepted original integer 0.50 assigned pair')
        if all(anchor['by_GT'][variant]['raw'] < localization.THRESHOLD for variant in VARIANTS):
            cohort.append(anchor)

    totals = raw_geometry.decompose(accepted_raw_private_pairs)
    need(all(tuple(totals[variant][threshold][name] for name in ('raw', 'retained', 'assigned')) == counts
             for variant, thresholds in localization.ACCEPTED_COUNTS.items() for threshold, counts in thresholds.items()),
         'Exact previously accepted raw, retained and assigned coverage anchors')
    need(len(cohort) == COHORT_PAIRS and len({row['image_id'] for row in cohort}) == COHORT_IMAGES,
         'Exact archived dual raw-miss population')

    rows = []
    for anchor, row in zip(cohort, accepted_localization_rows):
        control.check()
        rows.append(rebuild_pair(np, anchor, row, raw_by_image[anchor['image_id']], cocos, modules))
    actual_capacities = {region: tuple(sum(row['capacities'][region][sign] for row in rows) for sign in ('FN', 'FP'))
                         for region in REGIONS}
    need(actual_capacities == ACCEPTED_CAPACITIES, 'Exact accepted aggregate band and outside capacities')
    control.check()

    summary = dict(schema='Root-RFDETR-E24-dual-GT-signed-edit-oracle-v1',
                   status='SIGNED_EDIT_ORACLE_COMPLETE_PENDING_ROOT_ACCEPTANCE', qualified=False,
                   TEST=False, AP_recomputed=False, COCO_matching_recomputed=False, new_training=False,
                   scope=dict(images=DEV_IMAGES, GT_instances=GT_INSTANCES, fixed_pairs=FIXED_PAIRS,
                              cohort_pairs=COHORT_PAIRS, cohort_images=COHORT_IMAGES,
                              other_fixed_pairs_not_evaluated=FIXED_PAIRS-COHORT_PAIRS,
                              original_unpaired_GT=GT_INSTANCES-FIXED_PAIRS),
                   threshold=list(oracle.THRESHOLD), budgets=list(oracle.BUDGETS), radius_pixels=localization.RADIUS,
                   area_partition='Original integer COCO annotation area: [0,1024), [1024,9216), [9216,1e10]. '
                                  'Exact shared COCO boundaries are assigned to the larger group for disjoint totals.',
                   counts='Newly reaches threshold within the measured cohort. The fixed denominator includes '
                          'unevaluated pairs, so these counts are not a whole-population edited result.',
                   input_IoU_and_localization_exact=True, no_new_query_rank_search=True,
                   capacities={region: dict(FN=counts[0], FP=counts[1]) for region, counts in actual_capacities.items()},
                   groups=summarize(rows, fixed_sizes, oracle.BUDGETS))
    return dict(summary=summary, private_pair_records=rows)
