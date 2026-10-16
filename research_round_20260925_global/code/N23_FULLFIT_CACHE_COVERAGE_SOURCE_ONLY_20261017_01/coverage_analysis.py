# encoding: utf-8
"""UNBOUND coverage arithmetic only: no I/O, model, tensor-library or optimizer import.

The caller supplies already-validated N23 adapter output. Coverage does not prove
cache provenance, true target pixels, unbiased GT-reduced weights or training.
"""
from collections import Counter
import math
import operator


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def _int(value):
    require(not isinstance(value, bool), 'Boolean is not a count/id')
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise RuntimeError('Exact integer count/id required') from exc


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _weights(values):
    values = [float(v) for v in values]
    require(all(math.isfinite(v) and v > 0 for v in values), 'Finite positive original weights required')
    total = math.fsum(values)
    return dict(count=len(values), sum=total, min=min(values) if values else None,
                max=max(values) if values else None, mean=_ratio(total, len(values)))


def _merge_weights(stats):
    used = [s for s in stats if s['count']]
    count = sum(s['count'] for s in used)
    total = math.fsum(s['sum'] for s in used)
    return dict(count=count, sum=total, min=min(s['min'] for s in used) if used else None,
                max=max(s['max'] for s in used) if used else None, mean=_ratio(total, count))


def _hist(values):
    return {str(k): v for k, v in sorted(Counter(values).items())}


def _coordinate_weights(pixels, weights, flags):
    grouped = {}
    for pixel, weight, selected in zip(pixels, weights, flags):
        if selected:
            grouped.setdefault(pixel, []).append(weight)
    collapsed = [math.fsum(values) for values in grouped.values()]
    total = math.fsum(collapsed)
    squared = math.fsum(value*value for value in collapsed)
    return dict(unique=len(collapsed), sum=total, square_sum=squared,
                ESS=_ratio(total*total, squared))


def analyze_image(dataset_index, image_id, adapter_result):
    """Summarize one original4096-draw image; retain only small per-real-GT records."""
    index, image = _int(dataset_index), _int(image_id)
    require(index >= 0, 'Nonnegative original ordinal required')
    require(adapter_result['schema'] == 'N23-shared-original-C-matched-query-batches-v1',
            'Original shared-C adapter output required')
    audit, batches = adapter_result['audit'], adapter_result['batches']
    require(_int(audit['original_draws']) == 4096 and _int(audit['ignored_draws']) == 0
            and audit['K0_queries_present_in_original_sample'] is False,
            'Original4096 draws/no ignored/no sampled K0 contract required')
    GT, positions_seen, records, qids = _int(audit['real_GT_instances']), set(), [], []
    require(GT > 0, 'Original positive real-GT population required')
    for batch in batches:
        require(_int(batch['dataset_index']) == index and _int(batch['image_id']) == image,
                'Batch belongs to another image')
        n = _int(batch['n']); q = _int(batch['query_id'])
        require(n > 0 and 0 <= q < 100, 'Nonempty natural-query batch required')
        slots = [_int(v) for v in batch['sample_positions']]
        pixels = [_int(v) for v in batch['flat_indices']]
        omega = [bool(v) for v in batch['Omega_flag']]
        candidates = [bool(v) for v in batch['C_flag']]
        weights = [float(v) for v in batch['importance_weights']]
        require(all(len(v) == n for v in (slots, pixels, omega, candidates, weights)), 'Batch length drift')
        require(slots == sorted(set(slots)) and all(0 <= p < 4096 for p in slots)
                and not positions_seen.intersection(slots), 'Original draw slot reused or out of order')
        positions_seen.update(slots)
        require(all(p >= 0 for p in pixels) and all(candidates), 'Shared C coordinates/flag drift')
        stats = _weights(weights); editable = sum(omega); unique = len(set(pixels))
        C_weights = _coordinate_weights(pixels, weights, candidates)
        Omega_weights = _coordinate_weights(pixels, weights, omega)
        records.append(dict(GT_index=_int(batch['GT_index']), GT_annotation_id=_int(batch['GT_annotation_id']),
            query_id=q, sampled=True, K=None, matched_draws=n, Omega_points=editable,
            Omega_fraction=_ratio(editable, n), unique_query_pixel_pairs=unique,
            matched_duplicate_draws=n-unique, weight_stats=stats,
            Omega_weight_sum=Omega_weights['sum'],
            Omega_unique_query_pixel_pairs=Omega_weights['unique'],
            matched_coordinate_weight_sum=C_weights['sum'],
            matched_coordinate_weight_square_sum=C_weights['square_sum'],
            matched_coordinate_ESS=C_weights['ESS'],
            Omega_coordinate_weight_square_sum=Omega_weights['square_sum'],
            Omega_coordinate_ESS=Omega_weights['ESS']))
        qids.append(q)
    require(qids == audit['matched_query_ids'] and len(set(qids)) == len(qids)
            and len(batches) == _int(audit['nonempty_matched_batches']), 'Nonempty matched batch population drift')
    for missing in audit['unsampled_real_GT']:
        K = _int(missing['K']); q = _int(missing['query_id'])
        require(K >= 0 and 0 <= q < 100, 'Original unsampled query/K metadata required')
        records.append(dict(GT_index=_int(missing['GT_index']), GT_annotation_id=_int(missing['GT_annotation_id']),
            query_id=q, sampled=False, K=K, matched_draws=0, Omega_points=0, Omega_fraction=None,
            unique_query_pixel_pairs=0, matched_duplicate_draws=0, weight_stats=_weights([]), Omega_weight_sum=0.,
            Omega_unique_query_pixel_pairs=0, matched_coordinate_weight_sum=0.,
            matched_coordinate_weight_square_sum=0., matched_coordinate_ESS=None,
            Omega_coordinate_weight_square_sum=0., Omega_coordinate_ESS=None))
    require(len(records) == GT and sorted(r['GT_index'] for r in records) == list(range(GT))
            and len({r['GT_annotation_id'] for r in records}) == GT
            and len({r['query_id'] for r in records}) == GT, 'Real matched-GT coverage/identity mismatch')
    records.sort(key=lambda r: r['GT_index'])
    matched = sum(r['matched_draws'] for r in records)
    background = _int(audit['dropped_background_draws'])
    omega_points = sum(r['Omega_points'] for r in records)
    require(background >= 0 and matched == _int(audit['retained_matched_draws'])
            and matched + background == 4096 and matched == len(positions_seen)
            and omega_points == _int(audit['Omega_editable_draws'])
            and matched == _int(audit['C_editable_draws']), 'Original/matched/background/editable counts disagree')
    C_sum = math.fsum(r['matched_coordinate_weight_sum'] for r in records)
    C_square = math.fsum(r['matched_coordinate_weight_square_sum'] for r in records)
    O_sum = math.fsum(r['Omega_weight_sum'] for r in records)
    O_square = math.fsum(r['Omega_coordinate_weight_square_sum'] for r in records)
    return dict(schema='N23-image-shared-C-coverage-v1', dataset_index=index, image_id=image,
        original_draws=4096, real_GT_instances=GT, nonempty_matched_batches=len(batches),
        unsampled_GT_K0=sum(not r['sampled'] and r['K'] == 0 for r in records),
        unsampled_GT_K_positive=sum(not r['sampled'] and r['K'] > 0 for r in records),
        Omega_zero_point_batches=sum(r['sampled'] and r['Omega_points'] == 0 for r in records),
        matched_draws=matched, background_draws=background, Omega_points=omega_points,
        Omega_fraction_of_matched_draws=_ratio(omega_points, matched),
        matched_unique_query_pixel_pairs=sum(r['unique_query_pixel_pairs'] for r in records),
        matched_duplicate_draws=sum(r['matched_duplicate_draws'] for r in records),
        matched_weight_stats=_merge_weights([r['weight_stats'] for r in records]),
        Omega_weight_sum=O_sum,
        Omega_unique_query_pixel_pairs=sum(r['Omega_unique_query_pixel_pairs'] for r in records),
        matched_coordinate_weight_sum=C_sum, matched_coordinate_weight_square_sum=C_square,
        matched_coordinate_ESS=_ratio(C_sum*C_sum, C_square),
        Omega_coordinate_weight_square_sum=O_square, Omega_coordinate_ESS=_ratio(O_sum*O_sum, O_square),
        GT_records=records)


def aggregate_images(image_records, *, expected_images=741, expected_real_GT=1661):
    """Require complete declared image/GT coverage; do not choose loss/step policy."""
    expected_images, expected_real_GT = _int(expected_images), _int(expected_real_GT)
    require(expected_images > 0 and expected_real_GT > 0, 'Positive locked denominators required')
    images = list(image_records)
    require(len(images) == expected_images
            and sorted(r['dataset_index'] for r in images) == list(range(expected_images))
            and len({r['image_id'] for r in images}) == expected_images,
            'Full original image ordinal/identity coverage required')
    require(all(r['schema'] == 'N23-image-shared-C-coverage-v1' and r['original_draws'] == 4096 for r in images),
            'Only checked4096-draw image records accepted')
    GT_records = [g for image in images for g in image['GT_records']]
    require(len(GT_records) == expected_real_GT and sum(r['real_GT_instances'] for r in images) == expected_real_GT
            and len({g['GT_annotation_id'] for g in GT_records}) == expected_real_GT,
            'Complete unique real-GT denominator required')
    sum_keys = ('original_draws', 'real_GT_instances', 'nonempty_matched_batches', 'unsampled_GT_K0',
        'unsampled_GT_K_positive', 'Omega_zero_point_batches', 'matched_draws', 'background_draws',
        'Omega_points', 'matched_unique_query_pixel_pairs', 'matched_duplicate_draws')
    summary = {k: sum(image[k] for image in images) for k in sum_keys}
    C_sum = math.fsum(r['matched_coordinate_weight_sum'] for r in images)
    C_square = math.fsum(r['matched_coordinate_weight_square_sum'] for r in images)
    O_sum = math.fsum(r['Omega_weight_sum'] for r in images)
    O_square = math.fsum(r['Omega_coordinate_weight_square_sum'] for r in images)
    require(summary['original_draws'] == 4096*expected_images
            and summary['matched_draws'] + summary['background_draws'] == summary['original_draws']
            and summary['nonempty_matched_batches'] + summary['unsampled_GT_K0']
                + summary['unsampled_GT_K_positive'] == expected_real_GT,
            'Global point/real-GT accounting mismatch')
    ratios = [g['Omega_fraction'] for g in GT_records if g['sampled']]
    summary.update(images=expected_images,
        Omega_fraction_of_matched_draws=_ratio(summary['Omega_points'], summary['matched_draws']),
        Omega_fraction_per_nonempty_GT_mean=_ratio(math.fsum(ratios), len(ratios)),
        Omega_fraction_per_nonempty_GT_min=min(ratios) if ratios else None,
        Omega_fraction_per_nonempty_GT_max=max(ratios) if ratios else None,
        matched_draws_per_real_GT_histogram=_hist(g['matched_draws'] for g in GT_records),
        Omega_points_per_nonempty_GT_histogram=_hist(g['Omega_points'] for g in GT_records if g['sampled']),
        matched_weight_stats=_merge_weights([r['matched_weight_stats'] for r in images]),
        Omega_weight_sum=O_sum,
        Omega_unique_query_pixel_pairs=sum(r['Omega_unique_query_pixel_pairs'] for r in images),
        matched_coordinate_weight_sum=C_sum, matched_coordinate_weight_square_sum=C_square,
        matched_coordinate_ESS=_ratio(C_sum*C_sum, C_square),
        Omega_coordinate_weight_square_sum=O_square, Omega_coordinate_ESS=_ratio(O_sum*O_sum, O_square),
        coordinate_ESS_is_weight_concentration_only_not_statistical_independence=True,
        duplicate_scope='retained matched (query,pixel) draws only; distinct queries stay distinct',
        background_duplicate_statistics_available=False, background_weight_statistics_available=False,
        importance_weights_claimed_unbiased_after_GT_reduction=False,
        A_zero_data_gradient_step_rule='UNDETERMINED', loss_normalization_selected=False,
        real_intake_qualified_by_kernel=False, training_executed=False)
    summary['Omega_fraction_of_matched_weight_sum'] = _ratio(
        summary['Omega_weight_sum'], summary['matched_weight_stats']['sum'])
    return dict(schema='N23-fullFIT-shared-C-coverage-summary-v1', summary=summary,
                images=sorted(images, key=lambda r: r['dataset_index']))
