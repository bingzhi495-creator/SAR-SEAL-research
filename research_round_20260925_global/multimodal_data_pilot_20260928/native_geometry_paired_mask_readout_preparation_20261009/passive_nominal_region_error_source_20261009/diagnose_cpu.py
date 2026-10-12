"""Inert ROOT CPU nominal sampling-rectangle error association, no AP/model run.

CLI --spec ABS --out FRESH_ABS. Required SPEC schema:
ROOT-native-paired-passive-nominal-region-error-CPU-spec-v1. References have
{path:absolute,bytes:int,sha256:lowercase64hex}: entry_source,
association_source, association_spec, association_report,
population_GPU_receipt, evaluation_GPU_receipt, DEV. sources is the unique
code-only closure containing this entry and ALL original association code refs.
Other fields: phase, expected_epoch=12, evaluation_images=187, raw_GT=380,
queries=18700, expected_E0_queries=818, CPU_affinity=[0,1], CPU_threads=2,
CUDA_visible_devices='', automatic_retry=False. The original association SPEC
and original SG12 SPEC remain unchanged. Both raw actual0 receipts precede
scientific payload hashing/reads/imports; their original metadata gates still
hash all historical bindings, including checkpoint refs, without deserializing.

All18700 queries retain actual captured geometry/E identities. Error counts use
all380 paired-on stock-assigned annotations, fixed query IDs, never best-IoU
selection/rematching. The nominal frame is |major|<=1.5*sigma_major and
|minor|<=2.5*sigma_minor in F200 units. It includes unused/ineligible positions;
it is NOT an eligible hull, receptive field, GT domain or scattering footprint.
Original integer pixel centers map to F by
((pixel+.5)*actual_img_hw/ori_hw)/4-.5, not a nominal scale_factor.
E0 domains are undefined: full counts remain, region fields are None.
Regional raw integer counts partition full counts exactly; regional IoUs and
regional-GT-normalized ratios are deliberately omitted. Transition/fullGT
normalizations and event/opportunity rates have distinct explicit denominators.
The all818 E0 mask identity proof is inherited from the qualified association,
not replayed here. Only posterior spatial associations are reported. No model,
checkpoint deserialization, AP, optimizer, training, intervention or causal claim.
"""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path


SCHEMA = 'ROOT-native-paired-passive-nominal-region-error-CPU-spec-v1'
ASSOCIATION_BYTES = 26472
ASSOCIATION_SHA = '1d78b5884d69a8beda740df173f788c8388cc1bd260ce9871933cd811840f8c2'
INTEGER_KEYS = ('GT_pixels', 'on_mask_pixels', 'off_mask_pixels', 'on_TP', 'on_FP',
                'on_FN', 'off_TP', 'off_FP', 'off_FN', 'FN_added', 'FN_repaired',
                'FP_added', 'FP_removed', 'delta_TP', 'delta_FP', 'delta_FN')
EVENT_OPPORTUNITIES = {'FN_repaired': 'off_FN', 'FN_added': 'off_TP',
                       'FP_added': 'off_TN', 'FP_removed': 'off_FP'}
REGIONS = ('near', 'far')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def ref_path(row):
    require(isinstance(row, dict) and all(k in row for k in ('path', 'bytes', 'sha256')) and
            isinstance(row['path'], str) and Path(row['path']).is_absolute() and
            type(row['bytes']) is int and row['bytes'] >= 0 and isinstance(row['sha256'], str) and
            len(row['sha256']) == 64 and set(row['sha256']) <= set('0123456789abcdef'), 'Malformed absolute reference')
    return Path(row['path'])


def read_json(path):
    with Path(path).open('r', encoding='utf-8') as stream:
        return json.load(stream)


def reference(path):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def bound(row):
    path = ref_path(row)
    require(reference(path) == {key: row[key] for key in ('path', 'bytes', 'sha256')}, 'Reference changed: ' + str(path))
    return path


def terminal(receipt):
    require(receipt.get('status') == 'ACTUAL_OWNED_TERMINAL' and receipt.get('original_identity') is not None and
            type(receipt.get('returncode')) is int and receipt['returncode'] == 0 and receipt.get('root_wait_reaped') is True and
            receipt.get('held_handle_exit') == {'wait_state': 'WAIT_OBJECT_0', 'actual_exit_code': 0} and
            receipt.get('numeric_birth_absence', {}).get('original_birth_absent') is True and
            'abort' in receipt and receipt['abort'] is None and 'error' in receipt and receipt['error'] is None and
            receipt.get('wait_errors') is False and receipt.get('GPU_observation_errors') == [],
            'STOP before payloads/imports: actual owned0/reap/birth absence/no errors required')


def load_source(row, name):
    definition = importlib.util.spec_from_file_location(name, str(bound(row)))
    module = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(module)
    return module


def metadata_gate(spec_path, spec):
    require(spec.get('schema') == SCHEMA and isinstance(spec.get('phase'), str) and spec['phase'] and
            spec['expected_epoch'] == 12 and spec['evaluation_images'] == 187 and spec['raw_GT'] == 380 and
            spec['queries'] == 18700 and spec['expected_E0_queries'] == 818 and
            spec['CPU_affinity'] == [0, 1] and spec['CPU_threads'] == 2 and spec['CUDA_visible_devices'] == '' and
            spec['automatic_retry'] is False, 'Fixed E12/all-population/CPU nominal-region SPEC required')
    # These are raw receipt reads, before hashing qualifications or other payloads.
    terminal(read_json(ref_path(spec['population_GPU_receipt'])))
    terminal(read_json(ref_path(spec['evaluation_GPU_receipt'])))
    bound(spec['population_GPU_receipt']); bound(spec['evaluation_GPU_receipt'])
    require(spec['entry_source'] == reference(__file__) and spec['entry_source'] in spec['sources'] and
            len({row['path'] for row in spec['sources']}) == len(spec['sources']), 'Own unique code closure required')
    for row in spec['sources']:
        require(ref_path(row).suffix.lower() in ('.py', '.pyi'), 'Closure must contain code only')
        bound(row)
    require(spec['association_source'] in spec['sources'] and spec['association_source']['bytes'] == ASSOCIATION_BYTES and
            spec['association_source']['sha256'] == ASSOCIATION_SHA, 'Immutable qualified association source required')
    association_spec_path = bound(spec['association_spec']); association_spec = read_json(association_spec_path)
    require(association_spec['population_GPU_receipt'] == spec['population_GPU_receipt'] and
            association_spec['DEV'] == spec['DEV'] and all(row in spec['sources'] for row in association_spec['sources']),
            'Original association receipt/DEV/code closure must remain unchanged')
    original_SG_spec = read_json(bound(association_spec['SG12_spec']))
    require(original_SG_spec['GPU_receipt'] == spec['evaluation_GPU_receipt'], 'Evaluation actual0 receipt differs from original SG12')
    association = load_source(spec['association_source'], 'original_qualified_association_metadata')
    SG, SG_spec, evaluation, qualification, population, original_provenance = association.metadata_gate(
        association_spec_path, association_spec)
    report = read_json(bound(spec['association_report']))
    require(report['schema'] == 'ROOT-native-paired-passive-error-association-CPU-report-v1' and
            report['status'] == 'ROOT_PASSIVE_PAIRED_ERROR_ASSOCIATION_CPU_DIAGNOSED' and
            report['provenance'] == original_provenance and report['DEV'] == spec['DEV'] and
            report['expected_epoch'] == 12 and report['checkpoint'] == population['checkpoint'] and
            report['queries'] == 18700 and report['stock_matched_GT'] == 380 and report['E0_queries'] == 818 and
            report['E0_masks_independently_decoded_exact_same'] == 818 and
            report['CPU_affinity'] == [0, 1] and report['CPU_threads'] == 2 and report['CUDA_initialized'] is False and
            report['AP_recomputed'] is report['best_IoU_query_selection'] is report['posterior_assignment_recomputed'] is
            report['query_filter'] is report['GT_filter'] is report['model_loaded'] is report['checkpoint_loaded'] is
            report['model_forward'] is report['optimizer'] is report['automatic_retry'] is False and
            report['scientific_method_verdict'] is None, 'Qualified full fixed-q association contract differs')
    provenance = dict(spec=reference(spec_path), source=reference(__file__), sources=spec['sources'],
        association_source=spec['association_source'], association_spec=spec['association_spec'],
        association_report=spec['association_report'], population_GPU_receipt=spec['population_GPU_receipt'],
        evaluation_GPU_receipt=spec['evaluation_GPU_receipt'], original_association_provenance=original_provenance,
        original_metadata_gates_hash_historical_noncode_refs=True, checkpoint_deserialized=False)
    return association, SG, SG_spec, evaluation, qualification, population, report, association_spec, provenance


def counts_change(gt, on, off):
    """Exact binary pixel transitions; an empty GT has undefined /GT ratios."""
    import numpy as np
    arrays = [np.asarray(value) for value in (gt, on, off)]
    require(all(a.ndim == 2 and a.shape == arrays[0].shape and a.dtype.kind in ('b', 'i', 'u', 'f') and
                bool(np.isfinite(a).all()) and bool(((a == 0) | (a == 1)).all()) for a in arrays),
            'Same-shape finite exact binary GT/on/off masks required')
    g, a, b = [value.astype(bool, copy=False) for value in arrays]
    count = lambda value: int(value.sum(dtype=np.int64))
    area = count(g)
    result = dict(GT_pixels=area, on_mask_pixels=count(a), off_mask_pixels=count(b),
        on_TP=count(g & a), on_FP=count(~g & a), on_FN=count(g & ~a),
        off_TP=count(g & b), off_FP=count(~g & b), off_FN=count(g & ~b),
        FN_added=count(g & b & ~a), FN_repaired=count(g & ~b & a),
        FP_added=count(~g & ~b & a), FP_removed=count(~g & b & ~a))
    for label in ('on', 'off'):
        union = result[label + '_TP'] + result[label + '_FP'] + result[label + '_FN']
        result[label + '_IoU'] = result[label + '_TP'] / union if union else 1.0
    result['delta_IoU'] = result['on_IoU'] - result['off_IoU']
    for name in ('TP', 'FP', 'FN'):
        result['delta_' + name] = result['on_' + name] - result['off_' + name]
    require(result['delta_FN'] == result['FN_added'] - result['FN_repaired'] and
            result['delta_FP'] == result['FP_added'] - result['FP_removed'] and
            result['on_TP'] + result['on_FN'] == result['off_TP'] + result['off_FN'] == area,
            'Pixel transition conservation failed')
    for key in tuple(result):
        if key not in ('GT_pixels', 'on_IoU', 'off_IoU', 'delta_IoU'):
            result[key + '_per_GT_pixel'] = None if area == 0 else result[key] / area
    return result


def nominal_rectangle(np, metadata, centre, theta, sigmas):
    """Closed fixed rectangle on original pixel centers, F200 units, no GT."""
    require(isinstance(metadata, dict) and metadata.get('batch_input_shape') == [800, 800] and
            all(isinstance(metadata.get(name), (list, tuple)) and len(metadata[name]) >= 2 for name in ('img_shape', 'ori_shape')),
            'Actual image/original/canvas geometry metadata required')
    ih, iw = metadata['img_shape'][:2]; oh, ow = metadata['ori_shape'][:2]
    require(all(type(v) is int and v > 0 for v in (ih, iw, oh, ow)) and ih <= 800 and iw <= 800,
            'Positive integer actual image/original dimensions within800 required')
    require(isinstance(centre, (list, tuple)) and len(centre) == 2 and isinstance(sigmas, (list, tuple)) and len(sigmas) == 2,
            'Centre xy/two sigmas required')
    values = tuple(centre) + (theta,) + tuple(sigmas)
    require(all(type(v) in (int, float) and math.isfinite(v) for v in values) and all(v >= 1 for v in sigmas),
            'Finite center/theta and sigma floor1 required')
    cx, cy = centre; major_sigma, minor_sigma = sigmas
    x = ((np.arange(ow, dtype=np.float64) + .5) * (iw / ow)) / 4. - .5
    y = ((np.arange(oh, dtype=np.float64) + .5) * (ih / oh)) / 4. - .5
    cosine, sine = math.cos(theta), math.sin(theta)
    dx, dy = x[None, :] - cx, y[:, None] - cy
    major = cosine * dx + sine * dy
    minor = -sine * dx + cosine * dy
    domain = (np.abs(major) <= 1.5 * major_sigma) & (np.abs(minor) <= 2.5 * minor_sigma)
    require(domain.shape == (oh, ow) and domain.dtype == np.bool_, 'Original pixel-center nominal rectangle shape differs')
    return domain


def spatial_counts(gt, on, off, domain):
    """Raw near/far partition, fullGT-normalized events and opportunity rates."""
    import numpy as np
    full = counts_change(gt, on, off)
    full_GT = full['GT_pixels']; g, a, b = [np.asarray(value).astype(bool, copy=False) for value in (gt, on, off)]
    if domain is None:
        return dict(domain_defined=False, full_counts={key: full[key] for key in INTEGER_KEYS},
                    full_original_GT_pixels=full_GT, near=None, far=None)
    d = np.asarray(domain)
    require(d.shape == g.shape and d.ndim == 2 and d.dtype == np.bool_, 'Same-shape Boolean nominal domain required')
    image_area = int(d.size); result = dict(domain_defined=True,
        full_counts={key: full[key] for key in INTEGER_KEYS}, full_original_GT_pixels=full_GT)
    for label, region in (('near', d), ('far', ~d)):
        local = counts_change(g & region, a & region, b & region)
        integers = {key: local[key] for key in INTEGER_KEYS}
        area = int(region.sum(dtype=np.int64)); off_TN = area - integers['GT_pixels'] - integers['off_FP']
        require(off_TN >= 0, 'Regional true-negative opportunity became negative')
        rates = {}
        for event, opportunity in EVENT_OPPORTUNITIES.items():
            denominator = off_TN if opportunity == 'off_TN' else integers[opportunity]
            numerator = integers[event]
            require(0 <= numerator <= denominator, 'Pixel transition exceeds its off-endpoint opportunity')
            rates[event] = dict(events=numerator, opportunities=denominator,
                denominator_name=opportunity, rate=None if denominator == 0 else numerator / denominator)
        result[label] = dict(raw_counts=integers, area_pixels=area, image_area_pixels=image_area,
            image_area_fraction=area / image_area if image_area else None, GT_region_pixels=integers['GT_pixels'],
            full_original_GT_pixels=full_GT, GT_region_fraction_of_full_GT=None if full_GT == 0 else integers['GT_pixels'] / full_GT,
            off_TN=off_TN, transition_per_full_original_GT_pixel={event: None if full_GT == 0 else integers[event] / full_GT
                for event in EVENT_OPPORTUNITIES}, opportunity_rates=rates)
    require(result['near']['area_pixels'] + result['far']['area_pixels'] == image_area and
            all(result['near']['raw_counts'][key] + result['far']['raw_counts'][key] == full[key] for key in INTEGER_KEYS),
            'Nominal near/far integer counts do not partition full original-grid counts')
    return result


def aggregate_regions(rows):
    """Sum event/sum opportunity rates, not averages of per-GT rates."""
    result = dict(stock_GT=len(rows), defined_domain_GT=sum(row['regions']['domain_defined'] for row in rows),
        undefined_E0_domain_GT=sum(not row['regions']['domain_defined'] for row in rows), regions={})
    for label in REGIONS:
        available = [row['regions'][label] for row in rows if row['regions']['domain_defined']]
        summed = {key: sum(value['raw_counts'][key] for value in available) for key in INTEGER_KEYS}
        area = sum(value['area_pixels'] for value in available)
        image_area = sum(value['image_area_pixels'] for value in available)
        full_GT = sum(value['full_original_GT_pixels'] for value in available)
        rates = {}
        for event, opportunity in EVENT_OPPORTUNITIES.items():
            events = sum(value['opportunity_rates'][event]['events'] for value in available)
            opportunities = sum(value['opportunity_rates'][event]['opportunities'] for value in available)
            require(0 <= events <= opportunities, 'Aggregate transition/opportunity inconsistency')
            rates[event] = dict(events=events, opportunities=opportunities, denominator_name=opportunity,
                rate=None if opportunities == 0 else events / opportunities)
        result['regions'][label] = dict(available_stock_GT=len(available), raw_count_sums=summed,
            area_pixel_sum=area, image_area_pixel_sum=image_area,
            image_area_fraction_pooled=None if image_area == 0 else area / image_area,
            GT_region_pixel_sum=summed['GT_pixels'], full_original_GT_pixel_sum_of_defined_domains=full_GT,
            GT_region_fraction_pooled=None if full_GT == 0 else summed['GT_pixels'] / full_GT,
            off_TN_sum=sum(value['off_TN'] for value in available), opportunity_rates=rates,
            transition_per_full_original_GT_pixel_pooled={event: None if full_GT == 0 else summed[event] / full_GT
                for event in EVENT_OPPORTUNITIES},
            transition_per_full_original_GT_pixel_macro={event: dict(available_GT=sum(value['full_original_GT_pixels'] > 0 for value in available),
                mean=(math.fsum(value['transition_per_full_original_GT_pixel'][event] for value in available
                    if value['full_original_GT_pixels'] > 0) / sum(value['full_original_GT_pixels'] > 0 for value in available))
                    if any(value['full_original_GT_pixels'] > 0 for value in available) else None)
                for event in EVENT_OPPORTUNITIES})
    return result


def diagnose(spec_file, out_file):
    spec_path, out = Path(spec_file), Path(out_file)
    require(spec_path.is_absolute() and out.is_absolute() and not out.exists() and out.parent.is_dir(), 'Absolute SPEC/fresh output required')
    spec_path = spec_path.resolve(strict=True); spec = read_json(spec_path)
    A, SG, SG_spec, evaluation, qualification, population, previous, A_spec, provenance = metadata_gate(spec_path, spec)
    SG.cpu_environment(); endpoint_reader = SG.load_reader(SG_spec['evaluation_CPU_source'])
    pop_reader = A.load_source(A_spec['population_CPU_reader'], 'original_population_capture_array_helpers')
    import torch
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools import mask as masks
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    require(not torch.cuda.is_initialized(), 'CPU nominal-region diagnostic must not initialize CUDA')
    coco = COCO(str(bound(spec['DEV'])))
    require(len(coco.imgs) == 187 and len(coco.anns) == 380 and set(coco.cats) == {1}, 'All original DEV187/380/category1 required')
    ids, endpoints = SG.endpoints(endpoint_reader, evaluation, qualification, coco, np, masks)
    require(previous['endpoint_inputs'] == {label: value['input_refs'] for label, value in endpoints.items()},
            'Current official on/off endpoints differ from qualified association')
    image_rows = read_json(bound(population['source_outputs']['image_rows']))
    captures = read_json(bound(population['source_outputs']['capture_rows']))
    require(len(image_rows) == len(captures) == 187 and [row['image_id'] for row in image_rows] == ids and
            [row['image_id'] for row in captures] == ids, 'Complete ordered current population image/capture rows required')
    expected = {(image_id, qid) for image_id in ids for qid in range(100)}
    query_map = {(row['image_id'], row['query_id']): row for row in population['query_rows']}
    old_query_map = {(row['image_id'], row['query_id']): row for row in previous['all_query_rows']}
    assigned = {(row['image_id'], row['annotation_id']): row for row in population['raw_GT_assignment_rows']}
    old_GT = {(row['image_id'], row['annotation_id']): row for row in previous['raw_stock_GT_rows']}
    require(len(population['query_rows']) == len(previous['all_query_rows']) == len(query_map) == len(old_query_map) == 18700 and
            set(query_map) == set(old_query_map) == expected and len(population['raw_GT_assignment_rows']) ==
            len(previous['raw_stock_GT_rows']) == len(assigned) == len(old_GT) == 380 and set(assigned) == set(old_GT) ==
            {(int(a['image_id']), int(a['id'])) for a in coco.anns.values()}, 'All query/fixed-stock annotation identities required')
    require(sum(row['E0'] for row in query_map.values()) == previous['E0_masks_independently_decoded_exact_same'] == 818,
            'Inherited all818 E0 identity population differs')
    query_rows, GT_rows = [], []
    for index, (image_id, row, captured) in enumerate(zip(ids, image_rows, captures)):
        image = coco.imgs[image_id]; anns = coco.imgToAnns[image_id]
        ann_map = {ann['id']: ann for ann in anns}; ann_ids = [ann['id'] for ann in anns]
        metadata = row['metadata']
        scale = 800. / max(image['height'], image['width'])
        actual = [int(image['height'] * scale + .5), int(image['width'] * scale + .5)]
        require(row['dataset_index'] == captured['dataset_index'] == index and row['image_id'] == captured['image_id'] == image_id and
                row['ann_ids'] == captured['ann_ids'] == ann_ids and metadata['ori_shape'][:2] == [image['height'], image['width']] and
                metadata['img_shape'][:2] == actual and metadata['batch_input_shape'] == [800, 800] and
                row['checkpoint'] == population['checkpoint'] and row['mode'] == 'paired' and row['apply'] is True,
                'Actual all-GT image/capture/geometry/same-CP joins differ')
        parsed = pop_reader.capture_arrays(np, captured['capture'], metadata)
        require(parsed['eligible'].shape == (100, 16) and parsed['centre_xy'].shape == (100, 2) and
                all(parsed[name].shape == (100,) for name in ('used_theta', 'sigma_major', 'sigma_minor', 'rho_mass')),
                'Population capture_arrays already removes the singleton batch; actual query shapes differ')
        seen_ann = set()
        for qid in range(100):
            key = (image_id, qid); current, prior = query_map[key], old_query_map[key]
            annotation_id = current['stock_assigned_annotation_id']; E = parsed['eligible'][qid]
            E_count = int(E.sum(dtype=np.int64))
            require(current['query_id'] == prior['query_id'] == qid and current['unmatched_query'] == prior['unmatched_query'] ==
                    (annotation_id is None) and annotation_id == prior['stock_assigned_annotation_id'] and
                    type(current['E0']) is bool and current['E0'] == prior['E0'] == (E_count == 0) and
                    current['eligible_pairs'] == prior['eligible_pairs'] == E_count and current['all16_slots_preserved'] is True and
                    current['outside_donor_slots_not_applied'] is True and current['own_GT_is_posterior_stock_assignment'] is True,
                    'Captured E/all-slot/fixed-stock query identity differs')
            centre = [float(value) for value in parsed['centre_xy'][qid]]
            theta = float(parsed['used_theta'][qid])
            sigmas = [float(parsed['sigma_major'][qid]), float(parsed['sigma_minor'][qid])]
            geometry = dict(centre_xy=centre, used_theta=theta, sigma_major=sigmas[0], sigma_minor=sigmas[1],
                eligible_slots=E.tolist(), eligible_pairs=E_count, E0=current['E0'],
                rho_mass=float(parsed['rho_mass'][qid]), domain_defined=not current['E0'])
            query_rows.append(dict(image_id=image_id, query_id=qid, stock_assigned_annotation_id=annotation_id,
                unmatched_query=annotation_id is None, geometry=geometry, regions=None))
            if annotation_id is None:
                continue
            require(annotation_id in ann_map and annotation_id not in seen_ann, 'Stock annotation absent/duplicated')
            seen_ann.add(annotation_id); mapping = assigned[(image_id, annotation_id)]; old = old_GT[(image_id, annotation_id)]
            require(mapping['query_id'] == old['query_id'] == qid and mapping['GT_index'] == old['GT_index'] == ann_ids.index(annotation_id) and
                    mapping['E0'] == old['E0'] == current['E0'] and mapping['eligible_pairs'] == old['eligible_pairs'] == E_count,
                    'Qualified fixed-stock annotation order/E join differs')
            on = A.decode_mask(np, masks, endpoints['on']['predictions'][key], image)
            off = A.decode_mask(np, masks, endpoints['off']['predictions'][key], image)
            gt = coco.annToMask(ann_map[annotation_id])
            full = A.counts_change(gt, on, off)
            require(full == old['counts'] == prior['counts'] and full['GT_pixels'] == mapping['original_pixels'] ==
                    current['original_GT_pixels'], 'Whole original-grid counts differ from qualified fixed-q association')
            domain = None if current['E0'] else nominal_rectangle(np, metadata, centre, theta, sigmas)
            regions = spatial_counts(gt, on, off, domain)
            require(regions['full_counts'] == {name: old['counts'][name] for name in INTEGER_KEYS},
                    'Independent regional helper full integers differ from qualified association')
            GT_rows.append(dict(image_id=image_id, annotation_id=annotation_id, query_id=qid, GT_index=mapping['GT_index'],
                E0=current['E0'], eligible_pairs=E_count, geometry=geometry, full_counts=full, regions=regions,
                iscrowd=int(ann_map[annotation_id].get('iscrowd', 0)), ignore=int(ann_map[annotation_id].get('ignore', 0))))
            del gt, on, off, domain, regions
        require(seen_ann == set(ann_map), 'A raw annotation was omitted or rematched')
        del parsed
    require(len(query_rows) == 18700 and len(GT_rows) == 380 and
            {(row['image_id'], row['annotation_id']) for row in GT_rows} == set(assigned), 'Full denominators differ')
    aggregate = aggregate_regions(GT_rows)
    require(aggregate['defined_domain_GT'] + aggregate['undefined_E0_domain_GT'] == 380 and
            aggregate['undefined_E0_domain_GT'] == sum(row['E0'] for row in GT_rows), 'E0/defined domain GT partition differs')
    require(not torch.cuda.is_initialized(), 'Diagnostic initialized CUDA')
    result = dict(schema='ROOT-native-paired-passive-nominal-region-error-CPU-report-v1',
        status='ROOT_PASSIVE_PAIRED_NOMINAL_REGION_ERROR_CPU_DIAGNOSED', phase=spec['phase'], provenance=provenance,
        DEV=spec['DEV'], expected_epoch=12, checkpoint=population['checkpoint'], queries=18700, stock_matched_GT=380,
        E0_queries=818, E0_masks_identity_inherited_from_qualified_association=818, E0_masks_identity_replayed_here=False,
        CPU_affinity=[0, 1], CPU_threads=2, CUDA_initialized=False, all_query_rows=query_rows, raw_stock_GT_rows=GT_rows,
        aggregate=aggregate, full_original_GT_pixels_all380=sum(row['full_counts']['GT_pixels'] for row in GT_rows),
        domain_definition=dict(coordinate_space='F200 pixel centers', major_sigma_multiplier=1.5, minor_sigma_multiplier=2.5,
            comparisons='closed <=', mapping='((raw_pixel+.5)*actual_img_hw/ori_hw)/4-.5', E0='undefined, regional None'),
        capture_inputs={key: population['source_outputs'][key] for key in ('image_rows', 'capture_rows')},
        endpoint_inputs={label: value['input_refs'] for label, value in endpoints.items()},
        full_integer_count_partitions_exact_to_qualified_association=True, region_IoU_reported=False,
        regional_GT_normalized_transition_reported=False, opportunities_pooled_by_sum_events_over_sum_opportunities=True,
        AP_recomputed=False, posterior_assignment_recomputed=False, best_IoU_query_selection=False,
        query_filter=False, GT_filter=False, model_loaded=False, checkpoint_deserialized=False, model_forward=False,
        optimizer=False, automatic_retry=False, scientific_method_verdict=None, causal_claimed=False,
        limits=[
            'Nominal complete sampling rectangles include unused/ineligible positions; they are not eligible hulls, receptive fields, GT domains or physical scattering footprints.',
            'Own means paired-on stock final-cost assignment; fixed identity is conditional, not semantic correctness or detection recall.',
            'Area and opportunity denominators accompany spatial changes; a large far-domain count alone does not establish leakage.',
            'E0 region fields are None and full counts remain. The all818 mask identity proof is inherited, not replayed here.',
            'Near/far integer counts add to qualified full counts; IoU is not additive. /fullGT normalization differs from event/opportunity rates.',
            'Region statistics cover380 stock-matched GT only, with undefined E0 excluded from region rates explicitly; all18700 query geometry identities remain.',
            'On/off removes the complete inside+outside branch; spatial association is not outside contamination, feature purity, pairing specificity or a training cause.',
            'Testing a domain restriction would require a separate deployable logit intervention. No such intervention, width search, AP or new model is run here.'])
    with out.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(status=result['status'], output=reference(out), scientific_method_verdict=None), sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True); parser.add_argument('--out', required=True)
    args = parser.parse_args(); diagnose(args.spec, args.out)


if __name__ == '__main__':
    main()
