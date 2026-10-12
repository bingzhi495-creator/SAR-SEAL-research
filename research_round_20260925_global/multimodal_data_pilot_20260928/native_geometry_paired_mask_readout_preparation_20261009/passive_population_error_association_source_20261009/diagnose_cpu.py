"""Inert ROOT CPU posterior sampling/error association; no model or AP run.

CLI --spec ABS --out FRESH_ABS. SPEC schema:
ROOT-native-paired-passive-error-association-CPU-spec-v1. Required references
({path:absolute,bytes:int,sha256:lowercase64hex}): entry_source, population_spec,
population_qualification, population_GPU_receipt, population_GPU_admission,
population_producer, population_CPU_reader, SG12_spec, SG12_diagnostic_source,
SG12_diagnostic_report, DEV; sources is the code-only closure containing this
entry, both population sources and the original SG diagnostic/evaluation code.
Other fields: phase, expected_epoch=12, evaluation_images=187, raw_GT=380,
queries=18700, expected_E0_queries=818, CPU_affinity=[0,1], CPU_threads=2,
CUDA_visible_devices='', automatic_retry=False. Group definitions are fixed:
outside eligible mean own occupancy>0 crossed with other-union occupancy>0;
nearest_800 is primary, original_proxy is sensitivity, E0 is separate.

The original SG12 SPEC is passed unchanged to the original SG module's
metadata_gate: its __file__ remains that original source, still in old sources.
Current population and original evaluation actual0 receipts precede payload
hashing/qualification reads/scientific imports. Only old metadata gates and
prediction/score helpers are called: no checkpoint_state/model/forward/COCO AP.
The reused population metadata gate still hashes ALL its original bindings,
including historical checkpoint/non-code refs; none are deserialized here.
All100 queries and all380 stock-assigned annotations remain; no best-IoU query
selection or rematching. E0 mask identity is tested for ALL818 queries.
Own means posterior paired-on stock assignment, not semantic identity/recall.
On/off removes the entire inside+outside branch, so these groups are posterior
associations, not outside-specific effects, feature purity or training causes.
"""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path


SCHEMA = 'ROOT-native-paired-passive-error-association-CPU-spec-v1'
SG_BYTES = 24472
SG_SHA = 'd947c4d601c877a88b3852919b0ea9cd1e17e7930ff22b590f02bd20ceb260c1'
POP_BYTES = 36710
POP_SHA = '0124bdd73c91b9ebcb635ca686d0cf948413eab3012db8971da5173ef4a586da'
READER_BYTES = 57641
READER_SHA = '35c2b02bba3d498017f7205a06b28aaf541b3e2be0160779b5442e53136c3442'
MEASURES = ('nearest_800', 'original_proxy')
GROUPS = ('own0_other0', 'own1_other0', 'own0_other1', 'own1_other1', 'E0')
BASE_METRICS = ('GT_pixels', 'on_mask_pixels', 'off_mask_pixels', 'on_TP', 'on_FP', 'on_FN',
                'off_TP', 'off_FP', 'off_FN', 'FN_added', 'FN_repaired', 'FP_added', 'FP_removed',
                'on_IoU', 'off_IoU', 'delta_IoU', 'delta_TP', 'delta_FP', 'delta_FN')
METRIC_KEYS = BASE_METRICS + tuple(key + '_per_GT_pixel' for key in BASE_METRICS
                                 if key not in ('GT_pixels', 'on_IoU', 'off_IoU', 'delta_IoU'))


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
            spec['automatic_retry'] is False, 'Fixed E12/all-population/CPU association SPEC required')
    # Read raw actual receipts before even hashing scientific qualification refs.
    pop_receipt = read_json(ref_path(spec['population_GPU_receipt']))
    terminal(pop_receipt)
    SG_spec_path = ref_path(spec['SG12_spec']); SG_spec = read_json(SG_spec_path)
    terminal(read_json(ref_path(SG_spec['GPU_receipt'])))
    bound(spec['population_GPU_receipt']); bound(spec['SG12_spec']); bound(SG_spec['GPU_receipt'])
    require(spec['entry_source'] == reference(__file__) and spec['entry_source'] in spec['sources'] and
            len({row['path'] for row in spec['sources']}) == len(spec['sources']), 'Own unique code closure required')
    for row in spec['sources']:
        require(ref_path(row).suffix.lower() in ('.py', '.pyi'), 'Closure must contain code only')
        bound(row)
    for role in ('population_producer', 'population_CPU_reader', 'SG12_diagnostic_source'):
        require(spec[role] in spec['sources'], 'Missing source role: ' + role)
    require(spec['SG12_diagnostic_source']['bytes'] == SG_BYTES and spec['SG12_diagnostic_source']['sha256'] == SG_SHA and
            spec['population_producer']['bytes'] == POP_BYTES and spec['population_producer']['sha256'] == POP_SHA and
            spec['population_CPU_reader']['bytes'] == READER_BYTES and spec['population_CPU_reader']['sha256'] == READER_SHA,
            'Known immutable source roles differ')
    SG = load_source(spec['SG12_diagnostic_source'], 'original_SG12_association_metadata_helpers')
    evaluation, qualification, SG_provenance = SG.metadata_gate(SG_spec_path, SG_spec)
    require(evaluation['expected_epoch'] == 12 and evaluation['DEV'] == SG_spec['DEV'] == spec['DEV'] and
            set(SG_spec['sources'][i]['path'] for i in range(len(SG_spec['sources']))) <=
            set(row['path'] for row in spec['sources']), 'Original SG12 SPEC/source closure must stay unchanged')
    pop_spec_path = bound(spec['population_spec']); pop_spec = read_json(pop_spec_path)
    pop_reader = load_source(spec['population_CPU_reader'], 'qualified_population_source_metadata_gate')
    entry, reader, pop_spec_ref, admission, owner = pop_reader.source_gate(pop_spec_path, pop_spec, pop_receipt)
    require(entry == spec['population_producer'] and reader == spec['population_CPU_reader'] and
            pop_spec_ref == spec['population_spec'] and admission == spec['population_GPU_admission'] and
            pop_spec['expected_epoch'] == 12 and pop_spec['DEV'] == spec['DEV'] and
            pop_spec['checkpointmap'] == evaluation['checkpointmap'] and
            pop_spec['qualificationmap'] == evaluation['qualificationmap'] and
            pop_spec['formal_evaluation_qualification'] == SG_spec['evaluation_qualification'] and
            all(row in spec['sources'] for row in pop_spec['sources']
                if ref_path(row).suffix.lower() in ('.py', '.pyi')), 'Population/SG formal CP/code/DEV joins differ')
    population = read_json(bound(spec['population_qualification']))
    require(population['schema'] == 'ROOT-native-geometry-paired-passive-population-CPU-diagnostic-v1' and
            population['status'] == 'ROOT_PASSIVE_PAIRED_POPULATION_SAMPLING_CPU_DIAGNOSED' and
            population['source'] == entry and population['reader_source'] == reader and population['spec'] == pop_spec_ref and
            population['parent_receipt'] == spec['population_GPU_receipt'] and population['GPU_admission'] == admission and
            population['owner_driver'] == owner and population['expected_epoch'] == 12 and
            population['formal_evaluation_qualification'] == SG_spec['evaluation_qualification'] and
            population['checkpoint'] == qualification['checkpointmap']['paired'] and
            population['qualificationmap'] == qualification['qualificationmap'] and
            population['CPU_affinity'] == [0, 1] and population['torch_threads'] == 2 and population['CUDA_initialized'] is False and
            population['AP_computed'] is False and population['scientific_method_verdict'] is None,
            'Accepted population CPU qualification/source/receipt scope differs')
    require(population['independent_CPU_native_nearest800_allGT_digest_exact'] is True and
            population['independent_CPU_full_paired612_CP_digest_matches_before_after'] is True and
            population['official_endpoint_regression']['entries'] == 18700 and
            population['official_endpoint_regression']['exact_same_CP_RLE_factor_score_regression'] is True and
            population['outside_donor_slots_applied'] is False, 'Prior exact same-CP/population proofs required')
    bound(population['report']); bound(population['start'])
    SG_result = read_json(bound(spec['SG12_diagnostic_report']))
    require(SG_result['schema'] == 'ROOT-native-paired-score-geometry-CPU-diagnostic-report-v1' and
            SG_result['status'] == 'COMPLETED_FINITE_OUTPUT_INTERVENTION_NO_SCIENCE_VERDICT' and
            SG_result['provenance'] == SG_provenance and SG_result['DEV'] == spec['DEV'] and
            SG_result['expected_epoch'] == 12 and SG_result['same_checkpoint'] == population['checkpoint'] and
            SG_result['all187_images_all380_raw_GT_preserved'] is True and
            SG_result['all18700_predictions_preserved_in_each_cell'] is True and
            SG_result['exact_class_probability_equality_count'] == 18700 and
            set(SG_result['cells']) == {'M0Q0', 'M0Q1', 'M1Q0', 'M1Q1'}, 'Existing SG12 finite-output source/population differs')
    return SG, SG_spec, evaluation, qualification, population, dict(
        spec=reference(spec_path), source=reference(__file__), sources=spec['sources'],
        population_qualification=spec['population_qualification'], population_spec=spec['population_spec'],
        population_GPU_receipt=spec['population_GPU_receipt'], population_GPU_admission=admission,
        population_producer=entry, population_CPU_reader=reader, SG12_spec=spec['SG12_spec'],
        original_population_metadata_gate_checks_all_bindings=True,
        historical_noncode_refs_hashed_not_deserialized=[row for row in pop_spec['sources']
            if ref_path(row).suffix.lower() not in ('.py', '.pyi')],
        SG12_diagnostic_source=spec['SG12_diagnostic_source'], SG12_diagnostic_report=spec['SG12_diagnostic_report'],
        formal_evaluation_qualification=SG_spec['evaluation_qualification'])


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


def aggregate(rows):
    """Integer sums exact; floating sums fsum; undefined ratios stay missing."""
    keys = METRIC_KEYS
    require(all(set(row['counts']) == set(keys) for row in rows), 'Metric schemas differ across annotations')
    metrics = {}
    for key in keys:
        values = [row['counts'][key] for row in rows if row['counts'][key] is not None]
        require(all(type(value) in (int, float) and math.isfinite(value) for value in values), 'Nonfinite association statistic')
        total = sum(values) if all(type(value) is int for value in values) else math.fsum(values)
        metrics[key] = dict(sum=total, mean=None if not values else total / len(values),
                            available=len(values), missing=len(rows) - len(values), denominator=len(values))
    return dict(annotations=len(rows), empty_original_GT=sum(row['counts']['GT_pixels'] == 0 for row in rows), metrics=metrics)


def occupancy_groups(row):
    require(type(row['eligible_pairs']) is int and 0 <= row['eligible_pairs'] <= 16 and
            type(row['E0']) is bool and row['E0'] == (row['eligible_pairs'] == 0), 'Actual E0/count identity differs')
    matched = not row['unmatched_query']; groups = {}
    for measure in MEASURES:
        source = row['measures'][measure]
        own = source['own_GT']; other = source['other_GT_union']['outside']
        require((own is not None) == matched and other['eligible_points'] == row['eligible_pairs'], 'Own/other actual slot population differs')
        other_value = other['eligible_mean']
        own_value = None if own is None else own['outside']['eligible_mean']
        if matched:
            require(own['outside']['eligible_points'] == row['eligible_pairs'], 'Own outside actual E differs')
        for value in (own_value, other_value):
            require(value is None or (type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1),
                    'Outside occupancy must be a finite probability or undefined')
        if row['E0']:
            require(own_value is None and other_value is None, 'E0 eligible means must remain None')
            group = 'E0'
        elif not matched:
            require(own_value is None and other_value is not None, 'Unmatched own must be undefined')
            group = 'UNMATCHED_OWN_UNDEFINED'
        else:
            require(own_value is not None and other_value is not None, 'Matched nonempty E means required')
            group = 'own%d_other%d' % (int(own_value > 0), int(other_value > 0))
        groups[measure] = dict(group=group, eligible_outside_own_mean=own_value,
                               eligible_outside_other_union_mean=other_value, eligible_pairs=row['eligible_pairs'])
    return groups


def decode_mask(np, masks, prediction, image):
    rle = prediction['segmentation']
    require(rle['size'] == [image['height'], image['width']] and isinstance(rle['counts'], str), 'Original-size official RLE required')
    value = masks.decode(dict(size=rle['size'], counts=rle['counts'].encode('ascii')))
    require(value.shape == (image['height'], image['width']) and value.dtype == np.uint8 and
            bool(((value == 0) | (value == 1)).all()), 'Official mask must decode as full binary original grid')
    return value.astype(bool, copy=False)


def diagnose(spec_file, out_file):
    spec_path, out = Path(spec_file), Path(out_file)
    require(spec_path.is_absolute() and out.is_absolute() and not out.exists() and out.parent.is_dir(), 'Absolute SPEC/fresh output required')
    spec_path = spec_path.resolve(strict=True); spec = read_json(spec_path)
    SG, SG_spec, evaluation, qualification, population, provenance = metadata_gate(spec_path, spec)
    SG.cpu_environment()
    reader = SG.load_reader(SG_spec['evaluation_CPU_source'])
    import torch
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools import mask as masks
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    require(not torch.cuda.is_initialized(), 'CPU diagnostic must not initialize CUDA')
    coco = COCO(str(bound(spec['DEV'])))
    require(len(coco.imgs) == 187 and len(coco.anns) == 380 and set(coco.cats) == {1}, 'All original DEV187/380/category1 required')
    ids, endpoints = SG.endpoints(reader, evaluation, qualification, coco, np, masks)
    expected = {(image_id, qid) for image_id in ids for qid in range(100)}
    queries = population['query_rows']; assignments = population['raw_GT_assignment_rows']
    query_map = {(r['image_id'], r['query_id']): r for r in queries}
    GT_map = {(r['image_id'], r['annotation_id']): r for r in assignments}
    require(len(queries) == len(query_map) == 18700 and set(query_map) == expected and
            len(assignments) == len(GT_map) == 380 and set(GT_map) ==
            {(int(a['image_id']), int(a['id'])) for a in coco.anns.values()}, 'Full sampling/stock-GT identities required')
    require(sum(r['E0'] for r in queries) == population['occupancy_summary']['E0_queries'] == spec['expected_E0_queries'] == 818,
            'Fixed complete E0 query population differs')
    require(population['occupancy_summary']['queries'] == 18700 and population['occupancy_summary']['raw_GT'] == 380 and
            population['occupancy_summary']['matched_queries'] == 380, 'Prior full denominators differ')
    full_query_rows, GT_rows, E0_identity = [], [], 0
    for image_id in ids:
        image = coco.imgs[image_id]; anns = coco.imgToAnns[image_id]
        annotations = {ann['id']: ann for ann in anns}
        require(len(annotations) == len(anns), 'Duplicate raw annotation identity')
        seen_ann = set()
        for qid in range(100):
            key = (image_id, qid); row = query_map[key]; assigned = row['stock_assigned_annotation_id']
            require(type(row['query_id']) is int and type(row['unmatched_query']) is bool and
                    row['unmatched_query'] == (assigned is None) and row['all16_slots_preserved'] is True and
                    row['outside_donor_slots_not_applied'] is True and row['own_GT_is_posterior_stock_assignment'] is True,
                    'Actual all-slot stock query identity/sampling policy differs')
            groups = occupancy_groups(row)
            on_factor, off_factor = (endpoints[label]['factors'][key] for label in ('on', 'off'))
            require(np.asarray(on_factor['class_probability'], dtype=np.float32).tobytes() ==
                    np.asarray(off_factor['class_probability'], dtype=np.float32).tobytes(), 'Class probability FP32 bytes differ on/off')
            compact = dict(image_id=image_id, query_id=qid, stock_assigned_annotation_id=assigned,
                unmatched_query=row['unmatched_query'], E0=row['E0'], eligible_pairs=row['eligible_pairs'], groups=groups,
                class_probability_exact_same=True, counts=None)
            if row['E0'] or assigned is not None:
                on = decode_mask(np, masks, endpoints['on']['predictions'][key], image)
                off = decode_mask(np, masks, endpoints['off']['predictions'][key], image)
                if row['E0']:
                    require(bool(np.array_equal(on, off)), 'E0 on/off mask changed; fail, do not treat as a branch effect')
                    E0_identity += 1
                    compact['E0_full_original_binary_mask_exact_same'] = True
                if assigned is not None:
                    require(assigned in annotations and assigned not in seen_ann, 'Stock annotation duplicated or absent')
                    seen_ann.add(assigned); assignment = GT_map[(image_id, assigned)]
                    require(assignment['query_id'] == qid and assignment['GT_index'] == [ann['id'] for ann in anns].index(assigned) and
                            assignment['E0'] == row['E0'] and assignment['eligible_pairs'] == row['eligible_pairs'],
                            'Original stock annotation/query/order join differs')
                    gt = coco.annToMask(annotations[assigned])
                    changes = counts_change(gt, on, off)
                    require(changes['GT_pixels'] == assignment['original_pixels'] == row['original_GT_pixels'], 'Original GT area differs')
                    compact['counts'] = changes
                    GT_rows.append(dict(image_id=image_id, annotation_id=assigned, query_id=qid,
                        GT_index=assignment['GT_index'], E0=row['E0'], eligible_pairs=row['eligible_pairs'], groups=groups,
                        iscrowd=int(annotations[assigned].get('iscrowd', 0)), ignore=int(annotations[assigned].get('ignore', 0)),
                        counts=changes))
                    del gt
                del on, off
            full_query_rows.append(compact)
        require(seen_ann == set(annotations), 'A raw annotation was omitted or rematched')
    require(len(full_query_rows) == 18700 and len(GT_rows) == 380 and E0_identity == 818 and
            {(r['image_id'], r['annotation_id']) for r in GT_rows} == set(GT_map), 'Complete all-query/all-GT/E0 identity failed')
    group_summary = {}
    for measure in MEASURES:
        groups = {name: aggregate([r for r in GT_rows if r['groups'][measure]['group'] == name]) for name in GROUPS}
        require(sum(value['annotations'] for value in groups.values()) == 380, 'E0/four groups must partition all380 stock GT')
        query_counts = {name: sum(r['groups'][measure]['group'] == name for r in full_query_rows)
                        for name in GROUPS + ('UNMATCHED_OWN_UNDEFINED',)}
        require(sum(query_counts.values()) == 18700 and query_counts['E0'] == 818, 'All queries must retain their defined/undefined group')
        group_summary[measure] = dict(role='primary' if measure == 'nearest_800' else 'sensitivity',
                                     stock_GT_groups=groups, all_query_group_counts=query_counts)
    require(not torch.cuda.is_initialized(), 'Diagnostic initialized CUDA')
    result = dict(schema='ROOT-native-paired-passive-error-association-CPU-report-v1',
        status='ROOT_PASSIVE_PAIRED_ERROR_ASSOCIATION_CPU_DIAGNOSED', phase=spec['phase'], provenance=provenance,
        DEV=spec['DEV'], expected_epoch=12, checkpoint=population['checkpoint'], queries=18700, stock_matched_GT=380,
        E0_queries=818, E0_masks_independently_decoded_exact_same=E0_identity,
        CPU_affinity=[0, 1], CPU_threads=2, CUDA_initialized=False,
        grouping='eligible outside own mean>0 crossed with other-union mean>0; E0 separate; no threshold search',
        all_query_rows=full_query_rows, raw_stock_GT_rows=GT_rows, overall_stock_GT=aggregate(GT_rows), group_summaries=group_summary,
        endpoint_inputs={label: value['input_refs'] for label, value in endpoints.items()},
        original_SG12_2x2_external_source=spec['SG12_diagnostic_report'], AP_recomputed=False,
        best_IoU_query_selection=False, posterior_assignment_recomputed=False, query_filter=False, GT_filter=False,
        model_loaded=False, checkpoint_loaded=False, model_forward=False, optimizer=False, automatic_retry=False,
        scientific_method_verdict=None, limits=[
            'Groups and own annotation are posterior paired-on stock assignments, not independent semantic identities or recall.',
            'On/off removes both inside and outside contributions; associations cannot identify an outside-only or paired-specific effect.',
            'GT-position occupancy is not feature purity or sea background; original_proxy is a distinct sensitivity measure.',
            'All818 E0 masks must be identical. Other queries can affect global ranking, so this is not an E0 AP-zero-effect certificate.',
            'No best query is selected and no query is rematched. Crowds, ignored or empty GT remain in the raw denominator.',
            'Empty GT ratios are None; empty-union IoU is1. Floating means use available-count denominators shown explicitly.',
            'SG12 2x2 metrics are referenced as existing external results; no AP is recomputed or interpreted as a new method result.',
            'A single checkpoint association is not training causality, sampling benefit, innovation or a convergence finding.'])
    with out.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(status=result['status'], output=reference(out), scientific_method_verdict=None), sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True); parser.add_argument('--out', required=True)
    args = parser.parse_args(); diagnose(args.spec, args.out)


if __name__ == '__main__':
    main()
