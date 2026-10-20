"""Unbound U66 population wiring; no library imports, I/O, owner or qualification.

The caller admits actual endpoints, source bytes, four original terminals and
original COCO inputs. Only injected original scientific helpers do numerical work.
"""
from copy import deepcopy
import json
import math

ARMS = ('PLAIN', 'FULL')
RANGES = [[0, 47], [47, 94], [94, 141], [141, 187]]
MODULE_ROLES = ('population_contract', 'eval_body', 'official', 'cocoeval',
                'mask_api', 'geometry_diagnostic', 'population_metrics')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def signature(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _modules(modules):
    need(type(modules) is dict and set(modules) == set(MODULE_ROLES), 'Exact original helper roles required')
    for role, name in (('population_contract', 'merge_predictions'), ('eval_body', 'official_metrics'),
                       ('official', 'coco_metrics'), ('cocoeval', 'COCOeval'), ('mask_api', 'decode'),
                       ('geometry_diagnostic', 'diagnose'), ('population_metrics', 'relabel_geometry'),
                       ('population_metrics', '_delta_stats')):
        need(callable(getattr(modules[role], name, None)), 'Original source API required: '+role+'.'+name)


def _shards(shards):
    need(type(shards) is list and len(shards) == 4, 'Four ordered original prediction shards required')
    endpoint = shards[0]['endpoint']
    need(type(endpoint) is dict and endpoint['schema'] == 'U58-U57-DGX-fixed-endpoint-metadata-v1'
         and endpoint['mode'] == 'FORMAL' and endpoint['formal_endpoint'] is True
         and signature(endpoint['cursor']) == signature(dict(global_updates=37050, completed_epochs=50, within_epoch=0)),
         'Exact FORMAL U57 E50 endpoint metadata required')
    for index, (shard, span) in enumerate(zip(shards, RANGES)):
        need(type(shard) is dict and shard['schema'] == 'U66-DGX-fixed50-DEV-shard-pending-v1'
             and shard['status'] == 'PREDICTIONS_COMPLETE_PENDING_ROOT_ACCEPTANCE',
             'Completed U66 prediction output required; a reader check is insufficient')
        need(signature(shard['endpoint']) == signature(endpoint), 'All four full endpoint dictionaries must be exact')
        need(type(shard['shard_index']) is int and shard['shard_index'] == index
             and signature(shard['dataset_range']) == signature(span)
             and type(shard['images']) is int and shard['images'] == span[1]-span[0]
             and type(shard['queries']) is int and shard['queries'] == 100*(span[1]-span[0]),
             'Canonical four ranges, all187 images and all18700 queries required')
        need(all(type(shard[k]) is int and shard[k] == v for k, v in
                 (('fixed_epoch', 50), ('updates_per_arm', 37050), ('query_batch', 4)))
             and all(shard[k] is True for k in ('DEV', 'DEV_technical_input', 'original_full_S0_class_order_preserved'))
             and all(shard[k] is False for k in ('qualified', 'AP', 'TEST', 'new_training',
                 'DEV_metrics_computed', 'whole_population_predicted', 'GT_used_for_prediction', 'GT_rasterization_executed')),
             'Original E50 full-query prediction-only scope; no self qualification')
        need(type(shard['source_refs']) is dict
             and signature(shard['source_refs']) == signature(shards[0]['source_refs']),
             'Identical explicit prediction sources across all four shards')
        need(type(shard['arms']) is dict and set(shard['arms']) == set(ARMS)
             and type(shard['predictions']) is dict and set(shard['predictions']) == set(ARMS)
             and all(shard['arms'][arm]['status'] == 'PREDICTIONS_COMPLETE_PENDING_ROOT_ACCEPTANCE' for arm in ARMS),
             'Both completed original arms required')
        need(type(shard['cost']) is dict and set(shard['cost']) == set(ARMS), 'Original two-arm costs required')
        for arm in ARMS:
            record = shard['arms'][arm]
            need(signature(record['predictions']) == signature(shard['predictions'][arm])
                 and type(record['cost']) is dict and signature(record['cost']) == signature(shard['cost'][arm]),
                 'U66 top and arm prediction/cost aliases must agree; never add both copies')
            need(signature(record['identity_predictions']) == signature(shard['identity_predictions']),
                 'Both arms and top must preserve one exact original identity population')
            count = span[1]-span[0]
            need(all(type(record['cost'][key]) is int and record['cost'][key] == value
                     for key, value in (('encode_calls', count), ('readout_calls', 25*count), ('queries', 100*count))),
                 'Completed original one encode/25 readouts/full100 query costs per image required')
            need(type(record['proofs']) is list and len(record['proofs']) == count
                 and type(record['identity_predictions']) is list and len(record['identity_predictions']) == count*100,
                 'Every original per-image calibration proof required')
            for offset, proof in enumerate(record['proofs']):
                need(type(proof) is dict and type(proof['dataset_index']) is int
                     and proof['dataset_index'] == span[0]+offset
                     and type(proof['image_id']) is int and proof['image_id'] == record['identity_predictions'][100*offset]['image_id']
                     and type(proof['queries']) is int and proof['queries'] == 100
                     and all(proof[key] is True for key in ('zero_residual_native_L_exact',
                         'native_RLE_and_prediction_dictionaries_exact', 'original_full_S0_class_order_preserved')),
                     'Original image/full100/zero-native-RLE calibration proof must be complete')
        need(signature(shard['Omega_support']) == signature(shard['arms']['PLAIN']['Omega_support'])
             and shard['arms']['FULL']['Omega_support'] == [], 'Only original PLAIN top Omega support is permitted')


def evaluate_u66_population(np, c, *, shards, native, strong, reference_COCO, coco, ids, modules):
    """Return pending JSON summary plus separate, never-public pair records.

    U66 pending status is a numerical input contract, not a Root acceptance.
    No tensor/annotation loading, process supervision, output writing or gate is
    implemented here. Exceptions from the original functions propagate unchanged.
    """
    _modules(modules)
    _shards(shards)
    need(type(native) is list and type(strong) is list and len(native) == len(strong) == 18700,
         'Exactly18700 original native and strong_R candidates; never truncate extras')
    need(type(ids) is list and len(ids) == len(set(ids)) == 187 and all(type(i) is int for i in ids)
         and list(coco.imgs) == ids and len(coco.anns) == 380 and set(coco.cats) == {1}
         and [native[i*100]['image_id'] for i in range(187)] == ids,
         'Original canonical187 images/380 annotations/category1 and native export order required')
    need(type(reference_COCO) is dict and set(reference_COCO) == {'native_official', 'R'}
         and all(type(value) is dict and type(value['stats']) is list and len(value['stats']) == 12
                 for value in reference_COCO.values()), 'Two complete original reference COCO12 dictionaries required')
    signature(reference_COCO)
    population = modules['population_contract']
    c.check()
    strong = population.merge_predictions([
        dict(dataset_range=span, predictions=strong[span[0]*100:span[1]*100]) for span in RANGES], native, RANGES)
    pieces = {arm: [] for arm in ARMS}
    identity_pieces, supports = [], []
    for shard, span in zip(shards, RANGES):
        for arm in ARMS:
            pieces[arm].append(dict(dataset_range=span, predictions=shard['predictions'][arm]))
        identity_pieces.append(dict(dataset_range=span, predictions=shard['identity_predictions']))
        support = shard['Omega_support']
        first, stop = span
        need(type(support) is list and len(support) == (stop-first)*100, 'Complete original Omega support required')
        for offset, (record, original) in enumerate(zip(support, native[first*100:stop*100])):
            need(type(record) is dict and type(record['dataset_index']) is int
                 and record['dataset_index'] == first+offset//100
                 and type(record['image_id']) is int and type(record['query_id']) is int
                 and (record['image_id'], record['query_id']) == (original['image_id'], original['query_id'])
                 and type(record['Omega']) is dict, 'Omega must retain original dataset/image/query export order')
        supports.extend(support)
    predictions = {arm: population.merge_predictions(pieces[arm], native, RANGES) for arm in ARMS}
    identity = population.merge_predictions(identity_pieces, native, RANGES)
    need(signature(identity) == signature(native), 'Every original zero-residual dictionary must be exact')
    predictions.update(native_official=identity, R=strong)
    results, native_evaluation = {}, None
    for method in ('native_official', 'R', 'PLAIN', 'FULL'):
        c.check()
        metrics, per_iou, evaluation = modules['eval_body'].official_metrics(
            np, c, modules['official'], modules['cocoeval'].COCOeval, coco, predictions[method], ids)
        need(len(metrics['stats']) == 12 and all(math.isfinite(float(v)) for v in metrics['stats']),
             'Original complete finite COCO12, including undefined -1')
        if method in ('native_official', 'R'):
            need(signature(metrics) == signature(reference_COCO[method]),
                 'Original native/R complete official COCO12 dictionaries must reproduce exactly')
        if method == 'native_official':
            native_evaluation = evaluation
        results[method] = dict(COCO=metrics, per_IoU=per_iou)
    legacy_labels = dict(native_official=identity, R=strong,
                         SELECTIVE=predictions['FULL'], WHOLE_NODE=predictions['PLAIN'])
    geometry = modules['population_metrics'].relabel_geometry(modules['geometry_diagnostic'].diagnose(
        np, modules['mask_api'], coco, native_evaluation, legacy_labels, supports))
    stats = {method: value['COCO']['stats'] for method, value in results.items()}
    differences = {left+'_minus_'+right: modules['population_metrics']._delta_stats(stats[left], stats[right])
        for left, right in (('FULL', 'PLAIN'), ('FULL', 'native_official'),
                            ('PLAIN', 'native_official'), ('FULL', 'R'), ('PLAIN', 'R'))}
    summary = dict(schema='U67-DGX-fixed50-population-metrics-pending-v1',
        status='METRICS_COMPLETE_PENDING_ROOT_ACCEPTANCE', endpoint=deepcopy(shards[0]['endpoint']),
        images=187, GT=380, queries=18700, fixed_epoch=50, updates_per_arm=37050,
        results=results, Delta_COCO_stats=differences, geometry=geometry['summary'],
        geometry_legacy_schema_is_not_execution_qualification=True,
        geometry_method_label_adapter={'SELECTIVE': 'FULL', 'WHOLE_NODE': 'PLAIN'},
        GPU_costs_by_shard=[dict(dataset_range=deepcopy(s['dataset_range']), cost=deepcopy(s['cost'])) for s in shards],
        GPU_timing_scope='Original per-shard arm costs retained; no aggregate FPS or teacher precomputation estimate.',
        qualified=False, ROOT_actual_terminal_self_certified=False, utility_gate_decided=False,
        AP=True, DEV=True, DEV_technical_input=True, TEST=False, new_training=False,
        formal_training_budget_credit=0, standalone_FPS_claimed=False, no_shard_AP_average=True,
        geometry_pair_selection_uses_new_method=False, private_pair_records_not_for_publication=True)
    summary.update(original_native_R_COCO12_exact=True, zero_residual_identity_exact=True,
                   fixed_full_S0_class_query_order_preserved=True)
    result = dict(summary=summary, private_pair_records=geometry['private_pair_records'])
    signature(result)
    c.check()
    return result
