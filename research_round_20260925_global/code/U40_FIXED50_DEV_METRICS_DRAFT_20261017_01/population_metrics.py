"""UNBOUND fixed-E50 full-DEV CPU metrics; numerical path not executed.

Only stdlib is imported here. Future Root-owned callers inject all scientific
libraries and closed modules. Metadata tests do not establish AP or eligibility.
"""
from copy import deepcopy
import gc
import hashlib
import json
import math
from pathlib import Path
import time

SEED = 202610173101
ARMS = ('PLAIN', 'FULL')
RANGES = [[0, 47], [47, 94], [94, 141], [141, 187]]
CURSOR = dict(global_updates=37050, completed_epochs=50, within_epoch=0)
PREDICTION_ROLES = ('shared_initialization', 'full_legacy', 'full_dsconv', 'full_head',
    'full_feature', 'plain_legacy', 'plain_feature', 'instance_adapter',
    'u36_helpers', 'u39_training_ops', 'image_geometry', 'training_state',
    'dev_reader', 'dev_rgb_reader', 'population_contract', 'encoder', 'mask_api')
SHARD_TRUE_FLAGS = ('native_all_RLE_and_identity_prediction_dictionaries_exact',
    'zero_residual_native_L_exact', 'original_full_S0_class_order_preserved',
    'historical_target_fields_may_be_in_bundle', 'model_BN_RNG_unchanged')
SHARD_FALSE_FLAGS = ('original_teacher_forward', 'GT_used_for_prediction',
    'DEV_COCO_annotation_opened', 'new_training')
SHARD_RESULT_KEYS = ('predictions', 'identity_predictions', 'Omega_support',
    'dataset_range', 'shard_index', 'images', 'queries', 'seed', 'fixed_epoch',
    'updates_per_arm', 'training_qualified', 'pair_commit_ref', 'checkpoint_refs',
    'order_ref', 'dev_context', 'training_sources', 'prediction_sources',
    'new_training', 'AP', 'DEV', 'TEST')
METRICS_ROLES = ('legacy_eval', 'population', 'geometry', 'official',
                 'coco', 'cocoeval', 'masks')
PINNED_SOURCE_SHA256 = {
    'legacy_eval': '0e4ec8c999b0e954ce6d588b3ffae97510be1051c4465f9d4510c3c2ec3b34bf',
    'population': '733a208c5df49113e9dff3338b2b88ad2d0d4b26b2bbdd3676ce10417e05c9a5',
    'geometry': '6f6669eb91d0842f7c23eb09b992624642d16c9abe8d145c258ec9aae5ff36ac',
}


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def _reference(value):
    need(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'}
         and type(value['path']) is str and Path(value['path']).is_absolute()
         and type(value['bytes']) is int and value['bytes'] >= 0
         and type(value['sha256']) is str and len(value['sha256']) == 64
         and set(value['sha256']) <= set('0123456789abcdef'),
         'Strict absolute path/bytes/lowercase SHA256 reference required')
    return value


def validate_fixed_endpoint(spec, training):
    """Check preselected Root provenance only; never advance a training gate."""
    need(spec['seed'] == SEED and type(spec['seed']) is int
         and spec['fixed_epoch'] == 50 and spec['updates_per_arm'] == 37050,
         'Only the preregistered first-seed E50 endpoint is permitted')
    need(training['schema'] == 'ROOT-U40-continuous-training-segment-qualified-v1'
         and training['qualified'] is True and training['paired_prefix_advanced'] is True,
         'Actual Root-accepted paired endpoint required; pending workers do not qualify')
    result = training['result']
    need(result['seed'] == SEED and result['span'] == dict(start=36864, stop=37050)
         and result['paired_checkpoint_prefix'] == CURSOR,
         'Fixed final segment and complete E50 paired cursor required')
    need(type(spec['checkpoint_refs']) is dict and set(spec['checkpoint_refs']) == set(ARMS),
         'Both fixed paired checkpoints required')
    for key in ('pair_commit_ref', 'order_ref'):
        _reference(spec[key])
        need(result[key] == spec[key], 'Endpoint '+key+' differs from the accepted pair')
    for arm in ARMS:
        _reference(spec['checkpoint_refs'][arm])
    need(result['checkpoint_refs'] == spec['checkpoint_refs'],
         'No checkpoint substitution or cross-pair combination permitted')
    _reference(spec['training_qualified'])


def _source_map(value, expected_count, roles=None):
    need(type(value) is dict and len(value) == expected_count
         and all(type(key) is str for key in value), 'Complete explicit source-ref map required')
    if roles is not None:
        need(set(value) == set(roles), 'Exact reviewed source roles required')
    for reference in value.values():
        _reference(reference)


def validate_shard_metadata(spec, accepted, shard, ordinal):
    """Join Root-accepted shard metadata; no pixel, NN or terminal re-execution."""
    need(type(ordinal) is int and 0 <= ordinal < 4, 'Four fixed ordered shards required')
    need(accepted['schema'] == 'ROOT-U40-fixed50-full-method-DEV-shard-qualified-v1'
         and accepted['qualified'] is True, 'Actual Root-accepted prediction shard required')
    result = accepted['result']
    _reference(result['report_ref'])
    need(shard['schema'] == 'U40-fixed50-full-method-DEV-shard-pending-v1',
         'Exact new E50 shard report schema required')
    start, stop = RANGES[ordinal]
    need(shard['dataset_range'] == [start, stop] and shard['shard_index'] == ordinal
         and type(shard['images']) is int and shard['images'] == stop-start
         and type(shard['queries']) is int and shard['queries'] == (stop-start)*100,
         'Complete fixed shard range and all-100 population required')
    for key in ('seed', 'fixed_epoch', 'updates_per_arm', 'training_qualified',
                'pair_commit_ref', 'checkpoint_refs', 'order_ref', 'dev_context'):
        need(shard[key] == spec[key], 'Shard '+key+' differs from fixed common provenance')
    need(shard['query_batch'] == 4 and shard['AP'] is False
         and shard['DEV'] is True and shard['TEST'] is False,
         'Query-batch4 prediction only; no per-shard AP or TEST')
    need(all(shard[key] is True for key in SHARD_TRUE_FLAGS)
         and all(shard[key] is False for key in SHARD_FALSE_FLAGS),
         'Original native calibration/S0/order/model invariance and prediction scope required')
    for key in SHARD_RESULT_KEYS:
        need(result[key] == shard[key], 'Root result/report mismatch: '+key)
    need(type(shard['predictions']) is dict and set(shard['predictions']) == set(ARMS),
         'Real PLAIN and FULL exports required, without old model aliases')
    for reference in list(shard['predictions'].values()) + [shard['identity_predictions'], shard['Omega_support']]:
        _reference(reference)
    for key, count, roles in (('training_sources', 17, None),
                             ('prediction_sources', 17, PREDICTION_ROLES)):
        _source_map(spec[key], count, roles)
        need(shard[key] == spec[key], 'Every shard must use the exact common '+key)
    _reference(spec['prediction_core_source'])
    need(shard['prediction_core_source'] == spec['prediction_core_source'],
         'One exact prediction core source across all accepted shards')


def relabel_geometry(value):
    """Only adapt hard-coded helper labels; preserve its original mathematics."""
    result = deepcopy(value)
    summary = result['summary']
    need(summary['schema'] == 'N28-original-native-fixed-pair-geometry-summary-v1'
         and summary['scope']['total_DEV_GT'] == 380,
         'Original full-DEV denominator and fixed native-pair diagnostic required')
    selected = summary['scope']['selected_native_nonignored_GT']
    need(type(selected) is int and 0 <= selected <= 380
         and len(result['private_pair_records']) == selected,
         'Report conditional selected coverage; it need not equal all 380 GT')
    aliases = {'SELECTIVE': 'FULL', 'WHOLE_NODE': 'PLAIN'}
    expected = {'native_official', 'R', 'SELECTIVE', 'WHOLE_NODE'}
    for container in [summary] + result['private_pair_records']:
        need(type(container['methods']) is dict and set(container['methods']) == expected,
             'Exactly four helper method labels required before explicit adaptation')
        container['methods'] = {aliases.get(name, name): values
                                for name, values in container['methods'].items()}
    summary.update(schema='U40-fixed50-native-fixed-pair-geometry-summary-v1',
        algorithm_schema='N28-original-native-fixed-pair-geometry-summary-v1',
        method_label_adapter=aliases, old_method_models_evaluated=False)
    return result


def _read(c, reference):
    _reference(reference)
    return json.loads(Path(c.closed(reference)).read_bytes())


def _signature(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _closed_source(c, module, reference, role):
    _reference(reference)
    path = Path(c.closed(reference)).resolve(strict=True)
    raw = path.read_bytes()
    need(len(raw) == reference['bytes'] and hashlib.sha256(raw).hexdigest() == reference['sha256'],
         'Exact closed '+role+' source bytes required')
    filename = getattr(module, '__file__', None)
    need(type(filename) is str and Path(filename).resolve(strict=True) == path,
         role+' must be injected from its explicit admitted source')
    if role in PINNED_SOURCE_SHA256:
        need(reference['sha256'] == PINNED_SOURCE_SHA256[role], 'Reviewed original '+role+' source required')


def _check_prelock(spec, pre):
    need(spec['schema'] == 'ROOT-U40-fixed50-complete-DEV-metrics-spec-v1'
         and pre['schema'] == 'U40-fixed50-complete-DEV-metrics-prelock-v1',
         'New fixed50 complete-DEV metrics contract required')
    for values in (spec, pre):
        need(values['seed'] == SEED and values['fixed_epoch'] == 50
             and values['updates_per_arm'] == 37050, 'Only fixed E50, not a DEV-selected endpoint')
    need(pre['images'] == 187 and pre['GT'] == 380 and pre['queries'] == 18700
         and pre['GPU_ranges'] == RANGES and pre['no_new_training'] is True
         and pre['no_TEST'] is True, 'Complete standard DEV population and no training/TEST')
    need(type(spec['accepted_GPU_shards']) is list and len(spec['accepted_GPU_shards']) == 4,
         'All four Root-accepted shards must be supplied in canonical range order')
    for reference in spec['accepted_GPU_shards']:
        _reference(reference)
    need(len({_signature(value) for value in spec['accepted_GPU_shards']}) == 4,
         'Four distinct actual Root shard acceptances required')


def _delta_stats(left, right):
    # Keep standard undefined COCO -1 entries undefined; do not turn them into
    # a fabricated zero difference or introduce a replacement metric.
    return [float(a-b) if a >= 0 and b >= 0 else None for a, b in zip(left, right)]


def evaluate_population(torch, np, c, spec, pre, modules, report):
    """PENDING real CPU execution: complete187/380 COCO and fixed-native pairs.

    Caller owns Source/admission/terminal/resource checks. This function never
    loads a checkpoint, runs a network, advances a seed gate, or certifies itself.
    All scientific modules are injected; only old official_metrics is called.
    """
    need(type(report) is dict, 'Caller-owned partial report dictionary required')
    need(not torch.cuda.is_initialized(), 'CPU metrics must start CUDA cold')
    started = time.monotonic()
    report.update(schema='U40-fixed50-complete187-COCO-pending-v1', status='PENDING',
        seed=SEED, fixed_epoch=50, updates_per_arm=37050, images=187, GT=380, queries=18700,
        qualified=False, ROOT_actual_terminal_self_certified=False, new_training=False,
        AP=True, DEV=True, TEST=False, innovation_established=False, goal_complete=False,
        utility_gate_decided=False, no_shard_AP_average=True,
        prediction_fields_may_have_historical_target_associations=True)
    try:
        _check_prelock(spec, pre)
        need(type(modules) is dict and set(modules) == set(METRICS_ROLES),
             'Inject exactly seven explicit helper/framework modules')
        _source_map(spec['metrics_module_sources'], 7, METRICS_ROLES)
        for role in METRICS_ROLES:
            _closed_source(c, modules[role], spec['metrics_module_sources'][role], role)
        _reference(spec['metrics_source'])
        own_path = Path(c.closed(spec['metrics_source'])).resolve(strict=True)
        need(own_path == Path(__file__).resolve(strict=True), 'Execute the admitted metrics source itself')
        c.check()
        training = _read(c, spec['training_qualified'])
        validate_fixed_endpoint(spec, training)
        shards = []
        for ordinal, reference in enumerate(spec['accepted_GPU_shards']):
            accepted = _read(c, reference)
            shard = _read(c, accepted['result']['report_ref'])
            validate_shard_metadata(spec, accepted, shard, ordinal)
            shards.append(shard)
        gpu_costs = []
        for shard in shards:
            need(type(shard['cost']) is dict and set(shard['cost']) == set(ARMS)
                 and all(type(value) is dict for value in shard['cost'].values()),
                 'Preserve each actual arm cost dictionary from the accepted prediction shard')
            gpu_costs.append(dict(dataset_range=deepcopy(shard['dataset_range']),
                                  cost=deepcopy(shard['cost'])))
        # All four actual Root acceptances are joined before the COCO annotation
        # file is opened. Existing prediction JSON can contain historical fields.
        context = _read(c, spec['dev_context'])
        coco_context = _read(c, spec['coco_context'])
        need(context['schema'] == 'ROOT-N28-fixed24-dense-native-DEV-context-v1'
             and coco_context['schema'] == 'ROOT-N28-fixed24-original-COCO-context-v1'
             and coco_context['image_ids'] == context['image_ids']
             and coco_context['legacy_ROOT_result'] == context['legacy_ROOT_result'],
             'Reuse original DEV prediction/annotation association, not old model endpoint')
        ids = context['image_ids']
        need(type(ids) is list and len(ids) == len(set(ids)) == 187
             and all(type(value) is int for value in ids), 'Canonical original 187 image ids')
        need(coco_context['COCO_source'] == spec['metrics_module_sources']['official']
             and spec['prediction_sources']['encoder'] == context['encoder_source']
             and spec['prediction_sources']['population_contract'] == spec['metrics_module_sources']['population']
             and spec['prediction_sources']['mask_api'] == spec['metrics_module_sources']['masks'],
             'Original official helper/encoder/population/mask API source identities required')
        for role in set(spec['training_sources']).intersection(spec['prediction_sources']):
            need(spec['training_sources'][role] == spec['prediction_sources'][role],
                 'Shared training/prediction component source changed: '+role)
        need(spec['prediction_sources']['training_state'] in spec['training_sources'].values(),
             'Restore schema must come from the exact original training core source')
        legacy = _read(c, context['legacy_ROOT_result'])
        need(legacy['qualified'] is True, 'Original native/R official results must be Root accepted')
        native = _read(c, context['native_predictions'])
        strong = _read(c, context['R_predictions'])
        need(type(native) is list and type(strong) is list and len(native) == len(strong) == 18700,
             'Complete original native and R candidates required')
        need([native[i*100]['image_id'] for i in range(187)] == ids,
             'Original image/export population order must match the context')
        population = modules['population']
        strong = population.merge_predictions(
            [dict(dataset_range=span, predictions=strong[span[0]*100:span[1]*100]) for span in RANGES],
            native, RANGES)
        pieces = {arm: [] for arm in ARMS}
        identity_pieces, supports = [], []
        for shard, span in zip(shards, RANGES):
            first, stop = span
            for arm in ARMS:
                pieces[arm].append(dict(dataset_range=span, predictions=_read(c, shard['predictions'][arm])))
            identity_pieces.append(dict(dataset_range=span, predictions=_read(c, shard['identity_predictions'])))
            support = _read(c, shard['Omega_support'])
            original = native[first*100:stop*100]
            need(type(support) is list and len(support) == (stop-first)*100,
                 'Complete original native Omega support population required')
            for offset, (record, value) in enumerate(zip(support, original)):
                need(type(record) is dict and type(record['dataset_index']) is int
                     and record['dataset_index'] == first+offset//100
                     and (record['image_id'], record['query_id']) == (value['image_id'], value['query_id'])
                     and type(record['Omega']) is dict,
                     'Omega original image/query/export order must remain exact')
            supports.extend(support)
        predictions = {arm: population.merge_predictions(pieces[arm], native, RANGES) for arm in ARMS}
        identity = population.merge_predictions(identity_pieces, native, RANGES)
        need(_signature(identity) == _signature(native) and len(supports) == 18700,
             'Every zero-residual prediction dictionary and original query must be exact')
        predictions.update(native_official=identity, R=strong)
        del pieces, identity_pieces, support, original
        c.check()

        # This is the only COCO GT open in this function, after four Root proofs.
        coco = modules['coco'].COCO(str(c.closed(_reference(coco_context['DEV']))))
        need(list(coco.imgs) == ids and len(coco.anns) == 380 and set(coco.cats) == {1},
             'Unconverted original DEV187/380/category1 annotations required')
        out = Path(spec['output_directory'])
        out.mkdir()  # Fresh owned output only; do not overwrite any prior run.
        results, native_evaluation = {}, None
        report.update(source=spec['metrics_source'], spec=c.execution_spec_ref,
            CPU_admission=c.execution_admission_ref, phase=spec['phase'], prelock=spec['prelock'],
            training_qualified=spec['training_qualified'], checkpoint_refs=spec['checkpoint_refs'],
            pair_commit_ref=spec['pair_commit_ref'], order_ref=spec['order_ref'],
            paired_checkpoint_prefix=deepcopy(CURSOR), dev_context=spec['dev_context'],
            coco_context=spec['coco_context'], accepted_GPU_shards=spec['accepted_GPU_shards'],
            training_sources=spec['training_sources'], prediction_sources=spec['prediction_sources'],
            prediction_core_source=spec['prediction_core_source'],
            metrics_module_sources=spec['metrics_module_sources'], results=results,
            GPU_costs_by_shard=gpu_costs,
            GPU_timing_scope=('Actual prediction shard costs include the new UNet image branch; '
                'cached teacher precomputation is excluded. They do not measure complete-pipeline FPS.'),
            teacher_precompute_cost_included=False, standalone_FPS_claimed=False)
        for method in ('native_official', 'R', 'PLAIN', 'FULL'):
            c.check()
            tick = time.monotonic()
            metrics, per_iou, evaluation = modules['legacy_eval'].official_metrics(
                np, c, modules['official'], modules['cocoeval'].COCOeval, coco, predictions[method], ids)
            need(len(metrics['stats']) == 12 and all(math.isfinite(float(value)) for value in metrics['stats']),
                 'Standard finite COCO12 including original undefined -1 values required')
            if method in ('native_official', 'R'):
                key = 'native_official' if method == 'native_official' else 'strong_fixed_E24_Omega'
                need(_signature(metrics) == _signature(legacy['control_results'][key]['COCO']),
                     'Original native/R complete official COCO12 dictionary must reproduce exactly')
            if method == 'native_official':
                native_evaluation = evaluation
            results[method] = dict(COCO=metrics, per_IoU=per_iou,
                official_COCO_seconds=time.monotonic()-tick,
                predictions=c.write_json(out/(method+'_PREDICTIONS.json'), predictions[method]))
            del evaluation
        # Explicit label adapter: no old SELECTIVE/WHOLE_NODE model is restored.
        legacy_labels = dict(native_official=predictions['native_official'], R=predictions['R'],
                             SELECTIVE=predictions['FULL'], WHOLE_NODE=predictions['PLAIN'])
        geometry_full = relabel_geometry(modules['geometry'].diagnose(
            np, modules['masks'], coco, native_evaluation, legacy_labels, supports))
        pair_ref = c.write_json(out/'FIXED_NATIVE_MATCHED_GEOMETRY_PRIVATE.json',
                                geometry_full['private_pair_records'])
        stats = {method: value['COCO']['stats'] for method, value in results.items()}
        scores = {method: 100*float(values[0]) for method, values in stats.items()}
        differences = {left+'_minus_'+right: _delta_stats(stats[left], stats[right])
                       for left, right in (('FULL', 'PLAIN'), ('FULL', 'native_official'),
                                           ('PLAIN', 'native_official'), ('FULL', 'R'), ('PLAIN', 'R'))}
        # Existing non-segmentation bbox/area metadata is deliberately retained.
        # loadRes may use a nonempty bbox for dt.area/area-group FP ignore; mask
        # IoU stays segmentation-based. Never silently strip or rewrite fields.
        metadata_counts = dict(
            predictions_with_bbox=sum('bbox' in value for value in native),
            predictions_with_nonempty_bbox=sum('bbox' in value and bool(value['bbox']) for value in native),
            predictions_with_area=sum('area' in value for value in native))
        need(not torch.cuda.is_initialized(), 'COCO and fixed-pair diagnostics must remain CUDA cold')
        report.update(status='COMPLETE_PENDING_ROOT_ACTUAL_TERMINAL', AP_points=scores,
            paired_Delta_AP_points=scores['FULL']-scores['PLAIN'],
            Delta_AP_points={name: 100*values[0] if values[0] is not None else None
                             for name, values in differences.items()},
            Delta_COCO_stats=differences, geometry=geometry_full['summary'],
            private_pair_records=pair_ref, original_native_R_COCO12_exact=True,
            fixed_full_S0_class_query_order_preserved=True, zero_residual_identity_exact=True,
            geometry_pair_selection_uses_new_method=False, epoch_selected_on_DEV=False,
            current_COCO_annotations_opened_only_in_CPU_metrics=True,
            original_non_segmentation_metadata_counts=metadata_counts,
            original_loadRes_bbox_area_policy_preserved=True,
            loadRes_area_interpretation=('Original helper is unchanged: nonempty bbox can determine DT area '
                'and unmatched detection area-group ignore; segmentation mask still determines mask IoU.'),
            teacher_forward_reexecuted=False, CUDA_initialized=False,
            end_to_end_pipeline_FPS_measured=False,
            elapsed_seconds=time.monotonic()-started)
        report_ref = c.write_json(out/'COCO_REPORT.json', report)
        del predictions, native, strong, identity, supports, shards, geometry_full, native_evaluation
        del coco, legacy_labels
        gc.collect()
        c.check()
        return dict(report_ref=report_ref, seed=SEED, fixed_epoch=50, updates_per_arm=37050,
            images=187, GT=380, queries=18700, AP=True, DEV=True, TEST=False, new_training=False,
            AP_points=scores, paired_Delta_AP_points=report['paired_Delta_AP_points'],
            Delta_AP_points=report['Delta_AP_points'], qualified=False,
            utility_gate_decided=False)
    except BaseException as error:
        report.update(status='FAILED', error=dict(type=type(error).__name__, message=str(error)),
                      qualified=False)
        raise
