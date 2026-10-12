"""Inert ROOT CPU reader of five same-checkpoint residual-off endpoints.

Runtime first checks the held owned terminal and fresh admission/source chain,
then all three original training qualifications. The original training reader
is the explicit training_CPU_reader ref, exactly60171 bytes/5f1eac59; fresh
location never substitutes a sibling or bypasses that prior qualification.
Only then can scientific payload hashes/imports occur. Ordinary COCO and
score-free coverage are separate: neither proves training-path causality.
Off skips adapter.forward but retains612 states; not compute-matched, not a
pruned610 deployment or a zero-cost claim. No numerical execution was done
while authoring this SOURCE; ROOT must independently qualify before use.
This successor never replaces or retypes the original four-endpoint report.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path

SPEC_SCHEMA = 'ROOT-native-geometry-paired-residual-off-controls-evaluation-spec-v1'
QUAL_SCHEMA = 'ROOT-native-geometry-paired-residual-off-controls-evaluation-CPU-qualification-v1'
TRAIN_QUAL_SCHEMA = 'ROOT-native-geometry-paired-common-schedule-CPU-qualification-v1'
TRAIN_QUAL_STATUS = 'ROOT_NATIVE_GEOMETRY_PAIRED_COMMON_SEGMENT_CPU_QUALIFIED'
CP_SCHEMA = 'native-geometry-paired-common-schedule-checkpoint-v1'
MODES = ('native0', 'paired', 'fg_fine')
ENDPOINTS = ('native0', 'paired', 'paired_residual_off', 'fg_fine', 'fg_fine_residual_off')
OFF_ENDPOINTS = ('paired_residual_off', 'fg_fine_residual_off')
ENDPOINT_MODE = {label: label.replace('_residual_off', '') for label in ENDPOINTS}
ADAPTER_PREFIX = 'panoptic_head.paired_mask_readout.'
EVALUATOR_BYTES = 33702
EVALUATOR_SHA = 'a55fdd56317ba2bb549175ce811d5be64e419226ba5a07a0d051f24cb960b137'
TRAIN_READER_BYTES = 60171
TRAIN_READER_SHA = '5f1eac5900e314556b5041b9c2446b9a872ab0f6875d87f544945102e0d1a559'
METRIC_NAMES = ('AP', 'AP50', 'AP75', 'APS', 'APM', 'APL', 'AR1', 'AR10', 'AR100', 'ARS', 'ARM', 'ARL')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def reference(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def bound(row):
    require(isinstance(row, dict) and {'path', 'bytes', 'sha256'} <= set(row) and Path(row['path']).is_absolute(),
        'Complete absolute artifact reference required')
    require(reference(row['path']) == {key: row[key] for key in ('path', 'bytes', 'sha256')}, 'Artifact pin differs: ' + row['path'])
    return Path(row['path'])


def read_ref(row):
    return read_json(bound(row))


def finite(value, label, nonnegative=False):
    require(type(value) in (int, float) and math.isfinite(value) and (not nonnegative or value >= 0), label + ' must be finite')


def terminal(receipt, phase):
    require(receipt['status'] == 'ACTUAL_OWNED_TERMINAL' and receipt['original_identity'] is not None and
        type(receipt['returncode']) is int and receipt['returncode'] == 0 and receipt['root_wait_reaped'] is True and
        receipt['held_handle_exit'] == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=0) and
        receipt['numeric_birth_absence']['original_birth_absent'] is True and receipt['abort'] is None and receipt['error'] is None and
        receipt['wait_errors'] is False and receipt['GPU_observation_errors'] == [],
        'STOP before scientific access: actual owned exit0/held reap/birth absence/no errors required')
    if 'phase' in receipt:
        require(receipt['phase'] == phase, 'Owner receipt phase differs')
    peak = receipt['whole_GPU_observed_peak_MiB']
    finite(peak, 'Observed whole-GPU peak', True)
    require(peak <= 8192 and receipt['whole_GPU_used_limit_MiB'] == 8192 and receipt['whole_leaf_VRAM_certified'] is False and
        receipt['GPU_scope'] == 'whole GPU observed, not per-process WDDM allocation', 'Whole-GPU scope/resource contract differs')


def admission_gate(spec_path, spec, receipt):
    require(isinstance(spec['GPU_admission_file'], str) and Path(spec['GPU_admission_file']).is_absolute(), 'Absolute admission path required')
    admission_ref = reference(spec['GPU_admission_file'])
    admission = read_json(spec['GPU_admission_file'])
    producer = reference(Path(__file__).resolve().with_name('evaluate_common.py'))
    consumer = reference(__file__)
    spec_ref = reference(spec_path)
    require(EVALUATOR_BYTES > 0 and producer['bytes'] == EVALUATOR_BYTES and producer['sha256'] == EVALUATOR_SHA,
        'Independently reviewed frozen evaluator differs')
    require(receipt['admission_sha256'] == admission_ref['sha256'] and
        admission['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and admission['automatic_retry'] is False and
        admission['phase'] == spec['phase'] and admission['allowed_cpus'] == spec['CPU_affinity'] == [0, 1] and
        admission['cuda_visible_devices'] == '0' and admission['qualified_leaf_deadline_seconds'] == spec['deadline_seconds'] and
        admission['whole_GPU_used_limit_MiB'] == spec['whole_GPU_owner_limit_MiB'] == 8192 and
        Path(admission['entry']).resolve(strict=True) == Path(producer['path']).resolve(strict=True) and
        admission['entry_args'] == ['--spec', str(spec_path)], 'Actual evaluation admission/SPEC/source contract differs')
    owner = reference(spec_path.parent / 'finite_point_cache_execution_20261005_v2' / 'run_owned_gpu_limited.py')
    require(owner['sha256'] == 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924' and
        admission['driver_sha256'] == owner['sha256'], 'Actual owned driver differs')
    hashes = admission['bound_source_sha256']
    require(hashes.get(str(spec_path)) == spec_ref['sha256'] and producer in spec['sources'] and consumer in spec['sources'],
        'SPEC/evaluator/CPU reader not bound in actual admission')
    for source in spec['sources']:
        bound(source)
        require(hashes.get(source['path']) == source['sha256'], 'Current source absent from evaluation admission')
    if 'scientific_spec' in admission:
        require(admission['scientific_spec'] == spec_ref, 'Evaluation admission scientific SPEC differs')
    return producer, consumer, spec_ref, admission_ref, owner


def qualified_training(spec):
    qualifications, common = {}, None
    for mode in MODES:
        q = read_ref(spec['qualificationmap'][mode])
        require(q['schema'] == TRAIN_QUAL_SCHEMA and q['status'] == TRAIN_QUAL_STATUS and q['kind'] == 'mask2former' and
            q['mode'] == mode and q['source'] == spec['training_source'] and q['checkpoint'] == spec['checkpointmap'][mode] and
            q['checkpointmap'] == {mode: q['checkpoint']} and q['technical_training_checkpoint_eligible'] is True and
            q['completed_epoch'] == spec['expected_epoch'] and q['global_microsteps'] == 741 * spec['expected_epoch'] and
            q['global_updates'] == 741 * spec['expected_epoch'] // 2 and q['consumed_raw_GT'] == 1661 * spec['expected_epoch'],
            'Mode/complete epoch/qualified full checkpoint join differs')
        terminal(read_ref(q['parent_receipt']), q['phase'])
        for key in ('source', 'reader_source', 'spec', 'report', 'start', 'GPU_admission', 'owner_driver'):
            bound(q[key])
        require(q['reader_source']['bytes'] == TRAIN_READER_BYTES and q['reader_source']['sha256'] == TRAIN_READER_SHA and
            q['reader_source'] == spec['training_CPU_reader'] and spec['training_CPU_reader'] in spec['sources'],
            'Independent full-native training reader source differs')
        training = read_ref(q['spec'])
        admission = read_ref(q['GPU_admission'])
        require(read_ref(q['parent_receipt'])['admission_sha256'] == q['GPU_admission']['sha256'] and
            admission['phase'] == training['phase'] == q['phase'] and admission['automatic_retry'] is False and
            admission['entry_args'] == ['--spec', q['spec']['path']] and
            Path(admission['entry']).resolve() == Path(spec['training_source']['path']).resolve() and
            admission['driver_sha256'] == q['owner_driver']['sha256'], 'Qualified training receipt/admission lineage differs')
        for source in training['sources']:
            bound(source)
            require(source in spec['sources'] and admission['bound_source_sha256'].get(source['path']) == source['sha256'],
                'Qualified training current source closure differs')
        require(admission['bound_source_sha256'].get(q['spec']['path']) == q['spec']['sha256'], 'Qualified training SPEC not admission-bound')
        protocol = q['training_protocol']
        require(protocol['mode'] == mode and protocol['training_source'] == spec['training_source'] and
            protocol['native_loss_only'] is True and protocol['native_loss_atoms'] == 30 and protocol['Sobel'] is False and
            protocol['total_epochs'] == 24 and protocol['total_microsteps'] == 17784 and protocol['total_updates'] == 8892,
            'Full-native training protocol differs from common schedule')
        for key in ('initializer_source', 'micro_preflight_source', 'native_helpers', 'adapter_source', 'hooks_source', 'image_root'):
            require(training[key] == spec[key] == protocol[key], 'Evaluation/training source/input differs: ' + key)
        signature = {key: value for key, value in protocol.items() if key not in
            ('mode', 'adapter_factory_hidden_seed', 'final_coefficient_adapter')}
        if common is None:
            common = signature
        require(signature == common, 'Models do not share a common declared training protocol')
        qproof = q['model_proof']
        require(qproof['checkpoint'] == q['checkpoint'] and qproof['training_protocol'] == protocol and
            qproof['completed_epoch'] == q['completed_epoch'] and
            qproof['technical_proof']['full_native_state_CPU_finite_verified'] is True and
            qproof['technical_proof']['original_typed_BN_affine_and_buffers_exact'] is True,
            'Actual independent trained-state proof differs')
        qualifications[mode] = q
    return qualifications


def tensor_digest(value):
    array = value.detach().cpu().contiguous().numpy()
    return dict(shape=list(array.shape), dtype=str(array.dtype), sha256=hashlib.sha256(array.tobytes(order='C')).hexdigest())


def checkpoint_state(torch, spec, qualification, mode):
    cp = torch.load(str(bound(spec['checkpointmap'][mode])), map_location='cpu')
    require(set(cp) == {'state_dict', 'optimizer_state_dict', 'external_rng_state', 'private_order_state', 'meta'},
        'A complete native training checkpoint is required, never a compact frozen adapter CP')
    state, meta, q = cp['state_dict'], cp['meta'], qualification
    expected = dict(schema=CP_SCHEMA, kind='mask2former', mode=mode, phase=q['phase'], spec=q['spec'], source=spec['training_source'],
        training_protocol=q['training_protocol'], completed_epoch=spec['expected_epoch'], global_microsteps=q['global_microsteps'],
        global_updates=q['global_updates'], consumed_raw_GT=q['consumed_raw_GT'], no_pending_accumulation_gradients=True,
        not_technical_preflight_checkpoint=True)
    for key, value in expected.items():
        require(meta[key] == value, 'Full checkpoint metadata differs: ' + mode + '/' + key)
    proof = read_ref(q['model_proof']['proof_refs']['state_proof'])
    require(proof['schema'] == 'native-geometry-paired-common-segment-state-proof-v1' and proof['kind'] == 'mask2former' and
        proof['mode'] == mode and proof['frozen_BN_all_exact'] is True and proof['all_final_state_finite'] is True,
        'Independent qualified training proof contract differs')
    require(isinstance(state, dict) and len(state) == (610 if mode == 'native0' else 612) and state.keys() == proof['after'].keys(),
        'Actual full trained native/adapter state key set differs')
    for name, value in state.items():
        require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and bool(torch.isfinite(value).all()) and
            tensor_digest(value) == proof['after'][name], 'Actual trained checkpoint tensor differs from independently qualified CPU proof: ' + name)
    native_names = set(proof['original_native_cold'])
    require(len(native_names) == 610 and native_names == {name for name in state if not name.startswith(ADAPTER_PREFIX)},
        'Native610/new2 state boundary differs')
    shape_map = {} if mode == 'native0' else ({'hidden.weight': [512, 512], 'final.weight': [256, 512]} if mode == 'paired' else
        {'hidden.weight': [768, 256], 'final.weight': [256, 768]})
    require({name[len(ADAPTER_PREFIX):]: list(state[name].shape) for name in state if name.startswith(ADAPTER_PREFIX)} == shape_map,
        'Mode-specific trained adapter shape differs')
    policy = proof['policy']
    BN_params, BN_buffers = policy['frozen_names'], policy['BN_buffer_names']
    require(len(BN_params) == 106 and len(BN_buffers) == 159 and len(set(BN_params + BN_buffers)) == 265 and
        policy['state_entries'] == len(state) and policy['parameter_tensors'] == (451 if mode == 'native0' else 453) and
        policy['trainable_tensors'] == (345 if mode == 'native0' else 347) and
        all(tensor_digest(state[name]) == proof['original_native_cold'][name] for name in BN_params + BN_buffers),
        'Frozen native BN boundary/counts/original values differ')
    digests = {name: tensor_digest(value) for name, value in state.items()}
    require(proof['changed'] == q['model_proof']['changed'] and set(proof['changed']) == set(state), 'Actual native change proof differs')
    result = dict(checkpoint=spec['checkpointmap'][mode], qualification=spec['qualificationmap'][mode], fullstate=digests,
        initial_native_state=proof['original_native_cold'], native_changed_from_COCO_ship_initializer={name:
            digests[name] != proof['original_native_cold'][name] for name in native_names},
        trained_segment_changed=proof['changed'], BN_parameter_names=BN_params, BN_buffer_names=BN_buffers,
        state_tensors=len(state), native_state_tensors=610, adapter_state_tensors=len(shape_map),
        full_trained_CP_independent_CPU_digest_exact=True, original_BN_exact=True, trained_native_compared_to_initializer_for_change_only=True)
    del cp, state
    return result


def evaluation_state(record, qualified, spec, qualification, label):
    mode = ENDPOINT_MODE[label]
    proof = read_ref(record['state_proof'])
    loading = proof['loading_proof']
    require(proof['schema'] == 'native-geometry-paired-residual-off-controls-evaluation-state-proof-v1' and proof['mode'] == label and
        proof['checkpoint_mode'] == mode and proof['before'] == proof['after'] == qualified['fullstate'] and
        proof['full_native_and_adapter_state_exact'] is True and proof['params_gradNone'] is True and
        proof['all_parameters_frozen'] is True and proof['all_modules_eval'] is True and
        proof['same_CP_inference_ablation'] is (label in OFF_ENDPOINTS), 'Actual evaluation state proof differs')
    for key, value in dict(mode=mode, checkpoint=spec['checkpointmap'][mode], qualification=spec['qualificationmap'][mode], CP_qualified=True,
        strict_fullstate_loaded=True, native_state_tensors=610, adapter_state_tensors=qualified['adapter_state_tensors'],
        registered_parameters=451 if mode == 'native0' else 453, trained_fullstate=qualified['fullstate'],
        initial_native_state=qualified['initial_native_state'], native_changed_from_COCO_ship_initializer=qualified['native_changed_from_COCO_ship_initializer'],
        trained_segment_changed=qualified['trained_segment_changed'], original_frozen_BN_exact=True,
        inference_backbone_with_cp=False, training_backbone_with_cp=qualification['training_protocol']['backbone_with_cp'],
        no_random_missing_keys=True).items():
        require(loading[key] == value, 'Actual full native loading proof differs: ' + key)
    require(set(loading['BN_parameter_names']) == set(qualified['BN_parameter_names']) and
        set(loading['BN_buffer_names']) == set(qualified['BN_buffer_names']), 'Typed BN state membership differs')
    return dict(full_native_state_independently_CPU_verified=True, adapter_state_independently_CPU_verified=True,
        evaluation_before_after_equal_CPU_CP=True, all_parameter_grads_None_source_observed=True,
        native_changed_from_COCO_ship_initializer=qualified['native_changed_from_COCO_ship_initializer'],
        trained_segment_changed=qualified['trained_segment_changed'], source=record['state_proof'])


def image_population(record, spec, coco, label):
    ids = list(coco.imgs)
    rows = read_ref(record['image_rows'])
    mode = ENDPOINT_MODE[label]
    require(len(rows) == 187 and [row['image_id'] for row in rows] == ids and record['images'] == 187 and record['raw_GT'] == 380,
        'Complete ordered DEV187/380 image population differs')
    for index, row in enumerate(rows):
        image = coco.imgs[ids[index]]
        h, w = image['height'], image['width']
        scale = 800. / max(h, w)
        actual = [int(h * scale + .5), int(w * scale + .5)]
        require(row['mode'] == label and row['checkpoint_mode'] == mode and row['dataset_index'] == index and
            row['ann_ids'] == [ann['id'] for ann in coco.imgToAnns[ids[index]]] and row['annotation_count'] == len(row['ann_ids']) and
            row['original_shape'] == [h, w] and row['actual_img_shape'] == actual and row['batch_input_shape'] == [800, 800] and
            Path(row['image']['path']).resolve() == (Path(spec['image_root']) / image['file_name']).resolve() and
            row['checkpoint'] == spec['checkpointmap'][mode] and row['CP_qualified'] is True, 'Image/GT metadata/preprocessing/CP identity differs')
        bound(row['image'])
        require(row['scale_factor'] == [actual[1] / w, actual[0] / h], 'Native scale_factor differs from actual rounded image size')
        require(row['head_seed'] == 100000 + row['image_id'] and row['external_RNG_before'] == row['external_RNG_after'] and
            row['external_RNG_restored'] is True, 'GT-free head private seed/external RNG differs')
        rng = row['external_RNG_before']
        require(isinstance(rng, list) and len(rng) == 2 and all(isinstance(value, dict) and set(value) == {'shape', 'dtype', 'sha256'} and
            len(value['shape']) == 1 and type(value['shape'][0]) is int and value['shape'][0] > 0 and value['dtype'] == 'uint8' and
            isinstance(value['sha256'], str) and len(value['sha256']) == 64 and set(value['sha256']) <= set('0123456789abcdef') for value in rng),
            'Actual CPU/CUDA RNG tensor digests malformed')
        for key in ('GT_removed_from_forward', 'loader_accesses_annotations', 'official_instance_postprocess',
            'source_native_query_ID_mapping_verified', 'official_masks_query_ID_bitwise_equal', 'official_scores_query_ID_bitwise_equal',
            'query_score_factors_verified', 'class_probability_mask_quality_product_TorchEqual_official_score'):
            require(row[key] is True, 'Actual GT-free official inference proof differs: ' + key)
        require(row['matching_or_loss_executed'] is False and row['added_NMS'] is False and row['added_score_cutoff'] is False and
            row['query_filter'] is False and row['predictions'] == 100 and
            row['inference_ablation_not_training_arm'] is (label in OFF_ENDPOINTS), 'Query/loss/filter/ablation scope differs')
        query_ids = row['prediction_query_ids']
        require(len(query_ids) == 100 and all(type(q) is int for q in query_ids) and set(query_ids) == set(range(100)),
            'Every original query ordinal must appear exactly once')
        for key, shape in (('classes', [1, 100, 2]), ('masks', [1, 100, 200, 200])):
            value = row['final_fields'][key]
            require(value['shape'] == shape and value['dtype'] == 'float32' and len(value['sha256']) == 64 and
                set(value['sha256']) <= set('0123456789abcdef'), 'Actual final FP32 fields summary differs')
        hook = row['paired_hook']
        if mode == 'native0':
            require(hook is None, 'Native baseline cannot claim adapter hook invocation')
        else:
            require(hook['forward_calls'] == hook['completed_forwards'] == hook['pixel_decoder_calls'] == 1 and
                hook['coefficient_calls'] == 10 and hook['unchanged_early_readouts'] == 9 and hook['mode'] == mode and
                hook['apply'] is (label not in OFF_ENDPOINTS) and
                hook['final_coefficient_replacements'] == (0 if label in OFF_ENDPOINTS else 1) and
                hook['native_class_path_untouched'] is True and hook['earlier_gate_paths_untouched'] is True and
                hook['all_queries_retained'] is True and hook['cache_cleared_after_forward'] is True and
                hook['hooks_removed'] is True and hook['paired_source'] == spec['adapter_source'] and
                hook['feature_shape'] == [1, 256, 200, 200] and hook['coefficient_shape'] == [1, 100, 256] and
                hook['metadata_fields_used'] == ['img_shape', 'batch_input_shape'],
                'Actual final-only adapter scope/cleanup differs')
    return ids, rows


def score_factors(record, predictions, rows, np):
    values = read_ref(record['query_score_factors'])
    require(len(values) == len(predictions) == record['query_score_factors_count'] == 18700 and
        record['query_score_factors_verified'] is True and record['class_probability_mask_quality_product_TorchEqual_official_score'] is True,
        'Complete official score-factor population differs')
    seen, exact_count, max_diff = set(), 0, 0.
    for value, prediction in zip(values, predictions):
        require(set(value) == {'image_id', 'query_id', 'class_probability', 'mask_quality', 'official_score'} and
            (value['image_id'], value['query_id']) == (prediction['image_id'], prediction['query_id']) and
            value['official_score'] == prediction['score'], 'Factor/actual prediction identity differs')
        key = (value['image_id'], value['query_id'])
        require(key not in seen, 'Duplicate native factor identity')
        seen.add(key)
        for name in ('class_probability', 'mask_quality', 'official_score'):
            finite(value[name], 'Factor ' + name)
            require(0 <= value[name] <= 1 and float(np.float32(value[name])) == value[name], 'Factor is not an exact bounded FP32 export')
        c, quality, score = value['class_probability'], value['mask_quality'], value['official_score']
        cpu_product = float(np.float32(c) * np.float32(quality))
        exact_product = c * quality
        error_bound = (2. ** -24 / (1. - 2. ** -24)) * abs(exact_product) + 2. ** -126
        require(abs(cpu_product - exact_product) <= error_bound and abs(score - exact_product) <= error_bound,
            'Recorded official/CPU score product violates FP32 multiplication rounding/underflow bound')
        exact_count += int(cpu_product == score)
        max_diff = max(max_diff, abs(cpu_product - score))
    require(seen == {(row['image_id'], q) for row in rows for q in range(100)}, 'Incomplete image/query factor joint')
    return dict(source=record['query_score_factors'], entries=len(values), CPU_FP32_product_exact_official_count=exact_count,
        CPU_FP32_product_all_exact_official=exact_count == len(values), max_abs_CPU_product_minus_official=max_diff,
        factors_independent_CPU_arithmetic_checked=True, original_GPU_TorchEqual_checks_source_observed=True,
        native_mask_quality_is_mean_probability_not_GT_IoU=True, omitted_pixel_logits_quality_not_independently_recomputed=True)


def same_checkpoint_on_off(spec, report, np):
    """Within each trained model only; masks/quality/scores may legitimately differ."""
    proofs = {}
    for mode, off_label in (('paired', 'paired_residual_off'), ('fg_fine', 'fg_fine_residual_off')):
        on, off = (report['mode_reports'][label] for label in (mode, off_label))
        require(on['checkpoint'] == off['checkpoint'] == spec['checkpointmap'][mode] and
            on['qualification'] == off['qualification'] == spec['qualificationmap'][mode] and
            on['checkpoint_mode'] == off['checkpoint_mode'] == mode,
            'Residual-on/off is not the same independently qualified checkpoint/mode')
        on_rows, off_rows = read_ref(on['image_rows']), read_ref(off['image_rows'])
        require(len(on_rows) == len(off_rows) == 187, 'Same-CP on/off image count differs')
        identity_keys = ('dataset_index', 'image_id', 'image', 'ann_ids', 'annotation_count',
            'original_shape', 'actual_img_shape', 'batch_input_shape', 'scale_factor', 'head_seed')
        for a, b in zip(on_rows, off_rows):
            require(all(a[key] == b[key] for key in identity_keys) and
                a['final_fields']['classes'] == b['final_fields']['classes'] and
                a['prediction_query_ids'] == b['prediction_query_ids'],
                'Same-CP on/off input geometry, final class fields or native query order differs')
        populations = [read_ref(record['query_score_factors']) for record in (on, off)]
        indexed = [{(row['image_id'], row['query_id']): row for row in population} for population in populations]
        expected = {(row['image_id'], query_id) for row in on_rows for query_id in range(100)}
        require(len(expected) == 18700 and all(len(population) == len(index) == 18700 and set(index) == expected
            for population, index in zip(populations, indexed)), 'Same-CP all18700 qid factor join differs')
        require(all(np.asarray(indexed[0][key]['class_probability'], dtype=np.float32).tobytes() ==
            np.asarray(indexed[1][key]['class_probability'], dtype=np.float32).tobytes() for key in expected),
            'Same-CP on/off class-probability FP32 bytes differ')
        proofs[mode] = dict(checkpoint=on['checkpoint'], qualification=on['qualification'], off_endpoint=off_label,
            images=187, query_entries=18700, input_geometry_and_final_class_digests_exact=True,
            original_qid_order_exact=True, class_probability_FP32_bytes_exact=True,
            masks_or_quality_or_scores_required_equal=False, cross_model_query_exchange=False,
            inference_only_ablation_not_training_arm=True)
    return proofs


def run(spec_file, acceptance_file):
    spec_path, target = Path(spec_file), Path(acceptance_file)
    require(spec_path.is_absolute() and target.is_absolute() and not target.exists() and target.parent.is_dir(),
        'Absolute SPEC/fresh absolute acceptance in existing directory required')
    spec_path = spec_path.resolve(strict=True)
    spec = read_json(spec_path)
    require(spec['schema'] == SPEC_SCHEMA and spec['phase'].startswith('native_geometry_paired_residual_off_controls_evaluation_') and
        spec['modes'] == list(MODES) and set(spec['checkpointmap']) == set(spec['qualificationmap']) == set(MODES) and
        type(spec['expected_epoch']) is int and spec['expected_epoch'] in range(2, 25, 2) and spec['evaluation_images'] == 187 and
        spec['paired_residual_off_probe'] is True and spec['fg_fine_residual_off_probe'] is True and
        spec['endpoints'] == list(ENDPOINTS), 'Explicit three trained modes/five inference endpoints/even-epoch DEV187 required')
    receipt_path = spec_path.parent / 'finite_point_cache_execution_20261005_v2' / 'runs' / spec['phase'] / 'PARENT_RECEIPT.json'
    receipt = read_json(receipt_path)
    terminal(receipt, spec['phase'])
    # Only the SPEC and actual owner receipt have been opened above this gate.
    source_ref, reader_ref, spec_ref, admission_ref, owner_ref = admission_gate(spec_path, spec, receipt)
    require(spec['allocator_limit_MiB'] == 4096 and spec['evaluation_images'] == 187, 'Locked native evaluation resource/population differs')
    qualifications = qualified_training(spec)
    output = Path(spec['output_directory'])
    report_ref = reference(output / 'REPORT.json')
    report = read_json(output / 'REPORT.json')
    require(report['schema'] == 'native-geometry-paired-residual-off-controls-evaluation-report-v1' and
        report['status'] == 'TECHNICAL_FIVE_ENDPOINT_RESIDUAL_OFF_EXPORT_COMPLETE_NO_AP' and report['error'] is None and
        report['source'] == source_ref and report['spec'] == spec_ref and report['phase'] == spec['phase'] and
        report['modes'] == list(MODES) and report['expected_epoch'] == spec['expected_epoch'] and
        report['checkpointmap'] == spec['checkpointmap'] and report['qualificationmap'] == spec['qualificationmap'] and
        report['DEV'] == spec['DEV'] and report['paired_residual_off_probe'] is True and
        report['fg_fine_residual_off_probe'] is True and report['endpoints'] == list(ENDPOINTS),
        'Actual full-native export report identity/status differs')
    for key, value in dict(evaluation_images=187, raw_GT=380, predictions_per_image=100, optional_off_is_same_CP_inference_ablation=True,
        all_training_terminal_gates_passed_before_scientific_access=True, loader_accesses_annotations=True,
        GT_removed_from_forward=True, matching_executed=False, loss_executed=False, no_TEST=True, AP_computed=False,
        DEV_historically_exposed=True, cross_model_query_ID_object_identity_claimed=False, cross_model_factor_exchange=False,
        added_NMS=False, added_score_cutoff=False, scientific_method_decision_computed=False,
        trained_modes_count=3, inference_endpoints_count=5, training_path_or_regularization_mechanism_claimed=False,
        residual_off_skips_adapter_forward=True, residual_off_compute_matched=False,
        pruned610_deployment_qualified=False, zero_additional_inference_cost_claimed=False).items():
        require(report[key] == value, 'Actual evaluation scope differs: ' + key)
    finite(report['elapsed_seconds'], 'Actual elapsed time', True)
    require(report['elapsed_seconds'] < spec['deadline_seconds'] and type(report['torch_peak_allocated_bytes']) is int and
        0 <= report['torch_peak_allocated_bytes'] <= 4096 * 1024 ** 2, 'Actual evaluation resource guard failed')
    expected_start = dict(schema='native-geometry-paired-residual-off-controls-evaluation-start-v1', phase=spec['phase'], source=source_ref, spec=spec_ref,
        modes=list(MODES), expected_epoch=spec['expected_epoch'], evaluation_images=187, checkpointmap=spec['checkpointmap'],
        qualificationmap=spec['qualificationmap'], paired_residual_off_probe=True, fg_fine_residual_off_probe=True,
        endpoints=list(ENDPOINTS), no_AP=True,
        no_TEST=True, no_retry=True, all_training_terminal_gates_passed_before_scientific_access=True)
    require(read_json(output / 'START.json') == expected_start, 'Actual evaluation START differs')
    labels = list(ENDPOINTS)
    require(set(report['mode_reports']) == set(labels), 'Actual endpoint population differs')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '2'
    os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
    import psutil
    psutil.Process().cpu_affinity([0, 1])
    import numpy as np
    import torch
    from pycocotools import mask as masks
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    require(psutil.Process().cpu_affinity() == [0, 1] and not torch.cuda.is_initialized(), 'CPU01/no initialized CUDA required')
    coco = COCO(str(bound(spec['DEV'])))
    require(len(coco.imgs) == 187 and len(coco.anns) == 380 and set(coco.cats) == {1}, 'Full original DEV187/380/category1 required')
    states = {mode: checkpoint_state(torch, spec, qualifications[mode], mode) for mode in MODES}
    results = {}
    for label in labels:
        mode = ENDPOINT_MODE[label]
        record = report['mode_reports'][label]
        expected = dict(mode=label, checkpoint_mode=mode, checkpoint=spec['checkpointmap'][mode], qualification=spec['qualificationmap'][mode],
            completed_epoch=spec['expected_epoch'], CP_qualified=True, images=187, raw_GT=380, predictions_count=18700,
            query_score_factors_count=18700, fullstate_tensors=610 if mode == 'native0' else 612, full_native_state_exact=True,
            adapter_state_exact=True, params_gradNone=True, loader_accesses_annotations=True, GT_removed_from_forward=True,
            native_official_fusion=True, query_score_factors_verified=True, source_native_query_ID_mapping_verified=True,
            class_probability_mask_quality_product_TorchEqual_official_score=True, cross_model_query_ID_object_identity_claimed=False,
            cross_model_factor_exchange=False, inference_ablation_not_training_arm=label in OFF_ENDPOINTS,
            adapter_forward_executed=mode != 'native0' and label not in OFF_ENDPOINTS,
            retained_adapter_state_tensors=0 if mode == 'native0' else 2,
            residual_off_is_compute_matched=False if label in OFF_ENDPOINTS else None,
            pruned_deployment_qualified=False, added_NMS=False, added_score_cutoff=False)
        for key, value in expected.items():
            require(record[key] == value, 'Actual endpoint record differs: ' + label + '/' + key)
        require(record['cleanup']['allocated_after'] == 0 and type(record['peak_allocated_bytes']) is int and
            0 <= record['peak_allocated_bytes'] <= 4096 * 1024 ** 2, 'Endpoint cleanup/allocator guard failed')
        state_proof = evaluation_state(record, states[mode], spec, qualifications[mode], label)
        ids, image_rows = image_population(record, spec, coco, label)
        predictions, grouped = prediction_population(record, ids, coco, np, masks, image_rows)
        factor_proof = score_factors(record, predictions, image_rows, np)
        rows, empty_predictions = geometry_rows(ids, grouped, coco, np, masks)
        results[label] = dict(checkpoint=spec['checkpointmap'][mode], qualification=spec['qualificationmap'][mode],
            checkpoint_mode=mode, predictions=record['predictions'], image_rows=record['image_rows'], state_proof=state_proof,
            score_factor_proof=factor_proof, COCO=coco_metrics(coco, predictions, ids, COCOeval),
            score_free_all100_best_geometry=geometry_summary(rows), score_free_geometry_rows=rows, empty_predictions_retained=empty_predictions,
            same_CP_inference_ablation=label in OFF_ENDPOINTS, independently_trained_endpoint=label not in OFF_ENDPOINTS)
        del predictions, grouped, image_rows, rows
    on_off_proof = same_checkpoint_on_off(spec, report, np)
    require(not torch.cuda.is_initialized(), 'CPU consumer initialized CUDA')
    acceptance = dict(schema=QUAL_SCHEMA, status='ROOT_NATIVE_GEOMETRY_PAIRED_FIVE_ENDPOINT_RESIDUAL_OFF_DEV_CPU_EVALUATED', phase=spec['phase'],
        spec=spec_ref, source=source_ref, reader_source=reader_ref, report=report_ref, start=reference(output / 'START.json'),
        parent_receipt=reference(receipt_path), GPU_admission=admission_ref, owner_driver=owner_ref,
        expected_epoch=spec['expected_epoch'], checkpointmap=spec['checkpointmap'], qualificationmap=spec['qualificationmap'],
        modes=list(MODES), endpoints=list(ENDPOINTS), mode_results=results, paired_residual_off_probe=True,
        fg_fine_residual_off_probe=True, same_checkpoint_on_off=on_off_proof,
        trained_modes_count=3, inference_endpoints_count=5, training_path_or_regularization_mechanism_claimed=False,
        residual_off_skips_adapter_forward=True, residual_off_compute_matched=False,
        pruned610_deployment_qualified=False, zero_additional_inference_cost_claimed=False,
        full_native_state_CPU_checkpoint_verified=True, all187_images_all380_original_GT_preserved=True,
        all100_predictions_per_image_preserved=True, no_added_NMS_or_score_cutoff=True, official_COCO_maxDets=[1, 10, 100],
        official_segmentation_metrics_computed=True, no_cross_model_query_identity_or_factor_exchange=True,
        loader_accesses_annotations=True, GT_removed_from_model_forward_source_observed=True, CPU_affinity=[0, 1], torch_threads=2,
        CUDA_initialized=False, scientific_method_verdict=None, convergence_claimed=False, SOTA_claimed=False,
        DEV_historically_exposed=True, blind_TEST_claimed=False, evidence_limits=[
            'Full trained native states are checked against prior independent CPU qualification; the CUDA trajectory and inference are not replayed.',
            'GPU official masks/query IDs/factor equality are source-observed; CPU validates the emitted RLE population and factor arithmetic, not omitted logits.',
            'Mask quality is mean predicted foreground probability, not GT IoU or a separately learned quality head.',
            'Best-per-GT all100 coverage reuses predictions and GT after inference; it is not AP, deployable routing or one-to-one recall.',
            'Both residual-off endpoints retain their own model training history/checkpoint and are inference ablations, not fourth or fifth training arms.',
            'apply=False skips geometry/phi but leaves612 states registered; this is not compute-matched or a qualified pruned610 deployment and gives no zero-cost claim.',
            'Same-CP inputs/final classes and all18700 class-probability FP32 bytes are checked; masks/quality/scores may change, without proving a training-path or regularization mechanism.',
            'Independent training changes query semantics, F/e0, supports and class/mask scores; no cross-model query exchange is performed.',
            'This is historically exposed public DEV development evaluation; it is not a blind test or reproduced SOTA claim.'])
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(acceptance, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=acceptance['status'], acceptance=reference(target)), sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    parser.add_argument('--acceptance', required=True)
    args = parser.parse_args()
    run(args.spec, args.acceptance)


# The four original-resolution population/COCO functions below are copied
# whole from the already used factor CPU reader. No warm-start CP logic is
# reused, and none of its cross-query-factor counterfactual functions is copied.

def prediction_population(record, ids, coco, np, masks, image_rows):
    predictions = read_ref(record['predictions'])
    require(isinstance(predictions, list) and len(predictions) == len(ids) * 100,
            'Complete unfiltered all100 prediction population required')
    grouped = {image_id: [] for image_id in ids}
    for index, p in enumerate(predictions):
        require(isinstance(p, dict) and p['image_id'] in grouped and p['category_id'] == 1 and
                p['image_id'] == ids[index // 100], 'Prediction image order/single ship category differs')
        finite(p['score'], 'Official score')
        require(0 <= p['score'] <= 1 and
                type(p['query_id']) is int and 0 <= p['query_id'] < 100 and
                set(p) == {'image_id', 'category_id', 'score', 'segmentation', 'query_id'},
                'Official score bounds/export schema differ')
        image = coco.imgs[p['image_id']]
        rle = p['segmentation']
        require(set(rle) == {'size', 'counts'} and rle['size'] == [image['height'], image['width']] and
                isinstance(rle['counts'], str), 'Original-size compressed COCO RLE required')
        # Decode once here for technical qualification; geometry decodes one
        # image at a time later, never retaining all18700 dense masks.
        encoded = dict(size=rle['size'], counts=rle['counts'].encode('ascii'))
        binary = masks.decode(encoded)
        require(binary.shape == (image['height'], image['width']) and binary.dtype == np.uint8 and
                bool(((binary == 0) | (binary == 1)).all()) and
                int(masks.area(encoded)) == int(binary.sum(dtype=np.int64)),
                'RLE decode shape/binary/pixel count differs')
        grouped[p['image_id']].append(p)
    require(all(len(row) == 100 for row in grouped.values()), 'Per-image queries were dropped')
    require(all({p['query_id'] for p in row} == set(range(100)) for row in grouped.values()),
            'Each image must export every original query ID0..99 exactly once')
    require(all([p['query_id'] for p in grouped[row['image_id']]] == row['prediction_query_ids']
                for row in image_rows), 'Prediction query order differs from its actually verified mapping row')
    return predictions, grouped

def geometry_rows(ids, grouped, coco, np, masks):
    rows, empty_predictions = [], 0
    for image_id in ids:
        population = grouped[image_id]
        binary = np.stack([masks.decode(dict(size=p['segmentation']['size'],
                            counts=p['segmentation']['counts'].encode('ascii'))).astype(bool)
                           for p in population], axis=0)
        areas = binary.sum(axis=(1, 2), dtype=np.int64)
        empty_predictions += int((areas == 0).sum())
        for ann in coco.imgToAnns[image_id]:
            gt = coco.annToMask(ann)
            require(gt.shape == binary.shape[1:] and bool(((gt == 0) | (gt == 1)).all()),
                    'Original GT raster differs from image geometry')
            gt = gt.astype(bool)
            area = int(gt.sum(dtype=np.int64))
            inter = (binary & gt[None]).sum(axis=(1, 2), dtype=np.int64)
            union = areas + area - inter
            iou = np.divide(inter, union, out=np.ones(100, dtype=np.float64), where=union > 0)
            # np.argmax deterministically keeps the first exported row on a
            # tie. Record its verified original query ID separately.
            best = int(np.argmax(iou))
            tp, fp, fn = int(inter[best]), int(areas[best] - inter[best]), int(area - inter[best])
            rows.append(dict(image_id=image_id, annotation_id=ann['id'], category_id=ann['category_id'],
                             iscrowd=int(ann.get('iscrowd', 0)), ignore=int(ann.get('ignore', 0)),
                             original_shape=list(gt.shape), GT_pixels=area, empty_GT_raster=area == 0,
                             best_export_row=best, best_query_id=population[best]['query_id'],
                             best_official_score=population[best]['score'],
                             max_mask_IoU=float(iou[best]), TP=tp, FP=fp, FN=fn,
                             FN_over_GT_area=fn / max(area, 1), FP_over_GT_area=fp / max(area, 1)))
    require(len(rows) == sum(len(coco.imgToAnns[i]) for i in ids), 'A raw GT was discarded')
    return rows, empty_predictions

def geometry_summary(rows):
    values = [row['max_mask_IoU'] for row in rows]
    require(values, 'Formal DEV must contain raw GT')
    return dict(raw_GT=len(rows), empty_GT_retained=sum(row['empty_GT_raster'] for row in rows),
                min_IoU=min(values), mean_IoU=math.fsum(values) / len(values),
                threshold_counts={str(t): sum(v >= t for v in values) for t in (.5, .75, .9)},
                threshold_fractions={str(t): sum(v >= t for v in values) / len(values) for t in (.5, .75, .9)},
                mean_FN_over_GT_area=math.fsum(r['FN_over_GT_area'] for r in rows) / len(rows),
                mean_FP_over_GT_area=math.fsum(r['FP_over_GT_area'] for r in rows) / len(rows),
                denominator='all original GT, including crowd/ignore/empty raster',
                union_empty_IoU=1.0, area_denominator_floor=1,
                reusable_prediction_pairing=True, score_threshold=None,
                export_row_is_native_query_ID=False, original_query_ID_separately_recorded=True,
                diagnostic_only_not_AP_or_one_to_one_recall=True)

def coco_metrics(coco, predictions, ids, COCOeval):
    # loadRes may add fields to its argument; leave the validated export list
    # unchanged and retain the original GT dataset with all its flags.
    dataset = copy.deepcopy(coco)
    dataset.dataset.setdefault('info', {})
    evaluator = COCOeval(dataset, dataset.loadRes(copy.deepcopy(predictions)), 'segm')
    evaluator.params.imgIds = list(ids)
    evaluator.params.catIds = [1]
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    stats = evaluator.stats.tolist()
    require(len(stats) == 12, 'Official COCO must produce12 metrics')
    for value in stats:
        finite(value, 'Official COCO stat')
    return dict(stats=stats, metrics=dict(zip(METRIC_NAMES, stats)), iouType='segm',
                maxDets=[1, 10, 100], original_GT_preserved=True, score_filter=None,
                official_COCO_crowd_ignore_semantics=True)


if __name__ == '__main__':
    main()
