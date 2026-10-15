"""UNBOUND N20 endpoint CPU front only; no typed RAW prefix, tail, CLI or run.

ROOT concatenates after its preserved typed RAW front and replaces only the
inherited_source_gate CURRENT CPU branch with N20_TRAIN_CPU_SCHEMA. Historical
SCHEMA/CPU_SCHEMA and all preserved failures remain unchanged. This front never
calls/imports the GPU source_gate and never loads tensors, Torch or a renderer.

New spec refs: N20_completed_training_GPU_record (exact eight-field owner140),
qualifier_body_source, plus the unchanged N20 GPU training cache/init/lock refs.
Returned inherited parent is the original endpoint parent, never GPU140's spec.
New keys: training_parent/report/report_ref/GPU_record/start/endpoints,
training_binding_keys, initialization_metadata. Cache/common/binding keys retain
the GPU training front ABI. ROOT alone qualifies actual terminals afterwards.

Required failure22 prefix hooks: n20_failed22_raw_only(spec) before inherited
gate; n20_failed22_closed(spec, gate, failed22) afterwards. ROOT supplies their
actual preserved-RAW implementation. No failure22 ref or historical count is
invented here; GPU138/CPU139 retain21 and successful training GPU140 has22.
"""

import json
from pathlib import Path


N20_TRAIN_SCHEMA = 'ROOT-N20-paired-local-center-feature-error-training-GPU-spec-v1'
N20_TRAIN_CPU_SCHEMA = 'ROOT-N20-paired-local-center-feature-error-training-CPU-spec-v1'
N20_TRAIN_BINDING_KEYS = (
    'source', 'spec', 'GPU_admission', 'phase', 'N20_locked_protocol', 'locked_protocol',
    'parent_cache_report', 'neighbor_report', 'scout_module_source', 'base_scout_source',
    'paired_training_body_source', 'original_paired_training_body_source',
    'initialization_seed', 'order_seed', 'expected_N20_initialization_digest',
    'frozen_teacher_checkpoint', 'frozen_PR_checkpoint', 'N20_root_cache_qualified_result',
    'N20_initialization_lock', 'N20_cache_CPU_Q')


def _n20_cpu_exact(left, right):
    return json.dumps(left, sort_keys=True, separators=(',', ':'), allow_nan=False) == json.dumps(
        right, sort_keys=True, separators=(',', ':'), allow_nan=False)


def n20_completed_cache_record(record, output_key, owner_index, cuda, gate, spec):
    """Close an original GPU138/CPU139/cache or GPU140/training completion."""
    require(type(record) is dict and set(record) == {
        'source', 'spec', output_key, 'RAW', 'admission', 'phase_binding', 'bootstrap', 'owner_index'}
        and type(record['owner_index']) is int and record['owner_index'] == owner_index
        and output_key in ('report', 'Q') and cuda == ('0' if output_key == 'report' else ''),
        'Exact eight-field original completion record required')
    closure = gate['closure']
    for key in set(record) - {'owner_index'}:
        closed(closure, record[key])
    row = spec['RAW_inputs'][owner_index]
    require(all(row[key] == record[record_key] for key, record_key in (
        ('receipt', 'RAW'), ('admission', 'admission'), ('phase_binding', 'phase_binding'),
        ('bootstrap', 'bootstrap'))), 'Completed record must equal its original RAW prefix row')
    joined = gate['lineage'].owner_ref_join(gate['context'], gate['owners'],
        record['RAW'], record['admission'], record['source'], record['spec'], cuda)
    parent = read(closed(closure, record['spec']))
    output = read(closed(closure, record[output_key]))
    expected_failed = 22 if owner_index == 140 else 21
    admission_key = 'GPU_admission' if cuda == '0' else 'CPU_admission'
    filename = 'REPORT.json' if output_key == 'report' else 'Q.json'
    require(type(joined) is int and joined == owner_index == parent['normal_RAW_count']
        and parent['expected_normal_RAW_count'] == owner_index
        and parent['RAW_inputs'] == spec['RAW_inputs'][:owner_index]
        and parent['entry_source'] == record['source'] and parent['phase'] == row['phase']
        and parent['total_failed_RAW_count'] == expected_failed
        and parent['total_terminal_identity_pin_count'] == owner_index + expected_failed
        and all(closure.get(ref['path']) == ref for ref in parent['sources'])
        and all(spec['terminal_identity_pins'].get(key) == value
                for key, value in parent['terminal_identity_pins'].items())
        and Path(parent[admission_key + '_file']).resolve(strict=True) ==
            Path(record['admission']['path']).resolve(strict=True)
        and Path(record[output_key]['path']).resolve(strict=True) ==
            (Path(parent['output_directory']) / filename).resolve(strict=True),
        'Completed original owner/source/spec/admission/pins/prefix/output locator must be exact')
    require(output['source'] == record['source'] and output['spec'] == record['spec']
        and output[admission_key] == record['admission'] and output['phase'] == row['phase'],
        'Completed pending output must bind the actual original owner')
    return parent, output


def _n20_cpu_training_binding(value, report):
    require(type(value) is dict and all(value[key] == report[key] for key in N20_TRAIN_BINDING_KEYS),
            'Complete original GPU START/endpoint training binding required')


def source_gate(path, spec):
    require(spec['schema'] == N20_TRAIN_CPU_SCHEMA
        and all(type(spec[key]) is int for key in ('normal_RAW_count', 'expected_normal_RAW_count',
            'total_failed_RAW_count', 'total_terminal_identity_pin_count', 'initialization_seed',
            'deadline_seconds', 'maximum_process_RSS_bytes'))
        and spec['normal_RAW_count'] == spec['expected_normal_RAW_count'] == 141
        and spec['total_failed_RAW_count'] == 22 and spec['total_terminal_identity_pin_count'] == 163
        and spec['initialization_seed'] == 202610150201 and spec['deadline_seconds'] == 1800
        and spec['maximum_process_RSS_bytes'] == 16 * 1024**3 and spec['automatic_retry'] is False
        and spec['allocator_limit_MiB'] == 4096 and spec['whole_GPU_limit_MiB'] == 8192
        and Path(__file__).resolve(strict=True) == Path(spec['entry_source']['path']).resolve(strict=True),
        'Independent current N20 CPU entry requires own141/failed22/pins163/resources')
    failed22 = n20_failed22_raw_only(spec)
    gate = inherited_source_gate(path, spec, cpu=True)
    n20_failed22_closed(spec, gate, failed22)
    closure = gate['closure']
    model_keys = ('scout_module_source', 'base_scout_source', 'paired_training_body_source',
                  'original_paired_training_body_source')
    for key in model_keys + ('N20_root_cache_qualified_result', 'N20_initialization_lock',
            'N20_locked_protocol', 'cache_GPU_report', 'region_common_source', 'qualifier_body_source'):
        closed(closure, spec[key])
    protocol = read(closed(closure, spec['N20_locked_protocol']))
    require(_n20_cpu_exact(protocol, spec['expected_N20_locked_protocol'])
        and protocol['schema'] == 'N20-true-feature-neighborhood-fixed24-protocol-v1',
        'Exact complete prelocked N20 protocol; old N16 protocol is separate')
    root = read(closed(closure, spec['N20_root_cache_qualified_result']))
    require(_n20_cpu_exact(root, spec['expected_N20_root_cache_qualified_result'])
        and root['schema'] == 'Root-N20-fullFIT-true-feature-neighbor-cache-post-actual-qualified-v1'
        and root['status'] == 'FULLFIT_TRUE_FEATURE_NEIGHBOR_CACHE_QUALIFIED'
        and root['qualified'] is True and root['original_GPU_CPU_actual0_verified'] is True
        and root['full_proofs_ROOT_verified'] is True
        and root['images'] == 741 and root['sampled_points'] == 3035136
        and root['unique_centers'] == 1786820 and root['parent_cache'] == spec['cache_GPU_report']
        and root['protocol'] == spec['N20_locked_protocol']
        and root['GPU_report'] == root['gpu_record']['report'] and root['CPU_Q'] == root['CPU_record']['Q'],
        'Keep exact independently qualified GPU138/CPU139 cache ROOT envelope')
    gpu_spec, cache_report = n20_completed_cache_record(root['gpu_record'], 'report', 138, '0', gate, spec)
    cpu_spec, cpu_q = n20_completed_cache_record(root['CPU_record'], 'Q', 139, '', gate, spec)
    old_report = read(closed(closure, spec['cache_GPU_report']))
    require(cache_report['schema'] == 'N20-true-feature-neighbor-cache-GPU-report-v1'
        and cache_report['status'] == 'N20_FULLFIT_TRUE_FEATURE_NEIGHBORS_COMPLETE_PENDING_INDEPENDENT_CPU'
        and cache_report['parent_cache'] == spec['cache_GPU_report']
        and cache_report['protocol'] == spec['N20_locked_protocol']
        and cpu_q['schema'] == 'N20-fullFIT-feature-neighbor-cache-CPU-qualification-v1'
        and cpu_q['status'] == 'N20_FULLFIT_FEATURE_NEIGHBOR_CACHE_CPU_COMPLETE_PENDING_ROOT_ACTUAL0'
        and cpu_q['capture_GPU_report'] == root['GPU_report']
        and cpu_q['protocol'] == spec['N20_locked_protocol']
        and cpu_q['qualified'] is False and cpu_q['ROOT_actual_CPU0_verified'] is False
        and cpu_q['original_owned_terminal_self_certified'] is False
        and cpu_q['images'] == 741 and cpu_q['sampled_points'] == 3035136
        and cpu_q['unique_centers'] == 1786820
        and cpu_q['extra_feature_bytes'] == cache_report['extra_feature_bytes']
        and gpu_spec['N20_locked_protocol'] == cpu_spec['N20_locked_protocol'] == spec['N20_locked_protocol'],
        'Original cache pending GPU report/CPU Q remain unchanged under their ROOT envelope')
    closed(closure, cpu_q['image_proofs'])
    require(old_report['images'] == cache_report['images'] == 741
        and old_report['sampled_points'] == cache_report['sampled_points'] == 3035136
        and len(old_report['rows']) == len(old_report['packet_refs']) ==
            len(cache_report['rows']) == len(cache_report['packet_refs']) == 741,
        'Complete unchanged original/new FIT741 physical cache populations')
    for index, (oldrow, row) in enumerate(zip(old_report['rows'], cache_report['rows'])):
        require(index == oldrow['dataset_index'] == row['dataset_index']
            and oldrow['packet'] == old_report['packet_refs'][index]
            and row['packet'] == cache_report['packet_refs'][index] and row['parent_packet'] == oldrow['packet']
            and row['identity'] == oldrow['identity'] and row['image_id'] == oldrow['image_id']
            and row['points'] == oldrow['points'] == 4096
            and row['unique_centers'] == oldrow['unique_sampled_F_points'],
            'Qualified N20 packet retains original identity/sample row')
        closed(closure, row['packet'])
        closed(closure, oldrow['packet'])
    metadata = read(closed(closure, spec['N20_initialization_lock']))
    require(metadata['schema'] == 'Root-N20-source-only-private-initialization-lock-v1'
        and type(metadata['seed']) is int and metadata['seed'] == 202610150201
        and metadata['combined_digest'] == spec['expected_N20_initialization_digest']
        and all(metadata[key] is True for key in ('paired_state_exact', 'paired_storage_independent',
            'spatial_weights_zero', 'external_CPU_RNG_unchanged'))
        and metadata['CUDA_initialized'] is False and metadata['training_executed'] is False
        and metadata['torch_version'] == '2.0.1+cu118' and metadata['CPU_affinity'] == [0, 1]
        and type(metadata['threads']) is int and metadata['threads'] == 2,
        'Exact new N20 source-only five-tensor initialization metadata')
    initialization_sources = [spec[key] for key in model_keys]
    if 'N20_training_tail_source' in spec:
        closed(closure, spec['N20_training_tail_source'])
        initialization_sources.append(spec['N20_training_tail_source'])
    require(type(metadata['source_sha256']) is dict and all(
        metadata['source_sha256'].get(Path(ref['path']).name) == ref['sha256']
        for ref in initialization_sources), 'Preserved initialization basenames and model-source hashes')

    record = spec['N20_completed_training_GPU_record']
    training_parent, report = n20_completed_cache_record(record, 'report', 140, '0', gate, spec)
    require(training_parent['schema'] == N20_TRAIN_SCHEMA,
            'Actual completed owner140 must be original N20 training GPU specification')
    shared_keys = model_keys + ('N20_root_cache_qualified_result', 'N20_initialization_lock',
        'N20_locked_protocol', 'expected_N20_locked_protocol', 'expected_N20_root_cache_qualified_result',
        'expected_N20_initialization_digest', 'locked_protocol', 'cache_GPU_report',
        'region_common_source', 'initialization_seed')
    require(all(_n20_cpu_exact(training_parent[key], spec[key]) for key in shared_keys),
            'CPU must retain original training model/cache/initialization/protocol literals')
    require(report['schema'] == 'N20-paired-local-center-feature-error-training-GPU-report-v1'
        and report['status'] == 'N20_FULLFIT_PAIRED_FEATURE_ERROR_E8_E24_COMPLETE_PENDING_INDEPENDENT_CPU'
        and report['normal_RAW_count'] == 140 and report['total_failed_RAW_count'] == 22
        and report['total_terminal_identity_pin_count'] == 162
        and report['images'] == 741 and report['natural_queries'] == 74100 and report['epochs'] == 24
        and report['sampled_points_per_epoch'] == 3035136 and report['paired_updates_per_arm'] == 17784
        and report['point_exposure_per_arm'] == 72843264 and report['parameters_per_arm'] == 18851
        and report['tensors_per_arm'] == 5 and report['arms'] == ['LOCAL_ERROR', 'CENTER_ERROR']
        and report['initialization_seed'] == 202610150201 and report['order_seed'] == 202610140160001
        and report['expected_N20_initialization_digest'] == spec['expected_N20_initialization_digest']
        and all(report[key] is False for key in ('qualified', 'ROOT_actual_GPU0_verified',
            'current_owned_terminal_self_certified', 'teacher_or_PR_loaded', 'full_native_L_loaded',
            'DEV_AP_evaluated', 'best_endpoint_selected', 'old_cache_Source_or_protocol_modified',
            'additional_FIT_annotations', 'new_primitive_claimed'))
        and report['CPU_endpoint_qualification_pending'] is True
        and report['old_point_query_label_weight_order_preserved'] is True
        and report['cleanup'] == {'allocated_after': 0, 'reserved_after': 0},
        'Original GPU140 complete fixed24 report stays pending; no terminal self-certification')
    expected_report_bindings = dict(
        N20_locked_protocol=spec['N20_locked_protocol'], locked_protocol=spec['locked_protocol'],
        parent_cache_report=spec['cache_GPU_report'], neighbor_report=root['GPU_report'],
        N20_root_cache_qualified_result=spec['N20_root_cache_qualified_result'],
        N20_initialization_lock=spec['N20_initialization_lock'], N20_cache_CPU_Q=root['CPU_Q'],
        frozen_teacher_checkpoint=old_report['teacher_checkpoint'],
        frozen_PR_checkpoint=old_report['fixed_PR_checkpoint'])
    expected_report_bindings.update({key: spec[key] for key in model_keys})
    require(all(report[key] == value for key, value in expected_report_bindings.items()),
            'Training report complete model/cache/frozen-teacher/PR bindings exact')
    for key in ('START', 'UPDATE_TRACE'):
        closed(closure, report[key])
    require(type(report['checkpoints']) is dict and type(report['FIT_endpoints']) is dict
        and set(report['checkpoints']) == set(report['FIT_endpoints']) == {'8', '24'},
        'Only actual complete E8/E24 checkpoint and endpoint references')
    for ref in tuple(report['checkpoints'].values()) + tuple(report['FIT_endpoints'].values()):
        closed(closure, ref)
    start = read(closed(closure, report['START']))
    _n20_cpu_training_binding(start, report)
    require(start['schema'] == 'N20-paired-feature-error-training-start-v1'
        and start['initial_states'] == report['initial_states']
        and start['parameters_per_arm'] == 18851 and start['tensors_per_arm'] == 5
        and start['arms'] == ['LOCAL_ERROR', 'CENTER_ERROR'] and start['fixed_epochs'] == 24
        and start['updates_each_arm'] == 17784 and start['point_exposure_each_arm'] == 72843264
        and all(start[key] is False for key in ('AMP', 'DEV_AP_evaluated', 'qualified',
                                               'teacher_or_PR_loaded', 'full_native_L_loaded')),
        'Actual original GPU START uses locked N20 initialization and complete fixed24 recipe')
    endpoints = {}
    for epoch in (8, 24):
        payload = read(closed(closure, report['FIT_endpoints'][str(epoch)]))
        _n20_cpu_training_binding(payload, report)
        endpoint = payload['endpoint']
        require(type(endpoint) is dict and endpoint['epoch'] == epoch
            and endpoint['initialization_seed'] == 202610150201
            and set(endpoint['arms']) == {'LOCAL_ERROR', 'CENTER_ERROR'},
            'Pass actual FIT endpoint inner dictionary only')
        endpoints[epoch] = endpoint
    require('parent' in gate and 'lineage' in gate,
            'Original inherited endpoint parent remains available for final_endpoint_check')
    common = load(spec['region_common_source'], '_N20_CPU_original_region_library_gate')
    require(callable(getattr(common, 'imported_files', None)), 'Original imported_files ABI required')
    gate.update(common=common, parent_cache_report=old_report,
        parent_cache_report_ref=spec['cache_GPU_report'], neighbor_report=cache_report,
        neighbor_report_ref=root['GPU_report'], protocol=protocol, initialization_metadata=metadata,
        training_parent=training_parent, training_report=report, training_report_ref=record['report'],
        training_GPU_record=record, training_start=start, training_endpoints=endpoints,
        training_binding_keys=N20_TRAIN_BINDING_KEYS,
        training_binding_refs=dict(N20_root_cache_qualified_result=spec['N20_root_cache_qualified_result'],
            N20_initialization_lock=spec['N20_initialization_lock'], N20_cache_CPU_Q=root['CPU_Q']))
    return gate
