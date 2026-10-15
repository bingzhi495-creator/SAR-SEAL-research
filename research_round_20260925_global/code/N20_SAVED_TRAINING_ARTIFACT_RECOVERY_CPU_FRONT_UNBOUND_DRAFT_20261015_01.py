"""UNBOUND saved-artifact recovery CPU front; no prefix, tail, CLI or execution.

ROOT replaces only the inherited CURRENT CPU schema/counts: own140/failed24/
pins164. Historical cache GPU138/CPU139 remain21; the failed training GPU
specification remains own140/failed23/pins163 and is never a normal completion.
Required prefix hooks run raw22/raw23/raw24 before inherited_source_gate(...,
cpu=True), then their closed joins after it. Helper24 directly consumes the
exact10-field N20_failed_training_record and closes N20_failed_training_stderr.

New required spec fields: N20_failed_training_record, N20_failed_training_stderr,
N20_saved_training_artifacts={START,checkpoints:{'8','24'},FIT_endpoints:{'8','24'},
UPDATE_TRACE}. All six saved artifacts are closed refs to prescribed filenames
inside the preserved original output. Its REPORT.json must remain absent.
Cache/model/init/protocol fields keep the original CPU front contract, including
N20_training_tail_source (the original GPU tail, never this recovery tail).

Returns recovery_artifact_refs, recovery_start, recovery_training_binding (the
original20 fields), recovery_failed_GPU_record, recovery_training_parent, and
unchanged cache/common/parent bindings. No GPU REPORT is created or inferred.
ROOT alone qualifies the new CPU terminal; this front never imports Torch.
"""

import json
from pathlib import Path


N20_TRAIN_SCHEMA = 'ROOT-N20-paired-local-center-feature-error-training-GPU-spec-v1'
N20_TRAIN_CPU_SCHEMA = 'ROOT-N20-paired-feature-error-saved-artifact-recovery-CPU-spec-v1'
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
    """Only original qualified cache GPU138/CPU139; no completed GPU140 exists."""
    require(type(record) is dict and set(record) == {
        'source', 'spec', output_key, 'RAW', 'admission', 'phase_binding', 'bootstrap', 'owner_index'}
        and type(record['owner_index']) is int and record['owner_index'] == owner_index
        and (owner_index, output_key, cuda) in ((138, 'report', '0'), (139, 'Q', '')),
        'Exact eight-field original qualified cache completion record required')
    closure = gate['closure']
    for key in set(record) - {'owner_index'}:
        closed(closure, record[key])
    row = spec['RAW_inputs'][owner_index]
    require(all(row[key] == record[record_key] for key, record_key in (
        ('receipt', 'RAW'), ('admission', 'admission'), ('phase_binding', 'phase_binding'),
        ('bootstrap', 'bootstrap'))), 'Completed cache record must equal its original RAW prefix row')
    joined = gate['lineage'].owner_ref_join(gate['context'], gate['owners'],
        record['RAW'], record['admission'], record['source'], record['spec'], cuda)
    parent = read(closed(closure, record['spec']))
    output = read(closed(closure, record[output_key]))
    admission_key = 'GPU_admission' if cuda == '0' else 'CPU_admission'
    filename = 'REPORT.json' if output_key == 'report' else 'Q.json'
    require(type(joined) is int and joined == owner_index == parent['normal_RAW_count']
        and parent['expected_normal_RAW_count'] == owner_index
        and parent['RAW_inputs'] == spec['RAW_inputs'][:owner_index]
        and parent['entry_source'] == record['source'] and parent['phase'] == row['phase']
        and parent['total_failed_RAW_count'] == 21
        and parent['total_terminal_identity_pin_count'] == owner_index + 21
        and all(closure.get(ref['path']) == ref for ref in parent['sources'])
        and all(spec['terminal_identity_pins'].get(key) == value
                for key, value in parent['terminal_identity_pins'].items())
        and Path(parent[admission_key + '_file']).resolve(strict=True) ==
            Path(record['admission']['path']).resolve(strict=True)
        and Path(record[output_key]['path']).resolve(strict=True) ==
            (Path(parent['output_directory']) / filename).resolve(strict=True),
        'Cache completion exact original owner/source/spec/admission/pins/prefix/output')
    require(output['source'] == record['source'] and output['spec'] == record['spec']
        and output[admission_key] == record['admission'] and output['phase'] == row['phase'],
        'Cache pending output must bind the actual original owner')
    return parent, output


def source_gate(path, spec):
    require(spec['schema'] == N20_TRAIN_CPU_SCHEMA
        and all(type(spec[key]) is int for key in ('normal_RAW_count', 'expected_normal_RAW_count',
            'total_failed_RAW_count', 'total_terminal_identity_pin_count', 'initialization_seed',
            'deadline_seconds', 'maximum_process_RSS_bytes'))
        and spec['normal_RAW_count'] == spec['expected_normal_RAW_count'] == 140
        and spec['total_failed_RAW_count'] == 24 and spec['total_terminal_identity_pin_count'] == 164
        and spec['initialization_seed'] == 202610150201 and spec['deadline_seconds'] == 1800
        and spec['maximum_process_RSS_bytes'] == 16 * 1024**3 and spec['automatic_retry'] is False
        and spec['allocator_limit_MiB'] == 4096 and spec['whole_GPU_limit_MiB'] == 8192
        and Path(__file__).resolve(strict=True) == Path(spec['entry_source']['path']).resolve(strict=True),
        'Independent recovery CPU requires own140/failed24/pins164 and original resources')
    failed22 = n20_failed22_raw_only(spec)
    failed23 = n20_failed23_raw_only(spec)
    failed24 = n20_failed24_raw_only(spec)
    gate = inherited_source_gate(path, spec, cpu=True)
    n20_failed22_closed(spec, gate, failed22)
    n20_failed23_closed(spec, gate, failed23)
    n20_failed24_closed(spec, gate, failed24)
    closure = gate['closure']
    model_keys = ('scout_module_source', 'base_scout_source', 'paired_training_body_source',
                  'original_paired_training_body_source')
    for key in model_keys + ('N20_root_cache_qualified_result', 'N20_initialization_lock',
            'N20_locked_protocol', 'cache_GPU_report', 'region_common_source', 'qualifier_body_source',
            'N20_training_tail_source', 'N20_failed_training_stderr'):
        closed(closure, spec[key])
    protocol = read(closed(closure, spec['N20_locked_protocol']))
    require(_n20_cpu_exact(protocol, spec['expected_N20_locked_protocol'])
        and protocol['schema'] == 'N20-true-feature-neighborhood-fixed24-protocol-v1',
        'Exact complete prelocked N20 protocol; old N16 protocol stays separate')
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
        'Cache pending GPU report/CPU Q stay unchanged under original ROOT envelope')
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
        'Exact original source-only five-tensor initialization metadata')
    initialization_sources = [spec[key] for key in model_keys] + [spec['N20_training_tail_source']]
    require(type(metadata['source_sha256']) is dict and all(
        metadata['source_sha256'].get(Path(ref['path']).name) == ref['sha256']
        for ref in initialization_sources), 'Preserved original initialization source basenames/hashes')

    record = spec['N20_failed_training_record']
    require(type(record) is dict and set(record) == {
        'RAW_ref', 'phase', 'pin', 'returncode', 'admission', 'phase_binding', 'bootstrap',
        'entry_source', 'spec', 'original_report_absent'}
        and type(record['returncode']) is int and record['returncode'] == 1
        and record['original_report_absent'] is True,
        'Only preserved actual1 training record; no normal GPU140 completion')
    for key in ('RAW_ref', 'admission', 'phase_binding', 'bootstrap', 'entry_source', 'spec'):
        closed(closure, record[key])
    training_parent = read(closed(closure, record['spec']))
    output = Path(training_parent['output_directory'])
    shared_keys = model_keys + ('N20_root_cache_qualified_result', 'N20_initialization_lock',
        'N20_locked_protocol', 'expected_N20_locked_protocol', 'expected_N20_root_cache_qualified_result',
        'expected_N20_initialization_digest', 'locked_protocol', 'cache_GPU_report',
        'region_common_source', 'initialization_seed', 'N20_training_tail_source',
        'additional_preserved_N20_training_output_precondition_failure',
        'additional_preserved_N20_current_failure_counter_failure')
    require(training_parent['schema'] == N20_TRAIN_SCHEMA
        and training_parent['entry_source'] == record['entry_source']
        and training_parent['phase'] == record['phase']
        and training_parent['normal_RAW_count'] == training_parent['expected_normal_RAW_count'] == 140
        and training_parent['total_failed_RAW_count'] == 23
        and training_parent['total_terminal_identity_pin_count'] == 163
        and training_parent['RAW_inputs'] == spec['RAW_inputs'][:140]
        and all(closure.get(ref['path']) == ref for ref in training_parent['sources'])
        and all(spec['terminal_identity_pins'].get(key) == value
                for key, value in training_parent['terminal_identity_pins'].items())
        and all(_n20_cpu_exact(training_parent[key], spec[key]) for key in shared_keys)
        and training_parent['deadline_seconds'] == 1800 and training_parent['automatic_retry'] is False
        and training_parent['maximum_process_RSS_bytes'] == 16 * 1024**3
        and training_parent['allocator_limit_MiB'] == 4096 and training_parent['whole_GPU_limit_MiB'] == 8192
        and Path(training_parent['GPU_admission_file']).resolve(strict=True) ==
            Path(record['admission']['path']).resolve(strict=True)
        and output.is_absolute() and output.is_dir() and not output.is_symlink()
        and not (output / 'REPORT.json').exists() and not (output / 'REPORT.json').is_symlink(),
        'Preserved failed GPU original23 prefix/source/admission/science/output and absent REPORT')

    artifacts = spec['N20_saved_training_artifacts']
    require(type(artifacts) is dict and set(artifacts) == {'START', 'checkpoints', 'FIT_endpoints', 'UPDATE_TRACE'}
        and type(artifacts['checkpoints']) is dict and type(artifacts['FIT_endpoints']) is dict
        and set(artifacts['checkpoints']) == set(artifacts['FIT_endpoints']) == {'8', '24'},
        'Six actual saved artifacts only; no absent GPU REPORT reference')
    named_refs = [
        (artifacts['START'], 'START.json'), (artifacts['UPDATE_TRACE'], 'UPDATE_TRACE.jsonl'),
        (artifacts['checkpoints']['8'], 'E08_COMPLETE_PAIRED_SNAPSHOT.pth'),
        (artifacts['checkpoints']['24'], 'E24_COMPLETE_PAIRED_SNAPSHOT.pth'),
        (artifacts['FIT_endpoints']['8'], 'E08_FIT_ENDPOINT.json'),
        (artifacts['FIT_endpoints']['24'], 'E24_FIT_ENDPOINT.json')]
    require(len({ref['path'] for ref, _ in named_refs}) == 6,
        'Saved START, trace and two snapshots/endpoints must be six distinct physical artifacts')
    for ref, filename in named_refs:
        actual = closed(closure, ref)
        require(Path(actual).resolve(strict=True) == (output / filename).resolve(strict=True),
            'Saved artifact must be its prescribed original failed-training output: ' + filename)
    binding = dict(source=record['entry_source'], spec=record['spec'],
        GPU_admission=record['admission'], phase=record['phase'],
        N20_locked_protocol=spec['N20_locked_protocol'], locked_protocol=spec['locked_protocol'],
        parent_cache_report=spec['cache_GPU_report'], neighbor_report=root['GPU_report'],
        initialization_seed=202610150201, order_seed=202610140160001,
        expected_N20_initialization_digest=spec['expected_N20_initialization_digest'],
        frozen_teacher_checkpoint=old_report['teacher_checkpoint'],
        frozen_PR_checkpoint=old_report['fixed_PR_checkpoint'],
        N20_root_cache_qualified_result=spec['N20_root_cache_qualified_result'],
        N20_initialization_lock=spec['N20_initialization_lock'], N20_cache_CPU_Q=root['CPU_Q'])
    binding.update({key: spec[key] for key in model_keys})
    start = read(closed(closure, artifacts['START']))
    require(set(binding) == set(N20_TRAIN_BINDING_KEYS) and type(start) is dict
        and all(start[key] == value for key, value in binding.items())
        and start['schema'] == 'N20-paired-feature-error-training-start-v1'
        and start['parameters_per_arm'] == 18851 and start['tensors_per_arm'] == 5
        and start['arms'] == ['LOCAL_ERROR', 'CENTER_ERROR'] and start['fixed_epochs'] == 24
        and start['updates_each_arm'] == 17784 and start['point_exposure_each_arm'] == 72843264
        and all(start[key] is False for key in ('AMP', 'DEV_AP_evaluated', 'qualified',
                                               'teacher_or_PR_loaded', 'full_native_L_loaded')),
        'Actual original START uses locked five-tensor initialization and fixed24 recipe/binding')
    require('parent' in gate and 'lineage' in gate,
        'Preserve original inherited endpoint parent for final_endpoint_check')
    common = load(spec['region_common_source'], '_N20_recovery_CPU_original_region_library_gate')
    require(callable(getattr(common, 'imported_files', None)), 'Original imported_files ABI required')
    gate.update(common=common, parent_cache_report=old_report,
        parent_cache_report_ref=spec['cache_GPU_report'], neighbor_report=cache_report,
        neighbor_report_ref=root['GPU_report'], protocol=protocol, initialization_metadata=metadata,
        recovery_training_parent=training_parent, recovery_failed_GPU_record=record,
        recovery_artifact_refs=artifacts, recovery_start=start, recovery_training_binding=binding,
        training_binding_keys=N20_TRAIN_BINDING_KEYS,
        training_binding_refs=dict(N20_root_cache_qualified_result=spec['N20_root_cache_qualified_result'],
            N20_initialization_lock=spec['N20_initialization_lock'], N20_cache_CPU_Q=root['CPU_Q']))
    return gate
