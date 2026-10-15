"""UNBOUND N20 fixed DEV consumer FRONT only; no typed prefix, tail or CLI.

ROOT replaces only inherited_source_gate's current GPU schema with N20_DEV_SCHEMA
and current ninth counts with23/+23. Historical schemas/counts remain intact.
Helpers22/23 must precede this front. No Torch, tensor load, model or IO on import.
The exact record join below is copied from ordinary CPUfront02, extended only
for completed CPU141. Pending GPU reports/CPU Q are immutable under ROOT envelopes.

Required ordinary legacy helper: legacy_gate(spec,gate) copied from N17 Source04,
with its validate_legacy_COCO/OLD_CHECKS/COCO_METRIC_NAMES/LEGACY_SHARD_RANGES.
It consumes existing spec.legacy_qualified_result and spec.legacy_shards and
returns legacy_root/rows/reports/manifests/populations. Do not invoke old N17
source_gate/training_gate/model_load_cpu or change the old four logical ranges.

Required ROOT-defined METADATA-ONLY hook n20_prior_DEV_carrier_gate(spec,gate)
returns the same gate after binding the actual old native_packet/shared carrier,
priorpacket_gate (original prior-report argument), N17 qualified common-PR bundles
and decoder refs. Actual field names/refs are not guessed here. No default true
hook is provided. Actual tensors/neighbors/PR still need the owned leaf's checks.
This front leaves gate.protocol as the inherited N16 protocol; N20_protocol is new.
"""

import json
import math
from pathlib import Path

N20_DEV_SCHEMA = 'ROOT-N20-fixedDEV-local-center-feature-error-GPU-shard-spec-v1'
N20_TRAIN_SCHEMA = 'ROOT-N20-paired-local-center-feature-error-training-GPU-spec-v1'
N20_TRAIN_CPU_SCHEMA = 'ROOT-N20-paired-local-center-feature-error-training-CPU-spec-v1'
N20_DEV_RANGES = ((0, 47), (47, 94), (94, 141), (141, 187))


def _n20_dev_exact(left, right):
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
    expected_failed = {138: 21, 139: 21, 140: 23, 141: 23}[owner_index]
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


def source_gate(path, spec):
    require(spec['schema'] == N20_DEV_SCHEMA and type(spec['shard_id']) is int
        and 0 <= spec['shard_id'] < 4
        and spec['dataset_ordinal_range'] == list(N20_DEV_RANGES[spec['shard_id']])
        and all(type(spec[key]) is int for key in ('normal_RAW_count', 'expected_normal_RAW_count',
            'total_failed_RAW_count', 'total_terminal_identity_pin_count', 'initialization_seed',
            'deadline_seconds', 'maximum_process_RSS_bytes'))
        and spec['normal_RAW_count'] == spec['expected_normal_RAW_count'] == 142 + spec['shard_id']
        and spec['total_failed_RAW_count'] == 23
        and spec['total_terminal_identity_pin_count'] == spec['normal_RAW_count'] + 23
        and spec['initialization_seed'] == 202610150201 and spec['deadline_seconds'] == 1800
        and spec['automatic_retry'] is False and spec['maximum_process_RSS_bytes'] == 16 * 1024**3
        and spec['allocator_limit_MiB'] == 4096 and spec['whole_GPU_limit_MiB'] == 8192
        and Path(__file__).resolve(strict=True) == Path(spec['entry_source']['path']).resolve(strict=True),
        'New N20 DEV GPU142+sid/failed23/fixed four ranges and exact own resource contract')
    failed22 = n20_failed22_raw_only(spec)
    failed23 = n20_failed23_raw_only(spec)
    gate = inherited_source_gate(path, spec, cpu=False)
    n20_failed22_closed(spec, gate, failed22)
    n20_failed23_closed(spec, gate, failed23)
    closure = gate['closure']
    model_keys = ('scout_module_source', 'base_scout_source', 'paired_training_body_source',
                  'original_paired_training_body_source')
    for key in model_keys + ('N20_root_training_qualified_result', 'N20_root_cache_qualified_result',
            'N20_initialization_lock', 'N20_locked_protocol', 'locked_protocol', 'cache_GPU_report',
            'region_common_source', 'new_eval_body_source', 'legacy_qualified_result'):
        closed(closure, spec[key])
    protocol = read(closed(closure, spec['N20_locked_protocol']))
    require(_n20_dev_exact(protocol, spec['expected_N20_locked_protocol'])
        and protocol['schema'] == 'N20-true-feature-neighborhood-fixed24-protocol-v1',
        'Complete original N20 protocol literal, separate from inherited N16 protocol')
    metadata = read(closed(closure, spec['N20_initialization_lock']))
    digest = spec['expected_N20_initialization_digest']
    require(type(digest) is str and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest)
        and metadata['schema'] == 'Root-N20-source-only-private-initialization-lock-v1'
        and metadata['seed'] == 202610150201 and metadata['combined_digest'] == digest
        and metadata['CUDA_initialized'] is False and metadata['training_executed'] is False
        and all(metadata[key] is True for key in ('paired_state_exact', 'paired_storage_independent',
            'spatial_weights_zero', 'external_CPU_RNG_unchanged')),
        'Use original prelocked source-only initialization; never synthesize unknown digest')
    initialization_sources = [spec[key] for key in model_keys]
    if 'N20_training_tail_source' in spec:
        closed(closure, spec['N20_training_tail_source'])
        initialization_sources.append(spec['N20_training_tail_source'])
    require(type(metadata['source_sha256']) is dict and all(
        metadata['source_sha256'].get(Path(ref['path']).name) == ref['sha256']
        for ref in initialization_sources), 'All original initialization source basenames and hashes')
    cache = read(closed(closure, spec['N20_root_cache_qualified_result']))
    require(_n20_dev_exact(cache, spec['expected_N20_root_cache_qualified_result'])
        and cache['schema'] == 'Root-N20-fullFIT-true-feature-neighbor-cache-post-actual-qualified-v1'
        and cache['status'] == 'FULLFIT_TRUE_FEATURE_NEIGHBOR_CACHE_QUALIFIED'
        and cache['qualified'] is True and cache['original_GPU_CPU_actual0_verified'] is True
        and cache['full_proofs_ROOT_verified'] is True and cache['images'] == 741
        and cache['sampled_points'] == 3035136 and cache['unique_centers'] == 1786820
        and cache['parent_cache'] == spec['cache_GPU_report'] and cache['protocol'] == spec['N20_locked_protocol']
        and cache['GPU_report'] == cache['gpu_record']['report'] and cache['CPU_Q'] == cache['CPU_record']['Q'],
        'Original qualified cache envelope and GPU138/CPU139 refs remain unchanged')
    cache_gpu, cache_report = n20_completed_cache_record(cache['gpu_record'], 'report', 138, '0', gate, spec)
    cache_cpu, cache_q = n20_completed_cache_record(cache['CPU_record'], 'Q', 139, '', gate, spec)
    require(cache_gpu['N20_locked_protocol'] == cache_cpu['N20_locked_protocol'] == spec['N20_locked_protocol']
        and cache_report['protocol'] == cache_q['protocol'] == spec['N20_locked_protocol']
        and cache_q['capture_GPU_report'] == cache['GPU_report'] and cache_q['qualified'] is False
        and cache_q['ROOT_actual_CPU0_verified'] is False,
        'Do not rewrite old cache pending outputs as current qualified records')
    root = read(closed(closure, spec['N20_root_training_qualified_result']))
    require(_n20_dev_exact(root, spec['expected_N20_root_training_qualified_result'])
        and root['schema'] == 'Root-N20-fullFIT-paired-feature-error-fixed24-training-post-actual-qualified-v1'
        and root['status'] == 'N20_PAIRED_FEATURE_ERROR_E8_E24_TRAINING_QUALIFIED'
        and all(root[key] is True for key in ('qualified', 'original_GPU_CPU_actual0_verified',
                                             'full_endpoint_ROOT_verified'))
        and root['images'] == 741 and root['epochs'] == 24
        and root['sampled_points_per_epoch'] == 3035136 and root['updates_each_arm'] == 17784
        and root['parameters_per_arm'] == 18851 and root['tensors_per_arm'] == 5
        and root['cache_qualified_result'] == spec['N20_root_cache_qualified_result']
        and root['protocol'] == spec['N20_locked_protocol']
        and root['initialization_lock'] == spec['N20_initialization_lock']
        and root['initialization_digest'] == digest
        and root['GPU_report'] == root['gpu_record']['report'] and root['CPU_Q'] == root['CPU_record']['Q'],
        'Future actual0-qualified training ROOT envelope is mandatory before DEV consumption')
    gpu, report = n20_completed_cache_record(root['gpu_record'], 'report', 140, '0', gate, spec)
    cpu, q = n20_completed_cache_record(root['CPU_record'], 'Q', 141, '', gate, spec)
    require(gpu['schema'] == N20_TRAIN_SCHEMA and cpu['schema'] == N20_TRAIN_CPU_SCHEMA,
            'Exact original N20 training GPU140 and independent endpoint CPU141 schemas')
    shared_keys = model_keys + ('N20_root_cache_qualified_result', 'expected_N20_root_cache_qualified_result',
        'N20_initialization_lock', 'expected_N20_initialization_digest', 'N20_locked_protocol',
        'expected_N20_locked_protocol', 'locked_protocol', 'cache_GPU_report', 'region_common_source',
        'initialization_seed')
    require(all(_n20_dev_exact(parent[key], spec[key]) for parent in (gpu, cpu) for key in shared_keys)
        and cpu['N20_completed_training_GPU_record'] == root['gpu_record'],
        'Both completed training parents retain identical original cache/model/init/protocol refs')
    require(report['schema'] == 'N20-paired-local-center-feature-error-training-GPU-report-v1'
        and report['status'] == 'N20_FULLFIT_PAIRED_FEATURE_ERROR_E8_E24_COMPLETE_PENDING_INDEPENDENT_CPU'
        and report['normal_RAW_count'] == 140 and report['total_failed_RAW_count'] == 23
        and report['total_terminal_identity_pin_count'] == 163
        and report['images'] == 741 and report['epochs'] == 24 and report['parameters_per_arm'] == 18851
        and report['tensors_per_arm'] == 5 and report['paired_updates_per_arm'] == 17784
        and report['point_exposure_per_arm'] == 72843264
        and report['arms'] == ['LOCAL_ERROR', 'CENTER_ERROR']
        and report['initialization_seed'] == 202610150201 and report['order_seed'] == 202610140160001
        and report['expected_N20_initialization_digest'] == digest
        and report['qualified'] is False and report['ROOT_actual_GPU0_verified'] is False
        and report['current_owned_terminal_self_certified'] is False
        and report['CPU_endpoint_qualification_pending'] is True,
        'Original GPU140 report stays pending below the separate ROOT actual0 envelope')
    require(q['schema'] == 'N20-paired-local-center-feature-error-training-endpoint-CPU-qualification-v1'
        and q['status'] == 'N20_PAIRED_FEATURE_ERROR_E8_E24_ENDPOINT_CPU_COMPLETE_PENDING_ROOT_ACTUAL0'
        and q['normal_RAW_count'] == 141 and q['total_failed_RAW_count'] == 23
        and q['total_terminal_identity_pin_count'] == 164
        and q['training_GPU_record'] == root['gpu_record'] and q['training_GPU_report'] == root['GPU_report']
        and q['images'] == 741 and q['fixed_endpoints'] == [8, 24]
        and q['parameters_per_arm'] == 18851 and q['tensors_per_arm'] == 5
        and q['E8_updates_per_arm'] == 5928 and q['E24_updates_per_arm'] == 17784
        and q['qualified'] is False and q['ROOT_actual_CPU0_verified'] is False
        and q['original_owned_terminal_self_certified'] is False
        and all(q[key] is False for key in ('GPU_pointwise_logits_qualified', 'optimizer_trajectory_replayed',
            'selected_indices_qualified', 'mask_RLE_qualified', 'AP_evaluated',
            'teacher_or_PR_loaded', 'full_native_L_loaded'))
        and type(q['checks']) is dict and bool(q['checks']) and all(v is True for v in q['checks'].values()),
        'Immutable independent CPU Q scope is endpoint-only, not GPU/ranking/mask/AP qualification')
    closed(closure, q['details'])
    closed(closure, cpu['qualifier_body_source'])
    require(q['qualifier_body_source'] == cpu['qualifier_body_source'], 'Original CPU numerical body exact')
    for key in model_keys + ('N20_locked_protocol', 'locked_protocol', 'N20_root_cache_qualified_result',
            'N20_initialization_lock', 'initialization_seed'):
        require(report[key] == q[key] == spec[key], 'Original GPU/CPU training binding differs: ' + key)
    require(report['neighbor_report'] == q['neighbor_report'] == cache['GPU_report']
        and report['parent_cache_report'] == q['parent_cache_report'] == spec['cache_GPU_report']
        and report['N20_cache_CPU_Q'] == q['N20_cache_CPU_Q'] == cache['CPU_Q']
        and q['expected_N20_initialization_digest'] == digest,
        'Actual training uses the same independently qualified extra-F and original center caches')
    for key in ('START', 'UPDATE_TRACE'):
        closed(closure, report[key])
    require(type(report['checkpoints']) is dict and type(report['FIT_endpoints']) is dict
        and set(report['checkpoints']) == set(report['FIT_endpoints']) == {'8', '24'},
        'Both original trained endpoints, never a DEV-selected checkpoint')
    for ref in tuple(report['checkpoints'].values()) + tuple(report['FIT_endpoints'].values()):
        closed(closure, ref)
    require(callable(globals().get('legacy_gate')), 'ROOT must append the known ordinary N17 legacy metadata gate')
    require(legacy_gate(spec, gate) is gate, 'Preserve original fully qualified DEV shared carrier metadata gate')
    require(len(gate['legacy_rows']) == 187 and len(gate['legacy_reports']) == 4
        and len(gate['legacy_manifests']) == 4 and set(gate['populations']) == {
            'native_official', 'strong_fixed_E24_Omega'}, 'Original full187 native/R and physical carrier ABI')
    common = load(spec['region_common_source'], '_N20_DEV_original_region_library_gate')
    require(callable(getattr(common, 'imported_files', None)), 'Original library gate ABI remains exact')
    gate.update(common=common, N20_protocol=protocol, initialization_metadata=metadata,
        N20_cache_root=cache, N20_training_root=root,
        training=dict(root=root, gpu=gpu, cpu=cpu, report=report, q=q,
            checkpoints=report['checkpoints'], FIT_endpoints=report['FIT_endpoints']),
        trained_checkpoints=report['checkpoints'], N20_training_report_ref=root['GPU_report'],
        N20_training_CPU_Q_ref=root['CPU_Q'])
    require('parent' in gate and 'lineage' in gate and 'evaluation' in gate,
            'Inherited original endpoint parent/lineage/fixed PR loader must remain available')
    require(callable(globals().get('n20_prior_DEV_carrier_gate')),
        'UNBOUND: ROOT must supply exact metadata-only native/shared/common-PR/decoder reference gate')
    require(n20_prior_DEV_carrier_gate(spec, gate) is gate,
        'ROOT prior DEV carrier hook must preserve this verified gate; no default true or guessed fields')
    return gate


