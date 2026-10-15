"""UNBOUND N20 training front; concatenate after immutable typed21 RAW front.

ROOT changes only inherited_source_gate's CURRENT GPU schema branch to
N20_TRAIN_SCHEMA. Its historical SCHEMA and all preserved failure checks stay
unchanged. This file has no CLI/run and performs no reads on import. Required
existing helpers: require/read/closed/load; inherited gate provides lineage,
context, owners and the original endpoint_gate parent. Strict initialization
source-map keys are the four model source refs' actual basenames. Optional
N20_training_tail_source, when supplied, is also checked against that map.
"""

import json
from pathlib import Path


N20_TRAIN_SCHEMA = 'ROOT-N20-paired-local-center-feature-error-training-GPU-spec-v1'


def _n20_front_exact(left, right):
    return json.dumps(left, sort_keys=True, separators=(',', ':'), allow_nan=False) == json.dumps(
        right, sort_keys=True, separators=(',', ':'), allow_nan=False)


def n20_completed_cache_record(record, output_key, owner_index, cuda, gate, spec):
    """Rejoin one original completed cache owner; pending output is not rewritten."""
    require(type(record) is dict and set(record) == {
        'source', 'spec', output_key, 'RAW', 'admission', 'phase_binding', 'bootstrap', 'owner_index'}
        and type(record['owner_index']) is int and record['owner_index'] == owner_index
        and output_key in ('report', 'Q') and cuda == ('0' if output_key == 'report' else ''),
        'Exact eight-field original GPU/CPU cache completion record required')
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
        'Original completed cache owner/source/spec/admission/pins/prefix/output locator must be exact')
    require(output['source'] == record['source'] and output['spec'] == record['spec']
        and output[admission_key] == record['admission'] and output['phase'] == row['phase'],
        'Completed output JSON must bind the same actual original owner')
    return parent, output


def source_gate(path, spec):
    require(spec['schema'] == N20_TRAIN_SCHEMA
        and all(type(spec[key]) is int for key in ('normal_RAW_count', 'expected_normal_RAW_count',
            'total_failed_RAW_count', 'total_terminal_identity_pin_count', 'initialization_seed'))
        and spec['normal_RAW_count'] == spec['expected_normal_RAW_count'] == 140
        and spec['total_failed_RAW_count'] == 23 and spec['total_terminal_identity_pin_count'] == 163
        and spec['initialization_seed'] == 202610150201
        and Path(__file__).resolve(strict=True) == Path(spec['entry_source']['path']).resolve(strict=True),
        'Current N20 first-seed GPU entry needs own140/failed21/pins161 and exact frozen entry')
    failed22 = n20_failed22_raw_only(spec)
    failed23 = n20_failed23_raw_only(spec)
    gate = inherited_source_gate(path, spec, cpu=False)
    n20_failed22_closed(spec, gate, failed22)
    n20_failed23_closed(spec, gate, failed23)
    closure = gate['closure']
    model_keys = ('scout_module_source', 'base_scout_source', 'paired_training_body_source',
                  'original_paired_training_body_source')
    for key in model_keys + ('N20_root_cache_qualified_result', 'N20_initialization_lock',
                            'N20_locked_protocol', 'cache_GPU_report', 'region_common_source'):
        closed(closure, spec[key])
    protocol = read(closed(closure, spec['N20_locked_protocol']))
    require(_n20_front_exact(protocol, spec['expected_N20_locked_protocol'])
        and protocol['schema'] == 'N20-true-feature-neighborhood-fixed24-protocol-v1',
        'New N20 protocol must equal its complete prelocked expected JSON')
    root = read(closed(closure, spec['N20_root_cache_qualified_result']))
    require(_n20_front_exact(root, spec['expected_N20_root_cache_qualified_result'])
        and root['schema'] == 'Root-N20-fullFIT-true-feature-neighbor-cache-post-actual-qualified-v1'
        and root['status'] == 'FULLFIT_TRUE_FEATURE_NEIGHBOR_CACHE_QUALIFIED'
        and root['qualified'] is True and root['original_GPU_CPU_actual0_verified'] is True
        and root['full_proofs_ROOT_verified'] is True
        and root['images'] == 741 and root['sampled_points'] == 3035136
        and root['unique_centers'] == 1786820
        and root['parent_cache'] == spec['cache_GPU_report']
        and root['protocol'] == spec['N20_locked_protocol']
        and root['GPU_report'] == root['gpu_record']['report']
        and root['CPU_Q'] == root['CPU_record']['Q'],
        'Exact fullFIT independently qualified GPU/CPU cache ROOT envelope required')
    gpu_spec, report = n20_completed_cache_record(root['gpu_record'], 'report', 138, '0', gate, spec)
    cpu_spec, cpu_q = n20_completed_cache_record(root['CPU_record'], 'Q', 139, '', gate, spec)
    old_report = read(closed(closure, spec['cache_GPU_report']))
    require(report['schema'] == 'N20-true-feature-neighbor-cache-GPU-report-v1'
        and report['status'] == 'N20_FULLFIT_TRUE_FEATURE_NEIGHBORS_COMPLETE_PENDING_INDEPENDENT_CPU'
        and report['parent_cache'] == spec['cache_GPU_report']
        and report['protocol'] == spec['N20_locked_protocol']
        and cpu_q['schema'] == 'N20-fullFIT-feature-neighbor-cache-CPU-qualification-v1'
        and cpu_q['status'] == 'N20_FULLFIT_FEATURE_NEIGHBOR_CACHE_CPU_COMPLETE_PENDING_ROOT_ACTUAL0'
        and cpu_q['capture_GPU_report'] == root['GPU_report']
        and cpu_q['protocol'] == spec['N20_locked_protocol']
        and cpu_q['qualified'] is False and cpu_q['ROOT_actual_CPU0_verified'] is False
        and cpu_q['original_owned_terminal_self_certified'] is False
        and cpu_q['images'] == 741 and cpu_q['sampled_points'] == 3035136
        and cpu_q['unique_centers'] == 1786820
        and cpu_q['extra_feature_bytes'] == report['extra_feature_bytes']
        and gpu_spec['N20_locked_protocol'] == cpu_spec['N20_locked_protocol'] == spec['N20_locked_protocol'],
        'Original pending GPU report/CPU Q remain unchanged beneath separate ROOT actual0 qualification')
    closed(closure, cpu_q['image_proofs'])
    require(old_report['images'] == report['images'] == 741
        and old_report['sampled_points'] == report['sampled_points'] == 3035136
        and len(old_report['rows']) == len(old_report['packet_refs']) ==
            len(report['rows']) == len(report['packet_refs']) == 741,
        'Complete ordered original and N20 physical cache populations required')
    for index, (oldrow, row) in enumerate(zip(old_report['rows'], report['rows'])):
        require(index == oldrow['dataset_index'] == row['dataset_index']
            and oldrow['packet'] == old_report['packet_refs'][index]
            and row['packet'] == report['packet_refs'][index] and row['parent_packet'] == oldrow['packet']
            and row['identity'] == oldrow['identity'] and row['image_id'] == oldrow['image_id']
            and row['points'] == oldrow['points'] == 4096
            and row['unique_centers'] == oldrow['unique_sampled_F_points'],
            'Each qualified new packet must bind its complete original FIT identity/sample row')
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
        'Only exact new N20 source-only paired initialization lock is allowed')
    initialization_sources = [spec[key] for key in model_keys]
    if 'N20_training_tail_source' in spec:
        closed(closure, spec['N20_training_tail_source'])
        initialization_sources.append(spec['N20_training_tail_source'])
    require(type(metadata['source_sha256']) is dict and all(
        metadata['source_sha256'].get(Path(ref['path']).name) == ref['sha256']
        for ref in initialization_sources), 'Initialization source basenames and all four model hashes must be exact')
    require('parent' in gate and 'lineage' in gate,
            'Preserve inherited endpoint_gate parent for tail final_endpoint_check')
    common = load(spec['region_common_source'], '_N20_original_region_imported_library_gate')
    require(callable(getattr(common, 'imported_files', None)), 'Exact region_common imported_files ABI required')
    gate.update(common=common, parent_cache_report=old_report, parent_cache_report_ref=spec['cache_GPU_report'],
        neighbor_report=report, neighbor_report_ref=root['GPU_report'], protocol=protocol,
        training_binding_refs=dict(N20_root_cache_qualified_result=spec['N20_root_cache_qualified_result'],
            N20_initialization_lock=spec['N20_initialization_lock'], N20_cache_CPU_Q=root['CPU_Q']))
    return gate
